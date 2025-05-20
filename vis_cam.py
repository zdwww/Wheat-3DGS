import numpy as np
import collections
import struct
import open3d as o3d
import matplotlib.pyplot as plt

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_extrinsics_text(path):
    images = {}
    cam_centers = {}

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                
                R = np.transpose(qvec2rotmat(qvec))
                T = np.array(tvec)

                camera_id = int(elems[8])
                image_name = elems[9]

                cam_center = -R.dot(T)
                cam_centers[image_name] = cam_center

                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images, cam_centers

if __name__ == "__main__":
    xyz, _, _ = read_points3D_text("plot_461/sparse/0/points3D.txt")
    print(f"\nxyz centroid {np.mean(xyz, axis=0)}\nMin{np.min(xyz, axis=0)}\nMax{np.max(xyz, axis=0)}")

    _, cam_centers = read_extrinsics_text("plot_461/sparse/0/images.txt")
    cam_centers_lst = np.stack(list(cam_centers.values()), axis=0)
    print(f"\nCam Centroid{np.mean(cam_centers_lst, axis=0)}\nMin{np.min(cam_centers_lst, axis=0)}\nMax{np.max(cam_centers_lst, axis=0)}")

    labels = list(cam_centers.keys())
    labels = [label.replace("FPWW036_SR0461_", "").replace("FIP2_", "").replace(".png", "") for label in labels]
    print(labels)
    coordinates = list(cam_centers.values())
    
    # Convert coordinates to separate lists for x, y, and z
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    z_coords = [coord[2] for coord in coordinates]

    # Create a 3D plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot the points
    # ax.scatter(x_coords, y_coords, z_coords, color='b', marker='o')

    # Add labels to each point
    for label, (x, y, z) in zip(labels, coordinates):
        if label.startswith("c"):
            color = "red"
        elif label.startswith("1"):
            color = "green"
        else:
            color = "blue"
        ax.scatter(x, y, z, color=color, marker='o')  
        ax.text(x, y, z, label, color=color, fontsize=7)  
    
    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points with Labels from Dictionary')

    plt.show()