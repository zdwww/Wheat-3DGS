#!/usr/bin/env python3
"""
Apply ICP transformations to all .ply point clouds in subfolders.

For each subfolder inside `tls_old/`, this script:
  1) Looks for an ICP matrix named "<subfolder_name>_ICP.txt" in `matrices_ICP/`.
  2) Loads that 4×4 matrix via numpy.loadtxt.
  3) Reads every .ply in the subfolder using Open3D,
     applies the transformation, and writes the result to `tls_new/<subfolder_name>/`.
  4) Preserves each original filename.

Adjust the three Path variables below as needed.
"""

from pathlib import Path
import numpy as np
import open3d as o3d

# === USER‐CONFIGURABLE PARAMETERS ===
# folder containing the original point‐cloud subfolders:
DATA_RELATIVE_PATH = "data/corrupted_point_clouds_in_gscs"
# folder containing ICP matrices, named like "461_1707_ICP.txt", "462_1707_ICP.txt", etc.:
TRANSFORMATIONS_RELATIVE_PATH = "data/transformations_gscs_to_pcs"
# folder where transformed PLYs will be written, preserving subfolder names:
OUTPUT_RELATIVE_PATH = "results/corrected_wheat3dgs_point_clouds_in_pcs"


def main(data_relative_path=DATA_RELATIVE_PATH, output_relative_path=OUTPUT_RELATIVE_PATH,
         transformations_relative_path=TRANSFORMATIONS_RELATIVE_PATH):
    # Set pathlib Paths from strings
    point_clouds_old_root = Path(data_relative_path)
    matrices_folder = Path(transformations_relative_path)
    point_clouds_new_root = Path(output_relative_path)

    # Make sure the output root exists (will also create subfolders on demand).
    point_clouds_old_root.mkdir(parents=True, exist_ok=True)

    # Iterate over each subfolder in tls_old_root
    for subfolder in point_clouds_old_root.iterdir():
        if not subfolder.is_dir():
            continue

        sub_id = subfolder.name  # e.g. "461_1707"
        icp_filename = f"{sub_id}_ICP.txt"
        icp_path = matrices_folder / icp_filename

        if not icp_path.exists():
            print(f"[WARNING] ICP file not found for '{sub_id}': expected '{icp_filename}'. Skipping this subfolder.")
            continue

        # Load the 4×4 transformation matrix
        try:
            T = np.loadtxt(icp_path)
        except Exception as e:
            print(f"[ERROR] Failed to load ICP matrix from '{icp_path}': {e}")
            continue

        if T.shape != (4, 4):
            print(f"[ERROR] ICP matrix '{icp_path}' is not 4×4 (found shape {T.shape}). Skipping '{sub_id}'.")
            continue

        # Create corresponding output subfolder, e.g. "tls_new/461_1707"
        out_subfolder = point_clouds_new_root / sub_id
        out_subfolder.mkdir(parents=True, exist_ok=True)

        # Process every .ply file inside the current subfolder
        ply_files = list(subfolder.glob("*.ply"))
        if not ply_files:
            print(f"[INFO] No .ply files found in '{subfolder}'. Continuing to next.")
            continue

        for ply_path in ply_files:
            try:
                # Read point cloud with Open3D
                pcd = o3d.io.read_point_cloud(str(ply_path))
                if pcd.is_empty():
                    print(f"  [WARNING] '{ply_path.name}' loaded an empty point cloud. Skipping.")
                    continue

                # Apply the 4×4 ICP transformation in place

                pcd.translate([-2693000, -1256000, 0])
                pcd.transform(T)  # modifies pcd.points (and normals, if present)

                # Write the transformed point cloud to the new location
                out_ply = out_subfolder / ply_path.name
                o3d.io.write_point_cloud(str(out_ply), pcd)
                # print(f"  ✔ Transformed '{ply_path.name}' → '{out_ply.relative_to(Path.cwd())}'")

            except Exception as e:
                print(f"  [ERROR] Failed to process '{ply_path.name}': {e}")

    print("Done processing all subfolders.")


if __name__ == "__main__":
    main()
