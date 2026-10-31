import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
from scipy import stats
from functions.visualization_utils import *
from plyfile import PlyElement, PlyData
from datetime import datetime
from pathlib import Path
import open3d as o3d
import re


def load_ply(pcd_path: Path, retain_colors: bool = True, retain_normals: bool = True, scalar_fields: list[str] = None,
             normalize_intensities: bool = False, **kwargs) -> list:
    """
    Loads a ply file using *dranjan/python-plyfile*.

    Parameters
    ----------
    pcd_path : pathlib.Path
    scalar_fields : list[str], optional
                    List of scalar fields to keep (will be intersected against the available scalar fields from the
                    *ply-file*). `None` retains all available scalar fields.
    retain_colors : bool, default=True
    retain_normals : bool, default=True
    normalize_intensities: bool, default=False

    Returns
    -------
    pcd : DeSpAn.geometry.PointCloudData
    """
    with open(pcd_path, "rb") as f:
        plydata = PlyData.read(f)
    xyz = np.empty((plydata['vertex'].count, 3,), dtype=float)
    xyz[:, 0] = plydata["vertex"]["x"]
    xyz[:, 1] = plydata["vertex"]["y"]
    xyz[:, 2] = plydata["vertex"]["z"]

    ply_scalar_fields = [pe.name for pe in plydata["vertex"].properties]

    # ply_scalar_fields_lower = [ply_sf.lower() for ply_sf in ply_scalar_fields]
    # scalar_fields = None if scalar_fields is None else [sf for sf in scalar_fields]

    colors = None
    if retain_colors and len(set(ply_scalar_fields) & set(["r", "g", "b", "red", "green", "blue"])) == 3:
        colors = np.empty((plydata['vertex'].count, 3,), dtype=np.uint8)
        colors[:, 0] = plydata["vertex"]["r"] if "r" in ply_scalar_fields else plydata["vertex"]["red"]
        colors[:, 1] = plydata["vertex"]["g"] if "g" in ply_scalar_fields else plydata["vertex"]["green"]
        colors[:, 2] = plydata["vertex"]["b"] if "b" in ply_scalar_fields else plydata["vertex"]["blue"]

    normals = None
    if retain_normals and len(set(ply_scalar_fields) & set(["nx", "ny", "nz"])) == 3:
        normals = np.empty((plydata['vertex'].count, 3,), dtype=float)
        normals[:, 0] = plydata["vertex"]["nx"]
        normals[:, 1] = plydata["vertex"]["ny"]
        normals[:, 2] = plydata["vertex"]["nz"]

    common_scalar_fields = ply_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                set(ply_scalar_fields))

    scalar_fields_dict = dict()
    for sf in common_scalar_fields:
        if sf.lower() not in ["x", "y", "z", "r", "g", "b", "red", "green", "blue", "nx", "ny", "nz"]:
            scalar_fields_dict[sf] = np.array(plydata["vertex"][sf]).squeeze()

    if "scalar_Intensity" in scalar_fields_dict and normalize_intensities:
        scalar_fields_dict["scalar_Intensity"] = (
                scalar_fields_dict["scalar_Intensity"] / scalar_fields_dict["scalar_Intensity"].max())

    return [xyz, colors, normals, scalar_fields_dict]


def save_ply(pcd_path: Path, pcd: list, retain_colors: bool = True, retain_normals: bool = True,
             scalar_fields: list[str] = None) -> None:
    """
    Input:point cloud in a list
    Output: save such a point cloud list in a .ply file format
    """
    nb_points = pcd[0].shape[0]
    dtype_list = [("x", "f4"), ("y", "f4"), ("z", "f4")]


    if retain_colors and pcd[1] is not None:
        assert pcd[1].shape == (nb_points, 3)
        dtype_list.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

    if retain_normals and pcd[2] is not None:
        assert pcd[2].shape == (nb_points, 3)
        dtype_list.extend([("nx", "f8"), ("ny", "f8"), ("nz", "f8")])

    if pcd[3] is not None:
        pcd_scalar_fields = pcd[3].keys()
        common_scalar_fields = pcd_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                    set(pcd_scalar_fields))
        for sf in common_scalar_fields:
            assert pcd[3][sf].shape == (nb_points, )
            dtype_list.append((sf, pcd[3][sf].dtype.str))
    else:
        common_scalar_fields = []

    pcd_np_st = np.empty((nb_points,), dtype=dtype_list)

    pcd_np_st["x"] = pcd[0][:, 0]
    pcd_np_st["y"] = pcd[0][:, 1]
    pcd_np_st["z"] = pcd[0][:, 2]

    if retain_colors and pcd[1] is not None:
        pcd_np_st["red"] = pcd[1][:, 0]
        pcd_np_st["green"] = pcd[1][:, 1]
        pcd_np_st["blue"] = pcd[1][:, 2]

    if retain_normals and pcd[2] is not None:
        pcd_np_st["nx"] = pcd[2][:, 0]
        pcd_np_st["ny"] = pcd[2][:, 1]
        pcd_np_st["nz"] = pcd[2][:, 2]

    for sf in common_scalar_fields:
        pcd_np_st[sf] = pcd[3][sf]

    # TODO: Rename program in comment
    el = PlyElement.describe(pcd_np_st, "vertex", comments=["Created with dranjan/python-plyfile in REASSESS program",
                                                            f"Created {datetime.now():%Y-%m-%dT%H:%M:%S}"])

    if not pcd_path.parent.exists():
        pcd_path.parent.mkdir(parents=True, exist_ok=True)

    PlyData([el]).write(f"{pcd_path}")
    return None


def extract_id_date(filename: str) -> str:
    """
    Extracts the 'ID_DATE' substring (e.g. "461_1707") from a filename.
    Args: filename (str): The filename (without path).
    Returns: str: The extracted identifier.
    """
    match = re.search(r"(\d{3}_\d{4})", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Filename {filename} does not contain a valid ID_DATE pattern.")


def statistical_outlier_removal(data: np.ndarray, k: int = 10, std_ratio: [int, float] = 2.0):
    """
    Perform statistical outlier removal on a point cloud.
    Parameters:
    - point_cloud: numpy array of shape (n_points, 3), the input point cloud.
    - k: int, the number of neighbors to consider for each point.
    - std_ratio: float, the threshold for determining outliers based on standard deviation.
    Returns:
    - filtered_point_cloud: numpy array of the filtered point cloud.
    - outliers: numpy array of the points that were removed.
    """
    point_cloud = data[:, :3]
    # Compute the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(point_cloud)
    distances, _ = nbrs.kneighbors(point_cloud)

    # Exclude the distance to the point itself (first column)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # Compute mean and standard deviation of average distances
    # mean_dist = np.mean(avg_distances)
    # std_dist = np.std(avg_distances)

    mean_dist = np.median(avg_distances)
    std_dist = stats.median_abs_deviation(avg_distances) * 1.4826

    # Define a threshold to detect outliers
    threshold = mean_dist + std_ratio * std_dist

    # Filter points
    mask = avg_distances < threshold
    data_filtered = data[mask]
    data_outliers = data[~mask]

    return data_filtered, data_outliers


def subsample_pcd(data: np.ndarray, subsampling_threshold: float) -> np.ndarray:
    if data.shape[0] > subsampling_threshold:
        # Randomly select subsampling_threshold indices without replacement
        indices = np.random.choice(data.shape[0], subsampling_threshold, replace=False)
        # Subsample data using the selected indices
        data = data[indices]
    return data


def main_cluster_extraction(data: np.ndarray, clusterer_definition: dict) -> np.ndarray:
    # Run DBSCAN or HDBSCAN
    algorithm_type = clusterer_definition['type']
    min_samples = clusterer_definition['min_samples']
    cluster_selection_epsilon = clusterer_definition['epsilon_hdbscan']
    if algorithm_type == 'dbscan':
        epsilon = clusterer_definition['epsilon']
        clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)  # Adjust eps and min_samples based on your data
    elif algorithm_type == 'hdbscan':
        min_cluster_size = clusterer_definition['min_cluster_size']
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    allow_single_cluster=True, cluster_selection_epsilon=cluster_selection_epsilon)
    else:
        print('Incorrect clusterer type, algorithm will crash!')

    labels = clusterer.fit_predict(data[:, :3])

    # Identify the largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    # Retain only points in the largest cluster
    data = data[labels == largest_cluster_label]
    return data


def combine_point_clouds(gs_subfolder: Path) -> o3d.geometry.PointCloud:
    """
    Load all .ply files inside `gs_subfolder` and merge them into a single PointCloud.
    """
    all_points = []
    all_colors = []
    all_normals = []

    for ply_file in gs_subfolder.glob("*.ply"):
        pc = o3d.io.read_point_cloud(str(ply_file))
        pts = np.asarray(pc.points)
        all_points.append(pts)

        # If colors exist, store them
        if pc.has_colors():
            cols = np.asarray(pc.colors)
            all_colors.append(cols)

        # If normals exist, store them
        if pc.has_normals():
            norms = np.asarray(pc.normals)
            all_normals.append(norms)

    if len(all_points) == 0:
        raise RuntimeError(f"No .ply files found in {gs_subfolder}")

    combined_pc = o3d.geometry.PointCloud()
    combined_pc.points = o3d.utility.Vector3dVector(np.vstack(all_points))

    if len(all_colors) > 0:
        combined_pc.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))

    if len(all_normals) > 0:
        combined_pc.normals = o3d.utility.Vector3dVector(np.vstack(all_normals))

    return combined_pc


def filter_pcd_against_gs3d(
    tls_pc: o3d.geometry.PointCloud,
    gs_combined_pc: o3d.geometry.PointCloud,
    distance_threshold: float
) -> o3d.geometry.PointCloud:


    # 1)Compute a list of per‐point distances (length == number of points in tls_pc)
    all_distances = np.asarray(tls_pc.compute_point_cloud_distance(gs_combined_pc))

    # 2) Create a boolean mask of which points to keep
    keep_mask = all_distances <= distance_threshold

    # 3) Build a new filtered PointCloud
    filtered_pc = o3d.geometry.PointCloud()
    filtered_points = np.asarray(tls_pc.points)[keep_mask]
    filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)

    # Copy colors if present
    if tls_pc.has_colors():
        tls_colors = np.asarray(tls_pc.colors)
        filtered_pc.colors = o3d.utility.Vector3dVector(tls_colors[keep_mask])

    # Copy normals if present
    if tls_pc.has_normals():
        tls_normals = np.asarray(tls_pc.normals)
        filtered_pc.normals = o3d.utility.Vector3dVector(tls_normals[keep_mask])

    return filtered_pc
