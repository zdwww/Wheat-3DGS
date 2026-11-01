from pathlib import Path
import re

import numpy as np
import open3d as o3d


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
