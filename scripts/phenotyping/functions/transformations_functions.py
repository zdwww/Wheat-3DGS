import re
from typing import Dict, Tuple
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import pandas as pd
import open3d as o3d
import csv

# PCS = LV95 + this shift (coordinates reduced to fit single precision)
LV95_TO_PCS_SHIFT = np.array([-2693000.0, -1256000.0, 0.0])


def read_points_from_csv(file_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Read a CSV file containing point data and organize it by set ID and point ID.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Nested dictionary with set IDs as keys,
        containing dictionaries with point IDs as keys and 3D coordinates as values.
    """
    df = pd.read_csv(file_path)
    data = {}
    for set_id, group in df.groupby('Set_ID'):
        data[set_id] = dict(zip(group['Target_ID'], group[['X', 'Y', 'Z']].values))
    return data


def estimate_transformation(query_points: np.ndarray, target_points: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the transformation parameters between two sets of corresponding points
    using the Kabsch algorithm via scipy.spatial.align_vectors.

    Args:
        query_points (np.ndarray): Array of source/query points (Nx3).
        target_points (np.ndarray): Array of target points (Nx3).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - 3x3 rotation matrix
            - 3x1 translation vector
            - 3x1 array of Euler angles (in radians)
    """
    # A and B are Nx3 matrices of corresponding points
    # Calculate centroids
    centroid_t = np.mean(target_points, axis=0)
    centroid_q = np.mean(query_points, axis=0)

    # Center the points
    t_centered = target_points - centroid_t
    q_centered = query_points - centroid_q

    # Estimate rotation
    rotation, _ = Rotation.align_vectors(t_centered, q_centered)
    R = rotation.as_matrix()

    # Calculate translation
    t = centroid_t - R @ centroid_q

    # Convert rotation matrix to Euler angles
    euler_angles = rotation.as_euler('xyz', degrees=False)

    return R, t, euler_angles


def save_transformation(R: np.ndarray, t: np.ndarray, euler_angles: np.ndarray, set_id: str,
                        output_folder: Path) -> None:
    """
    Save the transformation parameters to a text file.

    Args:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
        euler_angles (np.ndarray): 3x1 array of Euler angles (in radians).
        set_id (str): Identifier for the transformation set.
        output_folder (Path): Folder to save the output file.
    """
    output_file = output_folder / f"{set_id}.txt"
    with open(output_file, 'w') as f:
        f.write("Rotation matrix:\n")
        np.savetxt(f, R, fmt='%.6f')
        f.write("\nTranslation vector:\n")
        np.savetxt(f, t, fmt='%.6f')
        f.write("\nEuler angles (radians):\n")
        np.savetxt(f, euler_angles, fmt='%.6f')
        f.write("\nEuler angles (degrees):\n")
        np.savetxt(f, np.degrees(euler_angles), fmt='%.6f')

    return


def load_transformation(file_path):
    """
    Load transformation matrix data from a .txt file.
        - format is project specific human-readable, not a common 4x4 transformation matrix
        - see examples under "results_transformations/matrices"
    """
    # Initialize variables to store rotation matrix and translation vector
    rotation_matrix = np.zeros((3, 3))
    translation_vector = np.zeros((3,))

    # Read the file and parse the values
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Extract rotation matrix
        rotation_matrix[0] = list(map(float, lines[1].strip().split()))
        rotation_matrix[1] = list(map(float, lines[2].strip().split()))
        rotation_matrix[2] = list(map(float, lines[3].strip().split()))

        # Extract translation vector, skip the header line "Translation vector:"
        translation_vector[:] = list(map(float, lines[6:9]))

    return rotation_matrix, translation_vector


def apply_global_shift(pcd, threshold=1e6, shift=np.array([-2693000, -1256000, 0])):
    """
    If the point cloud is in project coordinate system (PCS), which equals Swiss national CS (LV95),
    apply the global shift (coordinate reduction) to allow working with singles instead of doubles!
    """
    # Convert the point cloud to numpy array for easier manipulation
    points = np.asarray(pcd.points)

    # Check if any x or y coordinates exceed the threshold
    if np.any(np.abs(points[:, 0]) > threshold) or np.any(np.abs(points[:, 1]) > threshold):
        print("Applying global shift to point cloud...")
        # Apply the shift to each point
        points += shift
        # Update the point cloud with shifted points
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        print("No shift needed for point cloud.")

    return pcd


def apply_transformation_to_pcd(pcd: o3d.geometry.PointCloud, R: np.ndarray, t: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Applies a transformation to a point cloud.
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to transform.
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3-element translation vector.
    Returns:
        o3d.geometry.PointCloud: The transformed point cloud.
    """
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    pcd.transform(transformation)
    return pcd


def load_icp_matrix(matrix_file: Path) -> np.ndarray:
    """
    Loads a 4x4 ICP transformation matrix from a text file.
    The file is assumed to contain four rows with four space‐separated numbers each.

    Args:
        matrix_file (Path): Path to the ICP matrix file.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    # You can adjust the reading logic if your file format differs.
    try:
        T = np.loadtxt(matrix_file)
    except Exception as e:
        raise RuntimeError(f"Error loading ICP matrix from {matrix_file}: {e}")

    if T.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix in {matrix_file}, got shape {T.shape}")
    return T


def apply_inverse_transformation_4x4_numpy(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Applies the inverse of a 4x4 transformation matrix T to the xyz points.

    Args:
        xyz (np.ndarray): An (N x 3) array of point coordinates.
        T (np.ndarray): A 4x4 transformation matrix.

    Returns:
        np.ndarray: The transformed (N x 3) array.
    """
    T_inv = np.linalg.inv(T)
    N = xyz.shape[0]
    ones = np.ones((N, 1))
    xyz_hom = np.hstack([xyz, ones])
    xyz_transformed_hom = (T_inv @ xyz_hom.T).T
    return xyz_transformed_hom[:, :3]


def apply_transformation_4x4_numpy(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix T to the xyz points.
    """
    N = xyz.shape[0]
    ones = np.ones((N, 1))
    xyz_hom = np.hstack([xyz, ones])
    xyz_transformed_hom = (T @ xyz_hom.T).T
    return xyz_transformed_hom[:, :3]


def get_4x4_tmatrix_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Created 4x4 transformation matrix from 3x3 Rotation matrix and 3x1 translation vector
    """
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    return transformation


def load_matrix(file_path: Path) -> np.ndarray:
    """
    Loads a transformation matrix from a text file.
    """
    return np.loadtxt(file_path)