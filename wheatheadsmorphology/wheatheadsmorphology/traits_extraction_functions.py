from typing import Tuple, Union
import numpy as np
from wheatheadsmorphology.visualization_utils import *
from scipy.interpolate import splrep, splev
from sklearn.decomposition import PCA
import open3d as o3d


def run_pca(data: np.ndarray) -> Tuple:
    #   - Separate only point cloud coordinates (xyz)
    pcd = data[:, :3]
    #   - Center the point cloud
    pcd_centered = pcd - np.mean(pcd, axis=0)
    #   - Do PCA on the point cloud
    pca = PCA(n_components=3)
    pca.fit(pcd_centered)
    #   - Transform pcd_centered point cloud into PCA-coordinate system
    pcd_pca_3d = pca.transform(pcd_centered)
    return pcd_pca_3d, pca


def compute_length(pcd_pca_3d: np.ndarray, splines_smoothing_value: float) -> Tuple:
    # Calculate Length
    #   - Project points into P1-P2 plane (cross-section of the object)
    pcd_pca_2d = pcd_pca_3d[:, :2]
    #   - Prepare data_example for spline fitting
    x_pca_2d, y_pca_2d = pcd_pca_2d[:, 0], pcd_pca_2d[:, 1]
    sorted_indices = np.argsort(x_pca_2d)
    x_sorted, y_sorted = x_pca_2d[sorted_indices], y_pca_2d[sorted_indices]
    #   - Fit smoothing spline into 2d points within P1-P2 plane
    tck = splrep(x_sorted, y_sorted, s=splines_smoothing_value)
    #   - Evaluate spline (between "robustified" min. and max. of all datapoints along x)
    x_fine = np.linspace(np.percentile(x_sorted, 0.5), np.percentile(x_sorted, 99.5), 1000)
    y_fine = splev(x_fine, tck)
    #   - Combine the x,y coordinates into a single array
    central_axis_points = np.vstack((x_fine, y_fine)).T
    #   - Calculate distances between consecutive points & sum up (integrate)
    diffs = np.diff(central_axis_points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    length = np.sum(segment_lengths)
    return length, tck, central_axis_points


def compute_curvature(length: float, central_axis_points: np.ndarray) -> float:
    # Calculate Chord Length
    start_point = central_axis_points[0]
    end_point = central_axis_points[-1]
    chord_length = np.linalg.norm(end_point - start_point)
    # Calculate Curvature
    curvature_ratio = length / chord_length
    return curvature_ratio


def compute_inclination_angle(pca: PCA) -> float:
    # Compute inclination as the angle between Z-axis and 1st principal component
    p1 = pca.components_[0]  # 1st principal component
    dot_product = np.dot(p1 / np.linalg.norm(p1), np.array([0, 0, 1]))
    inclination_angle_rad = np.arccos(dot_product)
    inclination_angle_deg = np.degrees(inclination_angle_rad)
    return inclination_angle_deg


def estimate_convex_hull_volume_o3d(data: np.ndarray) -> float:
    """
    Estimate the volume of the convex hull of a 3D point cloud using Open3D.
    In -> data_example : np.ndarray Nx3 (point cloud)
    Out -> volume
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])

    hull, _ = pcd.compute_convex_hull()
    hull.orient_triangles()
    volume = hull.get_volume()

    return volume


def compute_width_pca(data: np.ndarray, percentile: float = 95) -> float:
    """
    Compute an alternative width measure for a point cloud using PCA.

    The point cloud is first centered, then PCA is performed to determine the best-fit
    plane spanned by the first two principal components. The orthogonal distance of each
    point to this plane (which is given by the absolute projection onto the third component)
    is computed. A robust radius is estimated as the given percentile (default 95th) of these
    distances, and the width is defined as twice this value.

    Parameters:
        data (np.ndarray): The point cloud data_example (Nx3 or more columns).
        percentile (float): The percentile to use for robust estimation (default is 95).

    Returns:
        float: The computed width (diameter) of the point cloud.
    """
    # Extract the 3D coordinates and center the data_example.
    points = data[:, :3]
    centered_points = points - np.mean(points, axis=0)

    # Perform PCA to obtain the principal components.
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # The third principal component is orthogonal to the plane spanned by the first two.
    normal_vector = pca.components_[2]

    # Compute orthogonal distances of all points from the plane.
    distances = np.abs(np.dot(centered_points, normal_vector))

    # Use the specified percentile to get a robust "radius" estimate.
    robust_radius = np.percentile(distances, percentile)

    # Multiply by 2 to convert radius into a diameter.
    width = 2 * robust_radius
    return width


def compute_traits(data: np.ndarray, distance_percentile: float,
                   splines_smoothing_value: float) -> list:
    # do PCA on 3d point cloud & transform points into pcd_pca_3d
    pcd_pca_3d, pca = run_pca(data)

    # Length from 2D spline representing the skeleton of the object
    length, spline_param, central_axis_points = compute_length(pcd_pca_3d,
                                                               splines_smoothing_value=splines_smoothing_value)

    # Curvature from length and chord distance
    curvature = compute_curvature(length, central_axis_points)

    # Width as robust max. 2d distance between points within slices along the skeleton of the object
    width = compute_width_pca(data, percentile=distance_percentile)

    # Estimate volume from alpha-shapes concave-hull
    try:
        volume = estimate_convex_hull_volume_o3d(data)
    except Exception as e:
        # volume = float('nan')
        volume = 0
        print(f"Error estimating volume: {e}")

    inclination_angle = compute_inclination_angle(pca)

    return [length, width, volume, inclination_angle, curvature]
