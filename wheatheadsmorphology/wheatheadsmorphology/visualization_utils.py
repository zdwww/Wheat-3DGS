import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
matplotlib.use("TkAgg")


def set_axes_equal(ax):
    """Set equal scaling for a 3D plot."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def plot_3d_scatter(points):
    """
    Visualizes a 3D scatter plot of 3D points.
    Parameters:
    points (numpy.ndarray): Nx3 numpy array where each row represents a 3D point (x, y, z).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Scatter plot
    ax.scatter(x, y, z)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Use the custom function to set the axes to equal
    set_axes_equal(ax)
    # Show the plot
    plt.show()
    return

def plot_2_point_clouds(source_points, target_points, source_color='r', target_color='b', title='Point Cloud Comparison'):
    """
    Plots two point clouds using Matplotlib.

    Parameters:
        source_points (numpy.ndarray): The source point cloud as a NumPy array of shape (N, 3).
        target_points (numpy.ndarray): The target point cloud as a NumPy array of shape (M, 3).
        source_color (str): Color for the source point cloud (default is 'r' for red).
        target_color (str): Color for the target point cloud (default is 'b' for blue).
        title (str): Title of the plot (default is 'Point Cloud Comparison').
    """
    # Check if both point clouds have three dimensions
    assert source_points.shape[1] == 3, "Source point cloud must have shape (N, 3)"
    assert target_points.shape[1] == 3, "Target point cloud must have shape (M, 3)"

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the source point cloud
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2],
               c=source_color, label='Source', alpha=0.5)

    #target_points = target_points + 0.05
    # Plot the target point cloud
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
               c=target_color, label='Target', alpha=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add legend
    ax.legend()
    # Use the custom function to set the axes to equal
    set_axes_equal(ax)
    # Show the plot
    plt.show()
    return


def display_histogram(d, bins=10, title="Histogram", xlabel="Values", ylabel="Frequency"):
    """
    Displays a histogram of the values stored in vector d.

    Parameters:
    - d: Input data_example (array-like)
    - bins: Number of bins for the histogram (default is 10)
    - title: Title of the histogram plot (default is "Histogram")
    - xlabel: Label for the x-axis (default is "Values")
    - ylabel: Label for the y-axis (default is "Frequency")
    """
    plt.figure(figsize=(8, 6))
    plt.hist(d, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    return


def compute_bbox_corners(bbox_data):
    """
    Computes the eight corners of a bounding box in a consistent order.
    For an oriented box, the local corners are computed using the half extents
    and then transformed to global coordinates.
    """
    if bbox_data['type'] == 'axis_aligned':
        min_bound = np.array(bbox_data['min_bound'])
        max_bound = np.array(bbox_data['max_bound'])
        # Directly define the corners.
        v0 = np.array([min_bound[0], min_bound[1], min_bound[2]])
        v1 = np.array([max_bound[0], min_bound[1], min_bound[2]])
        v2 = np.array([max_bound[0], max_bound[1], min_bound[2]])
        v3 = np.array([min_bound[0], max_bound[1], min_bound[2]])
        v4 = np.array([min_bound[0], min_bound[1], max_bound[2]])
        v5 = np.array([max_bound[0], min_bound[1], max_bound[2]])
        v6 = np.array([max_bound[0], max_bound[1], max_bound[2]])
        v7 = np.array([min_bound[0], max_bound[1], max_bound[2]])
        corners = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
    elif bbox_data['type'] == 'oriented':
        center = np.array(bbox_data['center'])
        extent = np.array(bbox_data['extent'])
        R = np.array(bbox_data['R'])
        half = extent / 2.0

        # Define local corners using the fixed order.
        v0_local = np.array([-half[0], -half[1], -half[2]])
        v1_local = np.array([half[0], -half[1], -half[2]])
        v2_local = np.array([half[0], half[1], -half[2]])
        v3_local = np.array([-half[0], half[1], -half[2]])
        v4_local = np.array([-half[0], -half[1], half[2]])
        v5_local = np.array([half[0], -half[1], half[2]])
        v6_local = np.array([half[0], half[1], half[2]])
        v7_local = np.array([-half[0], half[1], half[2]])
        local_corners = np.array([v0_local, v1_local, v2_local, v3_local,
                                  v4_local, v5_local, v6_local, v7_local])
        # Transform local corners into global coordinates.
        corners = (R @ local_corners.T).T + center
    else:
        raise ValueError("Unknown bounding box type. Expected 'axis_aligned' or 'oriented'.")
    return corners


def visualize_bbox_3d(point_cloud: np.ndarray, bbox_data: dict) -> None:
    """
    Visualizes a 3D point cloud and its bounding box using Matplotlib.
        bbox_data details in bbox_functions -> extract_bounding_box
    """
    # Visualization parameters.
    point_size = 1
    bbox_opacity = 0.3
    bbox_edge_color = 'k'
    bbox_face_color = 'red'

    # Create figure and 3D axes.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud.
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               s=point_size, color='blue', alpha=0.6)

    # Compute the eight corners of the bounding box.
    corners = compute_bbox_corners(bbox_data)

    # Define the six faces using the indices of the corners.
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom face
        [corners[4], corners[5], corners[6], corners[7]],  # top face
        [corners[0], corners[1], corners[5], corners[4]],  # front face
        [corners[3], corners[2], corners[6], corners[7]],  # back face
        [corners[1], corners[2], corners[6], corners[5]],  # right face
        [corners[0], corners[3], corners[7], corners[4]]  # left face
    ]

    # Create a Poly3DCollection for the bounding box faces.
    bbox_mesh = Poly3DCollection(faces, alpha=bbox_opacity,
                                 facecolor=bbox_face_color, edgecolor=bbox_edge_color)
    ax.add_collection3d(bbox_mesh)

    # Set equal aspect ratio.
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

    # Label axes.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    return
