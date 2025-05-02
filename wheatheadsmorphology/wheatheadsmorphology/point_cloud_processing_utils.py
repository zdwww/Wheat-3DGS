import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
from scipy import stats


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
        # Subsample data_example using the selected indices
        data = data[indices]
    return data


def main_cluster_extraction(data: np.ndarray, clusterer_definition: dict) -> np.ndarray:
    # Run DBSCAN or HDBSCAN
    algorithm_type = clusterer_definition['type']
    min_samples = clusterer_definition['min_samples']
    cluster_selection_epsilon = clusterer_definition['epsilon_hdbscan']
    if algorithm_type == 'dbscan':
        epsilon = clusterer_definition['epsilon']
        clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)  # Adjust eps and min_samples based on your data_example
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
