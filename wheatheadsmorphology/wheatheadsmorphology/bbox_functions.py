import json

import numpy as np
import numpy.typing as npt
import open3d as o3d


def extract_bounding_box(data: npt.NDArray, oriented: bool = False) -> dict:
    """
    Extract a bounding box from a 3D point cloud.

    Parameters:
        data (np.ndarray): Nx3 array of 3D points.
        oriented (bool): If True, compute an oriented bounding box (via PCA).
                         Otherwise, compute an axis‐aligned bounding box.

    Returns:
        bbox_data (dict): Dictionary with the minimal parameters to reconstruct the bounding box.
            For an axis‐aligned box, keys are:
                - "type": "axis_aligned"
                - "min_bound": [x_min, y_min, z_min]
                - "max_bound": [x_max, y_max, z_max]
            For an oriented box, keys are:
                - "type": "oriented"
                - "center": [x, y, z]
                - "extent": [width, height, depth] (the full lengths along each local axis)
                - "R": 3x3 rotation matrix (as a list of lists)
    """
    # Create an Open3D point cloud from the numpy array.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])

    if oriented:
        bbox = pcd.get_oriented_bounding_box()
        # Apply buffer by increasing each dimension (both sides) by buffer.

        bbox_data = {
            "type": "oriented",
            "center": list(bbox.center),
            "extent": list(bbox.extent),
            "R": bbox.R.tolist(),  # Convert rotation matrix to list-of-lists
        }
    else:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_data = {
            "type": "axis_aligned",
            "min_bound": list(bbox.min_bound),
            "max_bound": list(bbox.max_bound),
        }

    return bbox_data


def save_bounding_boxes(bboxes: dict, filename: str) -> None:
    """
    Save bounding box data_example for many point clouds to a JSON file.
    Parameters:
        bboxes (dict): Dictionary of bounding boxes, e.g. { id1: bbox_data1, id2: bbox_data2, ... }
        filename (str): Path to the JSON file where the data_example will be stored.
    """
    with open(filename, "w") as f:
        json.dump(bboxes, f, indent=2)


def load_bounding_boxes(filename: str) -> dict:
    """
    Load bounding box data_example from a JSON file.
    Parameters:
        filename (str): Path to the JSON file.
    Returns:
        bboxes (dict): Dictionary of bounding boxes.
    """
    with open(filename, "r") as f:
        bboxes = json.load(f)
    return bboxes


def points_in_bbox(pcd: np.ndarray, bbox: dict) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Determine which points in a point cloud lie within a bounding box (either axis aligned or oriented).
    Parameters:
        pcd (np.ndarray): Nx3 array of 3D points.
        bbox (dict): Bounding box data_example.
            - For an axis-aligned box:
                  keys: 'type', 'min_bound', 'max_bound'
            - For an oriented box:
                  keys: 'type', 'center', 'extent', 'R'
    Returns:
        inside_points (np.ndarray): Points that lie within the bounding box.
        mask (np.ndarray): Boolean mask for the input points.
    """
    pcd = np.asarray(pcd[:, :3])

    if bbox["type"] == "axis_aligned":
        min_bound = np.array(bbox["min_bound"])
        max_bound = np.array(bbox["max_bound"])
        mask = np.all((pcd >= min_bound) & (pcd <= max_bound), axis=1)
        return pcd[mask], mask

    elif bbox["type"] == "oriented":
        center = np.array(bbox["center"])
        extent = np.array(bbox["extent"])
        R = np.array(bbox["R"])
        half_extent = extent / 2.0

        # Transform all points to the OBB's local coordinate frame.
        local_points = (pcd - center) @ R.T
        mask = np.all(np.abs(local_points) <= half_extent, axis=1)
        return pcd[mask], mask

    else:
        raise ValueError("Unknown bounding box type.")


def points_in_oriented_bbox_optimized(
    pcd: np.ndarray, bbox: dict
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Efficiently determine which points in a (potentially huge) point cloud pcd lie within an
    oriented bounding box bbox.

    Parameters:
        pcd (np.ndarray): Nx3 array of points.
        bbox_data (dict): Bounding box data_example (for an oriented box only):
            - 'center': [x, y, z]
            - 'extent': [width, height, depth] (full extents)
            - 'R': 3x3 rotation matrix (list-of-lists)
    Returns:
        inside_points (np.ndarray): Points that lie within the oriented bounding box.
        mask (np.ndarray): Boolean mask for the input points.
        :param pcd:
        :param bbox:
    """
    pcd = np.asarray(pcd[:, :3])

    # Unpack parameters.
    center = np.array(bbox["center"])
    extent = np.array(bbox["extent"])
    R = np.array(bbox["R"])
    half_extent = extent / 2.0

    # Compute the 8 corners in the local frame (each coordinate is ±half_extent).
    corners_local = np.array(
        [
            [sx, sy, sz]
            for sx in (-half_extent[0], half_extent[0])
            for sy in (-half_extent[1], half_extent[1])
            for sz in (-half_extent[2], half_extent[2])
        ]
    )

    # Transform corners to global coordinates.
    # corners_global = (R @ corners_local.T).T + center
    corners_global = (corners_local @ R.T) + center

    # Compute the axis-aligned bounding box (AABB) of these corners.
    aabb_min = np.min(corners_global, axis=0)
    aabb_max = np.max(corners_global, axis=0)

    # Prefilter: select points that fall within this AABB.
    aabb_mask = np.all((pcd >= aabb_min) & (pcd <= aabb_max), axis=1)
    data_prefilter = pcd[aabb_mask]

    # Refined check: transform the prefiltered points to the local coordinate frame.
    local_points = (data_prefilter - center) @ R
    refined_mask = np.all(np.abs(local_points) <= half_extent, axis=1)

    # Build the final mask (initialize all as False, then set the indices from the prefilter).
    final_mask = np.zeros(len(pcd), dtype=bool)
    final_mask[np.nonzero(aabb_mask)[0]] = refined_mask
    return pcd[final_mask], final_mask


def points_in_bbox_wrapper(
    pcd: np.ndarray, bbox: dict, pt_nr_threshold: int = 2500
) -> np.ndarray:
    """
    Wrapper function - choose between the optimized and naive approaches for generating the
    points-in-bounding-box mask and filtering point cloud "pcd" based on this "bbox" mask.

    pt_nr_threshold - threshold for choosing the approach based on the definition of "large" point cloud
    Output: pcd_subset, mask [of len(pcd)]
    """
    if bbox["type"] == "oriented" and pcd.shape[0] > pt_nr_threshold:
        return points_in_oriented_bbox_optimized(pcd, bbox)
    else:
        return points_in_bbox(pcd, bbox)


def apply_buffer_to_bbox(bbox_data: dict, buffer: list) -> dict:
    """
    Apply a buffer (inflate) to an existing bounding box.
        - see extract_bounding_box() output for more details
    Input: buffer size to extend in all directions (unit of bbox)
    Returns: resized_bbox (dict)
    """

    if bbox_data["type"] == "axis_aligned":
        min_bound = np.array(bbox_data["min_bound"]) - np.array(buffer)
        max_bound = np.array(bbox_data["max_bound"]) + np.array(buffer)
        bbox_data["min_bound"] = list(min_bound)
        bbox_data["max_bound"] = list(max_bound)
    elif bbox_data["type"] == "oriented":
        # For an oriented box, simply add 2*buffer to each dimension of the extent.
        extent = np.array(bbox_data["extent"]) + np.array(buffer) * 2
        bbox_data["extent"] = list(extent)
    else:
        raise ValueError("Unknown bounding box type.")

    return bbox_data


def compute_bbox_volume(bbox_data: dict) -> float:
    """
    Computes the volume of a bounding box based on its stored parameters.

    Parameters:
        bbox_data (dict): Dictionary containing bounding box information.
            For an axis-aligned box, expected keys:
                - "type": "axis_aligned"
                - "min_bound": [x_min, y_min, z_min]
                - "max_bound": [x_max, y_max, z_max]
            For an oriented box, expected keys:
                - "type": "oriented"
                - "extent": [width, height, depth] (the full lengths along each local axis)

    Returns:
        float: The volume of the bounding box.
    """
    if bbox_data["type"] == "axis_aligned":
        min_bound = np.array(bbox_data["min_bound"])
        max_bound = np.array(bbox_data["max_bound"])
        differences = max_bound - min_bound
        volume = np.prod(differences)
    elif bbox_data["type"] == "oriented":
        extent = np.array(bbox_data["extent"])
        volume = np.prod(extent)
    else:
        raise ValueError(
            "Unknown bounding box type. Expected 'axis_aligned' or 'oriented'."
        )

    return volume
