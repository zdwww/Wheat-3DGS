import json
import numpy as np
from pathlib import Path

from functions.toolbox import load_ply, save_ply, apply_buffer_to_bbox, points_in_bbox_wrapper
from functions.point_cloud_processing_utils import extract_id_date
from functions.bbox_functions import find_bbox_json_file

# === USER‐CONFIGURABLE PARAMETERS ===
# I/0 settings
DATA_RELATIVE_PATH = "results/tls_point_clouds_in_pcs_trimmed"
OUTPUT_RELATIVE_PATH = "results/tls_wheat_heads"
BBOXES_RELATIVE_PATH = "results/gs_traits_and_bboxes"
# Parameters
USE_OBB = True  # Process oriented bounding boxes?
USE_AABB = False  # Process axis-aligned bounding boxes? (set one or both to True as needed)
BBOX_BUFFER = [0.01, 0.01, 0.01]  # Buffer (in same units as your point cloud) to expand each bbox
PT_NR_THRESHOLD = 2500  # Threshold passed to points_in_bbox_wrapper


def main(data_relative_path=DATA_RELATIVE_PATH, output_relative_path=OUTPUT_RELATIVE_PATH,
         bboxes_relative_path=BBOXES_RELATIVE_PATH,
         use_obb=USE_OBB, use_aabb=USE_AABB, bbox_buffer=BBOX_BUFFER, pt_nr_threshold=PT_NR_THRESHOLD):
    # Define the folders (adjust the paths as needed)
    project_folder = Path(__file__).parent
    # Folder with full point clouds in .ply format
    pcd_folder = project_folder / data_relative_path
    # Folder with JSON files containing bounding boxes (named like "ID_DATE_bboxes.json")
    bboxes_folder = project_folder / bboxes_relative_path
    # Output folder root for the small point clouds
    output_root = project_folder / output_relative_path

    # Process each point cloud file in pcd_folder
    for pcd_file in pcd_folder.glob("*.ply"):
        try:
            id_date = extract_id_date(pcd_file.stem)
        except ValueError as e:
            print(f"Skipping file {pcd_file.name}: {e}")
            continue

        print(f"Processing scan {id_date} from file {pcd_file.name}")

        # Load the full point cloud (list: [xyz, colors, normals, scalar_fields_dict])
        pcd_data = load_ply(pcd_file)
        # pcd_data[0] is the (N x 3) xyz np.ndarray

        # Load the JSON file with bounding boxes for this scan.
        # bbox_json_file = bboxes_folder / f"{id_date}_bboxes.json"
        bbox_json_file = find_bbox_json_file(bboxes_folder, id_date)

        if not bbox_json_file.exists():
            print(f"Bounding boxes JSON {bbox_json_file.name} not found for {pcd_file.name}. Skipping scan.")
            continue
        with open(bbox_json_file, 'r') as f:
            bboxes_dict = json.load(f)
        # Prepare output folder for this scan: results/TLS_wheat_heads/ID_DATE
        output_scan_folder = output_root / id_date
        output_scan_folder.mkdir(parents=True, exist_ok=True)

        # Loop through each bounding box in the JSON
        for bbox_key, bbox in bboxes_dict.items():
            # Determine the bbox type from the key (e.g. "0045_obb" or "0045_aabb")
            bbox_id = bbox_key[:4]
            bbox_type_indicator = bbox_key.split("_")[-1].lower()

            if ("obb" in bbox_type_indicator and not use_obb) or ("aabb" in bbox_type_indicator and not use_aabb):
                continue  # Skip bounding boxes of types we are not processing

            # Apply buffer to expand the bounding box
            bbox_buffered = apply_buffer_to_bbox(bbox, bbox_buffer)

            # Detect points in bbox (using the (N x 3) xyz array)
            # points_in_bbox_wrapper returns a tuple: (inside_points, mask)
            inside_pts, mask = points_in_bbox_wrapper(pcd_data[0], bbox_buffered, pt_nr_threshold=pt_nr_threshold)
            num_points = np.count_nonzero(mask)
            print(f"  BBox {bbox_key}: {num_points} points selected")
            if num_points == 0:
                continue  # Skip saving an empty point cloud

            # Use the same boolean mask to extract corresponding points from all fields
            new_xyz = pcd_data[0][mask]

            # Inspection of applied bounding boxes:
            # from functions.visualization_utils import plot_2_point_clouds, visualize_bbox_3d
            # from functions.point_cloud_processing_utils import subsample_pcd
            # subsampled_old_pcd = subsample_pcd(pcd_data[0], subsampling_threshold=25_000)
            # subsampled_new_xyz = subsample_pcd(new_xyz, subsampling_threshold=5000)
            # visualize_bbox_3d(subsampled_new_xyz, bbox_buffered)
            # visualize_bbox_3d(subsampled_old_pcd, bbox_buffered)

            new_colors = pcd_data[1][mask] if pcd_data[1] is not None and len(pcd_data[1]) == len(pcd_data[0]) else None
            new_normals = pcd_data[2][mask] if pcd_data[2] is not None and len(pcd_data[2]) == len(
                pcd_data[0]) else None
            new_scalar_fields = {}
            if pcd_data[3] is not None:
                for key, arr in pcd_data[3].items():
                    # Assume each scalar field is an array with the same first dimension as xyz
                    new_scalar_fields[key] = arr[mask]

            # Create the new point cloud list in the same format as load_ply returns it
            new_pcd = [new_xyz, new_colors, new_normals, new_scalar_fields]

            # Build the output filename: "<bbox_id>.ply" (4-digit identifier)
            output_filename = output_scan_folder / f"{bbox_id}.ply"
            save_ply(output_filename, new_pcd)
            print(f"    Saved small point cloud for bbox {bbox_key} as {output_filename.name}")


if __name__ == '__main__':
    main()
