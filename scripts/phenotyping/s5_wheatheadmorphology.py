"""
Wheat head structural trait extraction module (MAIN SCRIPT)

This version scans _all_ immediate subfolders under `data_folder` (e.g. "461_1707", "462_1707", …),
processes all point clouds in each subfolder, and writes one set of outputs per subfolder into the
single `output_folder` defined in the config.

Input:
    - A parent folder containing subfolders named like "461_1707", "462_1707", etc.
    - Inside each subfolder: a set of point clouds of individual wheat heads (*.ply or *.txt)
    - Other hyperparameters as before (in config)

Output (per subfolder):
    - An .xlsx table of structural traits for all wheat heads in that subfolder
    - (Optional) A .json file with all bounding‐boxes
    - (Optional) A single merged .ply of all “processed” clouds (concatenated) for inspection

Authors: Tomislav Medic & ChatGPT, updated June 2025
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import sys
import re

import numpy as np
from functions.traits_extraction_functions import compute_traits
from functions.point_cloud_processing_utils import (
    load_ply, subsample_pcd, main_cluster_extraction,
    statistical_outlier_removal, save_ply
)
from functions.bbox_functions import extract_bounding_box, save_bounding_boxes

# === USER‐CONFIGURABLE PARAMETERS ===
DEFAULT_CFG = {
    # DATA (paths & I/O)
    # This should now be the _parent_ directory containing multiple subfolders (each named like "461_1707", etc.)
    "data_folder": "results/corrected_wheat3dgs_point_clouds_in_pcs",
    "output_folder": "results/gs_traits_and_bboxes",
    "file_format": "ply",          # Allowed: 'ply' or 'txt'
    "get_bboxes": True,            # whether to extract & save bounding boxes
    "get_processed_pcd": True,     # whether to concatenate & save processed PCDs

    # TUNING
    "subsampling_threshold": 5000,
    "clusterer_definition": {
        "type": "hdbscan",
        "epsilon": 0.005,
        "min_samples": 15,
        "min_cluster_size": 250,
        "epsilon_hdbscan": 0.015
    },
    "sor_parameters": {
        "k": 200,
        "std_ratio": 3
    },
    "splines_smoothing_value": 1,
    "distance_percentile": 99
}


def main(cfg: dict = DEFAULT_CFG) -> None:
    # ----------------------------------------------------------------
    # Unpack configuration parameters
    # ----------------------------------------------------------------
    parent_data_folder = Path(cfg['data_folder']).resolve()
    output_folder = Path(cfg['output_folder']).resolve()
    file_format = cfg['file_format']
    subsampling_threshold = cfg['subsampling_threshold']
    clusterer_definition = cfg['clusterer_definition']
    sor_parameters = cfg['sor_parameters']
    splines_smoothing_value = cfg['splines_smoothing_value']
    distance_percentile = cfg['distance_percentile']
    get_bboxes = cfg['get_bboxes']
    get_processed_pcd = cfg['get_processed_pcd']

    # Create the common output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Loop over each immediate subfolder of parent_data_folder
    # ----------------------------------------------------------------
    # We assume each subfolder is named like "461_1707", "462_1707", etc.
    for subfolder in sorted(parent_data_folder.iterdir()):
        if not subfolder.is_dir():
            continue

        subfolder_name = subfolder.name
        print(f"\n--- Processing subfolder: {subfolder_name} ---")

        # ------------------------------------------------------------------------------------
        # Step 1a: Find all point cloud files (*.ply or *.txt) inside this subfolder
        # ------------------------------------------------------------------------------------
        file_paths = list(subfolder.glob(f"*.{file_format}"))
        if len(file_paths) == 0:
            print(f"  No '*.{file_format}' files found in {subfolder_name}, skipping.")
            continue

        # ------------------------------------------------------------------------------------
        # Step 1b: Prepare data structures for this subfolder
        # ------------------------------------------------------------------------------------
        # Create a DataFrame with one row per point‐cloud file
        values_names = ['X', 'Y', 'Z', 'ptnr', 'length', 'width', 'volume', 'inclination', 'curvature']
        results_df = pd.DataFrame(index=range(len(file_paths)), columns=values_names)

        # Optional: collect all bounding boxes in a dict
        all_bboxes = {}

        # Optional: for saving processed PCDs (concatenate all downsampled/cleaned clouds)
        remaining_point_cloud = []
        per_point_file_id = []

        # ------------------------------------------------------------------------------------
        # Step 1c: Iterate over each point cloud in this subfolder
        # ------------------------------------------------------------------------------------
        for file_i, file_path in enumerate(tqdm(file_paths,
                                               desc=f"Processing wheat heads in {subfolder_name}",
                                               total=len(file_paths))):

            # 1c.i: Load the point cloud (either .txt or .ply)
            if file_format == 'txt':
                data_pd = pd.read_csv(file_path, sep=' ', header=0)
                data = data_pd.to_numpy()[:, :3]  # only XYZ columns
            elif file_format == 'ply':
                pcd_data = load_ply(file_path)
                data = pcd_data[0]  # XYZ array
            else:
                sys.exit("Point cloud file_format not supported, supported: .txt, .ply ")

            # 1c.ii: Apply a “global shift” if needed (for numerical stability)
            if np.any(data[0] > 10_000):
                centroid = data.mean(axis=0)
                global_shift = np.floor(centroid / 10_000) * 10_000
                data = data - global_shift
            else:
                global_shift = np.zeros(3)

            # 1c.iii: POINT CLOUD REFINEMENT
            #   1) Subsample if too many points
            data = subsample_pcd(data, subsampling_threshold)

            #   2) Keep only largest cluster (denoise)
            data = main_cluster_extraction(data, clusterer_definition)

            #   3) Statistical outlier removal (SOR)
            if data.shape[0] > sor_parameters['k']:
                data, _ = statistical_outlier_removal(
                    data,
                    k=sor_parameters['k'],
                    std_ratio=sor_parameters['std_ratio']
                )

            # 1c.iv: (Optional) Extract bounding boxes
            if get_bboxes:
                bbox_obb = extract_bounding_box(data, oriented=True)
                bbox_aabb = extract_bounding_box(data, oriented=False)
                key_base = file_path.stem
                all_bboxes[f"{key_base}_obb"] = bbox_obb
                all_bboxes[f"{key_base}_aabb"] = bbox_aabb

            # 1c.v: TRAIT COMPUTATION
            #  - Centroid (apply back global shift)
            centroid_xyz = np.mean(data[:, :3], axis=0) + global_shift
            results_df.loc[file_i, ['X', 'Y', 'Z']] = centroid_xyz
            #  - Number of points
            results_df.loc[file_i, ['ptnr']] = data.shape[0]
            #  - Length, width, volume, inclination, curvature
            results_df.loc[file_i, ['length', 'width', 'volume', 'inclination', 'curvature']] = (
                compute_traits(data, distance_percentile, splines_smoothing_value)
            )

            # 1c.vi: (Optional) Keep processed‐PCD for later merging
            if get_processed_pcd:
                remaining_point_cloud.append(data)
                # If the filename stem is an integer, use that; otherwise keep as string
                try:
                    file_number = int(file_path.stem)
                except ValueError:
                    file_number = file_path.stem
                per_point_file_id.append(np.full((data.shape[0],), file_number))

        # ------------------------------------------------------------------------------------
        # Step 2: After processing all files in this subfolder, SAVE RESULTS
        # ------------------------------------------------------------------------------------
        # 2a) Attach a “file_id” column to results_df (stem or integer)
        file_ids = []
        for file_path in file_paths:
            stem = file_path.stem
            if re.fullmatch(r'\d+', stem):
                file_ids.append(int(stem))
            else:
                file_ids.append(stem)
        results_df['file_id'] = file_ids

        # 2b) Build output filenames: we prefix everything with the subfolder name + timestamp
        current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        traits_filename = f"{subfolder_name}_{current_time}_traits.xlsx"
        bboxes_filename = f"{subfolder_name}_{current_time}_bboxes.json"
        pcd_filename = f"{subfolder_name}_{current_time}_pcd.ply"

        # 2c) Ensure output folder exists (we already made sure above, but just in case)
        output_folder.mkdir(parents=True, exist_ok=True)

        # 2d) Write out the traits DataFrame to Excel
        traits_path = output_folder / traits_filename
        results_df.to_excel(traits_path, index=False)
        print(f"  → Traits saved: {traits_path.name}")

        # 2e) (Optional) Save all bounding boxes to JSON
        if get_bboxes:
            bboxes_path = output_folder / bboxes_filename
            save_bounding_boxes(all_bboxes, str(bboxes_path))
            print(f"  → BBoxes JSON saved: {bboxes_path.name}")

        # 2f) (Optional) Save the concatenated “processed” point cloud (all subfolder files merged)
        if get_processed_pcd and remaining_point_cloud:
            xyz_all = np.vstack(remaining_point_cloud)
            file_id_array = np.concatenate(per_point_file_id)
            file_id_array = file_id_array.astype(np.int32)
            # We attach a scalar field "file_id" so that each point knows which head‐file it came from
            scalar_fields = {'file_id': file_id_array}
            pcd_out_path = output_folder / pcd_filename
            save_ply(pcd_out_path, [xyz_all, None, None, scalar_fields])
            print(f"  → Merged processed‐PCD saved: {pcd_out_path.name}")

        print(f"--- Done with subfolder: {subfolder_name} ---\n")

    return None


if __name__ == "__main__":
    main()
