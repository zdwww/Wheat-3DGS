#!/usr/bin/env python3
"""
This script iterates over .ply files in a designated (TLS) folder, finds the corresponding
subfolder of gs / wheat3dgs point clouds (based on the same identifier_date prefix), merges all wheat head instance
point clouds into one, and then filters out any TLS points whose nearest-neighbor distance
to the merged gs cloud exceeds a given threshold. The filtered clouds are saved to an output folder.
"""

import open3d as o3d
import numpy as np
from pathlib import Path
from functions.point_cloud_processing_utils import combine_point_clouds, filter_pcd_against_gs3d

# === USER‐CONFIGURABLE PARAMETERS ===
# Query data: TLS or MVS point clouds
QUERY_DATA_RELATIVE_PATH = "results/tls_point_clouds_in_pcs"
OUTPUT_RELATIVE_PATH = "results/tls_point_clouds_in_pcs_trimmed"
# Target data: Wheat3DGS output - per wheat head instance point cloud
TARGET_DATA_RELATIVE_PATH = "results/corrected_wheat3dgs_point_clouds_in_pcs"
DISTANCE_THRESHOLD = 0.015  # [m] e.g. 0.015 m (used in the paper)


def main(query_data_relative_path=QUERY_DATA_RELATIVE_PATH, output_relative_path=OUTPUT_RELATIVE_PATH,
         target_data_relative_path=TARGET_DATA_RELATIVE_PATH, distance_threshold=DISTANCE_THRESHOLD):
    # Set parameters
    tls_folder = Path(query_data_relative_path)
    gs_root_folder = Path(target_data_relative_path)
    results_folder = Path(output_relative_path)

    # Make sure results folder exists
    results_folder.mkdir(parents=True, exist_ok=True)

    # Iterate over all TLS .ply files
    for tls_file in tls_folder.glob("*.ply"):
        identifier_date = tls_file.stem[:8]  # e.g. "461_1707"

        # Find the corresponding GS subfolder
        gs_subfolder = gs_root_folder / identifier_date
        if not gs_subfolder.exists() or not gs_subfolder.is_dir():
            print(f"Warning: GS subfolder not found for '{identifier_date}'. Skipping TLS file.")
            continue

        print(f"\nProcessing file: {tls_file.name}")
        print(f" -> GS subfolder: {gs_subfolder.name}")

        # 1) Load and combine GS point clouds
        try:
            gs_combined = combine_point_clouds(gs_subfolder)
        except Exception as e:
            print(f"  Error combining GS clouds: {e}")
            continue

        # 2) Load the TLS point cloud
        tls_pc = o3d.io.read_point_cloud(str(tls_file))
        if len(tls_pc.points) == 0:
            print(f" Point cloud {tls_file.name} has no points. Skipping.")
            continue

        # 3) Filter TLS points by nearest‐neighbor distance
        filtered_tls = filter_pcd_against_gs3d(tls_pc, gs_combined, distance_threshold)

        # 4) Save the filtered TLS cloud
        out_file = results_folder / tls_file.name
        o3d.io.write_point_cloud(str(out_file), filtered_tls)
        print(f" -> Saved filtered point cloud to: {out_file}")


if __name__ == "__main__":
    main()
