"""
Point Cloud Transformation Estimation Script

This script processes sets of corresponding 3D points to estimate and save transformation parameters.

It reads input data (3d points with set_IDs) from CSV files, finds corresponding points and estimates
transformation matrices using the Kabsch algorithm (via scipy.spatial.align_vectors) for each set.

"""

from functions.transformations_functions import *
import numpy as np
from pathlib import Path

# === USER‐CONFIGURABLE PARAMETERS ===
DATA_RELATIVE_PATH = "data/target_centers"
OUTPUT_RELATIVE_PATH = "results/transformations_lcs_2_pcs"
TARGET_CS_POINT_COORDINATES = "target_pcs.csv"
QUERY_CS_POINT_COORDINATES = "query_lcs.csv"


def main(data_relative_path=DATA_RELATIVE_PATH, output_relative_path=OUTPUT_RELATIVE_PATH,
         target_cs_point_coordinates=TARGET_CS_POINT_COORDINATES,
         query_cs_point_coordinates=QUERY_CS_POINT_COORDINATES) -> None:
    # Set up file paths
    project_folder = Path(__file__).parent
    data_folder = project_folder / data_relative_path
    matrices_folder = project_folder / output_relative_path
    matrices_folder.mkdir(parents=True, exist_ok=True)

    # Read input data
    target_data = read_points_from_csv(data_folder / target_cs_point_coordinates)
    query_data = read_points_from_csv(data_folder / query_cs_point_coordinates)

    # Process each set defined with set_IDs
    for set_id in target_data.keys():
        if set_id in query_data:
            # Extract corresponding points
            target_points = np.array(list(target_data[set_id].values()))
            query_points = np.array(list(query_data[set_id].values()))

            # Estimate transformation
            R, t, euler_angles = estimate_transformation(query_points, target_points)

            # Save transformation parameters per set
            save_transformation(R, t, euler_angles, set_id, matrices_folder)


if __name__ == "__main__":
    main()
