#!/usr/bin/env python3
"""
Script for assigning genotype IDs to traits.
For each CSV file in traits_folder:
  - Loads the CSV and extracts (X, Y, Z) coordinates.
  - Applies a global shift and a transformation matrix (target-based PRS to LCS)
    using matrices loaded from corresponding files (named based on the three-digit ID).
For each matching JSON file in genotype_bboxes_folder:
  - Loads the JSON (GeoJSON-like) file, extracts its polygon coordinates,
    applies the same global shift and a different transformation matrix (from lv95_2_tls_matrix.txt)
    after expanding the 2D coordinates to 3D.
Then, for each point from the traits CSV, the script checks which transformed polygon (of 6)
contains the point (using the X–Y coordinates) and assigns a genotype identifier of the form “461_1”.
The new column “gen_row_id” is added to the dataframe and saved to a new file in results/genotypes_assigned/.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from functions.transformations_functions import (load_transformation, apply_transformation_4x4_numpy,
                                                 get_4x4_tmatrix_from_Rt, load_matrix, LV95_TO_PCS_SHIFT)
from functions.assign_genotype_utils import *
from itertools import chain

# === USER‐CONFIGURABLE PARAMETERS ===
DATA_RELATIVE_PATH = "results/gs_traits_and_bboxes"
GENOTYPE_BBOXES_RELATIVE_PATH = "data/genotype_bboxes_in_lv95"
OUTPUT_RELATIVE_PATH = "results/genotypes_assigned/gs"
TRANSFORMATIONS_LCS_2_PCS_PER_PLOT_RELATIVE_PATH = "results/transformations_lcs_2_pcs"
TRANSFORMATIONS_PCS_2_LCS_JOINT_RELATIVE_PATH = "data/transformations_pcs_to_lcs/tmatrix_4x4_pcs_to_lcs.txt"


def main(data_relative_path=DATA_RELATIVE_PATH, genotype_bboxes_relative_path=GENOTYPE_BBOXES_RELATIVE_PATH,
         output_relative_path=OUTPUT_RELATIVE_PATH,
         transformations_lcs2pcs_plots=TRANSFORMATIONS_LCS_2_PCS_PER_PLOT_RELATIVE_PATH,
         transformations_pcs2lcs_all=TRANSFORMATIONS_PCS_2_LCS_JOINT_RELATIVE_PATH):
    # Define global shift
    global_shift = LV95_TO_PCS_SHIFT

    # Define paths
    project_folder = Path(__file__).parent.resolve()
    traits_folder = Path(data_relative_path)   # update with your CSV folder path
    genotype_bboxes_folder = Path(genotype_bboxes_relative_path)  # update with your JSON folder path

    # Folder with transformation matrices pcs-2-lcs
    tb_matrices_folder = project_folder / transformations_lcs2pcs_plots

    # Transformation matrix for JSON polygons
    pcs2lcs_tmatrix_path = project_folder / transformations_pcs2lcs_all
    pcs2lcs_tmatrix = load_matrix(pcs2lcs_tmatrix_path)

    # Output folder
    output_folder = project_folder / output_relative_path
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each CSV file in traits_folder
    for traits_file in chain(traits_folder.glob("*.xlsx"), traits_folder.glob("*.csv")):
        print(f"Processing traits file: {traits_file.name}")
        # Extract three-digit identifier from CSV file name (e.g., "461")
        #m = re.match(r"(\d{3})", traits_file.stem)
        # Extract the part before the first underscore
        prefix = traits_file.stem.split("_")[0]

        # Search for a three-digit number within that prefix
        m = re.search(r"(\d{3})", prefix)


        if not m:
            print(f"Could not extract identifier from {traits_file.name}, skipping.")
            continue
        identifier = m.group(1)

        # Find corresponding JSON file in genotype_bboxes_folder containing pattern "SR0<identifier>"
        json_pattern = f"*SR0{identifier}*json"
        json_files = list(genotype_bboxes_folder.glob(json_pattern))
        if not json_files:
            print(f"No matching JSON file found for identifier {identifier} (pattern {json_pattern}), skipping.")
            continue
        json_file = json_files[0]  # assuming one match
        print(f"  Found JSON file: {json_file.name}")

        # --- Process traits CSV file ---
        try:
            if "csv" in str(traits_file):
                df_traits = pd.read_csv(traits_file)
            elif "xlsx" in str(traits_file):
                df_traits = pd.read_excel(traits_file)
            else:
                print("No csv or xlsx file with traits")

        except Exception as e:
            print(f"Error reading {traits_file.name}: {e}")
            continue

        # Check if required columns exist
        if not set(['X', 'Y', 'Z']).issubset(df_traits.columns):
            print(f"Columns X, Y, Z not found in {traits_file.name}, skipping.")
            continue

        # Extract coordinates and apply global shift
        xyz = df_traits[['X', 'Y', 'Z']].values.astype(float)
        # xyz += global_shift

        # Load and apply target-based transformation matrix (from file: "{identifier}_transformation.txt")
        identifier_and_date = identifier + "_1707"
        tb_file = tb_matrices_folder / f"{identifier_and_date}.txt"
        if not tb_file.exists():
            print(f"Target-based matrix file {tb_file.name} not found for {traits_file.name}, skipping.")
            continue
        R, t = load_transformation(tb_file)
        tb_matrix = get_4x4_tmatrix_from_Rt(R, t)
        xyz = apply_transformation_4x4_numpy(xyz, tb_matrix)
        # Now, xyz contains the transformed trait coordinates.

        # --- Process JSON file for genotype bounding boxes ---
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {json_file.name}: {e}")
            continue

        # Dictionary to store transformed polygons with their polygon number (as string)
        polygons = {}

        for feature in json_data.get("features", []):
            # Extract the id string (e.g., "FPWW036_SR0461_1_FIP2")
            prop_id = feature.get("properties", {}).get("id", "")
            # Extract polygon number (assumes the third part when splitting by "_" gives the number)
            parts = prop_id.split("_")
            if len(parts) < 3:
                print(f"Unexpected id format in JSON feature: {prop_id}, skipping feature.")
                continue
            poly_num = parts[2]  # e.g., "1" (expected to be from 1 to 6)

            # Extract polygon coordinates
            # Assuming geometry is like: { "type": "Polygon", "coordinates": [ [ [x,y], ... ] ] }
            try:
                poly_coords = feature["geometry"]["coordinates"][0]  # outer ring
            except Exception as e:
                print(f"Error extracting polygon coordinates for feature {prop_id}: {e}")
                continue
            poly_coords = np.array(poly_coords, dtype=float)  # shape (n_points, 2)

            # Expand to 3D (by adding a zero for Z)
            n_points = poly_coords.shape[0]
            poly_coords_3d = np.hstack([poly_coords, np.ones((n_points, 1))*554.9])
            # Apply the same global shift
            poly_coords_3d += global_shift

            # Apply lv95 transformation (the one for JSON polygons)
            poly_coords_transformed = apply_transformation_4x4_numpy(poly_coords_3d, pcs2lcs_tmatrix)
            # poly_coords_transformed = apply_transformation_4x4_numpy(poly_coords_3d, np.eye(4))
            # poly_coords_transformed = apply_transformation_4x4_numpy(poly_coords_3d, tb_matrix)
            # Convert back to 2D by taking the X, Y components
            poly_xy = poly_coords_transformed[:, :2]
            # Save in dictionary with key poly_num
            polygons[poly_num] = poly_xy

        if not polygons:
            print(f"No valid polygons found in {json_file.name}, skipping.")
            continue

        # --- Assign genotype row IDs ---
        # For each trait coordinate (from xyz) use its X, Y components to check inclusion in polygons.
        all_poly_xy = [polygons[str(i+1)] for i in range(6)]
        all_poly_xy = np.concatenate(all_poly_xy)

        # plot_polygon_and_traits(all_poly_xy, xyz[:, :2])

        gen_row_ids = []
        for pt in xyz:
            pt_xy = pt[:2]
            assigned_id = None
            for poly_num, poly_xy in polygons.items():
                if is_point_in_polygon(pt_xy, poly_xy):
                    assigned_id = f"{identifier}_{poly_num}"
                    break  # assume one match per point
            gen_row_ids.append(assigned_id)

        # Add new column to the dataframe
        df_traits["gen_row_id"] = gen_row_ids

        # Save the new dataframe to output folder with filename: original name + "_ga.csv"
        output_filename = traits_file.stem + "_ga.csv"
        output_path = output_folder / output_filename
        try:
            df_traits.to_csv(output_path, index=False)
            print(f"Saved genotype-assigned file: {output_filename}")
        except Exception as e:
            print(f"Error saving file {output_filename}: {e}")

        #  Check for plausability
        # create_scatter_plot_random_colors(df_traits, 'gen_row_id')


if __name__ == "__main__":
    main()
