"""
Wheat head structural trait extraction module (MAIN SCRIPT)
Input:
    A set of point clouds of individual wheat heads (path to folder) + other (hyper-) parameters defined
    in config.yaml file
Output:
    A table of structural traits per wheat head (.xlsx file) + other (optional) outputs
Traits:
    Position [m]: centroid of the point cloud (pcd), X,Y,Z
    Length [m]: 2S Spline length passing through pcd points projected onto P1-P2 plane of PCA
    Width [m]:  point-2-plane distance wrt. plane spawned by P1-P2 plane of PCA
    Volume [m3]: volume of a convex hull (using open3d library -> Qhull algorithm)
    Curvature [a.u.]: 2D spline length vs. chord distance between spline endpoints
    Inclination [deg]: angle between 1st PCA component (P1) and Z-axis
Authors: Tomislav Medic & ChatGPT, updated/shortened for public use on 2nd May 2025
"""
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from wheatheadsmorphology.traits_extraction_functions import compute_traits
from wheatheadsmorphology.point_cloud_processing_utils import *
from wheatheadsmorphology.bbox_functions import extract_bounding_box, save_bounding_boxes
from wheatheadsmorphology.io_utils import load_ply, save_ply
import sys
import re


def run_pipeline(cfg: dict) -> None:

    # Unpack configuration parameters
    data_folder = Path(cfg['data_folder']).resolve()
    output_folder =Path(cfg['output_folder']).resolve()
    file_format = cfg['file_format']
    subsampling_threshold = cfg['subsampling_threshold']
    clusterer_definition = cfg['clusterer_definition']
    sor_parameters = cfg['sor_parameters']
    splines_smoothing_value = cfg['splines_smoothing_value']
    distance_percentile = cfg['distance_percentile']
    get_bboxes = cfg['get_bboxes']
    get_processed_pcd = cfg['get_processed_pcd']

    # -------------------------- START OF THE ALGORITHM --------------------------------------------------

    # Input:
    # find all point cloud files containing individual wheat heads within a designated folder & save paths
    # get a list of all .txt files in the folder
    file_paths = list(data_folder.glob(f"*.{file_format}"))

    # Main Output:
    # create empty Pandas dataframe for storing the results (structural traits values)
    values_names = ['X', 'Y', 'Z', 'ptnr', 'length', 'width', 'volume', 'inclination', 'curvature']
    results_df = pd.DataFrame(index=range(len(file_paths)), columns=values_names)

    # Optional Output 1:
    # create empty dict for storing bounding boxes for wheat head point cloud and saving them in .json file
    all_bboxes = {}

    # Optional Output 2:
    # checking what pre-processing does to the clouds!
    remaining_point_cloud = []
    per_point_file_id = []

    # Iterate over point clouds and extract structural traits:
    for file_i, file_path in enumerate(tqdm(file_paths, desc="Processing wheat heads", total=len(file_paths))):

        # Load point cloud
        if file_format == 'txt':
            data_pd = pd.read_csv(file_path, sep=' ', header=0)
            data = data_pd.to_numpy()
            data = data[:, :3]  # remove any unnecessary scalar fields
        elif file_format == 'ply':
                pcd_data = load_ply(file_path)
                data = pcd_data[0]  # remove any unnecessary scalar fields
        else:
            sys.exit("Point cloud file_format not supported, supported: .txt, .ply ")

        # Apply global shift if necessary (for computational stability)
        if np.any(data[0] > 10_000):
            # compute the centroid of all points
            centroid = data.mean(axis=0)
            # round down each component to the nearest 10 000
            global_shift = np.floor(centroid / 10_000) * 10_000
            # subtract the shift from all points
            data = data - global_shift
        else:
            # no shift needed
            global_shift = np.zeros(3)

        # POINT CLOUD REFINEMENT (pre-processing / de-noising)
        # -----------------------------------------------------
        # 1 - subsample data_example (if necessary)
        data = subsample_pcd(data, subsampling_threshold)

        # 2 - retain only dominant point cluster
        data = main_cluster_extraction(data, clusterer_definition)

        # 3 - remove remaining outliers by "robustified SOR" filter
        if data.shape[0] > sor_parameters['k']:
            data, _ = statistical_outlier_removal(data, k=sor_parameters['k'], std_ratio=sor_parameters['std_ratio'])

        # Optional: Extract bounding boxes (axis aligned - aabb, and oriented - obb)
        # --------------------------------------------------------------------------
        if get_bboxes is True:
            bbox_obb = extract_bounding_box(data, oriented=True)
            bbox_aabb = extract_bounding_box(data, oriented=False)
            all_bboxes[file_path.stem + '_obb'] = bbox_obb
            all_bboxes[file_path.stem + '_aabb'] = bbox_aabb

        # TRAITS EXTRACTION
        # ------------------
        # Save center-point XYZ coordinates (position)
        #   (+ apply back global shift)
        results_df.loc[file_i, ['X', 'Y', 'Z']] = np.mean(data[:, :3], axis=0) + global_shift 
        # Save number of points
        results_df.loc[file_i, ['ptnr']] = data.shape[0]
        # Estimate Length, Width, Volume, Inclination, Curvature
        results_df.loc[file_i, ['length', 'width', 'volume', 'inclination', 'curvature']] = (
            compute_traits(data, distance_percentile, splines_smoothing_value))

        # Optional: Save resulting (processed) point clouds for later visual inspection, e.g. in CloudCompare
        # ---------------------------------------------------------------------------------------------------
        if get_processed_pcd is True:
            remaining_point_cloud.append(data)
            file_number = int(file_path.stem)
            per_point_file_id.append(file_number * np.ones(data.shape[0]))


    # SAVING RESULTS
    # ---------------
    # 1 - Save structural traits into .xlsx file
    # Get a list of str(file_paths) for each wheat head point cloud
    # (+ determine whether each filename stem is integer-like and convert accordingly)
    file_ids = []
    for file_path in file_paths:
        stem = file_path.stem
        if re.fullmatch(r'\d+', stem):
            file_ids.append(int(stem))
        else:
            file_ids.append(stem)

    # Add the file paths as a new column (unique wheat head identifier)
    results_df['file_id'] = file_ids
    # Get the current date and time
    current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    # Create the filenames with the current date and time
    file_path_i = file_paths[0]
    filename = f"{file_path_i.parent.name}_{current_time}_traits.xlsx"
    # Define the directory where you want to save the file
    output_dir = output_folder
    # Create the directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define the full file path
    file_path = output_dir / filename
    # Save the DataFrame to the Excel file
    results_df.to_excel(file_path, index=False)

    # Save bounding boxes to .json file
    if get_bboxes is True:
        filename_bboxes = f"{file_path_i.parent.name}_{current_time}_bboxes.json"
        file_path_bboxes = output_dir / filename_bboxes
        save_bounding_boxes(all_bboxes, str(file_path_bboxes))

    # Save resulting (processed) point clouds for later visual inspection to .ply
    if get_processed_pcd is True:
        xyz = np.concatenate(remaining_point_cloud, axis=0)
        per_point_file_id = np.concatenate(per_point_file_id, axis=0)
        pcd_output_name = f"{file_path_i.parent.name}_{current_time}_pcd.ply"
        output_checking_preprocessing = output_dir / pcd_output_name
        scalar_fields_dict = {'file_id': per_point_file_id}
        save_ply(output_checking_preprocessing, [xyz, None, None, scalar_fields_dict])
        print(f"Done! Results saved to {file_path}")

    return None

    