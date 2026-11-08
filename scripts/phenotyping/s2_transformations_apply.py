"""
Point Cloud Transformation Application Script

This script applies previously estimated transformation parameters on a set of corresponding  point clouds and
transforms them from local coordinate system (LCS) to project coordinate system(PCS), which in this case
corresponds to LV95 (Swiss CS).

Important: The data is stored with global shift (reduction of coordinates) to allow for working with single precision
numbers (float32). Applied shift:  pcd = pcd + [-2693000, -1256000, 0]
"""
from functions.transformations_functions import *
from functions.toolbox import save_ply, load_ply

# === USER‐CONFIGURABLE PARAMETERS ===
DATA_RELATIVE_PATH = "data/tls_point_clouds_in_lcs"
OUTPUT_RELATIVE_PATH = "results/tls_point_clouds_in_pcs"
TRANSFORMATIONS_RELATIVE_PATH = "results/transformations_lcs_2_pcs"


def main(data_relative_path=DATA_RELATIVE_PATH, output_relative_path=OUTPUT_RELATIVE_PATH,
         transformations_relative_path=TRANSFORMATIONS_RELATIVE_PATH) -> None:
    """
    Processes TLS point clouds one by one:
      • Loads a point cloud from the input folder.
      • Extracts the ID_DATE from the filename.
      • Loads the corresponding transformation matrix (from PCS → LCS),
        inverts it to obtain LCS → PCS,
      • Applies the transformation, and
      • Saves the transformed point cloud to the output folder.

    Expected file naming:
      - Point cloud: "ID_DATE" (e.g. "461_1707.ply" or "461_1707.e57")
      - Transformation file: "ID_DATE_transformation.txt"
      - Output file: "ID_DATE_TLS_LV95_Reduced.ply"
    """
    # Set up folder paths.
    project_folder = Path(__file__).parent
    data_folder = project_folder / data_relative_path
    matrices_folder = project_folder / transformations_relative_path
    output_folder = project_folder / output_relative_path
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each point cloud file.
    # (Assuming the files are either .e57 or .ply)
    for pcd_file in data_folder.glob("*.*"):
        if pcd_file.suffix.lower() not in [".e57", ".ply"]:
            continue

        # Extract the ID_DATE substring (3 digits, underscore, 4 digits) from the file name.
        match = re.search(r"(\d{3}_\d{4})", pcd_file.stem)
        if not match:
            print(f"Skipping file {pcd_file.name}: no valid ID_DATE pattern found.")
            continue
        id_date = match.group(1)

        # Build the corresponding transformation file path.
        transformation_file = matrices_folder / f"{id_date}.txt"
        if not transformation_file.exists():
            print(f"Transformation file {transformation_file.name} not found for {pcd_file.name}. Skipping.")
            continue

        # Load the transformation (which is defined from PCS to LCS).
        R, t = load_transformation(transformation_file)
        # Invert the transformation so that it maps from LCS to PCS:
        R_inv = R.T
        t_inv = -R_inv @ t

        # Load the point cloud.
        # pcd = o3d.io.read_point_cloud(str(pcd_file))
        pcd_list = load_ply(pcd_file)

        # Apply the inverted transformation.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_list[0])
        pcd = apply_transformation_to_pcd(pcd, R_inv, t_inv)
        pcd_list[0] = np.asarray(pcd.points)

        # Build the output file name: "ID_DATE_TLS_LV95_Reduced.ply"
        output_file = output_folder / f"{id_date}_pcs.ply"
        # o3d.io.write_point_cloud(str(output_file), pcd)
        save_ply(output_file, pcd_list)
        print(f"Processed {pcd_file.name} -> {output_file.name}")


if __name__ == '__main__':
    main()
