#!/usr/bin/env python3

from s1_transformations_estimate import main as estimate_transformations
from s2_transformations_apply import main as apply_transformations
from s3_transform_from_gscs_to_pcs import main as correct_wheat3dgs_cs
from s4_trim_point_clouds_by_proximity import main as trim_by_proximity
from s5_wheatheadmorphology import main as get_wheathead_morphology
from s6_extract_wheat_head_instances import main as get_wheat_heads
from s7_assign_genotype import main as assign_genotype
from s8_combine_plot_files import main as combine_per_plot_csvs
from s9_compute_traits_statistics import main as compute_traits_statistics

def main():
 
    # 1) Estimate transformations
    print("1) Estimating transformations LCS -> PCS")
    estimate_transformations()

    # 2) Apply transformations
    print("2) Applying transformations to TLS point clouds (LCS ->PCS)")
    apply_transformations()

    # 3) Correct Wheat3DGS output for a small shift occurred while running Splatfacto with Nerfstudio (?)
    print("3) Correcting Wheat3DGS results for small shift in coordinate systems relative to PCS")
    correct_wheat3dgs_cs()

    # 4) Trim point clouds to only points close to Wheat3DGS output point clouds (remove > 15 mm away)
    # TLS
    print(" 4) Trim TLS point clouds to only points close to Wheat3DGS output point clouds")
    trim_by_proximity(query_data_relative_path="results/tls_point_clouds_in_pcs",
                      output_relative_path="results/tls_point_clouds_in_pcs_trimmed")
    # MVS
    print("4) Trim MVS point clouds to only points close to Wheat3DGS output point clouds")
    trim_by_proximity(query_data_relative_path="data/mvs_point_clouds_in_pcs",
                      output_relative_path="results/mvs_point_clouds_in_pcs_trimmed")

    # 5) Estimate wheat head morphology of Wheat3DGS output and get bounding boxes
    print(" 5) Get traits and bounding boxes for Wheat3DGS")
    # IMPORTANT! - Running with default settings which are set for Wheat3DGS
    get_wheathead_morphology()

    # 6) Extract wheat head instances:
    # TLS
    print("6) Get wheat head instances from TLS point clouds")
    get_wheat_heads(data_relative_path="results/tls_point_clouds_in_pcs_trimmed",
                    output_relative_path="results/tls_wheat_heads")

    # MVS
    print("6) Get wheat head instances from MVS point clouds")
    get_wheat_heads(data_relative_path="results/mvs_point_clouds_in_pcs_trimmed",
                    output_relative_path="results/mvs_wheat_heads")

    # 5b) Estimate wheat head morphology for TLS and MVS point clouds:
    # TLS
    print("5b) Get traits and bounding boxes for TLS")
    tls_cfg = {
        # DATA (paths & I/O)
        "data_folder": "results/tls_wheat_heads",
        "output_folder": "results/tls_traits_and_bboxes",
        "file_format": "ply",
        "get_bboxes": True,
        "get_processed_pcd": True,
        # TUNING
        "subsampling_threshold": 5000,
        "clusterer_definition": {
            "type": "hdbscan",
            "epsilon": 0.005,
            "min_samples": 25,
            "min_cluster_size": 2000,
            "epsilon_hdbscan": 0.005},
        "sor_parameters": {
            "k": 200,
            "std_ratio": 3},
        "splines_smoothing_value": 1,
        "distance_percentile": 99}
    get_wheathead_morphology(cfg=tls_cfg)

    # MVS
    print("5b) Get traits and bounding boxes for MVS")
    mvs_cfg = {
        # DATA (paths & I/O)
        "data_folder": "results/mvs_wheat_heads",
        "output_folder": "results/mvs_traits_and_bboxes",
        "file_format": "ply",
        "get_bboxes": True,
        "get_processed_pcd": True,
        # TUNING
        "subsampling_threshold": 5000,
        "clusterer_definition": {
            "type": "hdbscan",
            "epsilon": 0.005,
            "min_samples": 15,
            "min_cluster_size": 250,
            "epsilon_hdbscan": 0.015},
        "sor_parameters": {
            "k": 200,
            "std_ratio": 3},
        "splines_smoothing_value": 1,
        "distance_percentile": 99}
    get_wheathead_morphology(cfg=mvs_cfg)

    # 7) Assign genotype (to Wheat3DGS, TLS, MVS)
    print("7) Assign genotypes to Wheat3DGS, TLS, MVS")
    # Wheat3DGS
    assign_genotype(data_relative_path="results/gs_traits_and_bboxes",
                    output_relative_path="results/genotypes_assigned/gs")
    # TLS
    assign_genotype(data_relative_path="results/tls_traits_and_bboxes",
                    output_relative_path="results/genotypes_assigned/tls")
    # MVS
    assign_genotype(data_relative_path="results/mvs_traits_and_bboxes",
                    output_relative_path="results/genotypes_assigned/mvs")

    # 8) Combine per-plot trait + genotypes files
    print("8) Combine per-plot 'trait + genotypes' files into 1 file per method")
    # Wheat3DGS
    combine_per_plot_csvs(data_relative_path="results/genotypes_assigned/gs")
    # TLS
    combine_per_plot_csvs(data_relative_path="results/genotypes_assigned/tls")
    # MVS
    combine_per_plot_csvs(data_relative_path="results/genotypes_assigned/mvs")

    # 9) Compute traits statistics
    print("9) Compute traits statistics - Export Table 3 and 4 (as in paper")
    compute_traits_statistics()
    print("Processing done!")


if __name__ == "__main__":
    main()

