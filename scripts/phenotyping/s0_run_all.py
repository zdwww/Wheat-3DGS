#!/usr/bin/env python3
"""
Runs the whole TLS comparison (steps 1-9, see Readme.txt) for one set of Wheat3DGS wheat heads.

--data_dir is the root of the data release:
    02_point_clouds/mvs/<plot>_<date>.ply                 dense MVS, in PCS
    02_point_clouds/tls_lcs/<plot>_<date>.ply             TLS, in LCS
    02_point_clouds/transformations/                      target_pcs.csv, query_lcs.csv, tmatrix_4x4_pcs_to_lcs.txt
                                                          (joint matrix for the genotype polygons, fit_joint_pcs_to_lcs.py),
                                                          gscs_to_pcs/ (the paper's ICP matrices, see --icp_dir)
    00_raw/<date>/genotype_polygons/*.json                genotype polygons, in LV95 (--date, default 20240717)

Wheat heads, one of:
    --heads_dir         Wheat3DGS heads only, <plot>_<date>/*.ply, in LV95 (or in PCS with --heads_in_pcs).
                        The TLS and MVS heads are cut out of the plot point clouds around them (steps 2-4 and 6).
    --paper_heads_dir     heads already cut for all three methods, {gs,tls,mvs}/<plot>_<date>/*.ply, in PCS:
                        only steps 1, 5 and 7-9 run. With the paper's heads (<data_dir>/03_results/paper_heads)
                        this reproduces Tables 3 and 4 of the paper.
Use an empty --results_dir: later steps read every file in the output folders of earlier ones.
"""
import argparse
from pathlib import Path

from s1_transformations_estimate import main as estimate_transformations
from s2_transformations_apply import main as apply_transformations
from s3_transform_from_gscs_to_pcs import main as correct_wheat3dgs_cs
from s4_trim_point_clouds_by_proximity import main as trim_by_proximity
from s5_wheatheadmorphology import main as get_wheathead_morphology, DEFAULT_CFG
from s6_extract_wheat_head_instances import main as get_wheat_heads
from s7_assign_genotype import main as assign_genotype
from s8_combine_plot_files import main as combine_per_plot_csvs
from s9_compute_traits_statistics import main as compute_traits_statistics


def parse_args():
    parser = argparse.ArgumentParser(description="TLS comparison of Wheat3DGS wheat heads (steps 1-9)")
    parser.add_argument("--data_dir", required=True, help="root of the data release")
    heads = parser.add_mutually_exclusive_group(required=True)
    heads.add_argument("--heads_dir", help="Wheat3DGS heads, <plot>_<date>/*.ply, in LV95: TLS and MVS heads are cut around them")
    heads.add_argument("--paper_heads_dir", help="heads of all methods, {gs,tls,mvs}/<plot>_<date>/*.ply, in PCS, "
                                               "e.g. the paper's: <data_dir>/03_results/paper_heads")
    parser.add_argument("--results_dir", required=True, help="output folder (empty)")
    parser.add_argument("--icp_dir", default=None,
                        help="<plot>_<date>_ICP.txt matrices applied after LV95 -> PCS: "
                             "<data_dir>/02_point_clouds/transformations/gscs_to_pcs for the paper's reconstruction "
                             "(D1); leave out for one without the principal point shift (D3)")
    parser.add_argument("--heads_in_pcs", action="store_true", help="--heads_dir is already in PCS: skip step 3")
    parser.add_argument("--date", default="20240717", help="acquisition date, for <data_dir>/00_raw/<date>/genotype_polygons")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random subsampling in step 5")
    parser.add_argument("--min_points", type=int, default=50, help="step 5 skips heads with fewer points")
    args = parser.parse_args()
    if args.heads_in_pcs and args.icp_dir:
        parser.error("--icp_dir needs heads in LV95, not --heads_in_pcs")
    if args.paper_heads_dir and (args.icp_dir or args.heads_in_pcs):
        parser.error("--icp_dir and --heads_in_pcs only apply to --heads_dir (--paper_heads_dir is in PCS)")
    return args


def main():
    args = parse_args()
    data = Path(args.data_dir).resolve()
    results = Path(args.results_dir).resolve()
    transformations = data / "02_point_clouds/transformations"
    lcs_2_pcs = str(results / "transformations_lcs_2_pcs")
    gs_bboxes = str(results / "gs_traits_and_bboxes")
    tables = str(results / "traits_and_genotypes_all_methods")
    run_cfg = {"seed": args.seed, "min_points": args.min_points}
    if args.paper_heads_dir:
        paper_heads = Path(args.paper_heads_dir).resolve()
        gs_pcs, tls_heads, mvs_heads = (str(paper_heads / m) for m in ["gs", "tls", "mvs"])
    else:
        gs_pcs = str(Path(args.heads_dir).resolve()) if args.heads_in_pcs else str(results / "corrected_wheat3dgs_point_clouds_in_pcs")
        tls_heads, mvs_heads = str(results / "tls_wheat_heads"), str(results / "mvs_wheat_heads")

    # 1) Estimate transformations (also needed by step 7)
    print("1) Estimating transformations LCS -> PCS")
    estimate_transformations(data_relative_path=str(transformations), output_relative_path=lcs_2_pcs)

    if not args.paper_heads_dir:
        # 2) Apply transformations
        print("2) Applying transformations to TLS point clouds (LCS ->PCS)")
        apply_transformations(data_relative_path=str(data / "02_point_clouds/tls_lcs"),
                              output_relative_path=str(results / "tls_point_clouds_in_pcs"),
                              transformations_relative_path=lcs_2_pcs)

        # 3) Bring Wheat3DGS output to PCS, correcting the small shift if ICP matrices are given
        if not args.heads_in_pcs:
            print("3) Correcting Wheat3DGS results for small shift in coordinate systems relative to PCS")
            correct_wheat3dgs_cs(data_relative_path=str(Path(args.heads_dir).resolve()), output_relative_path=gs_pcs,
                                 transformations_relative_path=str(Path(args.icp_dir).resolve()) if args.icp_dir else None)

        # 4) Trim point clouds to only points close to Wheat3DGS output point clouds (remove > 15 mm away)
        # TLS
        print(" 4) Trim TLS point clouds to only points close to Wheat3DGS output point clouds")
        trim_by_proximity(query_data_relative_path=str(results / "tls_point_clouds_in_pcs"),
                          output_relative_path=str(results / "tls_point_clouds_in_pcs_trimmed"),
                          target_data_relative_path=gs_pcs)
        # MVS
        print("4) Trim MVS point clouds to only points close to Wheat3DGS output point clouds")
        trim_by_proximity(query_data_relative_path=str(data / "02_point_clouds/mvs"),
                          output_relative_path=str(results / "mvs_point_clouds_in_pcs_trimmed"),
                          target_data_relative_path=gs_pcs)

    # 5) Estimate wheat head morphology of Wheat3DGS output and get bounding boxes
    print(" 5) Get traits and bounding boxes for Wheat3DGS")
    # IMPORTANT! - Running with default settings which are set for Wheat3DGS
    get_wheathead_morphology(cfg={**DEFAULT_CFG, **run_cfg, "data_folder": gs_pcs, "output_folder": gs_bboxes})

    if not args.paper_heads_dir:
        # 6) Extract wheat head instances:
        # TLS
        print("6) Get wheat head instances from TLS point clouds")
        get_wheat_heads(data_relative_path=str(results / "tls_point_clouds_in_pcs_trimmed"),
                        output_relative_path=tls_heads, bboxes_relative_path=gs_bboxes)

        # MVS
        print("6) Get wheat head instances from MVS point clouds")
        get_wheat_heads(data_relative_path=str(results / "mvs_point_clouds_in_pcs_trimmed"),
                        output_relative_path=mvs_heads, bboxes_relative_path=gs_bboxes)

    # 5b) Estimate wheat head morphology for TLS and MVS point clouds:
    # TLS
    print("5b) Get traits and bounding boxes for TLS")
    tls_cfg = {
        **DEFAULT_CFG,
        **run_cfg,
        "data_folder": tls_heads,
        "output_folder": str(results / "tls_traits_and_bboxes"),
        "clusterer_definition": {
            "type": "hdbscan",
            "epsilon": 0.005,
            "min_samples": 25,
            "min_cluster_size": 2000,
            "epsilon_hdbscan": 0.005}}
    get_wheathead_morphology(cfg=tls_cfg)

    # MVS
    print("5b) Get traits and bounding boxes for MVS")
    mvs_cfg = {
        **DEFAULT_CFG,
        **run_cfg,
        "data_folder": mvs_heads,
        "output_folder": str(results / "mvs_traits_and_bboxes"),
        "clusterer_definition": {
            "type": "hdbscan",
            "epsilon": 0.005,
            "min_samples": 15,
            "min_cluster_size": 250,
            "epsilon_hdbscan": 0.015}}
    get_wheathead_morphology(cfg=mvs_cfg)

    # 7) Assign genotype and 8) combine per-plot trait + genotypes files (Wheat3DGS, TLS, MVS)
    print("7) Assign genotypes to Wheat3DGS, TLS, MVS")
    print("8) Combine per-plot 'trait + genotypes' files into 1 file per method")
    for method in ["gs", "tls", "mvs"]:
        assigned = str(results / "genotypes_assigned" / method)
        assign_genotype(data_relative_path=str(results / f"{method}_traits_and_bboxes"),
                        genotype_bboxes_relative_path=str(data / "00_raw" / args.date / "genotype_polygons"),
                        output_relative_path=assigned,
                        transformations_lcs2pcs_plots=lcs_2_pcs,
                        transformations_pcs2lcs_all=str(transformations / "tmatrix_4x4_pcs_to_lcs.txt"))
        combine_per_plot_csvs(data_relative_path=assigned, output_relative_path=tables)

    # 9) Compute traits statistics
    print("9) Compute traits statistics - Export Table 3 and 4 (as in paper")
    compute_traits_statistics(data_relative_path=tables, output_relative_path=str(results / "statistics_tables_3_and_4"))
    print("Processing done!")


if __name__ == "__main__":
    main()
