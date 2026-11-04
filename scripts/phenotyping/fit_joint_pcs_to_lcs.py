#!/usr/bin/env python3
"""
Fit the joint PCS -> LCS matrix that s7_assign_genotype.py uses to move the genotype polygons into the TLS
coordinate system (tmatrix_4x4_pcs_to_lcs.txt).

Same target files as s1 (query_lcs.csv: target centres in PCS, target_pcs.csv: the same targets in LCS; the
names follow the original scripts) and the same Kabsch fit, but one rigid transformation over the targets of all
plots of one date. Check the printed residuals: a plot whose targets don't fit (about 2-3 cm is normal) can be left
out with --exclude. For 20240717, plot 466 is 18 cm off and the paper's matrix was fitted without it
(--date 20240717 --exclude 466 reproduces it exactly); on 20240719 all plots fit.

Usage:
    python scripts/phenotyping/fit_joint_pcs_to_lcs.py --transformations_dir <data_dir>/02_point_clouds/transformations \
        --date 20240719 [--exclude <plot> ...] [--out <transformations_dir>/tmatrix_4x4_pcs_to_lcs.txt]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from functions.transformations_functions import estimate_transformation, get_4x4_tmatrix_from_Rt


def main():
    parser = argparse.ArgumentParser(description="Joint PCS -> LCS matrix for the genotype polygons (step 7)")
    parser.add_argument("--transformations_dir", required=True, help="folder with query_lcs.csv and target_pcs.csv")
    parser.add_argument("--date", required=True, help="acquisition date YYYYMMDD, as in s0_run_all.py (Set_IDs <plot>_<DDMM>)")
    parser.add_argument("--exclude", nargs="*", default=[], help="plots left out of the fit, e.g. 466 for 20240717")
    parser.add_argument("--out", default=None, help="default: <transformations_dir>/tmatrix_4x4_pcs_to_lcs.txt")
    args = parser.parse_args()
    if len(args.date) != 8 or not args.date.isdigit():
        parser.error(f"--date must be YYYYMMDD, e.g. 20240719 (got {args.date})")

    folder = Path(args.transformations_dir)
    pcs = pd.read_csv(folder / "query_lcs.csv", encoding="utf-8-sig")
    lcs = pd.read_csv(folder / "target_pcs.csv", encoding="utf-8-sig")
    pairs = pcs.merge(lcs, on=["Set_ID", "Target_ID"], suffixes=("_pcs", "_lcs"))
    plot, date = pairs["Set_ID"].str.split("_").str[0], pairs["Set_ID"].str.split("_").str[1]
    ddmm = args.date[6:8] + args.date[4:6]  # 20240719 -> 1907, as in the Set_IDs
    pairs = pairs[(date == ddmm) & ~plot.isin(args.exclude)]
    if len(pairs) < 3:
        raise SystemExit(f"only {len(pairs)} target pairs for date {args.date} (Set_IDs <plot>_{ddmm})")

    query = pairs[["X_pcs", "Y_pcs", "Z_pcs"]].to_numpy()
    target = pairs[["X_lcs", "Y_lcs", "Z_lcs"]].to_numpy()
    R, t, _ = estimate_transformation(query, target)
    matrix = get_4x4_tmatrix_from_Rt(R, t)

    residuals = np.linalg.norm(query @ R.T + t - target, axis=1) * 1000
    for set_id, res in pd.Series(residuals, index=pairs["Set_ID"].to_numpy()).groupby(level=0):
        print(f"{set_id}: max residual {res.max():.1f} mm")
    out = Path(args.out) if args.out else folder / "tmatrix_4x4_pcs_to_lcs.txt"
    np.savetxt(out, matrix)
    print(f"{len(pairs)} targets of {pairs['Set_ID'].nunique()} plots -> {out}")


if __name__ == "__main__":
    main()
