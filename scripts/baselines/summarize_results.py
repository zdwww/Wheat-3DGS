"""Summarize the nerfstudio baseline runs per plot and method (paper NVS table: SSIM, PSNR, LPIPS, time, storage).

Reads <output_dir>/<plot>/<method>/[...]/<run>/test_results.json (from ns-eval) of the newest run per plot and method, the
training time (train_seconds.txt from run_baseline.sh, else the wandb summary) and the checkpoint size. Writes
<csv_dir>/<method>_per_scene_metrics.csv, <method>_summary_metrics.csv and all_methods_summary.csv, and prints the means.

    python scripts/baselines/summarize_results.py <output_dir> [--methods nerfacto splatfacto ...] [--csv_dir <dir>]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["PSNR", "SSIM", "LPIPS", "FPS", "Runtime (min)", "Storage (GB)"]


def read_run(run: Path):
    with open(run / "test_results.json") as f:
        results = json.load(f)["results"]
    row = {"PSNR": results.get("psnr"), "SSIM": results.get("ssim"), "LPIPS": results.get("lpips"),
           "FPS": results.get("fps")}
    seconds = None
    if (run / "train_seconds.txt").exists():
        seconds = float((run / "train_seconds.txt").read_text())
    else:
        summary = run / "wandb" / "latest-run" / "files" / "wandb-summary.json"
        if summary.exists():
            seconds = json.loads(summary.read_text()).get("_runtime")
    row["Runtime (min)"] = int(seconds / 60) if seconds is not None else None
    checkpoints = sorted((run / "nerfstudio_models").glob("*.ckpt"))
    row["Storage (GB)"] = round(checkpoints[-1].stat().st_size / 1024**3, 3) if checkpoints else None
    return row


def summarize_method(output_dir: Path, method: str, csv_dir: Path):
    rows = []
    for plot_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        runs = sorted((r.parent for r in (plot_dir / method).rglob("test_results.json")), key=lambda r: r.name)
        if not runs:
            continue
        rows.append({"Scene": plot_dir.name, "Method": method, "Run": runs[-1].name, **read_run(runs[-1])})
    if not rows:
        print(f"No runs with test_results.json for {method} in {output_dir}")
        return None
    per_scene = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"Metric": "Average", "Method": method, **per_scene[METRICS].mean().round(3).to_dict()},
        {"Metric": "Std Dev", "Method": method, **per_scene[METRICS].std(ddof=0).round(3).to_dict()},
    ])
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_scene.to_csv(csv_dir / f"{method}_per_scene_metrics.csv", index=False)
    summary.to_csv(csv_dir / f"{method}_summary_metrics.csv", index=False)
    print(per_scene.to_string(index=False))
    return summary.assign(Plots=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="output folder of run_baseline.sh (<plot>/<method>/<run>/)")
    parser.add_argument("--methods", nargs="+",
                        default=["instant-ngp", "nerfacto", "fruit_nerf", "fruit_nerf_big", "splatfacto",
                                 "splatfacto-sparse_pc"])
    parser.add_argument("--csv_dir", type=Path, help="where to write the CSVs (default: <output_dir>/summary)")
    args = parser.parse_args()
    csv_dir = args.csv_dir or args.output_dir / "summary"

    summaries = [s for m in args.methods if (s := summarize_method(args.output_dir, m, csv_dir)) is not None]
    if summaries:
        all_summaries = pd.concat(summaries)
        all_summaries.to_csv(csv_dir / "all_methods_summary.csv", index=False)
        means = all_summaries[all_summaries.Metric == "Average"].drop(columns="Metric")
        print("\nMean over plots:\n" + means.to_string(index=False))
        print(f"\nCSVs in {csv_dir}")


if __name__ == "__main__":
    main()
