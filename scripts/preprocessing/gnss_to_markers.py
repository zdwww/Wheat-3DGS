"""
Write the per-plot marker files for metashape_to_colmap.py from a GNSS survey of the targets (e.g. a Trimble export).

The input CSV has one target per line: <plot>-<target>,x,y,z[,...], e.g. "461-1,2693782.343,1256224.338,554.98,".
For each plot it writes <out>/plot_<plot>_coordinates.txt with "target <n>,x,y,z" lines, where "target <n>" is the label
Metashape gives its coded target n. No Metashape needed.

Other coordinates work the same way, e.g. tape measurements in a local system in metres: write the file by hand (or
with this script) and pass --crs local to metashape_to_colmap.py. Use at least 3 targets that are not on one line, a
right-handed system with z pointing up (the 3D segmentation and the trait extraction assume a vertical z).

Usage:
    python scripts/preprocessing/gnss_to_markers.py --gnss_csv SL20240717.csv --out <markers_dir> [--plots 461 462]
"""
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Per-plot Metashape marker files from a GNSS survey CSV")
    parser.add_argument("--gnss_csv", type=str, required=True, help="CSV with <plot>-<target>,x,y,z per line")
    parser.add_argument("--out", type=str, required=True, help="folder for plot_<plot>_coordinates.txt")
    parser.add_argument("--plots", type=str, nargs="*", default=None, help="plots to write (default: all in the CSV)")
    parser.add_argument("--crs", type=str, default="EPSG::2056", help="CRS of the coordinates, written to the header (default: Swiss LV95)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing marker files")
    args = parser.parse_args()

    markers = {}  # plot -> [(target, x, y, z)]
    with open(args.gnss_csv) as f:
        for line in f:
            fields = [v.strip() for v in line.strip().split(",")]
            if len(fields) < 4 or "-" not in fields[0]:
                continue
            plot, target = fields[0].split("-", 1)
            markers.setdefault(plot, []).append((target, *fields[1:4]))
    plots = args.plots if args.plots else sorted(markers)
    missing = [p for p in plots if p not in markers]
    assert not missing, f"Plots {missing} not in {args.gnss_csv}"

    os.makedirs(args.out, exist_ok=True)
    for plot in plots:
        path = os.path.join(args.out, f"plot_{plot}_coordinates.txt")
        if os.path.exists(path) and not args.overwrite:
            print(f"{path} exists, skipping (use --overwrite)")
            continue
        rows = sorted(markers[plot], key=lambda r: (not r[0].isdigit(), int(r[0]) if r[0].isdigit() else r[0]))
        assert len(rows) >= 3, f"Plot {plot} has {len(rows)} targets; georeferencing needs at least 3"
        with open(path, "w") as f:
            f.write(f"# CoordinateSystem: {args.crs}\n")
            f.write("#Label\tX/Easting\tY/Northing\tZ/Altitude\n")
            for target, x, y, z in rows:
                f.write(f"target {target},{x},{y},{z}\n")
        print(f"Wrote {len(rows)} targets of plot {plot} to {path}")
