"""
Export the wheat heads of a 3D segmentation run as point clouds of Gaussian centres in Swiss coordinates (LV95),
one .ply per head, for the trait comparison with the laser scans (wheatheadsmorphology / TLS scripts).

The 3DGS frame is LV95 minus the mean camera centre of the plot, stored in <dataset>/cam_centroid.npy
(LV95 = xyz + cam_centroid, translation only). Coordinates are written as float64: in float32, LV95 eastings
are quantised to 0.25 m.

Heads:
    saved  (default) the per-head files in ply/ written when each head is found, as used in the paper.
           For run_3d_seg.py, a head merged later is saved again as wh_XXXX_a.ply, _b.ply, ...: the last one is used.
    final  the head labels left in gaussians.ply (which_object) at the end of the run; heads don't overlap.

Writes <out>/<NNNN>.ply with x, y, z (double), by default <model_path>/wheat-head/<exp_name>/lv95_<heads>/.

Usage:
    python scripts/phenotyping/export_heads_lv95.py -m <model_path> --exp_name <exp_name> [-s <dataset_path>] [--heads saved|final]
"""
import os
import re
import glob
from argparse import ArgumentParser, Namespace

import numpy as np
from plyfile import PlyData, PlyElement


def read_xyz(path):
    vertex = PlyData.read(path)["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64), vertex


def write_xyz(path, xyz):
    vertices = np.empty(len(xyz), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def saved_heads(seg_dir):
    """{head id: xyz} from ply/wh_XXXX[_<letter>].ply, keeping the last saved version of each head"""
    latest = {}
    for path in sorted(glob.glob(os.path.join(seg_dir, "ply", "wh_*.ply"))):
        match = re.fullmatch(r"wh_(\d+)(?:_([a-z]))?\.ply", os.path.basename(path))
        if match is None:
            continue
        head_id, suffix = int(match.group(1)), match.group(2) or ""
        if head_id not in latest or suffix > latest[head_id][0]:
            latest[head_id] = (suffix, path)
    return {head_id: read_xyz(path)[0] for head_id, (_, path) in sorted(latest.items())}


def final_heads(seg_dir):
    """{head id: xyz} from the which_object labels in gaussians.ply (0 = background)"""
    xyz, vertex = read_xyz(os.path.join(seg_dir, "gaussians.ply"))
    which_object = np.asarray(vertex["which_object"]).astype(int)
    return {int(head_id): xyz[which_object == head_id] for head_id in np.unique(which_object) if head_id > 0}


def find_cam_centroid(args):
    if args.cam_centroid:
        return args.cam_centroid
    source_path = args.source_path
    if not source_path:
        with open(os.path.join(args.model_path, "cfg_args")) as f:
            source_path = eval(f.read()).source_path
    path = os.path.join(source_path, "cam_centroid.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: pass the dataset folder with -s or the file with --cam_centroid")
    return path


if __name__ == "__main__":
    parser = ArgumentParser(description="Export segmented wheat heads in LV95")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--exp_name", required=True, help="segmentation run in <model_path>/wheat-head/")
    parser.add_argument("-s", "--source_path", default="", help="dataset folder with cam_centroid.npy (default: from cfg_args)")
    parser.add_argument("--cam_centroid", default="", help="path to cam_centroid.npy (overrides -s)")
    parser.add_argument("--heads", choices=["saved", "final"], default="saved")
    parser.add_argument("--out", default="", help="output folder (default: <model_path>/wheat-head/<exp_name>/lv95_<heads>)")
    args = parser.parse_args()

    seg_dir = os.path.join(args.model_path, "wheat-head", args.exp_name)
    cam_centroid_path = find_cam_centroid(args)
    offset = np.load(cam_centroid_path).astype(np.float64)
    heads = saved_heads(seg_dir) if args.heads == "saved" else final_heads(seg_dir)
    if not heads:
        raise RuntimeError(f"No wheat heads found in {seg_dir} ({args.heads})")

    out_dir = args.out or os.path.join(seg_dir, f"lv95_{args.heads}")
    os.makedirs(out_dir, exist_ok=True)
    for head_id, xyz in heads.items():
        write_xyz(os.path.join(out_dir, f"{head_id:04}.ply"), xyz + offset)
    print(f"{len(heads)} heads ({args.heads}, {sum(len(xyz) for xyz in heads.values())} Gaussians) "
          f"+ {offset} from {cam_centroid_path} -> {out_dir}")
