"""
Move the origin of a COLMAP text model to the mean camera centre (translation only) and save that offset, as in the
released data: in each plot, sparse/0 is in metres with the origin at the mean camera centre, and cam_centroid.npy
holds the offset (float64), so that  world (e.g. LV95) = model + cam_centroid.

Why: projected coordinates are large (LV95 easting/northing ~ 2.7e6 / 1.3e6 m), and 3DGS stores Gaussian positions in
float32, which resolves only ~0.25 m at 2.7e6 m. Near the origin (a few metres) it's well below a micrometre.

Usage (after metashape_to_colmap.py):
    python scripts/preprocessing/recenter_colmap.py --src <export>/sparse/0 --out <plot>
writes <plot>/sparse/0/{cameras,images,points3D}.txt and <plot>/cam_centroid.npy (cameras.txt unchanged, except that
SIMPLE_PINHOLE is written as the equivalent PINHOLE, the only model the text loader in scene/colmap_loader.py accepts).
"""
import os
from argparse import ArgumentParser

import numpy as np


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_images(path):
    """Header lines and per-image [fields of the pose line, 2D points line]"""
    header, images = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("#"):
            header.append(lines[i])
        if lines[i].startswith("#") or not lines[i].strip():
            i += 1
            continue
        images.append([lines[i].split(), lines[i + 1] if i + 1 < len(lines) else ""])
        i += 2
    return header, images


def camera_center(fields):
    R = qvec2rotmat(np.array(fields[1:5], dtype=np.float64))
    t = np.array(fields[5:8], dtype=np.float64)
    return -R.T @ t, R, t


if __name__ == "__main__":
    parser = ArgumentParser(description="Recentre a COLMAP text model on its mean camera centre")
    parser.add_argument("--src", type=str, required=True, help="folder with cameras.txt, images.txt, points3D.txt")
    parser.add_argument("--out", type=str, required=True, help="plot folder: writes sparse/0/ and cam_centroid.npy there")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing <out>/sparse/0")
    args = parser.parse_args()

    dst = os.path.join(args.out, "sparse", "0")
    assert os.path.abspath(dst) != os.path.abspath(args.src), "--out must not contain --src"
    assert args.overwrite or not os.path.exists(os.path.join(dst, "images.txt")), f"{dst} exists (use --overwrite)"
    os.makedirs(dst, exist_ok=True)

    header, images = read_images(os.path.join(args.src, "images.txt"))
    centroid = np.mean([camera_center(fields)[0] for fields, _ in images], axis=0)

    # Cameras: same rotation, centre c - centroid, so t' = -R (c - centroid) = t + R centroid
    with open(os.path.join(dst, "images.txt"), "w") as f:
        f.write("\n".join(header) + "\n")
        for fields, points2d in images:
            _, R, t = camera_center(fields)
            fields[5:8] = [repr(float(v)) for v in t + R @ centroid]
            f.write(" ".join(fields) + "\n" + points2d + "\n")

    # Points: X - centroid, everything else (color, error, track) unchanged
    with open(os.path.join(args.src, "points3D.txt")) as fin, open(os.path.join(dst, "points3D.txt"), "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            fields = line.split()
            fields[1:4] = [repr(float(v)) for v in np.array(fields[1:4], dtype=np.float64) - centroid]
            fout.write(" ".join(fields) + "\n")

    # Cameras: SIMPLE_PINHOLE (Metashape) -> the same camera as PINHOLE (f, f, cx, cy), which the text loader expects
    with open(os.path.join(args.src, "cameras.txt")) as fin, open(os.path.join(dst, "cameras.txt"), "w") as fout:
        for line in fin:
            fields = line.split()
            if not line.startswith("#") and len(fields) == 7 and fields[1] == "SIMPLE_PINHOLE":
                line = " ".join(fields[:1] + ["PINHOLE"] + fields[2:5] + fields[4:]) + "\n"
            fout.write(line)
    np.save(os.path.join(args.out, "cam_centroid.npy"), centroid)

    # Check: camera centres now average to zero
    _, out_images = read_images(os.path.join(dst, "images.txt"))
    residual = np.mean([camera_center(fields)[0] for fields, _ in out_images], axis=0)
    print(f"{args.src} -> {dst}: {len(images)} cameras, cam_centroid {centroid.tolist()}, new mean camera centre {residual.tolist()}")
