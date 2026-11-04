"""Prepare a plot of the Wheat3DGS dataset for the nerfstudio baselines (run inside the baselines Docker image).

nerfacto, splatfacto (random init) and FruitNeRF read transforms.json (nerfstudio format, with the train/test split and
the FruitNeRF semantics). The original dataset ships it; for a plot without it (e.g. the corrected dataset) it is
written from sparse/0. This script writes:

  transforms.json                    if the plot has none: from sparse/0 like instant-ngp's colmap2nerf.py (which made
                                     the released ones; scene at the origin, as blender-data and splatfacto's random
                                     init expect), test views = cameras 11 and 12 (the split of the 3DGS code,
                                     scene/dataset_readers.py), semantic_path if there are semantics
  semantics/<image>.png              with --semantics_from <masks_dir>: per image the union of the instance masks
                                     <masks_dir>/masks/<image>_NNN.png (how the released semantics were made)
  transforms_{train,val,test}.json   instant-ngp with the `blender-data` dataparser, as in the paper
                                     (blender format: image paths without extension, val = test)
  sparse_pc.ply                      with --sparse_pc: the COLMAP points of sparse/0 in the frame of transforms.json,
                                     which then gets "ply_file_path" (splatfacto starts from them; nerfacto, FruitNeRF
                                     and instant-ngp ignore them)

Only instant-ngp needs the extra transforms_*.json; the rest read transforms.json.

Files go into the plot folder, or into <output_dir>/<plot> with --output_dir (then `images` and `semantics` symlinks
point to the plot's folders, and a transforms.json there takes precedence over the plot's; the dataset can stay
read-only). The symlinks
are relative: in Docker, mount the dataset and the output at the same paths for this script and for training.

    python scripts/baselines/prepare_data.py data/plot_46{1..7} [--sparse_pc] [--semantics_from yolosam_640]
        [--output_dir <dir>]
"""
import argparse
import copy
import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def link_images(plot_dir: Path, dest: Path, name="images"):
    """`dest/<name>` -> the plot's folder (relative symlink), unless dest is the plot folder itself"""
    if dest.resolve() == plot_dir.resolve():
        return
    link = dest / name
    if link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(plot_dir / name, dest))


def _rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (np.linalg.norm(v) ** 2 + 1e-10))


def _closest_point_2_lines(oa, da, ob, db):
    da, db = da / np.linalg.norm(da), db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = min(np.linalg.det([t, db, c]) / (denom + 1e-10), 0)
    tb = min(np.linalg.det([t, da, c]) / (denom + 1e-10), 0)
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def colmap2nerf_transforms(plot_dir: Path, aabb_scale=32, sharpness=True):
    """transforms.json from sparse/0 like instant-ngp's scripts/colmap2nerf.py, which made the released ones: OpenGL
    cameras, average camera up -> +z, the point the cameras look at -> origin, cameras on average 4 units from it.
    Also returns the COLMAP points in that frame (xyz, rgb)."""
    from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_model

    cameras, images, points3D = read_model(str(plot_dir / "sparse" / "0"))
    assert len(cameras) == 1, "one camera model expected"
    cam = next(iter(cameras.values()))
    assert cam.model in ("PINHOLE", "SIMPLE_PINHOLE"), f"undistorted cameras expected, got {cam.model}"
    fl_x, fl_y, cx, cy = cam.params if cam.model == "PINHOLE" else (cam.params[0], cam.params[0], *cam.params[1:])
    frames, up = [], np.zeros(3)
    for image in images.values():  # images.txt order, as colmap2nerf.py
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = qvec2rotmat(image.qvec), image.tvec
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1  # OpenCV -> OpenGL camera
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1
        up += c2w[0:3, 1]
        frame = {"file_path": f"./images/{image.name}"}
        if sharpness:
            gray = cv2.cvtColor(cv2.imread(str(plot_dir / "images" / image.name)), cv2.COLOR_BGR2GRAY)
            frame["sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()
        frames.append(dict(frame, transform_matrix=c2w))
    R = np.eye(4)
    R[:3, :3] = _rotmat(up, [0, 0, 1])
    for f in frames:
        f["transform_matrix"] = R @ f["transform_matrix"]
    totw, totp = 0.0, np.zeros(3)
    for f in frames:
        mf = f["transform_matrix"][0:3, :]
        for g in frames:
            mg = g["transform_matrix"][0:3, :]
            p, w = _closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.00001:
                totp += p * w
                totw += w
    for f in frames:
        f["transform_matrix"][0:3, 3] -= totp / totw
    avglen = np.mean([np.linalg.norm(f["transform_matrix"][0:3, 3]) for f in frames])
    for f in frames:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen
        f["transform_matrix"] = f["transform_matrix"].tolist()
    # the same change of world frame for the points: row swap/flip above, rotation, shift, scale
    xyz = np.array([p.xyz for p in points3D.values()])
    xyz = (4.0 / avglen) * ((R[:3, :3] @ (xyz[:, [1, 0, 2]] * [1, 1, -1]).T).T - totp / totw)
    rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)
    w, h = cam.width, cam.height
    return (xyz, rgb), {"camera_angle_x": 2 * math.atan(w / (2 * fl_x)), "camera_angle_y": 2 * math.atan(h / (2 * fl_y)),
            "fl_x": fl_x, "fl_y": fl_y, "k1": 0, "k2": 0, "k3": 0, "k4": 0, "p1": 0, "p2": 0, "is_fisheye": False,
            "cx": cx, "cy": cy, "w": float(w), "h": float(h), "aabb_scale": aabb_scale, "frames": frames}


def split_names(transforms):
    """Image file names per split, from train/val/test_filenames"""
    names = {split: {Path(p).name for p in transforms[f"{split}_filenames"]} for split in ("train", "val", "test")}
    assert names["train"] and names["test"] and not names["train"] & names["test"], "bad train/test split"
    return names


def set_split(transforms, names):
    """train/val/test_filenames in the paths of these transforms"""
    paths = [frame["file_path"] for frame in transforms["frames"]]
    assert {Path(p).name for p in paths} == set().union(*names.values()), "the images in sparse/0 and the split differ"
    for split in ("train", "val", "test"):  # in frame order, like the released transforms.json
        transforms[f"{split}_filenames"] = [p for p in paths if Path(p).name in names[split]]


def camera_split(transforms):
    """Test views = cameras 11 and 12 (image names end in _cam_<NN>), as in scene/dataset_readers.py; val = test"""
    names = [Path(frame["file_path"]).name for frame in transforms["frames"]]
    test = {n for n in names if int(Path(n).stem.split("_")[-1]) > 10}
    return {"train": set(names) - test, "val": test, "test": test}


def write_semantics(plot_dir: Path, masks_dir: str, dest: Path):
    masks = plot_dir / masks_dir
    masks = masks / "masks" if (masks / "masks").is_dir() else masks
    assert masks.is_dir(), f"No masks in {plot_dir / masks_dir}"
    out = dest / "semantics"
    assert not out.exists(), f"{out} exists already; use --output_dir to write a new set of semantics elsewhere"
    out.mkdir(parents=True)
    images = sorted((plot_dir / "images").glob("*.png"))
    for image in images:
        w, h = Image.open(image).size
        union = np.zeros((h, w), dtype=bool)
        for mask in glob.glob(str(masks / f"{image.stem}_[0-9][0-9][0-9].png")):
            union |= np.array(Image.open(mask).convert("L")) > 127
        Image.fromarray(union.astype(np.uint8) * 255).convert("RGB").save(out / image.name)  # RGB like the released ones
    print(f"  semantics/: {len(images)} images from {masks}")


def add_semantic_paths(transforms, dest: Path):
    if (dest / "semantics").is_dir():
        for frame in transforms["frames"]:
            frame["semantic_path"] = f"./semantics/{Path(frame['file_path']).name}"


def write_blender(plot_dir: Path, transforms, dest: Path):
    names = split_names(transforms)
    meta = {k: v for k, v in transforms.items() if k != "frames" and not k.endswith("_filenames")}
    if "camera_angle_x" not in meta:  # colmap_to_json stores the intrinsics per frame
        frame = transforms["frames"][0]
        meta["camera_angle_x"] = 2 * math.atan(frame["w"] / (2 * frame["fl_x"]))
    for split in ("train", "val", "test"):
        frames = []
        for frame in transforms["frames"]:
            name = Path(frame["file_path"]).name
            if name in names[split]:
                assert name.endswith(".png"), f"blender-data appends .png to the image paths: {name}"
                frames.append(dict(copy.deepcopy(frame), file_path=f"./images/{Path(name).stem}"))
        with open(dest / f"transforms_{split}.json", "w") as f:
            json.dump(dict(meta, frames=frames), f, indent=2)
        print(f"  transforms_{split}.json: {len(frames)} frames")
    link_images(plot_dir, dest)


def write_sparse_pc(plot_dir: Path, transforms, dest: Path):
    """sparse_pc.ply in the frame of transforms.json (checked against its poses); sets ply_file_path"""
    (xyz, rgb), ref = colmap2nerf_transforms(plot_dir, sharpness=False)
    poses = {Path(f["file_path"]).name: np.array(f["transform_matrix"]) for f in transforms["frames"]}
    diff = max(np.abs(np.array(f["transform_matrix"]) - poses[Path(f["file_path"]).name]).max() for f in ref["frames"])
    assert diff < 1e-6, f"transforms.json isn't in the colmap2nerf frame of sparse/0 (pose difference {diff:.3g})"
    with open(dest / "sparse_pc.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")
    transforms["ply_file_path"] = "sparse_pc.ply"
    print(f"  sparse_pc.ply: {len(xyz)} points (poses match transforms.json to {diff:.1g})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plot_dirs", nargs="+", type=Path, help="plot folders with images/ and sparse/0")
    parser.add_argument("--output_dir", type=Path, help="write to <output_dir>/<plot> instead of the plot folder")
    parser.add_argument("--sparse_pc", action="store_true", help="also write sparse_pc.ply (COLMAP points) for splatfacto")
    parser.add_argument("--semantics_from", help="mask set (folder with masks/, relative to the plot) for FruitNeRF's "
                                                 "semantics/, e.g. yolosam_640")
    args = parser.parse_args()

    for plot_dir in args.plot_dirs:
        dest = args.output_dir / plot_dir.resolve().name if args.output_dir else plot_dir
        dest.mkdir(parents=True, exist_ok=True)
        print(f"{plot_dir} -> {dest}")
        if args.semantics_from:
            write_semantics(plot_dir, args.semantics_from, dest)
        if (plot_dir / "transforms.json").exists():
            with open(plot_dir / "transforms.json") as f:
                transforms = json.load(f)
            # a copy pointing to the new semantics or the points (in place if dest is the plot folder)
            write = (args.semantics_from and dest.resolve() != plot_dir.resolve()) or args.sparse_pc
        else:
            _, transforms = colmap2nerf_transforms(plot_dir)
            set_split(transforms, camera_split(transforms))
            write = True
        if args.sparse_pc:
            write_sparse_pc(plot_dir, transforms, dest)
        if write:
            if not args.semantics_from and (plot_dir / "semantics").is_dir():
                link_images(plot_dir, dest, "semantics")  # the released ones, for the copy of transforms.json
            add_semantic_paths(transforms, dest)
            with open(dest / "transforms.json", "w") as f:
                json.dump(transforms, f, indent=2)
            link_images(plot_dir, dest)
            n = {s: len(transforms[f"{s}_filenames"]) for s in ("train", "test")}
            print(f"  transforms.json: {len(transforms['frames'])} frames, train/test {n['train']}/{n['test']}, "
                  f"semantics: {'semantic_path' in transforms['frames'][0]}")
        write_blender(plot_dir, transforms, dest)


if __name__ == "__main__":
    main()
