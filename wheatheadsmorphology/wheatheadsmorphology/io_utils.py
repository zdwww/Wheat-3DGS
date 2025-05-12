from datetime import datetime
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(
    pcd_path: Path,
    retain_colors: bool = True,
    retain_normals: bool = True,
    scalar_fields: list[str] = None,
    normalize_intensities: bool = False,
    **kwargs,
) -> list:
    """
    Loads a ply file using *dranjan/python-plyfile*.

    Parameters
    ----------
    pcd_path : pathlib.Path
    scalar_fields : list[str], optional
                    List of scalar fields to keep (will be intersected against the available scalar fields from the
                    *ply-file*). `None` retains all available scalar fields.
    retain_colors : bool, default=True
    retain_normals : bool, default=True
    normalize_intensities: bool, default=False

    Returns
    -------
    pcd : DeSpAn.geometry.PointCloudData
    """
    with open(pcd_path, "rb") as f:
        plydata = PlyData.read(f)
    xyz = np.empty(
        (
            plydata["vertex"].count,
            3,
        ),
        dtype=float,
    )
    xyz[:, 0] = plydata["vertex"]["x"]
    xyz[:, 1] = plydata["vertex"]["y"]
    xyz[:, 2] = plydata["vertex"]["z"]

    ply_scalar_fields = [pe.name for pe in plydata["vertex"].properties]

    # ply_scalar_fields_lower = [ply_sf.lower() for ply_sf in ply_scalar_fields]
    # scalar_fields = None if scalar_fields is None else [sf for sf in scalar_fields]

    colors = None
    if (
        retain_colors
        and len(set(ply_scalar_fields) & set(["r", "g", "b", "red", "green", "blue"]))
        == 3
    ):
        colors = np.empty(
            (
                plydata["vertex"].count,
                3,
            ),
            dtype=np.uint8,
        )
        colors[:, 0] = (
            plydata["vertex"]["r"]
            if "r" in ply_scalar_fields
            else plydata["vertex"]["red"]
        )
        colors[:, 1] = (
            plydata["vertex"]["g"]
            if "g" in ply_scalar_fields
            else plydata["vertex"]["green"]
        )
        colors[:, 2] = (
            plydata["vertex"]["b"]
            if "b" in ply_scalar_fields
            else plydata["vertex"]["blue"]
        )

    normals = None
    if retain_normals and len(set(ply_scalar_fields) & set(["nx", "ny", "nz"])) == 3:
        normals = np.empty(
            (
                plydata["vertex"].count,
                3,
            ),
            dtype=float,
        )
        normals[:, 0] = plydata["vertex"]["nx"]
        normals[:, 1] = plydata["vertex"]["ny"]
        normals[:, 2] = plydata["vertex"]["nz"]

    common_scalar_fields = (
        ply_scalar_fields
        if scalar_fields is None
        else list(set(scalar_fields) & set(ply_scalar_fields))
    )

    scalar_fields_dict = dict()
    for sf in common_scalar_fields:
        if sf.lower() not in [
            "x",
            "y",
            "z",
            "r",
            "g",
            "b",
            "red",
            "green",
            "blue",
            "nx",
            "ny",
            "nz",
        ]:
            scalar_fields_dict[sf] = np.array(plydata["vertex"][sf]).squeeze()

    if "scalar_Intensity" in scalar_fields_dict and normalize_intensities:
        scalar_fields_dict["scalar_Intensity"] = (
            scalar_fields_dict["scalar_Intensity"]
            / scalar_fields_dict["scalar_Intensity"].max()
        )

    return [xyz, colors, normals, scalar_fields_dict]


def save_ply(
    pcd_path: Path,
    pcd: list,
    retain_colors: bool = True,
    retain_normals: bool = True,
    scalar_fields: list[str] = None,
) -> None:
    """
    Input:point cloud in a list
    Output: save such a point cloud list in a .ply file format
    """
    nb_points = pcd[0].shape[0]
    dtype_list = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    if retain_colors and pcd[1] is not None:
        assert pcd[1].shape == (nb_points, 3)
        dtype_list.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

    if retain_normals and pcd[2] is not None:
        assert pcd[2].shape == (nb_points, 3)
        dtype_list.extend([("nx", "f8"), ("ny", "f8"), ("nz", "f8")])

    if pcd[3] is not None:
        pcd_scalar_fields = pcd[3].keys()
        common_scalar_fields = (
            pcd_scalar_fields
            if scalar_fields is None
            else list(set(scalar_fields) & set(pcd_scalar_fields))
        )
        for sf in common_scalar_fields:
            assert pcd[3][sf].shape == (nb_points,)
            dtype_list.append((sf, pcd[3][sf].dtype.str))
    else:
        common_scalar_fields = []

    pcd_np_st = np.empty((nb_points,), dtype=dtype_list)

    pcd_np_st["x"] = pcd[0][:, 0]
    pcd_np_st["y"] = pcd[0][:, 1]
    pcd_np_st["z"] = pcd[0][:, 2]

    if retain_colors and pcd[1] is not None:
        pcd_np_st["red"] = pcd[1][:, 0]
        pcd_np_st["green"] = pcd[1][:, 1]
        pcd_np_st["blue"] = pcd[1][:, 2]

    if retain_normals and pcd[2] is not None:
        pcd_np_st["nx"] = pcd[2][:, 0]
        pcd_np_st["ny"] = pcd[2][:, 1]
        pcd_np_st["nz"] = pcd[2][:, 2]

    for sf in common_scalar_fields:
        pcd_np_st[sf] = pcd[3][sf]

    # TODO: Rename program in comment
    el = PlyElement.describe(
        pcd_np_st,
        "vertex",
        comments=[
            "Created with dranjan/python-plyfile in REASSESS program",
            f"Created {datetime.now():%Y-%m-%dT%H:%M:%S}",
        ],
    )

    if not pcd_path.parent.exists():
        pcd_path.parent.mkdir(parents=True, exist_ok=True)

    PlyData([el]).write(f"{pcd_path}")
    return None
