"""
Wheat head structural trait extraction module (MAIN ENTRY POINT)
Input:
    A set of point clouds of individual wheat heads (path to folder) + other (hyper-) parameters defined
    in config.yaml file
Output:
    A table of structural traits per wheat head (.xlsx file) + other (optional) outputs
Traits:
    Position [m]: centroid of the point cloud (pcd), X,Y,Z
    Length [m]: 2S Spline length passing through pcd points projected onto P1-P2 plane of PCA
    Width [m]:  point-2-plane distance wrt. plane spawned by P1-P2 plane of PCA
    Volume [m3]: volume of a convex hull (using open3d library -> Qhull algorithm)
    Curvature [a.u.]: 2D spline length vs. chord distance between spline endpoints
    Inclination [deg]: angle between 1st PCA component (P1) and Z-axis
Authors: Tomislav Medic & ChatGPT, updated/shortened for public use on 2nd May 2025
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Literal, Union

import yaml
from pydantic import BaseModel, DirectoryPath, StrictInt, field_validator

from wheatheadsmorphology.pipeline import run_pipeline


class ConfigModel(BaseModel):
    """
    Class for checking if config.yaml is correctly formatted
    """

    data_folder: DirectoryPath
    output_folder: DirectoryPath
    file_format: Literal["ply", "txt"]
    subsampling_threshold: StrictInt
    clusterer_definition: Dict[str, Any]
    sor_parameters: Dict[str, Any]
    splines_smoothing_value: Union[int, float]
    distance_percentile: Union[int, float]
    get_bboxes: bool
    get_processed_pcd: bool

    @field_validator("clusterer_definition")
    def check_clusterer_definition(cls, v):
        # Each key is mandatory and must have the correct type
        if "type" not in v:
            raise ValueError('clusterer_definition must include a "type" key')
        if not isinstance(v["type"], str) or v["type"] not in ("dbscan", "hdbscan"):
            raise ValueError('clusterer_definition.type must be "dbscan" or "hdbscan"')

        if "epsilon" not in v:
            raise ValueError('clusterer_definition must include "epsilon"')
        if not isinstance(v["epsilon"], (int, float)):
            raise TypeError("epsilon must be a number")

        if "min_samples" not in v:
            raise ValueError('clusterer_definition must include "min_samples"')
        if not isinstance(v["min_samples"], int):
            raise TypeError("min_samples must be an int")

        if "min_cluster_size" not in v:
            raise ValueError('clusterer_definition must include "min_cluster_size"')
        if not isinstance(v["min_cluster_size"], int):
            raise TypeError("min_cluster_size must be an int")

        if "epsilon_hdbscan" not in v:
            raise ValueError('clusterer_definition must include "epsilon_hdbscan"')
        if not isinstance(v["epsilon_hdbscan"], (int, float)):
            raise TypeError("epsilon_hdbscan must be a number")

        return v

    @field_validator("sor_parameters")
    def check_sor_parameters(cls, v):
        # Ensure both keys are present
        if "k" not in v:
            raise ValueError('sor_parameters must include "k"')
        if not isinstance(v["k"], int):
            raise TypeError("k must be an int")

        if "std_ratio" not in v:
            raise ValueError('sor_parameters must include "std_ratio"')
        if not isinstance(v["std_ratio"], (int, float)):
            raise TypeError("std_ratio must be a number")

        return v

    @field_validator("distance_percentile")
    def check_distance_percentile(cls, v):
        try:
            val = float(v)
        except (TypeError, ValueError):
            raise TypeError("distance_percentile must be a number")
        if not (1 <= val <= 100):
            raise ValueError("distance_percentile must be between 1 and 100 inclusive")
        return v

    @classmethod
    def validate_paths(cls, raw_cfg: dict, base_dir: Path) -> dict:
        """
        Resolve environment vars and relative paths against the config file's directory.
        """
        resolved = raw_cfg.copy()
        if "data_folder" in raw_cfg:
            expanded = os.path.expandvars(str(raw_cfg["data_folder"]))
            resolved["data_folder"] = (base_dir / expanded).resolve()
        if "output_folder" in raw_cfg:
            expanded_out = os.path.expandvars(str(raw_cfg["output_folder"]))
            output_path = (base_dir / expanded_out).resolve()
            os.makedirs(output_path, exist_ok=True)
            resolved["output_folder"] = output_path
        return resolved


# Allowing running custom config.yaml using CLI
def get_args():
    p = argparse.ArgumentParser(
        description="Run the full pipeline with a given config YAML file"
    )
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to YAML config file",
    )
    return p.parse_args()


# Load config
def load_config(config_path: Path) -> ConfigModel:
    """Load, resolve, and validate the YAML config using Pydantic."""
    # Resolve and read YAML
    config_path = config_path.resolve()
    raw_cfg = yaml.safe_load(config_path.read_text())
    # Resolve any path-like entries relative to the config file
    resolved_cfg = ConfigModel.validate_paths(raw_cfg, config_path.parent)
    # Validate and cast types
    return ConfigModel(**resolved_cfg)


if __name__ == "__main__":
    args = get_args()
    cfg_model = load_config(args.config)
    # Pass a dict or the model directly, depending on run_pipeline signature
    run_pipeline(cfg_model.dict())
