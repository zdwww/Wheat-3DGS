#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def seg_load_iteration(model_path, exp_name, iteration=-1):
    # Model iteration that the 3D segmentation run <exp_name> was made on, as recorded in its experiment.txt.
    # Rendering or evaluating that run on another iteration would apply its labels to a different set of Gaussians.
    exp_file = os.path.join(model_path, "wheat-head", exp_name, "experiment.txt")
    recorded = None
    if os.path.exists(exp_file):
        with open(exp_file) as f:
            for line in f:
                key, _, value = line.strip().partition(" ")
                if key == "load_iteration":
                    recorded = int(value)
    if recorded is None:
        print(f"[Warning] No load_iteration in {exp_file}, using iteration {iteration}")
        return iteration
    if iteration != -1 and iteration != recorded:
        raise ValueError(f"Iteration {iteration} requested, but 3D segmentation {exp_name} was made on iteration {recorded}")
    return recorded
