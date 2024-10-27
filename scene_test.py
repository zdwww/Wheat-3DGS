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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe):

    torch.set_printoptions(precision=10, sci_mode=False)
    gaussians = GaussianModel(dataset.sh_degree)

    dataset.source_path = "../Wheat-GS-data/20240717/plot_461"
    print(dataset.source_path)
    scene = Scene(dataset, gaussians)
    print(f"Scene1 extent {scene.cameras_extent}")
    # print(f"{scene.gaussians.get_xyz}")
    print(f"Min xyz {torch.min(scene.gaussians.get_xyz, dim=0).values.detach().cpu()}")
    print(f"Max xyz {torch.max(scene.gaussians.get_xyz, dim=0).values.detach().cpu()}")

    dataset.source_path = "../Wheat-GS-data/20240717/plot_461_new"
    scene = Scene(dataset, gaussians)
    print(f"Scene2 extent {scene.cameras_extent}")
    # print(f"{scene.gaussians.get_xyz}")
    print(f"Min xyz {torch.min(scene.gaussians.get_xyz, dim=0).values.detach().cpu()}")
    print(f"Max xyz {torch.max(scene.gaussians.get_xyz, dim=0).values.detach().cpu()}")
     
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    print("Length of viewpoint stack", len(viewpoint_stack))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    
    print("Optimizing model" + args.model_path + "source" + args.source_path) 

    # Start GUI server, configure and run training
    training(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
