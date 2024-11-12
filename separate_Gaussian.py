import os
import random
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, Camera
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image, ImageDraw
from gaussian_renderer.render_helper import get_render_label
import torch.nn as nn
from typing import List
from copy import deepcopy

def rgb_tensor_to_PIL(tensor : torch.Tensor):
    return Image.fromarray((np.transpose(torch.clamp(tensor.detach().cpu(), 0, 1).numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))

def main(dataset, opt, pipe):
    out_dir = "/cluster/scratch/daizhang/Wheat-GS-output/plot_461_wheat_heads/point_cloud"
    os.makedirs(out_dir, exist_ok=True)
    
    gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, shuffle=False)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # viewpoint_stack = scene.getTrainCameras().copy()
    gaussians.load_ply("/cluster/scratch/daizhang/Wheat-GS-output/plot_461_flashsplat_sugar_coarse_fullres/gaussians.ply")
    print(torch.unique(gaussians.get_which_object, return_counts=True))
    obj_indices = torch.unique(gaussians.get_which_object)
    for obj_id in obj_indices:
        print(f"{obj_id}/{len(obj_indices)}")
        if obj_id != 0:
            # render_dir = os.path.join(out_dir, f"{obj_id:04d}", "render")
            # os.makedirs(render_dir, exist_ok=True)
            gaussians_obj = deepcopy(gaussians)
            gaussians_obj.prune_points(mask=torch.flatten(gaussians.get_which_object.detach() != obj_id), during_training=False)
            gaussians_obj.save_ply(f"{out_dir}/iteration_mask{str(obj_id.item())}/point_cloud.ply")

            # for viewpoint_cam in viewpoint_stack:
            #     render_pkg = render(viewpoint_cam, gaussians_obj, pipe, background)
            #     image = render_pkg["render"]
            #     rgb_tensor_to_PIL(image).save(os.path.join(render_dir, f"{viewpoint_cam.image_name}.png"))
                
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    main(lp.extract(args), op.extract(args), pp.extract(args))