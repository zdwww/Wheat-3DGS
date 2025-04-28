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
import gc
import torch
from scene import Scene
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import flashsplat_render
from utils.wheatgs_helper import multi_instance_opt
from utils.image_helper import *

def eval_obj_labels(all_obj_labels, viewpoint_cam, gaussians, pipe, background):
    from gaussian_renderer import flashsplat_render
    render_num = all_obj_labels.size(0)
    pred_mask = None
    max_alpha = None
    min_depth = None
    for obj_idx in range(render_num):
        obj_used_mask = (all_obj_labels[obj_idx]).bool()
        if obj_used_mask.sum().item() == 0 or obj_idx == 0: # obj 0 is background
            continue
        flashsplat_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask.to("cuda"))
        render_alpha = flashsplat_pkg["alpha"].detach().cpu()
        render_depth = flashsplat_pkg["depth"].detach().cpu()
        if pred_mask is None:
            pred_mask = torch.zeros_like(render_alpha)
            max_alpha = torch.zeros_like(render_alpha)
            min_depth = torch.ones_like(render_alpha)
        _pix_mask = (render_alpha > 0.5)
        pix_mask = _pix_mask.clone() 
        overlap_mask = (_pix_mask & (pred_mask > 0))
        if overlap_mask.sum().item() > 0:
            if (min_depth[overlap_mask].mean() < render_depth[overlap_mask].mean()):
                pix_mask[_pix_mask] = (~(pred_mask[_pix_mask] > 0)) 
        pred_mask[pix_mask] = obj_idx
        min_depth[pix_mask] = render_depth[pix_mask]
        max_alpha[pix_mask] = render_alpha[pix_mask]
    return pred_mask

def opt_w_masks(viewpoint_cam, gaussians, pipe, background, obj_masks, obj_num=None):
    if obj_num is None: # if None then it's the first view
        obj_num = torch.unique(obj_masks).numel() - 1
    obj_masks = obj_masks.to(torch.float32).to("cuda")
    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, gt_mask=obj_masks.squeeze(), obj_num=obj_num)
    # print(render_pkg["render"].shape)
    used_count = render_pkg["used_count"].detach().cpu()
    return used_count, obj_num

def render_set(model_path, name, views, gaussians, pipeline, background, all_obj_labels):
    render_path = os.path.join(model_path, name, "overlay")
    seg_path = os.path.join(model_path, name, "segmentation")

    makedirs(render_path, exist_ok=True)
    makedirs(seg_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"].detach().cpu()
        gt = view.original_image[0:3, :, :]
        pred_seg = eval_obj_labels(all_obj_labels, view, gaussians, pipeline, background)
        binary_array = (pred_seg.cpu().squeeze() != 0).to(torch.uint8) * 255 
        rgb_mask = visualize_obj(pred_seg) / 255.0
        rgb_image = overlay_image(rendering, rgb_mask)

        torchvision.utils.save_image(rgb_image, os.path.join(render_path, f"{view.image_name}.png"))
        image = Image.fromarray(binary_array.numpy(), mode="L")
        image.save(os.path.join(seg_path, f"{view.image_name}.png"))

def render_sets(dataset : ModelParams, pipeline : PipelineParams, exp_name, skip_train, load_counts):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False) # Load max iteration
        ply_path = os.path.join(dataset.model_path, "wheat-head", exp_name, "gaussians.ply")
        # gaussians.load_ply(ply_path)
        print(f"Gaussians successfully loaded from {ply_path}")
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ########### Begin of FlashSplat Optimization ########### 
        viewpoint_stack = scene.getTrainCameras().copy()
        if not load_counts:
            seg2d_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "2DSeg")
            obj_num = 0
            for viewpoint_cam in viewpoint_stack:
                seg2d = torch.load(os.path.join(seg2d_dir, f"{viewpoint_cam.image_name}.pt"))
                # print(torch.max(seg2d))
                if torch.max(seg2d) > obj_num:
                    obj_num = torch.max(seg2d)
                    print(f"Obj num updated to {obj_num}")
            all_counts = None

            for viewpoint_cam in viewpoint_stack:
                seg2d = torch.load(os.path.join(seg2d_dir, f"{viewpoint_cam.image_name}.pt"))
                used_count, flash_splat_obj_num = opt_w_masks(viewpoint_cam, gaussians, pipeline, background, seg2d, obj_num)
                if all_counts is None:
                    all_counts = torch.zeros_like(used_count)
                # print(used_count.shape, flash_splat_obj_num)
                all_counts += used_count
                gc.collect()
                torch.cuda.empty_cache()
        else:
            counts_pth = os.path.join(dataset.model_path, "wheat-head", exp_name, "all_counts.pt")
            assert os.path.exists(counts_pth)
            all_counts = torch.load(counts_pth).cuda()
            print(f"Load all counts shape {all_counts.shape}")

        slackness = 0.0
        all_obj_labels = multi_instance_opt(all_counts, slackness)
        ########### End of FlashSplat Optimization ########### 
        if not skip_train:
            render_set(dataset.model_path, "train", scene.getTrainCameras(), gaussians, pipeline, background, all_obj_labels)
        render_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, all_obj_labels)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--exp_name", type=str, help="Exp name")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--load_counts", action="store_true")
    args = get_combined_args(parser)
    print(f"Rendering {args.model_path}/{args.exp_name}")
    render_sets(model.extract(args), pipeline.extract(args), args.exp_name, args.skip_train, args.load_counts)