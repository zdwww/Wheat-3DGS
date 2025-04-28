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

import torch
import subprocess
import shutil
import math
from scene import Scene
import os
import gc
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import flashsplat_render
from utils.wheatgs_helper import render_360, multi_instance_opt

import numpy as np
import torch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam

#### Begin of 360-degree camera trajectory copied from gsgen ####
def get_c2w_from_up_and_look_at(up, look_at, pos):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    return c2w

def get_camera_path_fixed_elevation(n_frames, n_circles=1, camera_distance=2, cam_center=[0, 0, 0], elevation=0):
    azimuth = np.linspace(0, 2 * np.pi * n_circles, n_frames)
    elevation_rad = np.deg2rad(elevation)
    x = camera_distance * np.cos(azimuth) * np.cos(elevation_rad)
    y = camera_distance * np.sin(azimuth) * np.cos(elevation_rad)
    z = camera_distance * np.sin(elevation_rad) * np.ones_like(x)

    up = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.array(cam_center, dtype=np.float32)
    pos = np.stack([x, y, z], axis=1)

    c2ws = []
    for i in range(n_frames):
        c2ws.append(get_c2w_from_up_and_look_at(up, look_at, pos[i]))
    c2ws = np.stack(c2ws, axis=0)
    return c2ws

#### End of 360-degree camera trajectory copied from gsgen ####

def opt_w_masks(viewpoint_cam, gaussians, pipe, background, obj_masks, obj_num=None):
    if obj_num is None: # if None then it's the first view
        obj_num = torch.unique(obj_masks).numel() - 1
    obj_masks = obj_masks.to(torch.float32).to("cuda")
    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, gt_mask=obj_masks.squeeze(), obj_num=obj_num)
    print(render_pkg["render"].shape)
    used_count = render_pkg["used_count"].detach().cpu()
    return used_count, obj_num

def render_all(dataset : ModelParams, pipeline : PipelineParams, exp_name):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)
        bg_color = [1,1,1] # if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        og_views = scene.getTrainCameras()
        og_view = og_views[0]
        width, height = math.floor(og_view.image_width / 3), math.floor(og_view.image_height / 3)
        fovy, fovx = og_view.FoVy / 5, og_view.FoVx / 5
        znear, zfar = og_view.znear, og_view.zfar
        print(f"Fixed parameters width: {width} height {height} fovy {fovy} fovx {fovx}")

        wheat_head_dir = os.path.join(dataset.model_path, "wheat-head", exp_name, "ply")
        # wheat_head_folders = [
        #     name for name in os.listdir(wheat_head_dir)
        #     if os.path.isdir(os.path.join(wheat_head_dir, name)) and name.isdigit()
        # ]
        # wheat_head_folders = sorted(wheat_head_folders)
        ply_files = [f for f in os.listdir(wheat_head_dir) if f.startswith("wh") and f.endswith(".ply")]
        print("ply_files", len(ply_files), ply_files)

        # for idx, wheat_head in enumerate(tqdm(wheat_head_folders, desc="Rendering progress")):
        for idx, ply_file in enumerate(tqdm(ply_files, desc="Rendering progress")):
            if len(os.path.splitext(ply_file)[0].split("_")) > 2:
                print(f"Pass file {ply_file}")
                continue

            scene.load_ply(os.path.join(wheat_head_dir, ply_file))
            gs_centroid = torch.mean(gaussians.get_xyz, dim=0).cpu().tolist()
            scene_radius = scene.cameras_extent
            print(f"Gaussians centroid {gs_centroid}, Scene radius {scene_radius}")
            
            ply_id = ply_file.replace("wh_", "", 1).replace(".ply", "", 1)
            camera_distance = scene_radius * 0.65
            render_path = os.path.join(os.path.dirname(wheat_head_dir), "3DWheat", ply_id)
            makedirs(render_path, exist_ok=True)

            c2ws = get_camera_path_fixed_elevation(n_frames=100, n_circles=1, camera_distance=camera_distance, cam_center=gs_centroid, elevation=15)
            for idx, c2w in enumerate(c2ws):
                c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
                w2c = np.linalg.inv(np.float64(c2w))
                world_view_transform = torch.tensor(w2c.astype(np.float32)).transpose(0, 1).cuda()
                projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0,1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                view = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
                render_pkg = render(view, gaussians, pipeline, background)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            output_video = os.path.join(os.path.dirname(render_path), f"{ply_id}.mp4")
            framerate = 10
            subprocess.run([
                "ffmpeg",
                "-loglevel", "error",
                "-framerate", str(framerate),
                "-start_number", "0",
                "-i", "%05d.png",  
                "-vf", "scale=iw-mod(iw\,2):ih-mod(ih\,2)",
                "-r", str(framerate),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                output_video
            ], cwd=render_path)
            # shutil.rmtree(render_path)
            
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, which_wheat_head=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, which_wheat_head=which_wheat_head)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        og_views = scene.getTrainCameras()
        og_view = og_views[0]
        
        if which_wheat_head is not None:
            width, height = math.floor(og_view.image_width / 3), math.floor(og_view.image_height / 3)
            fovy, fovx = og_view.FoVy / 7, og_view.FoVx / 7
        else:
            width, height = math.floor(og_view.image_width / 2), math.floor(og_view.image_height / 2)
            fovy, fovx = og_view.FoVy, og_view.FoVx
        znear, zfar = og_view.znear, og_view.zfar
        print(f"Fixed parameters width: {width} height {height} fovy {fovy} fovx {fovx}")

        gs_centroid = torch.mean(gaussians.get_xyz, dim=0).cpu().tolist()
        scene_radius = scene.cameras_extent
        print(f"Gaussians centroid {gs_centroid}, Scene radius {scene_radius}")

        if which_wheat_head is None:
            camera_distance = scene_radius*4
            render_path = os.path.join(dataset.model_path, "render360", "wheat-field")
        else:
            camera_distance = scene_radius * 0.75
            render_path = os.path.join(dataset.model_path, "render360", str(int(which_wheat_head)).zfill(4))
        makedirs(render_path, exist_ok=True)
        c2ws = get_camera_path_fixed_elevation(n_frames=200, n_circles=1, camera_distance=camera_distance, cam_center=gs_centroid, elevation=45)
        for idx, c2w in enumerate(tqdm(c2ws, desc="Rendering progress")):
            c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
            w2c = np.linalg.inv(np.float64(c2w))
            world_view_transform = torch.tensor(w2c.astype(np.float32)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            view = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            render_pkg = render(view, gaussians, pipeline, background)
            rendering = render_pkg["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if which_wheat_head is not None:
            output_video = os.path.join(os.path.dirname(render_path), f"{int(which_wheat_head):04}.mp4")
        else:
            output_video = os.path.join(os.path.dirname(render_path), f"wheat-field-d5-e45.mp4")
        subprocess.run([
            "ffmpeg",
            "-loglevel", "error",
            "-start_number", "0",
            "-i", "%05d.png",  
            "-vf", "scale=iw-mod(iw\,2):ih-mod(ih\,2),fps=5",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ], cwd=render_path)
        shutil.rmtree(render_path)

def render_3d_seg(dataset : ModelParams, pipeline : PipelineParams, opt, load_iteration=-1):
    seg2d_dir = os.path.join(dataset.model_path, "wheat-head", "run_test", "2DSeg")
    out_dir = os.path.join(dataset.model_path, "wheat-head", "run_test", "3DSeg")
    os.makedirs(out_dir, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree)
    try:
        load_iteration = int(load_iteration)
    except:
        pass
    print(f"Load iteration {load_iteration}, Resolution {dataset.resolution}")
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_xyz)}")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack_eval = scene.getTestCameras().copy()
    # viewpoint_stack += viewpoint_stack_eval
    print(f"Length of viewpoint stack: {len(viewpoint_stack)}")

    obj_num = 0
    for viewpoint_cam in viewpoint_stack:
        seg2d = torch.load(os.path.join(seg2d_dir, f"{viewpoint_cam.image_name}.pt"))
        print(torch.max(seg2d))
        if torch.max(seg2d) > obj_num:
            obj_num = torch.max(seg2d)
            print(f"Obj num updated to {obj_num}")

    all_counts = None
    for viewpoint_cam in viewpoint_stack:
        seg2d = torch.load(os.path.join(seg2d_dir, f"{viewpoint_cam.image_name}.pt"))
        used_count, flash_splat_obj_num = opt_w_masks(viewpoint_cam, gaussians, pipeline, background, seg2d, obj_num)
        if all_counts is None:
            all_counts = torch.zeros_like(used_count)
        print(f"Used count: {used_count.shape}, flash_splat_obj_num {flash_splat_obj_num}")
        all_counts += used_count
        gc.collect()
        torch.cuda.empty_cache()
    print(f"All counts: {all_counts.shape}")
    slackness = 0.0
    torch.save(all_counts.detach().cpu(), os.path.join(dataset.model_path, "wheat-head", "run_test", "all_counts.pth"))
    all_obj_labels = multi_instance_opt(all_counts, slackness)
    print("All_obj_labels: ", all_obj_labels.shape)
    # Save for future rendering
    torch.save(all_obj_labels.detach().cpu(), os.path.join(dataset.model_path, "wheat-head", "run_test", "all_obj_labels.pth"))
    output_video = render_360(viewpoint_stack[0], scene.cameras_extent, out_dir, 200, 10, gaussians, pipeline, background, all_obj_labels=all_obj_labels)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--which_wheat_head", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    args = get_combined_args(parser)
    print(f"Rendering {args.model_path} for wheat head {args.which_wheat_head}")
    if args.which_wheat_head == "all":
        print("Render All Wheat Heads")
        render_all(model.extract(args), pipeline.extract(args), args.exp_name)
    elif args.which_wheat_head == "3dseg":
        print("Render 3D Segmentation")
        render_3d_seg(model.extract(args), pipeline.extract(args), opt.extract(args))
    else:
        pass
        # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.which_wheat_head)