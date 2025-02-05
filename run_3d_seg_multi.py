import os
import gc
import glob
import wandb
import ffmpeg
import csv
import random
import torch
import shutil
import subprocess
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from gaussian_renderer import flashsplat_render # FlashSplat Render
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
import torch.nn.functional as F
from typing import List
from copy import deepcopy
import matplotlib.pyplot as plt

import torchvision
from utils.image_helper import *
from utils.wheatgs_helper import get_interpolated_viewpts, nearest_k_viewpts, get_interpolated_viewpts_old, short_image_name, render_360, eval_obj_labels, find_best_match

out_dir = None

def find_cam(viewpoint_stack, cam_short_name):
    """
    Method for finding cameras from the viewpoint stack with its shorthand name
    """
    for vpt_cam in viewpoint_stack:
        if short_image_name(vpt_cam.image_name) == cam_short_name:
            return vpt_cam

########### Begin of Visualization Helper Functions, will migrate to utils dir later ###########

def vis_overlap(pred_seg, matched_seg, all_seg):
    binary_pred_seg = (pred_seg != 0)
    binary_matched_seg = (matched_seg != 0)
    binary_all_seg = (all_seg != 0)
    overlap_rgb = torch.zeros((*binary_pred_seg.shape, 3), dtype=torch.uint8)
    overlap_rgb[binary_pred_seg & ~binary_matched_seg] = torch.tensor([255, 255, 255], dtype=torch.uint8)
    overlap_rgb[binary_all_seg] = torch.tensor((100, 100, 100), dtype=torch.uint8)
    overlap_rgb[~binary_pred_seg & binary_matched_seg] = torch.tensor([255, 165, 0], dtype=torch.uint8)
    overlap_rgb[binary_pred_seg & binary_matched_seg] = torch.tensor([255, 0, 0], dtype=torch.uint8)
    overlap_rgb = overlap_rgb.squeeze().permute(2, 0, 1) / 255.0
    return overlap_rgb

def vis_image_w_overlay(img_tensor, save_dir, save_name, pred_seg, overlap_seg=None, resize_factor=1):
    """
    Args:
        pred_seg: segmentation rendered from 3DGS
        overlap_seg: seg obtained from SAM with largest IOU between pred_seg
    """
    image_pil = rgb_tensor_to_PIL(img_tensor)
    mask_pil = Image.fromarray(pred_seg.astype(np.uint8) * 255)
    image_with_overlay = overlay_img_w_mask(image_pil, mask_pil, color="red")
    if overlap_seg is not None:
        mask_pil = Image.fromarray(overlap_seg.astype(np.uint8) * 255)
        image_with_overlay = overlay_img_w_mask(image_with_overlay, mask_pil, color="blue")
    if resize_factor != 1:
        width, height = image_with_overlay.size                
        new_size = (width // resize_factor, height // resize_factor)
        image_with_overlay = image_with_overlay.resize(new_size)
    image_with_overlay.save(os.path.join(save_dir, f"{save_name}.jpg"))

########### End of Visualization Helper Functions ###########

def multi_instance_opt(all_contrib, gamma=0.):
    """
    Input:
    all_contrib: A_{e} with shape (obj_num, gs_num) 
    gamma: softening factor range from [-1, 1]
    
    Output: 
    all_obj_labels: results S with shape (obj_num, gs_num)
    where S_{i,j} denotes j-th gaussian belong i-th object
    """
    all_contrib_sum = all_contrib.sum(dim=0)
    all_obj_labels = torch.zeros_like(all_contrib).bool()
    for obj_idx, obj_contrib in tqdm(enumerate(all_contrib), desc="multi-view optimize"):
        obj_contrib = torch.stack([all_contrib_sum - obj_contrib, obj_contrib], dim=0)
        obj_contrib = F.normalize(obj_contrib, dim=0)
        obj_contrib[0, :] += gamma
        obj_label = torch.argmax(obj_contrib, dim=0)
        all_obj_labels[obj_idx] = obj_label
    return all_obj_labels

def opt_label_w_seg(gaussians : GaussianModel, 
                    viewpoint_stack : List[Camera], 
                    mask_paths : List[str], 
                    pipeline, background):
    """Helper function that wraps Gaussians label optimization schema into one function"""
    assert len(viewpoint_stack) == len(mask_paths)
    
    all_counts = None
    for idx, viewpoint_cam in enumerate(viewpoint_stack):
        with Image.open(mask_paths[idx]) as temp:
            gt_mask = binarize_mask(PILtoTorch(temp.copy(), viewpoint_cam.resolution)).squeeze().to("cuda")
            assert viewpoint_cam.original_image.shape[-2:] == gt_mask.shape
        with torch.no_grad():
            render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipeline, background, gt_mask=gt_mask, obj_num=1)
            rendering = render_pkg["render"]
            used_count = render_pkg["used_count"]
            if all_counts is None:
                all_counts = torch.zeros_like(used_count)
            all_counts += used_count
        gc.collect()
        torch.cuda.empty_cache()
        
    slackness = 0.0
    all_obj_labels = multi_instance_opt(all_counts, slackness)
    print(f"Optimized w.r.t {len(viewpoint_stack)} viewpoints, {torch.sum(all_obj_labels, dim=1)[1]} Gaussians identified")
    return all_obj_labels

def opt_w_masks(viewpoint_cam, gaussians, pipe, background, obj_masks, obj_num=None):
    if obj_num is None: # if None then it's the first view
        obj_num = torch.unique(obj_masks).numel() - 1
    obj_masks = obj_masks.to(torch.float32).to("cuda")
    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, gt_mask=obj_masks.squeeze(), obj_num=obj_num)
    used_count = render_pkg["used_count"].detach().cpu()
    return used_count, obj_num

def vis_one_cam(all_obj_labels, vpt_cam, gaussians, pipe, background, pred_mask=None):
    """Visualize overlay of rendered image and segmentation of one camera"""
    global out_dir
    # Render RGB
    render_pkg = render(vpt_cam, gaussians, pipe, background)
    render_image = render_pkg["render"].detach().cpu()
    del render_pkg
    torch.cuda.empty_cache()
    # Render Segmentation Mask
    if pred_mask is None:
        pred_mask = eval_obj_labels(all_obj_labels, vpt_cam, gaussians, pipe, background)
    rgb_mask = visualize_obj(pred_mask) / 255.0
    overlayed_image = overlay_image(render_image, rgb_mask)
    torch.cuda.empty_cache()
    return overlayed_image
    
def vis_interpolated_cams(all_obj_labels, vis_viewpoint_stack, gaussians, pipe, background, fps=5):
    """Visualization of 3D segmentation with interpolated camera poses"""
    global out_dir
    os.makedirs(os.path.join(out_dir, "interpolated"), exist_ok=True) 
    # os.makedirs(render_path, exist_ok=True)
    for idx in tqdm(range(len(vis_viewpoint_stack)), desc="Visualizing interpolated cameras progress"):
        inner_cam = vis_viewpoint_stack[idx]
        overlayed_image = vis_one_cam(all_obj_labels, inner_cam, gaussians, pipe, background)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, "interpolated", f"{idx:05}.png"))
        # torchvision.utils.save_image(overlayed_image, os.path.join(render_path, f"{idx:05}.png"))
    print("Len vis_viewpoint_stack", len(vis_viewpoint_stack), 'fps', fps)
    output_video = os.path.join(out_dir, "interpolate.mp4")
    try:
        (
            ffmpeg
            .input(f'{os.path.join(out_dir, "interpolated")}/%05d.png', framerate=fps, start_number=0)
            .filter("scale", "iw-mod(iw,2)", "ih-mod(ih,2)")  # Scale to ensure even dimensions
            .output(output_video, framerate=fps, vcodec="libx264", pix_fmt="yuv420p", vsync="cfr")
            .global_args("-loglevel", "error")  
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Video successfully created:", output_video)
    except ffmpeg.Error as e:
        print("Error during FFmpeg execution:")
        print("STDERR:", e.stderr.decode())
    shutil.rmtree(os.path.join(out_dir, "interpolated"))
    return output_video

def seg_first_view(viewpoint_cam, gaussians, pipe, background):
    """Initialize the segmentation w.r.t FIRST view"""
    obj_num = len(viewpoint_cam.mask_paths)
    print(f"Num of YOLO-SAM segs for {viewpoint_cam.image_name}: {obj_num}")
    obj_masks = torch.zeros((1, *viewpoint_cam.original_image.shape[-2:]), dtype=torch.uint8)
    # Integrate all individual segmentation into one matrix
    for i, path in enumerate(viewpoint_cam.mask_paths):
        with Image.open(path) as temp:
            mask = binarize_mask(PILtoTorch(temp.copy()))
            obj_masks[mask != 0] = i + 1
    assert torch.unique(obj_masks).numel() - 1 == obj_num
    used_count, flash_splat_obj_num = opt_w_masks(viewpoint_cam, gaussians, pipe, background, obj_masks) 
    assert obj_num == flash_splat_obj_num
    return used_count, obj_num
        
def training(dataset, opt, pipe, load_iteration, exp_name, iou_threshold, num_match):
    wandb.init(project="Wheat-GS", name=dataset.source_path.split("/")[-1], dir="/cluster/scratch/daizhang/wandb")
    
    global out_dir
    out_dir = os.path.join(dataset.model_path, "wheat-head", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    # with open(f"{out_dir}/experiment.txt", "w") as file:
    #     file.write(f"exp_name {exp_name}\niou_threshold {iou_threshold}\nnum_match {num_match}\n")
    
    results = open(os.path.join(out_dir, 'results.csv'), mode='w', newline='')
    # writer = csv.writer(results)
    # writer.writerow(["id", "init_mask", "num_matches", "mean_iou"])
    
    gaussians = GaussianModel(dataset.sh_degree)

    try:
        load_iteration = int(load_iteration)
    except:
        pass
    print(f"Load iteration {load_iteration}, Resolution {dataset.resolution}")
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_xyz)}")

    min_vals, _ = torch.min(gaussians.get_xyz.detach().cpu(), dim=0)  # Minimum along the rows (dim=0)
    max_vals, _ = torch.max(gaussians.get_xyz.detach().cpu(), dim=0)  # Maximum along the rows (dim=0)
    
    # Print results
    print("Minimum values (x, y, z):", min_vals)
    print("Maximum values (x, y, z):", max_vals)

    z_mean = torch.mean(gaussians.get_xyz.cpu()[:, 2])
    print(f"All Gaussians z_min: {torch.min(gaussians.get_xyz.cpu()[:, 2])} zmax: {torch.max(gaussians.get_xyz.cpu()[:, 2])}")
    pts_filter = (gaussians.get_xyz.cpu()[:, 2] < z_mean * 1.1) # points below avererage z value won't be conisdered in 3D Seg
    print(f"Num of Gaussians below average z {z_mean} * 1.1: {torch.sum(pts_filter)}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    print(f"Length of viewpoint stack: {len(viewpoint_stack)}")

    # Render360
    # render_360(viewpoint_stack[0], scene.cameras_extent, os.path.join(out_dir, "render360"), 50, gaussians, pipe, background)
    
    #### Iterate through all vpts for preprocessing
    # os.makedirs(f"{out_dir}/2DSeg", exist_ok=True)
    num_bboxes = 0
    all_mask_paths = [] # a list of saved binary masks in png format
    cam_centers = [] 
    twoD_seg_results = {} # 2D segmentation results update through the pipeline
    cam_cover_areas = []
    for viewpoint_cam in viewpoint_stack:
        print(f"{short_image_name(viewpoint_cam.image_name)}: {viewpoint_cam.rectangle}, {viewpoint_cam.rectangle.shape}")
        cam_cover_areas.append(viewpoint_cam.rectangle)
        
        bboxes = torch.load(viewpoint_cam.bbox_path)
        num_bboxes += len(bboxes)
        all_mask_paths += viewpoint_cam.mask_paths
        cam_centers.append(viewpoint_cam.camera_center.cpu())
        twoD_seg_results[viewpoint_cam.image_name] = torch.zeros(viewpoint_cam.original_image.shape[1:], dtype=torch.int)
        # torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")
    assert len(all_mask_paths) == num_bboxes
    print(f"Total of {len(all_mask_paths)} mask & bounding box pairs found")
    cam_centers = torch.stack(cam_centers, dim=0)
    print(f"Centroid of camera centers {torch.mean(cam_centers, dim=0)}")

    ########## BEGIN OF CAM PROP TEST ##########
    cam_list = ["6_cam_01", "6_cam_03", "6_cam_05", "6_cam_07", "6_cam_09", "6_cam_10", "6_cam_08", "6_cam_06", "6_cam_04", "6_cam_02",
"cam_02", "cam_04", "cam_06", "cam_08", "cam_10", "cam_09", "cam_07", "cam_05", "cam_03", "cam_01",
 "1_cam_01", "1_cam_03", "1_cam_05", "1_cam_07", "1_cam_09", "1_cam_10", "1_cam_08", "1_cam_06", "1_cam_04", "1_cam_02"]

    ########## END OF CAM PROP TEST ##########

    #### Main framework for propagate cameras
    # curr_vpt_cams = [viewpoint_stack[0]]
    # # viewpoint_stack = viewpoint_stack[1:]
    # # Two rows of cameras
    # odd_vpt_stack = [vpt for vpt in viewpoint_stack if int(vpt.image_name.split("_")[-1]) % 2 != 0]
    # even_vpt_stack = [vpt for vpt in viewpoint_stack if int(vpt.image_name.split("_")[-1]) % 2 == 0]
    # curr_vpt_cam = odd_vpt_stack[0]

    num_wheat_heads = 0
    count = 0
    all_counts = None
    weighted_all_counts = None
    seg_counts = None # Count the number of segmentation find by yolosam

    for cam_idx in range(len(cam_list)-1):
        count += 1
        print(f"CAM {count} / {len(cam_list)}")
        curr_vpt_cam = find_cam(viewpoint_stack, cam_list[cam_idx])
        ############## Begin of Selection of Cameras ##############
        print(f"Current camera {cam_list[cam_idx]}, {curr_vpt_cam.image_name}")
        curr_vpt_cam_idx = int(curr_vpt_cam.image_name.split("_")[-1])
        # nearest_odd_vpts, odd_vpt_stack  = nearest_k_viewpts(odd_vpt_stack, curr_vpt_cam.camera_center.cpu(), 1)
        target_vpt_cam = find_cam(viewpoint_stack, cam_list[cam_idx+1])
        print(f"Target camera {cam_list[cam_idx+1]}, {target_vpt_cam.image_name}")
        # print(f"Odd left {len(odd_vpt_stack)}")
        ############## End of Selection of Cameras ##############

        ########## Begin of Optimization w.r.t CURRENT cameras ##########
        if num_wheat_heads == 0: # Initial Segmentation from first view 
            used_count, obj_num = seg_first_view(curr_vpt_cam, gaussians, pipe, background)
            tensor = used_count[1:].detach().cpu()
            print(f"Used count Min: {torch.min(tensor[tensor != 0]):.4f}, Mean: {torch.mean(tensor[tensor != 0]):.4f}, Max: {torch.max(tensor[tensor != 0]):.4f}")
            all_counts = used_count
            weighted_all_counts = used_count
            seg_counts = torch.ones((used_count.shape[0]))
            num_wheat_heads += obj_num
        
        else: # Use matched segmentation from prev iter
            # used_count, obj_num = opt_w_masks(curr_vpt_cam, gaussians, pipe, background, obj_masks)
            obj_num = all_counts.shape[0] - 1
            
        print(f"Cam {curr_vpt_cam.image_name} with {obj_num} output all counts with shape {weighted_all_counts.shape}")

        ##### NOTE: USE weighted all counts for inference ! #####

        # Apply height filter to 3D Seg: Gaussians below threshold
        gauss_below_thre = pts_filter.nonzero(as_tuple=True)[0]
        print(f"Height filter applied on {gauss_below_thre} Gaussians.")
        weighted_all_counts[1:, gauss_below_thre] = 0
          
        slackness = 0.0
        # torch.save(all_obj_labels, os.path.join(out_dir, f"3DSeg.pt")) # save 3D seg
        ########## End of Optimization w.r.t CURRENT cameras ##########

        ###### Video evaluation & Remove redundant wheat heads every X cameras or last camera ######
        if (count - 1) % 5 == 0 or cam_idx == len(cam_list)-1:
            assert weighted_all_counts.shape[0] == seg_counts.shape[0]
            count_1_indices = torch.nonzero(seg_counts <= 1, as_tuple=True)[0]
            print(f"{len(count_1_indices)}/{seg_counts.shape[0]} wheat heads have count 1")

            # Erase un-matched wheat heads every x iterations
            # if count > 1:
            #     print(f"Removing count=1 wheat heads for {count}th cam {short_image_name(curr_vpt_cam.image_name)}")
            #     all_counts[count_1_indices] = 0
            #     seg_counts[count_1_indices] = 0 
            all_obj_labels = multi_instance_opt(weighted_all_counts, slackness)
            
            print(f"Start video evaluation for {count}th cam {short_image_name(curr_vpt_cam.image_name)}")
            # Evaluate 3D Segmentation in 360
            output_video = render_360(viewpoint_stack[0], scene.cameras_extent, os.path.join(out_dir, "render360"), 100, 10, gaussians, pipe, background, all_obj_labels=all_obj_labels)
            os.rename(output_video, os.path.join(out_dir, f"{count:02}_a_360.mp4")) 
            ## Full Scene Visualization
            # vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts_old(viewpoint_stack, N=200), gaussians, pipe, background)
            ########## Begin of Move to TARGET cam and find matches ##########
        else:
            all_obj_labels = multi_instance_opt(weighted_all_counts, slackness)

        output_video = vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts(curr_vpt_cam, target_vpt_cam, N=20), gaussians, pipe, background, fps=5)
        os.rename(output_video, os.path.join(out_dir, f"{count:02}_b_move.mp4"))
        
        # curr_vpt_cam = target_vpt_cam
        # continue
            
        pred_seg = eval_obj_labels(all_obj_labels, target_vpt_cam, gaussians, pipe, background)
        overlayed_image = vis_one_cam(all_obj_labels, target_vpt_cam, gaussians, pipe, background, pred_mask=pred_seg)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, f"{count:02}_c_pred_{short_image_name(target_vpt_cam.image_name)}.jpg"))

        # Matching Pipeline
        yolo_bboxes = torch.load(target_vpt_cam.bbox_path) # all bbox&seg pairs from target cam
        print("yolo_bboxes.shape", yolo_bboxes.shape)
        matched_seg = torch.zeros_like(pred_seg) 

        # All groudtruth segmentation from YOLOSAM
        all_seg = torch.zeros_like(pred_seg) # For visualization purpose only
        for mask_path in target_vpt_cam.mask_paths:
            mask = read_mask(mask_path)
            all_seg[mask] = True

        ##### Begin of iterating through existing predictions for matching, NO wheat head will be identified in this process #####
        matched_mask_idx = set()
        for obj_id in torch.unique(pred_seg).tolist():
            if obj_id == 0: # skip background
                continue 
            obj_seg = (pred_seg == obj_id)
            pred_bbox = get_bbox_from_mask(obj_seg.squeeze().numpy())
            overlap_bboxes = [tuple(bbox.tolist()) for bbox in yolo_bboxes if is_overlapping(pred_bbox, tuple(bbox.tolist()))]
            overlap_idx = [i for i, bbox in enumerate(yolo_bboxes) if is_overlapping(pred_bbox, tuple(bbox.tolist()))]
            # overlap_masks_paths = [os.path.join(os.path.dirname(target_vpt_cam.mask_paths[0]), f"{target_vpt_cam.image_name}_{str(i).zfill(3)}.png") for i in overlap_idx]
            recalls, precisions = [], []
            
            max_precision = 0.0
            max_pre_mask_idx = None

            ###### Begin of Finding the best match ######
            for mask_idx in overlap_idx: # or for i, mask_path in enumerate(overlap_masks_paths):
                mask_path = os.path.join(os.path.dirname(target_vpt_cam.mask_paths[0]), f"{target_vpt_cam.image_name}_{str(mask_idx).zfill(3)}.png")
                assert mask_path in target_vpt_cam.mask_paths, f"{mask_path} not found in current image's masks"
                mask = read_mask(mask_path)
                assert mask.size() == obj_seg.size()
                # if recall > 0.5, then candidate
                if calculate_recall(pred=obj_seg, gt=mask) > 0.5:
                    precision = calculate_precision(pred=obj_seg, gt=mask)
                    # for case where there're two candidates, select the one with larger precision
                    if precision > max_precision:
                        max_precision = precision
                        max_pre_mask_idx = mask_idx
            if max_pre_mask_idx is not None:
                if max_pre_mask_idx in matched_mask_idx:
                    print("Note that this mask is already matched! Please check manually!")
                else:
                    matched_mask_idx.add(max_pre_mask_idx)
                    mask_path = os.path.join(os.path.dirname(target_vpt_cam.mask_paths[0]), f"{target_vpt_cam.image_name}_{str(max_pre_mask_idx).zfill(3)}.png")
                    matched_mask = read_mask(mask_path)
                    matched_seg[matched_mask] = obj_id
            ###### End of Finding the best match ######
                
        print(f"Found {len(matched_mask_idx)}/{len(target_vpt_cam.mask_paths)} masks matched")
        ##### End of iterating through existing predictions #####
        gt_image = target_vpt_cam.original_image.cpu()
        rgb_mask = visualize_obj(matched_seg) / 255.0
        overlayed_image = overlay_image(gt_image, rgb_mask)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, f"{count:02}_e_matched_{short_image_name(target_vpt_cam.image_name)}.jpg"))
        
        unmatched_count = 0
        for mask_idx, _ in enumerate(yolo_bboxes): # conforms with overlap_idx
            if mask_idx in matched_mask_idx:
                continue
            else:
                unmatched_count += 1
                num_wheat_heads += 1
                new_mask_path = os.path.join(os.path.dirname(target_vpt_cam.mask_paths[0]), f"{target_vpt_cam.image_name}_{str(mask_idx).zfill(3)}.png")
                new_mask = read_mask(new_mask_path)
                matched_seg[new_mask] = num_wheat_heads
        print("Num Wheat head", num_wheat_heads)
        gt_image = target_vpt_cam.original_image.cpu()
        rgb_mask = visualize_obj(matched_seg) / 255.0
        overlayed_image = overlay_image(gt_image, rgb_mask)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, f"{count:02}_f_new_{short_image_name(target_vpt_cam.image_name)}.jpg"))

        new_used_count, _ = opt_w_masks(target_vpt_cam, gaussians, pipe, background, matched_seg, num_wheat_heads)
        print("new_used_count.shape", new_used_count.shape, num_wheat_heads)
        tensor = new_used_count[1:].detach().cpu()
        print(f"New Used count Min: {torch.min(tensor[tensor != 0]):.4f}, Mean: {torch.mean(tensor[tensor != 0]):.4f}, Max: {torch.max(tensor[tensor != 0]):.4f}")

        # updated ALL counts
        zero_padding = torch.zeros((new_used_count.shape[0]-all_counts.shape[0], all_counts.shape[1]), dtype=all_counts.dtype)
        updated_all_counts = torch.cat((all_counts, zero_padding), dim=0)
        assert new_used_count.shape == updated_all_counts.shape
        # updated_used_count *= 0.5 # Scale down the previous counts
        updated_all_counts += new_used_count
        all_counts = updated_all_counts # update ALL counts

        # Update counts of wheat head segmentation defined
        new_seg_counts = torch.zeros((new_used_count.shape[0]))
        non_zero_rows = torch.any(new_used_count != 0, dim=1)
        rows_with_non_zero = torch.nonzero(non_zero_rows, as_tuple=True)[0]
        new_seg_counts[rows_with_non_zero] = 1
        print("new_seg_counts.sum", torch.sum(new_seg_counts))

        zero_padding = torch.zeros(new_used_count.shape[0]-seg_counts.shape[0], dtype=seg_counts.dtype)
        updated_seg_counts = torch.cat((seg_counts, zero_padding), dim=0)
        assert updated_seg_counts.shape == new_seg_counts.shape
        updated_seg_counts += new_seg_counts
        print("updated_seg_counts.unique", torch.unique(updated_seg_counts, return_counts=True))
        seg_counts = updated_seg_counts

        # print("seg_counts.view(-1, 1).shape", seg_counts.view(-1, 1).shape, "all_counts.shape", all_counts.shape)
        weight = ((seg_counts / 10) + 1).view(-1, 1)
        weighted_all_counts = all_counts / seg_counts.view(-1, 1) # Update WEIGHTED all counts
        # weighted_all_counts *= weight

        # all_obj_labels = multi_instance_opt(updated_used_count, slackness)
        # vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts(target_vpt_cams[0], curr_vpt_cams[0], N=20), gaussians, pipe, background)
        
        # curr_vpt_cam = target_vpt_cam
        overlap_rgb = vis_overlap(pred_seg, matched_seg, all_seg)
        torchvision.utils.save_image(overlap_rgb, os.path.join(out_dir, f"{count:02}_d_overlap_{short_image_name(target_vpt_cam.image_name)}.jpg"))

        for file_path in glob.glob(os.path.join(out_dir, "*all_counts.pt")):
            if os.path.isfile(file_path):  
                os.remove(file_path)
                print(f"Removed: {file_path}")
        torch.save(weighted_all_counts, os.path.join(out_dir, f"{count:02}_g_w_all_counts.pt"))

    
    # print("Odd", len(odd_vpt_stack), [vpt.image_name for vpt in odd_vpt_stack]) 
    # print("Even", len(even_vpt_stack), [vpt.image_name for vpt in even_vpt_stack]) 
        
    gaussians.save_ply(f"{out_dir}/gaussians.ply")
    results.close()
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--load_iteration', type=str, default="-1")
    parser.add_argument("--exp_name", type=str, help="Exp name")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IOU threshold for matching")
    parser.add_argument("--num_match", type=int, default=5, help="Num of matches required")
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.load_iteration, args.exp_name, args.iou_threshold, args.num_match)
    