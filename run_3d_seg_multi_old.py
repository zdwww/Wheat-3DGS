import os
import gc
import csv
import random
import torch
import shutil
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
from utils.wheatgs_helper import get_interpolated_viewpts, nearest_k_viewpts, get_interpolated_viewpts_old, short_image_name

out_dir = None

########### Begin of Visualization Helper Functions, will migrate to utils dir later ###########

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
    print(render_pkg["render"].shape)
    used_count = render_pkg["used_count"].detach().cpu()
    return used_count

def eval_obj_labels(all_obj_labels, viewpoint_cam, gaussians, pipe, background):
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
    
def vis_interpolated_cams(all_obj_labels, vis_viewpoint_stack, gaussians, pipe, background):
    """Visualization of 3D segmentation with interpolated camera poses"""
    global out_dir
    os.makedirs(os.path.join(out_dir, "interpolated"), exist_ok=True) 
    for idx in tqdm(range(len(vis_viewpoint_stack)), desc="Visualizing interpolated cameras progress"):
        inner_cam = vis_viewpoint_stack[idx]
        overlayed_image = vis_one_cam(all_obj_labels, inner_cam, gaussians, pipe, background)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, "interpolated", f"{idx:03}.jpg"))

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
    used_count = opt_w_masks(viewpoint_cam, gaussians, pipe, background, obj_masks) 
    return used_count, obj_num
        
def training(dataset, opt, pipe, load_iteration, exp_name, iou_threshold, num_match):
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

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    print(f"Length of viewpoint stack: {len(viewpoint_stack)}")
    
    #### Iterate through all vpts for preprocessing
    # os.makedirs(f"{out_dir}/2DSeg", exist_ok=True)
    num_bboxes = 0
    all_mask_paths = [] # a list of saved binary masks in png format
    cam_centers = [] 
    twoD_seg_results = {} # 2D segmentation results update through the pipeline
    for viewpoint_cam in viewpoint_stack:
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

    #### Main framework for propagate cameras
    curr_vpt_cams = [viewpoint_stack[0]]
    # viewpoint_stack = viewpoint_stack[1:]
    # Two rows of cameras
    odd_vpt_stack = [vpt for vpt in viewpoint_stack if int(vpt.image_name.split("_")[-1]) % 2 != 0]
    even_vpt_stack = [vpt for vpt in viewpoint_stack if int(vpt.image_name.split("_")[-1]) % 2 == 0]

    num_wheat_heads = 0

    curr_vpt_cam = odd_vpt_stack[0]

    while len(odd_vpt_stack) > 0:
        print(f"Current CAM {short_image_name(curr_vpt_cam.image_name)}")
        curr_vpt_cam_idx = int(curr_vpt_cam.image_name.split("_")[-1])
        nearest_odd_vpts, odd_vpt_stack  = nearest_k_viewpts(odd_vpt_stack, curr_vpt_cam.camera_center.cpu(), 1)
        target_vpt_cam = nearest_odd_vpts[0]
        print(f"Target CAM {short_image_name(target_vpt_cam.image_name)}")
        curr_vpt_cam = target_vpt_cam

    return
    
    
    while len(odd_vpt_stack) > 0 and len(even_vpt_stack) > 0:
        ############## Begin of Selection of Cameras ##############
        # Curr_cams: optimize with FlashSplat; Target_cams: where find matches
        print(f"Current cams {[cam.image_name for cam in curr_vpt_cams]}")
        target_vpt_cams = []
        if len(curr_vpt_cams) == 1:
            curr_vpt_cam = curr_vpt_cams[0]
            curr_vpt_cam_idx = int(curr_vpt_cam.image_name.split("_")[-1])
            # Find nearest cam on the OPPOSITE row
            if curr_vpt_cam_idx % 2 == 0: # Even index
                nearest_odd_vpts, odd_vpt_stack  = nearest_k_viewpts(odd_vpt_stack, curr_vpt_cam.camera_center.cpu(), 1)
                target_vpt_cams += nearest_odd_vpts
            else: # Odd index 
                nearest_even_vpts, even_vpt_stack = nearest_k_viewpts(even_vpt_stack, curr_vpt_cam.camera_center.cpu(), 1)
                target_vpt_cams += nearest_even_vpts
        elif len(curr_vpt_cams) == 2:
            for curr_vpt_cam in curr_vpt_cams:
                curr_vpt_cam_idx = int(curr_vpt_cam.image_name.split("_")[-1])
                # Reverse of the previous block, find nearest cam in the SAME row
                if curr_vpt_cam_idx % 2 == 0: # Even index
                    nearest_even_vpts, even_vpt_stack = nearest_k_viewpts(even_vpt_stack, curr_vpt_cam.camera_center.cpu(), 1)
                    target_vpt_cams += nearest_even_vpts
                else: # Odd index 
                    nearest_odd_vpts, odd_vpt_stack  = nearest_k_viewpts(odd_vpt_stack, curr_vpt_cam.camera_center.cpu(), 1)
                    target_vpt_cams += nearest_odd_vpts
        else:
            raise ValueError("There shoud be only 1 or 2 current viewpoint cams")
        print("Target", len(target_vpt_cams), [cam.image_name for cam in target_vpt_cams])
        print(f"Odd left {len(odd_vpt_stack)} Even left {len(even_vpt_stack)}") 
        ############## End of Selection of Cameras ##############

        ########## Begin of Optimization w.r.t CURRENT cameras ##########
        assert len(curr_vpt_cams) == 1
        curr_vpt_cam = curr_vpt_cams[0]
        
        # Initial Segmentation from first view
        used_count, obj_num = seg_first_view(curr_vpt_cam, gaussians, pipe, background)
        print(f"Cam {curr_vpt_cam.image_name} with {obj_num} objs used count {used_count.shape} minmax {torch.min(used_count)} {torch.max(used_count)}")
        num_wheat_heads += obj_num
        slackness = 0.0
        all_obj_labels = multi_instance_opt(used_count, slackness)
        ########## End of Optimization w.r.t CURRENT cameras ##########

        ## Interpolation visualization
        # vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts(curr_vpt_cams[0], target_vpt_cams[0], N=20), gaussians, pipe, background)
        # vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts_old(viewpoint_stack, N=200), gaussians, pipe, background)

        ########## Begin of Move to TARGET cam and find matches ##########
        assert len(target_vpt_cams) == 1
        target_vpt_cam = target_vpt_cams[0]
        # vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts(curr_vpt_cams[0], target_vpt_cam, N=20), gaussians, pipe, background)
            
        pred_seg = eval_obj_labels(all_obj_labels, target_vpt_cam, gaussians, pipe, background)
        overlayed_image = vis_one_cam(all_obj_labels, target_vpt_cam, gaussians, pipe, background, pred_mask=pred_seg)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, f"pred_{target_vpt_cam.image_name}.jpg"))
        print("pred_seg", pred_seg.shape, torch.unique(pred_seg, return_counts=True))

        # Matching Pipeline
        yolo_bboxes = torch.load(target_vpt_cam.bbox_path) # all bbox&seg pairs from target cam
        print("yolo_bboxes.shape", yolo_bboxes.shape)
        matched_seg = torch.zeros_like(pred_seg) 
        all_seg = torch.zeros_like(pred_seg) # For visualization purpose only
        for mask_path in target_vpt_cam.mask_paths:
            mask = read_mask(mask_path)
            all_seg[mask] = True

        ##### Begin of iterating through existing predictions, NO wheat head will be identified in this process #####
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
                
        print("Old Matched seg", matched_seg.size(), torch.unique(matched_seg, return_counts=True))
        print(f"Found {len(matched_mask_idx)}/{len(target_vpt_cam.mask_paths)} masks matched")
        ##### End of iterating through existing predictions #####
        gt_image = target_vpt_cam.original_image.cpu()
        rgb_mask = visualize_obj(matched_seg) / 255.0
        overlayed_image = overlay_image(gt_image, rgb_mask)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, f"old_matched_{target_vpt_cam.image_name}.jpg"))
        
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
        print("New Matched seg", matched_seg.size(), torch.unique(matched_seg, return_counts=True))
        gt_image = target_vpt_cam.original_image.cpu()
        rgb_mask = visualize_obj(matched_seg) / 255.0
        overlayed_image = overlay_image(gt_image, rgb_mask)
        torchvision.utils.save_image(overlayed_image, os.path.join(out_dir, f"new_matched_{target_vpt_cam.image_name}.jpg"))

        new_used_count = opt_w_masks(target_vpt_cam, gaussians, pipe, background, matched_seg, num_wheat_heads)
        print("new_used_count.shape", new_used_count.shape, num_wheat_heads)

        zero_padding = torch.zeros((new_used_count.shape[0]-used_count.shape[0], used_count.shape[1]), dtype=used_count.dtype)
        updated_used_count = torch.cat((used_count, zero_padding), dim=0)

        assert new_used_count.shape == updated_used_count.shape

        updated_used_count += new_used_count
        all_obj_labels = multi_instance_opt(updated_used_count, slackness)
        vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts(target_vpt_cams[0], curr_vpt_cams[0], N=20), gaussians, pipe, background)
        

        
        # binary_pred_seg = (pred_seg != 0)
        # binary_matched_seg = (matched_seg != 0)
        # binary_all_seg = (all_seg != 0)
        # overlap_rgb = torch.zeros((*binary_pred_seg.shape, 3), dtype=torch.uint8)
        # overlap_rgb[binary_pred_seg & ~binary_matched_seg] = torch.tensor([255, 255, 255], dtype=torch.uint8)
        # overlap_rgb[binary_all_seg] = torch.tensor((100, 100, 100), dtype=torch.uint8)
        # overlap_rgb[~binary_pred_seg & binary_matched_seg] = torch.tensor([255, 165, 0], dtype=torch.uint8)
        # overlap_rgb[binary_pred_seg & binary_matched_seg] = torch.tensor([255, 0, 0], dtype=torch.uint8)
        # overlap_rgb = overlap_rgb.squeeze().permute(2, 0, 1) / 255.0
        # torchvision.utils.save_image(overlap_rgb, os.path.join(out_dir, f"overlap_{target_vpt_cam.image_name}.jpg"))
                 
        # if len(curr_vpt_cams) == 1:
        #     curr_vpt_cams += target_vpt_cams
        # else:
        #     curr_vpt_cams = target_vpt_cams

        break
            
    
    # print("Odd", len(odd_vpt_stack), [vpt.image_name for vpt in odd_vpt_stack]) 
    # print("Even", len(even_vpt_stack), [vpt.image_name for vpt in even_vpt_stack]) 

    return
    

    random.shuffle(all_mask_paths)
    processed_masks = set()
    num_wheat_head = 0

    return
    for viewpoint_cam in viewpoint_stack:
        print(viewpoint_cam.image_name)   
        all_obj_labels = seg_one_view(viewpoint_cam, gaussians, pipe, background)
        # Evaluation 3D segmentation 
        vis_interpolated_cams(all_obj_labels, get_interpolated_viewpts(viewpoint_stack, N=200), viewpoint_cam, gaussians, pipe, background)            
        break
    return
    
    #### Iterate through all YOLO/SAM bbox/seg pairs
    for exp_id, this_mask_path in enumerate(all_mask_paths):
        this_mask_name = os.path.splitext(os.path.basename(this_mask_path))[0]
        
        if this_mask_name in processed_masks:
            print(f"{this_mask_name} already processed and saved")
            continue
        
        processed_masks.add(this_mask_name)  
        this_image_name = this_mask_name[:-4]
        mask_idx = int(this_mask_name[-3:])
        print(f"Train 3D segmentation against {this_image_name}'s {mask_idx}th mask ({this_mask_name})")
        # Get the matched viewpoint cam corresponding to that seg
        this_viewpoint_cam = next(cam for cam in viewpoint_stack if cam.image_name == this_image_name)
            
        # Save the ground-truth target mask, for manual verification only
        this_mask_dir = f"{out_dir}/{num_wheat_head:04}"
        os.makedirs(this_mask_dir, exist_ok=True)
        
        with Image.open(this_mask_path) as temp:
            this_mask = binarize_mask(PILtoTorch(temp.copy(), this_viewpoint_cam.resolution))
        vis_image_w_overlay(img_tensor=this_viewpoint_cam.original_image, 
                            save_dir=this_mask_dir, 
                            save_name=this_mask_name,
                            pred_seg=this_mask.squeeze().numpy() > 0)
        
        # Optimize Gaussians' labels w.r.t ONE segmentation
        all_obj_labels = opt_label_w_seg(gaussians, [this_viewpoint_cam], [this_mask_path], pipe, background)
        obj_used_mask = (all_obj_labels[1]).bool()

        #### Render from other cameras
        # Initialize a list of consistent segmentation for future fine-tuning
        new_viewpoint_stack = [this_viewpoint_cam]
        match_mask_paths = [this_mask_path]
        sum_max_iou = 0.0
        
        for viewpoint_cam in viewpoint_stack:
            if viewpoint_cam.image_name == this_image_name:
                continue
            else:
                with torch.no_grad():
                    # Go through other cameras to find match
                    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask)
                    render_alpha = render_pkg["alpha"]
                    pred_seg = render_alpha.squeeze().detach().cpu().numpy() > 0.5
                pred_bbox = get_bbox_from_mask(pred_seg) # get outer bounding box of segmentation
                # Load YOLO bounding boxes
                bboxes = torch.load(viewpoint_cam.bbox_path) / viewpoint_cam.resolution_scale
                # Overlap boxes xyxy, id and mIOU
                overlap_bboxes = [tuple(box.tolist()) for box in bboxes if is_overlapping(pred_bbox, tuple(box.tolist()))]
                overlap_idx = [i for i, box in enumerate(bboxes) if is_overlapping(pred_bbox, tuple(box.tolist()))]
                # Infer SAM-generated segmentation from bounding boxes
                # overlap_masks_paths = [mask_path for mask_path in viewpoint_cam.mask_paths if int(os.path.basename(mask_path)[-7:-4]) in overlap_idx]
                overlap_masks_paths = [os.path.join(os.path.dirname(this_mask_path), f"{viewpoint_cam.image_name}_{str(i).zfill(3)}.png") for i in overlap_idx]
                for p in overlap_masks_paths:
                    assert p in viewpoint_cam.mask_paths, f"{p} not found in current image's masks"

                # Find the bbox/seg pair with largest Segmentation IOU between the rendering
                max_iou = 0.0
                max_overlap_mask = None
                max_overlap_mask_path = None
                for mask_path in overlap_masks_paths:
                    with Image.open(mask_path) as temp:
                        mask = binarize_mask(PILtoTorch(temp.copy(), this_viewpoint_cam.resolution)).squeeze().numpy() > 0
                        assert mask.shape == pred_seg.shape
                    iou = calculate_seg_iou(mask, pred_seg)
                    if iou > max_iou:
                        max_iou = iou
                        max_overlap_mask = mask
                        max_overlap_mask_path = mask_path
                                         
                if max_iou > iou_threshold: # Hyperparameters to modify
                    # Add matched viewpoint cam and matched seg to a list
                    new_viewpoint_stack.append(viewpoint_cam)
                    match_mask_paths.append(max_overlap_mask_path)
                    sum_max_iou += max_iou
                    match_mask_name = os.path.splitext(os.path.basename(max_overlap_mask_path))[0]
                    # processed_masks.add(match_mask_name) # Don't add matched to processed here!
                    print(f"Find a mathch with IOU={max_iou} with seg {match_mask_name}") 
        
        assert len(new_viewpoint_stack) == len(match_mask_paths)
        print(f"Total of {len(new_viewpoint_stack)} matches with IOU > {iou_threshold} found for refine training.")

        if len(new_viewpoint_stack) >= num_match:
            #### Only do Refine training w.r.t newly found segmentation  when 2 matches (3 corresponding seg) are found ####
            num_wheat_head += 1
            print(f"Add {len(match_mask_paths)} matched masks to processed_masks")
            writer.writerow([num_wheat_head, this_mask_name, str(len(new_viewpoint_stack)), f"{sum_max_iou/(len(new_viewpoint_stack)-1):.4f}"])
            results.flush()
            
            for match_mask_path in match_mask_paths:
                match_mask_name = os.path.splitext(os.path.basename(match_mask_path))[0]
                processed_masks.add(match_mask_name)

            print(f"Start refine training w.r.t the {num_wheat_head}th wheat head found")
            # train_label_w_seg(gaussians, new_viewpoint_stack, match_mask_paths, opt, background, iterations=6000)
            all_obj_labels = opt_label_w_seg(gaussians, new_viewpoint_stack, match_mask_paths, pipe, background)
            obj_used_mask = (all_obj_labels[1]).bool()
            
            #### Evaluation of refined training ####
            os.makedirs(f"{this_mask_dir}/masks", exist_ok=True)
            os.makedirs(f"{this_mask_dir}/refine", exist_ok=True)  
            for i, viewpoint_cam in enumerate(viewpoint_stack):
                with torch.no_grad():
                    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask)
                    render_alpha = render_pkg["alpha"].squeeze().detach().cpu()
                    pred_seg = render_alpha.numpy() > 0.5          
                    mask = Image.fromarray(np.where(pred_seg, 255, 0).astype(np.uint8), mode='L')
                    mask.save(f"{this_mask_dir}/masks/{viewpoint_cam.image_name}.png")
                    vis_image_w_overlay(img_tensor=viewpoint_cam.original_image, 
                                        save_dir=f"{this_mask_dir}/refine",
                                        save_name=viewpoint_cam.image_name,
                                        pred_seg=pred_seg,
                                        resize_factor=4)
                    # Update the 2D seg&count results
                    assert twoD_seg_results[viewpoint_cam.image_name].shape == render_alpha.shape
                    twoD_seg_results[viewpoint_cam.image_name][render_alpha > 0.5] = num_wheat_head
                    # Update the saved 2D seg saved
                    torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")
            gaussians.reset_label(obj_used_mask=obj_used_mask, set_which_object_to=num_wheat_head)
            
            gaussians_obj = deepcopy(gaussians)
            gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != num_wheat_head), during_training=False)
            gaussians_obj.save_ply(f"{this_mask_dir}/{num_wheat_head:04}.ply")
            # print(f"Num of Gaussians corresponding to each wheat head {torch.unique(gaussians.get_which_object.cpu(), return_counts=True)}")
        else:
            # Remove the saved segmentation if not enough matching is found 
            shutil.rmtree(this_mask_dir)
            print(f"Not enough matchings are found. Remove files at {this_mask_dir}")

        if exp_id % 5 == 0: # Save Gaussians every 5 distinct masks
            gaussians.save_ply(f"{out_dir}/gaussians.ply")
            print("Gaussians saved!")
        print(f"Processed masks {len(processed_masks)} / {len(all_mask_paths)}")
        
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
    