import os
import gc
import csv
import glob
import random
import torch
import string
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
from rich.console import Console
import torchvision
from utils.wheatgs_utils import (
    normalize_to_0_1,
    PILtoTorch,
    binarize_mask,
    gray_tensor_to_PIL,
    rgb_tensor_to_PIL,
    overlay_img_w_mask,
    get_bbox_from_mask,
    is_overlapping,
    calculate_bbox_iou,
    calculate_seg_iou
)
CONSOLE = Console()

def find_new_mask_dir(out_dir, num_wheat_head):
    base_dir = f"{out_dir}/{num_wheat_head:04}"
    existing_dirs = glob.glob(f"{base_dir}*")
    assert existing_dirs, f"Error: No existing directory found for {base_dir}*"
    for letter_suffix in string.ascii_lowercase:  # 'a' to 'z'
        candidate_dir = f"{base_dir}_{letter_suffix}"
        if candidate_dir not in existing_dirs:
            new_dir = candidate_dir
            break
    return letter_suffix

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

########### Begin of Find & Match helper methods ###########

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
                    pipeline, background, pts_filter=None):
    """
    Helper function that wraps Gaussians label optimization schema into one function
    return:
        all_counts: counts that are additive
        all_obj_labels:
    """
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

    # Filter points that are below threshold
    if pts_filter is not None:
        cols_to_modify = pts_filter.nonzero(as_tuple=True)[0]
        all_counts[1:, cols_to_modify] = 0
    return all_counts

def counts_to_obj_labels(all_counts, slackness=0.0):
    """
    Input: additive all_counts
    Output: all_obj_labels
    """
    all_obj_labels = multi_instance_opt(all_counts, slackness)
    print(f"{torch.sum(all_obj_labels, dim=1)[1]} Gaussians identified")
    return all_obj_labels

def find_match(target_viewpoint_stack, gs_params, obj_used_mask, iou_threshold, dir_name, save_dir):
    """
    Input: 
        target_viewpoint_stack: a list of viewpoints to iterate
        gs_params: gaussians, pipe, background
        obj_used_mask: pre-optimized flashsplat results
        
    """
    gaussians, pipe, background = gs_params
    new_viewpoint_stack = []
    match_mask_paths = []
    sum_max_iou = 0.0
    # print(f"Length of target vpt stack to be matched: {len(target_viewpoint_stack)}")
    for viewpoint_cam in target_viewpoint_stack:
        this_save_dir = os.path.join(save_dir) # , viewpoint_cam.image_name)
        os.makedirs(this_save_dir, exist_ok=True)
        with torch.no_grad():
            # Go through other cameras to find match
            render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask)
            render_alpha = render_pkg["alpha"]
            pred_seg = render_alpha.squeeze().detach().cpu().numpy() > 0.5          
            mask = Image.fromarray(np.where(pred_seg, 255, 0).astype(np.uint8), mode='L')
            mask.save(os.path.join(this_save_dir, f"{viewpoint_cam.image_name}_render.png"))
        pred_bbox = get_bbox_from_mask(pred_seg) # get outer bounding box of segmentation
        # Load YOLO bounding boxes
        bboxes = torch.load(viewpoint_cam.bbox_path) / viewpoint_cam.resolution_scale
        # Overlap boxes xyxy, id and mIOU
        overlap_bboxes = [tuple(box.tolist()) for box in bboxes if is_overlapping(pred_bbox, tuple(box.tolist()))]
        overlap_idx = [i for i, box in enumerate(bboxes) if is_overlapping(pred_bbox, tuple(box.tolist()))]
        # Infer SAM-generated segmentation from bounding boxes
        # overlap_masks_paths = [mask_path for mask_path in viewpoint_cam.mask_paths if int(os.path.basename(mask_path)[-7:-4]) in overlap_idx]
        overlap_masks_paths = [os.path.join(dir_name, f"{viewpoint_cam.image_name}_{str(i).zfill(3)}.png") for i in overlap_idx]
        for p in overlap_masks_paths:
            assert p in viewpoint_cam.mask_paths, f"{p} not found in current image's masks"
            # shutil.copy(p, this_save_dir) 

        # Find the bbox/seg pair with largest Segmentation IOU between the rendering
        max_iou = 0.0
        max_overlap_mask_path = None
        for mask_path in overlap_masks_paths:
            with Image.open(mask_path) as temp:
                mask = binarize_mask(PILtoTorch(temp.copy(), viewpoint_cam.resolution)).squeeze().numpy() > 0
                assert mask.shape == pred_seg.shape
            iou = calculate_seg_iou(mask, pred_seg)
            if iou > max_iou:
                max_iou = iou
                max_overlap_mask_path = mask_path
                                    
        if max_iou > iou_threshold: # Hyperparameters to modify
            # Add matched viewpoint cam and matched seg to a list
            new_viewpoint_stack.append(viewpoint_cam)
            match_mask_paths.append(max_overlap_mask_path)
            sum_max_iou += max_iou
            match_mask_name = os.path.splitext(os.path.basename(max_overlap_mask_path))[0]
            # processed_masks.add(match_mask_name) # Don't add matched to processed here!
            print(f"find a mathch with IOU={max_iou} with seg {match_mask_name}")
            shutil.copy(max_overlap_mask_path, os.path.join(this_save_dir, f"{match_mask_name}_match.png")) 
    
    assert len(new_viewpoint_stack) == len(match_mask_paths)
    print(f"Total of {len(new_viewpoint_stack)} / {len(target_viewpoint_stack)} matches" +
        (f" with mean IOU {sum_max_iou / len(new_viewpoint_stack)} > {iou_threshold} found for refine training." 
        if len(new_viewpoint_stack) > 0 else ""))
    return new_viewpoint_stack, match_mask_paths

def update_processed_masks(processed_masks, new_mask_paths):
    for new_mask_path in new_mask_paths:
        new_mask_name = os.path.splitext(os.path.basename(new_mask_path))[0]
        processed_masks.add(new_mask_name)
    return processed_masks

########### End of Find & Match helper methods ###########
        
def training(dataset, opt, pipe, load_iteration, exp_name, iou_threshold):
    out_dir = os.path.join(dataset.model_path, "wheat-head", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/experiment.txt", "w") as file:
        file.write(f"exp_name {exp_name}\niou_threshold {iou_threshold}\n")
    
    results = open(os.path.join(out_dir, 'results.csv'), mode='w', newline='')
    writer = csv.writer(results)
    writer.writerow(["id", "init_mask", "num_matches", "num_GS"])
    
    gaussians = GaussianModel(dataset.sh_degree)
    try:
        load_iteration = int(load_iteration)
    except:
        pass
    print(f"Load iteration {load_iteration}, Resolution {dataset.resolution}")
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_xyz)}")

    z_mean = torch.mean(gaussians.get_xyz.cpu()[:, 2])
    print(f"All Gaussians z_min: {torch.min(gaussians.get_xyz.cpu()[:, 2])} zmax: {torch.max(gaussians.get_xyz.cpu()[:, 2])}")
    pts_filter = (gaussians.get_xyz.cpu()[:, 2] < z_mean)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack_eval = scene.getTestCameras().copy()

    print(f"Length of viewpoint stack: {len(viewpoint_stack)}")
    
    twoD_seg_results = {} # 2D segmentation results update through the pipeline
    os.makedirs(f"{out_dir}/2DSeg", exist_ok=True)
    all_mask_paths = [] # a list of saved binary masks in png format
    num_bboxes = 0
    ### Initialize and save 2D Segmentation
    for viewpoint_cam in viewpoint_stack:
        bboxes = torch.load(viewpoint_cam.bbox_path)
        num_bboxes += len(bboxes)
        all_mask_paths += viewpoint_cam.mask_paths
        twoD_seg_results[viewpoint_cam.image_name] = torch.zeros(viewpoint_cam.original_image.shape[1:], dtype=torch.int)
        torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")
    # Save for eval images too
    for viewpoint_cam in viewpoint_stack_eval:
        twoD_seg_results[viewpoint_cam.image_name] = torch.zeros(viewpoint_cam.original_image.shape[1:], dtype=torch.int)
        torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")
    
    assert len(all_mask_paths) == num_bboxes
    print(f"Total of {len(all_mask_paths)} mask & bounding box pairs found")

    random.shuffle(all_mask_paths)
    processed_masks = set()
    buffered_masks = set()
    num_wheat_head = 0
    
    #### Iterate through all YOLO/SAM bbox/seg pairs
    # for exp_id, this_mask_path in tqdm(enumerate(all_mask_paths), total=len(all_mask_paths), desc="Processing Masks"):
    if True:
        exp_id = 0 
        this_mask_path = "/cluster/scratch/daizhang/Wheat-GS-data-scaled/20240717/plot_461/masks/FPWW036_SR0461_1_FIP2_cam_04_006.png"
        print("-" * 50)
        this_mask_name = os.path.splitext(os.path.basename(this_mask_path))[0]

        this_image_name = this_mask_name[:-4]
        mask_idx = int(this_mask_name[-3:])
        CONSOLE.print(f"==== Train 3D segmentation against {this_mask_name} ====")
        
        this_viewpoint_cam = next(cam for cam in viewpoint_stack if cam.image_name == this_image_name)
        
        # Optimize Gaussians' labels w.r.t ONE segmentation
        # NOTE: all_counts is additive
        all_counts = opt_label_w_seg(gaussians, [this_viewpoint_cam], [this_mask_path], pipe, background, pts_filter)
        all_obj_labels = counts_to_obj_labels(all_counts)
        if torch.sum(all_obj_labels, dim=1)[1] == 0:
            print(f"Can't identify any Gaussians above average height for this mask: {this_mask_name}), PASS")
        obj_used_mask = (all_obj_labels[1]).bool()

        # Save initial Gaussians
        gaussians_obj = deepcopy(gaussians)
        gaussians_obj.reset_label(obj_used_mask=obj_used_mask, set_which_object_to=1)
        gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != 1), during_training=False)
        gaussians_obj.save_ply(f"{out_dir}/wh_it0.ply")


        #### Render from other cameras
        # Initialize a list of consistent segmentation for future fine-tuning
        matched_viewpoint_stack = [this_viewpoint_cam]
        matched_mask_paths = [this_mask_path]
        
        os.makedirs(f"{out_dir}/match", exist_ok=True)
        new_viewpoint_stack, new_mask_paths = find_match(
            target_viewpoint_stack = [vpt for vpt in viewpoint_stack if vpt.image_name != this_image_name],
            gs_params = (gaussians, pipe, background),
            obj_used_mask = obj_used_mask,
            iou_threshold = iou_threshold,
            dir_name = os.path.dirname(this_mask_path),
            save_dir = os.path.join(out_dir, "it0")
        )
    
        matched_viewpoint_stack += new_viewpoint_stack # as a whole
        matched_mask_paths += new_mask_paths
        processed_masks = update_processed_masks(processed_masks, new_mask_paths)
        CONSOLE.print(f"==== Find {len(new_mask_paths)} newly matched masks. {len(matched_mask_paths)} matched in total ====")

        #### Only do Refine training w.r.t newly found segmentation when a pairf of matches is found ####
        if len(new_viewpoint_stack) > 0:
            num_wheat_head += 1 # Potential wheat head
            # if find a match, then it's processed and create a dir for it
            this_mask_dir = f"{out_dir}/{num_wheat_head:04}"
            os.makedirs(this_mask_dir, exist_ok=True)
            processed_masks.add(this_mask_name)
            
            CONSOLE.print(f"==== Start refine training w.r.t the {num_wheat_head}th potential wheat head found ====")
            
            for i in range(1, 100): 
                CONSOLE.print(f"-- fine-tuning iteration {i} --")
                assert len(new_viewpoint_stack) == len(new_mask_paths)
                # Update 3D Segmentation
                update_counts = opt_label_w_seg(gaussians, new_viewpoint_stack, new_mask_paths, pipe, background)
                assert update_counts.shape == all_counts.shape
                all_counts += update_counts # update all counts
                all_obj_labels = counts_to_obj_labels(all_counts)
                obj_used_mask = (all_obj_labels[1]).bool()
                gaussians_obj = deepcopy(gaussians)
                gaussians_obj.reset_label(obj_used_mask=obj_used_mask, set_which_object_to=1)
                gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != 1), during_training=False)
                gaussians_obj.save_ply(f"{out_dir}/wh_it{str(i)}.ply")
                # fine-tuning
                new_viewpoint_stack, new_mask_paths = find_match(
                    target_viewpoint_stack = [
                        vpt for vpt in viewpoint_stack if vpt.image_name not in {mpt.image_name for mpt in matched_viewpoint_stack}
                    ],
                    gs_params = (gaussians, pipe, background),
                    obj_used_mask = obj_used_mask,
                    iou_threshold = iou_threshold,
                    dir_name = os.path.dirname(this_mask_path),
                    save_dir = os.path.join(out_dir, f"it{str(i)}")
                )
                if len(new_viewpoint_stack) == 0:
                    CONSOLE.print(f"No new matched found from {len(viewpoint_stack) - len(matched_viewpoint_stack)} vpts")
                    break
                else:
                    matched_viewpoint_stack += new_viewpoint_stack # as a whole
                    matched_mask_paths += new_mask_paths
                    processed_masks = update_processed_masks(processed_masks, new_mask_paths)
                    CONSOLE.print(f"-- Find {len(new_mask_paths)} newly matched masks. {len(matched_mask_paths)} matched in total --")
            
            return

            # Check if Gaussians are largely overlap with previously identified wheat head
            which_overlap_object = gaussians.reset_label(obj_used_mask=obj_used_mask, set_which_object_to=num_wheat_head)
            gaussians_obj = deepcopy(gaussians)
            if which_overlap_object is not None:
                num_wheat_head -= 1 # if overlapping, then it's not a new wheat head
                shutil.rmtree(this_mask_dir)
                which_wheat_head = which_overlap_object
                CONSOLE.print(f"===== This wheat head is largely overlapped with previous one {which_overlap_object}, Remove created {this_mask_dir} =====")
                num_GS = torch.sum(gaussians_obj.get_which_object.detach() == which_wheat_head).item()
                gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != which_wheat_head), during_training=False)
                letter_suffix = find_new_mask_dir(out_dir, which_wheat_head)
                gaussians_obj.save_ply(f"{out_dir}/wh_{which_wheat_head:04}_{letter_suffix}.ply")
                this_mask_dir = f"{out_dir}/{which_wheat_head:04}_{letter_suffix}"
                os.makedirs(this_mask_dir, exist_ok=True)
                CONSOLE.print(f"Create new mask dir {this_mask_dir}")
                writer.writerow([f"{which_wheat_head:04}_{letter_suffix}", this_mask_name, str(len(matched_viewpoint_stack)), str(num_GS)])
                results.flush()
            else:
                CONSOLE.print(f"======== Identify {num_wheat_head}th new wheat head with {len(matched_viewpoint_stack)} matches ========")
                which_wheat_head = num_wheat_head
                num_GS = torch.sum(gaussians_obj.get_which_object.detach() == which_wheat_head).item()
                gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != which_wheat_head), during_training=False)
                gaussians_obj.save_ply(f"{out_dir}/wh_{which_wheat_head:04}.ply")
                writer.writerow([f"{which_wheat_head:04}", this_mask_name, str(len(matched_viewpoint_stack)), str(num_GS)])
                results.flush()

            #### Evaluation of refined training ####
            # os.makedirs(f"{this_mask_dir}/overlay", exist_ok=True)  
            for i, viewpoint_cam in enumerate(viewpoint_stack + viewpoint_stack_eval):
                with torch.no_grad():
                    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask)
                    render_alpha = render_pkg["alpha"].squeeze().detach().cpu()
                    pred_seg = render_alpha.numpy() > 0.5          
                    mask = Image.fromarray(np.where(pred_seg, 255, 0).astype(np.uint8), mode='L')
                    # mask.save(f"{this_mask_dir}/masks/{viewpoint_cam.image_name}.jpg")
                    vis_image_w_overlay(img_tensor=viewpoint_cam.original_image, 
                                        save_dir=f"{this_mask_dir}",
                                        save_name=viewpoint_cam.image_name,
                                        pred_seg=pred_seg,
                                        resize_factor=4)
                    # Update the 2D seg&count results
                    assert twoD_seg_results[viewpoint_cam.image_name].shape == render_alpha.shape
                    twoD_seg_results[viewpoint_cam.image_name][render_alpha > 0.5] = which_wheat_head
                    # Update the saved 2D seg saved
                    torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")
                    
        else:
            CONSOLE.print(f"==== Not matchings found for {this_mask_name}. Add to Buffer. ====")
            if this_mask_name not in processed_masks and this_mask_name not in buffered_masks:
                buffered_masks.add(this_mask_name)

        if exp_id % 5 == 0: # Save Gaussians every 5 distinct masks
            gaussians.save_ply(f"{out_dir}/gaussians.ply")
            print("Gaussians saved!")

        print(f"======== Processed masks {len(processed_masks.union(buffered_masks))} / {len(all_mask_paths)} ========")
        print("-" * 50)
        
    # gaussians.save_ply(f"{out_dir}/gaussians.ply")
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
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.load_iteration, args.exp_name, args.iou_threshold)