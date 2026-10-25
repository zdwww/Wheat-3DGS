"""FlashSplat-based helpers shared by the 3D segmentation scripts (run_3d_seg.py, run_3d_seg_paper.py):
lift 2D masks to per-Gaussian labels and match a lifted wheat head to the 2D masks of other views."""
import os
import gc
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from gaussian_renderer import flashsplat_render
from scene import GaussianModel, Camera
from utils.wheatgs_utils import (
    PILtoTorch,
    binarize_mask,
    get_bbox_from_mask,
    is_overlapping,
    calculate_seg_iou,
)

########### Begin of Find & Match helper methods ###########

# This function is adapted from the implementation in:
# "FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally" by Shen et. al.
# Paper: https://arxiv.org/abs/2409.08270
# Original Code: https://github.com/florinshen/FlashSplat
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

def find_match(target_viewpoint_stack, gs_params, obj_used_mask, iou_threshold, dir_name):
    """
    Input: 
        target_viewpoint_stack: a list of viewpoints to iterate
        gs_params: gaussians, pipe, background
        obj_used_mask: pre-optimized flashsplat results
    Output:
    """
    gaussians, pipe, background = gs_params
    new_viewpoint_stack = []
    match_mask_paths = []
    sum_max_iou = 0.0
    # print(f"Length of target vpt stack to be matched: {len(target_viewpoint_stack)}")
    for viewpoint_cam in target_viewpoint_stack:
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
        overlap_masks_paths = [os.path.join(dir_name, f"{viewpoint_cam.image_name}_{str(i).zfill(3)}.png") for i in overlap_idx]
        for p in overlap_masks_paths:
            assert p in viewpoint_cam.mask_paths, f"{p} not found in current image's masks"

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
            # print(f"find a mathch with IOU={max_iou} with seg {match_mask_name}") 
    
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
