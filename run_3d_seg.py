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

########### Begin of Image Helper Functions, will migrate to utils dir later ###########
def normalize_to_0_1(img_tensor):
    # Normalize PIL loaded image tensor from 0-255 to 0-1
    if torch.max(img_tensor) > 1.0:
        return (img_tensor / 255.0).clamp(0.0, 1.0)
    else:
        return img_tensor

def PILtoTorch(pil_image, resolution, normalize=True):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)).float()
    if normalize:
        resized_image = normalize_to_0_1(resized_image)
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1) # (H,W,3) -> (3,H,W)
    elif len(resized_image.shape) == 2:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    else:
        raise ValueError("PIL.Image shape not recognized")

def binarize_mask(mask_tensor):
    assert torch.min(mask_tensor) >= 0.0 and torch.max(mask_tensor) <= 1.0, "Mask tensor should be in the range [0, 1]"
    if mask_tensor.shape[0] == 1:
        mask_tensor = torch.where(mask_tensor > 0, 1.0, 0.0)
    elif mask_tensor.shape[0] == 3:
        mask_tensor = (mask_tensor > 0.0).any(dim=0).unsqueeze(dim=0).float()
        assert mask_tensor.shape[0] == 1
    else:
        raise ValueError("Mask tensor should have 1 or 3 channels")
    # assert mask_tensor has two unique value 0 and 1
    assert torch.all((mask_tensor == 0) | (mask_tensor == 1)), "Mask tensor should have two unique values 0 and 1"
    return mask_tensor

def gray_tensor_to_PIL(tensor : torch.Tensor):
    return Image.fromarray((torch.clamp(tensor.detach().cpu(), 0, 1).numpy().squeeze() * 255.0).astype(np.uint8))

def rgb_tensor_to_PIL(tensor : torch.Tensor):
    return Image.fromarray((np.transpose(torch.clamp(tensor.detach().cpu(), 0, 1).numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))

def overlay_img_w_mask(image_pil, mask_pil, color="red"):
    if color == "red":
        overlay = Image.new("RGBA", image_pil.size, (255, 0, 0, 0))    
        overlay = Image.composite(Image.new("RGBA", image_pil.size, (255, 0, 0, 128)), overlay, mask_pil)
    elif color == "blue":
        overlay = Image.new("RGBA", image_pil.size, (0, 0, 255, 0))    
        overlay = Image.composite(Image.new("RGBA", image_pil.size, (0, 0, 255, 128)), overlay, mask_pil)
    image_pil = image_pil.convert("RGBA")
    image_with_overlay = Image.alpha_composite(image_pil, overlay)
    image_with_overlay_rgb = image_with_overlay.convert("RGB")
    return image_with_overlay_rgb

def get_bbox_from_mask(mask):
    object_pixels = np.argwhere(mask == 1)
    if object_pixels.size == 0:
        return None
    # Get the min and max x and y coordinates
    y_min, x_min = object_pixels.min(axis=0)
    y_max, x_max = object_pixels.max(axis=0)
    # Return the bounding box in xyxy format
    return (x_min, y_min, x_max, y_max)

def is_overlapping(box1, box2):
    if box1 is None or box2 is None:
        return False
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    # Check if one box is to the left or right of the other, or if one is above or below the other
    if x_max1 < x_min2 or x_max2 < x_min1:
        return False  # One box is to the left of the other
    if y_max1 < y_min2 or y_max2 < y_min1:
        return False  # One box is above the other
    return True

def calculate_bbox_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the intersection coordinates
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate the area of the intersection
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    # Calculate the areas of each box
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_seg_iou(mask1, mask2):
    # Calculate intersection (logical AND)
    intersection = np.logical_and(mask1, mask2)
    
    # Calculate union (logical OR)
    union = np.logical_or(mask1, mask2)
    
    # Compute IoU
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
    return iou

########### End of Image Helper Functions ###########

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

def train_label_w_seg(gaussians : GaussianModel, 
                      viewpoint_stack : List[Camera], 
                      mask_paths : List[torch.Tensor], 
                      opt, background, iterations=10000, enable_progress_bar=False):
    """Helper function that wraps Gaussians label training schema into one function"""
    assert len(viewpoint_stack) == len(mask_paths)
    gaussians.training_setup(opt)
    first_iter = 0
    if enable_progress_bar:
        progress_bar = tqdm(range(first_iter, iterations), desc="Training progress")
    first_iter += 1
    gaussians.update_lr_for_label(label_lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Start training w.r.t seg; length of viewpoints: {len(viewpoint_stack)}")
    
    for iteration in range(first_iter, iterations + 1):
        random_index = random.randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack[random_index]
        render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=True)
        # Load binary mask
        with Image.open(mask_paths[random_index]) as temp:
            mask = binarize_mask(PILtoTorch(temp.copy(), viewpoint_cam.resolution).to("cuda"))
        loss = criterion(input=render_label, target=mask)
        loss.backward()
        with torch.no_grad():
            if enable_progress_bar:
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                    progress_bar.update(10)
                if iteration == iterations:
                    progress_bar.close()
            ## Optimizer step
            if iteration <= iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        
def training(dataset, opt, pipe, total_iterations=10000):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
    gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_label)}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()

    
    twoD_seg_results = {} # 2D segmentation results update through the pipeline
    os.makedirs(f"output/2DSeg", exist_ok=True)
    all_mask_paths = [] # a list of saved binary masks in png format
    num_bboxes = 0
    for viewpoint_cam in viewpoint_stack:
        bboxes = torch.load(viewpoint_cam.bbox_path)
        num_bboxes += len(bboxes)
        all_mask_paths += viewpoint_cam.mask_paths
        twoD_seg_results[viewpoint_cam.image_name] = torch.zeros(viewpoint_cam.original_image.shape[1:], dtype=torch.int)
        torch.save(twoD_seg_results[viewpoint_cam.image_name], f"output/2DSeg/{viewpoint_cam.image_name}.pt")
    
    assert len(all_mask_paths) == num_bboxes
    print(f"Total of {len(all_mask_paths)} mask & bounding box pairs found")

    random.shuffle(all_mask_paths)
    processed_masks = set()
    num_wheat_head = 0

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
        
        this_viewpoint_cam = next(cam for cam in viewpoint_stack if cam.image_name == this_image_name)
            
        # Save the ground-truth target mask, for manual verification only
        os.makedirs(f"output/{this_mask_name}", exist_ok=True)
        with Image.open(this_mask_path) as temp:
            this_mask = binarize_mask(PILtoTorch(temp.copy(), this_viewpoint_cam.resolution))
        vis_image_w_overlay(img_tensor=this_viewpoint_cam.original_image, 
                            save_dir=f"output/{this_mask_name}", 
                            save_name=this_mask_name,
                            pred_seg=this_mask.squeeze().numpy() > 0)
        
        # Train Gaussians' labels w.r.t ONE segmentation
        train_label_w_seg(gaussians, [this_viewpoint_cam], [this_mask_path], opt, background, iterations=2000)

        #### Render from other cameras
        # Initialize a list of consistent segmentation for future fine-tuning
        new_viewpoint_stack = [this_viewpoint_cam]
        match_mask_paths = [this_mask_path]
        
        for viewpoint_cam in viewpoint_stack:
            if viewpoint_cam.image_name == this_image_name:
                continue
            else:
                # Go through other cameras to find match
                render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=True)
                pred_seg = render_label.squeeze().detach().cpu().numpy() > 0.01
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
                                         
                if max_iou > 0.5: # Hyperparameters to modify
                    # Add matched viewpoint cam and matched seg to a list
                    new_viewpoint_stack.append(viewpoint_cam)
                    match_mask_paths.append(max_overlap_mask_path)
                    match_mask_name = os.path.splitext(os.path.basename(max_overlap_mask_path))[0]
                    processed_masks.add(match_mask_name)
                    print(f"Find a mathch with IOU={max_iou} with seg {match_mask_name}") 
                    # Create the saved match directory only when a macth is found
                    os.makedirs(f"output/{this_mask_name}/match", exist_ok=True)
                    vis_image_w_overlay(img_tensor=viewpoint_cam.original_image, 
                                        save_dir=f"output/{this_mask_name}/match", 
                                        save_name=match_mask_name,
                                        pred_seg=pred_seg,
                                        overlap_seg=max_overlap_mask,
                                        resize_factor=2
                                       )
        
        assert len(new_viewpoint_stack) == len(match_mask_paths)
        print(f"Total of {len(new_viewpoint_stack)} matches with IOU > 0.5 found for refine training.")
        # print("Set", processed_masks)

        if len(new_viewpoint_stack) > 1:
            #### Only do Refine training w.r.t newly found segmentation  when at least one match is found ####
            num_wheat_head += 1
            print(f"Start refine training w.r.t the {num_wheat_head}th wheat head found")
            train_label_w_seg(gaussians, new_viewpoint_stack, match_mask_paths, opt, background, iterations=7000)
            
            # gaussians.save_ply(f"output/{os.path.splitext(os.path.basename(mask_path))[0]}_refined.ply")
    
            #### Evaluation of refined training ####
            for viewpoint_cam in viewpoint_stack:
                with torch.no_grad():
                    render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=False).detach().cpu()
                    pred_seg = render_label.numpy() > 0.01
                    os.makedirs(f"output/{this_mask_name}/refine", exist_ok=True)
                    vis_image_w_overlay(img_tensor=viewpoint_cam.original_image, 
                                        save_dir=f"output/{this_mask_name}/refine",
                                        save_name=viewpoint_cam.image_name,
                                        pred_seg=pred_seg,
                                        resize_factor=2)
                    # Update the 2D seg&count results
                    assert twoD_seg_results[viewpoint_cam.image_name].shape == render_label.shape
                    twoD_seg_results[viewpoint_cam.image_name][render_label > 0.01] = num_wheat_head
                    # Update the saved 2D seg saved
                    torch.save(twoD_seg_results[viewpoint_cam.image_name], f"output/2DSeg/{viewpoint_cam.image_name}.pt")
            gaussians.reset_label(set_which_object_to=num_wheat_head)
            print(f"Num of Gaussians corresponding to each wheat head {torch.unique(gaussians.get_which_object.cpu(), return_counts=True)}")
        else:
            gaussians.reset_label()

        if exp_id % 5 == 0: # Save Gaussians every 5 distinct masks
            gaussians.save_ply("output/gaussians.ply")
            print("Gaussians saved!")
        print(f"Processed masks {len(processed_masks)} / {len(all_mask_paths)}")
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args))
    