import os
import random
import torch
import numpy as np
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
from PIL import Image, ImageDraw
from gaussian_renderer.render_helper import get_render_label
import torch.nn as nn

########### Begin of Image Helper Functions ###########
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

def training(dataset, opt, pipe, total_iterations=10000):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=7000, shuffle=False)
    gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_label)}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()

    # Initialize a list of saved binary masks in png format
    all_mask_paths = []
    all_bboxes = {}
    num_bboxes = 0
    for viewpoint_cam in viewpoint_stack:
        print(viewpoint_cam.image_name)
        bboxes = torch.load(viewpoint_cam.bbox_path)
        num_bboxes += len(bboxes)
        all_bboxes[viewpoint_cam.image_name] = bboxes
        all_mask_paths += viewpoint_cam.mask_paths
    assert len(all_mask_paths) == num_bboxes
    print(f"Total of {len(all_mask_paths)} masks & bounding boxes found")

    processed_masks = set()
    
    for exp_id, mask_path in enumerate(all_mask_paths):
        if os.path.basename(mask_path) in processed_masks:
            print(f"{os.path.basename(mask_path)} already processed")
            continue
        
        processed_masks.add(os.path.basename(mask_path))
        print(f"Train 3D segmentation against {os.path.basename(mask_path)}")
        this_image_name = os.path.basename(mask_path)[:-8]
        mask_idx = int(os.path.basename(mask_path)[-7:-4])
        this_viewpoint_cam = next(cam for cam in viewpoint_stack if cam.image_name == this_image_name)
        print("Image size", this_viewpoint_cam.original_image.shape)
        # Load binary mask
        with Image.open(mask_path) as temp:
            # print("OG size", temp.size)
            # print("Scale", this_viewpoint_cam.resolution_scale)
            # print("Resolution", this_viewpoint_cam.resolution)
            mask = binarize_mask(PILtoTorch(temp.copy(), this_viewpoint_cam.resolution).to("cuda"))
        print("Mask shape", mask.shape)
        
        #### Training ####
        first_iter = 0
        progress_bar = tqdm(range(first_iter, total_iterations), desc="Training progress")
        first_iter += 1
        gaussians.update_lr_for_label(label_lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        if exp_id == 0:
            gaussians.load_ply(f"output/{os.path.splitext(os.path.basename(mask_path))[0]}.ply")
            print(f"Trained Gaussians loaded for exp {exp_id}")
        else:
            for iteration in range(first_iter, total_iterations + 1):
                render_label = torch.mean(get_render_label(this_viewpoint_cam, gaussians, background), dim=0, keepdim=True)
                loss = criterion(input=render_label, target=mask)
                loss.backward()
                with torch.no_grad():
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                        progress_bar.update(10)
                    if iteration == total_iterations:
                        progress_bar.close()
                    ## Optimizer step
                    if iteration <= total_iterations:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)
    
            gaussians.save_ply(f"output/{os.path.splitext(os.path.basename(mask_path))[0]}.ply")

        #### Render from other cameras
        new_viewpoint_stack = [this_viewpoint_cam]
        match_mask_path = [mask_path]
        
        for viewpoint_cam in viewpoint_stack:
            if viewpoint_cam.image_name == this_image_name:
                continue
            else:
                # Go through other cameras to find match
                print(viewpoint_cam.image_name)
                # Save the rendered segmentation for visualization
                render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=True)
                pred_seg = render_label.squeeze().detach().cpu().numpy() > 0.01
                pred_bbox = get_bbox_from_mask(pred_seg)
                print("Pred", pred_bbox)
                # Load YOLO bounding boxes
                bboxes = torch.load(viewpoint_cam.bbox_path) / viewpoint_cam.resolution_scale
                # Overlap boxes xyxy, id and mIOU
                overlap_bboxes = [tuple(box.tolist()) for box in bboxes if is_overlapping(pred_bbox, tuple(box.tolist()))]
                overlap_idx = [i for i, box in enumerate(bboxes) if is_overlapping(pred_bbox, tuple(box.tolist()))]
                # Infer SAM-generated segmentation from bounding boxes
                overlap_masks_paths = [mask_path for mask_path in viewpoint_cam.mask_paths if int(os.path.basename(mask_path)[-7:-4]) in overlap_idx]
                overlap_masks = []
                ious = []
                for mask_path in overlap_masks_paths:
                    with Image.open(mask_path) as temp:
                        mask = binarize_mask(PILtoTorch(temp.copy(), this_viewpoint_cam.resolution)).squeeze().numpy() > 0
                        assert mask.shape == pred_seg.shape
                        overlap_masks.append(mask)
                        ious.append(calculate_seg_iou(mask, pred_seg))
                                         
                # ious = [calculate_iou(pred_bbox, box) for box in overlap_bboxes]
                # Find a match segmentation in other cameras
                print(f"IOUs: {ious}")
                match = None
                if len(ious) != 0:
                    max_iou = np.max(ious)
                    if max_iou > 0.2:
                        match = np.argmax(ious)
                        # Add matched viewpoint cam and matched seg to a list
                        new_viewpoint_stack.append(viewpoint_cam)
                        match_mask_path.append(overlap_masks_paths[match])
                        processed_masks.add(os.path.basename(overlap_masks_paths[match]))
                        print(f"Find a mathch with IOU={max_iou} at cam {viewpoint_cam.image_name} with seg {os.path.basename(overlap_masks_paths[match])}")

                #### Begin of visualization block ####
                visualize = True
                if visualize:
                    mask_pil = Image.fromarray(pred_seg.astype(np.uint8) * 255)
                    image_pil = rgb_tensor_to_PIL(viewpoint_cam.original_image)
                    image_with_overlay = overlay_img_w_mask(image_pil, mask_pil, color="red")
                    
                    # for overlap_mask in overlap_masks:
                    if match is not None:
                        mask_pil = Image.fromarray(overlap_masks[match].astype(np.uint8) * 255)
                        image_with_overlay = overlay_img_w_mask(image_with_overlay, mask_pil, color="blue")
                    
                    # Draw bounding box
                    if pred_bbox is not None:
                        draw = ImageDraw.Draw(image_with_overlay)
                        draw.rectangle(pred_bbox, outline="red", width=3)
                        # if len(overlap_bboxes) != 0:
                        #     for i, overlap_bbox in enumerate(overlap_bboxes):
                        #         draw.rectangle(overlap_bbox, outline="blue", width=1)
                        
                    image_with_overlay.save(f"output/vis/{viewpoint_cam.image_name}.png")
                #### End of visualization block ####
        
        assert len(new_viewpoint_stack) == len(match_mask_path)
        print(f"a total of {len(new_viewpoint_stack)} matches found")
        
        #### Begin of Refine training with newly found segmentation ####
        first_iter = 0
        progress_bar = tqdm(range(first_iter, total_iterations), desc="Refine Training")
        first_iter += 1
        gaussians.training_setup(opt)
        gaussians.update_lr_for_label(label_lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        for iteration in range(first_iter, total_iterations + 1):
            random_index = random.randint(0, len(new_viewpoint_stack) - 1)
            viewpoint_cam = new_viewpoint_stack[random_index]
            with Image.open(match_mask_path[random_index]) as temp:
                mask = binarize_mask(PILtoTorch(temp.copy(), viewpoint_cam.resolution).to("cuda"))
            
            render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=True)
            # gray_tensor_to_PIL(render_label).save(f"output/vis/{iteration}_render.png")
            # gray_tensor_to_PIL(mask).save(f"output/vis/{iteration}_mask.png")
            loss = criterion(input=render_label, target=mask)
            loss.backward()
            with torch.no_grad():
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                    progress_bar.update(10)
                if iteration == total_iterations:
                    progress_bar.close()
                ## Optimizer step
                if iteration <= total_iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
        #### End of Refine training with newly found segmentation ####
        gaussians.save_ply(f"output/{os.path.splitext(os.path.basename(mask_path))[0]}_refined.ply")

        #### Evaluation of refined training ####
        for viewpoint_cam in viewpoint_stack:
            with torch.no_grad():
                render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=True)
                pred_seg = render_label.squeeze().detach().cpu().numpy() > 0.01
                mask_pil = Image.fromarray(pred_seg.astype(np.uint8) * 255)
                image_pil = rgb_tensor_to_PIL(viewpoint_cam.original_image)
                image_with_overlay = overlay_img_w_mask(image_pil, mask_pil, color="red")
                os.makedirs(f"output/{exp_id}", exist_ok=True)
                image_with_overlay.save(f"output/{exp_id}/{viewpoint_cam.image_name}.png")
        # print("Render label shape", render_label.shape)
        gaussians.reset_label()
        print(f"Processed masks {len(processed_masks)} / {len(all_mask_paths)}: {processed_masks}")
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args))
    