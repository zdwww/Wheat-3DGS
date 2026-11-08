import os
import sys
import csv
import glob
import random
import shutil
import string
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import flashsplat_render
from scene import Scene, GaussianModel
from utils.wheatgs_utils import vis_image_w_overlay
from utils.seg_utils import (
    opt_label_w_seg,
    counts_to_obj_labels,
    find_match,
    update_processed_masks,
    check_yolosam,
)
CONSOLE = Console()

def find_new_mask_dir(img_dir, num_wheat_head):
    """
    Function used to find the previous-identified & overlapped segmentation's suffix
    and reuturn the new letter suffix in order.
    """
    base_dir = f"{img_dir}/{num_wheat_head:04}"
    existing_dirs = glob.glob(f"{base_dir}*")
    assert existing_dirs, f"Error: No existing directory found for {base_dir}*"
    for letter_suffix in string.ascii_lowercase:  # 'a' to 'z'
        candidate_dir = f"{base_dir}_{letter_suffix}"
        if candidate_dir not in existing_dirs:
            new_dir = candidate_dir
            break
    return letter_suffix

        
def training(dataset, opt, pipe, load_iteration, exp_name, iou_threshold, seed=0, height_filter=True, save_counts=False):
    # The order of the seed masks (shuffled below) drives the greedy match-and-fine-tune, so fix it
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # All 3DSeg results will be saved under 3dgs_model_path/wheat-head/(exp_name)
    out_dir = os.path.join(dataset.model_path, "wheat-head", exp_name)
    sub_dirs = ["ply", "img"] + (["count"] if save_counts else [])
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(out_dir, sub_dir), exist_ok=True)
    ply_dir, img_dir, count_dir = [os.path.join(out_dir, sub_dir) for sub_dir in ["ply", "img", "count"]]

    with open(f"{out_dir}/experiment.txt", "w") as file:
        file.write(f"exp_name {exp_name}\niou_threshold {iou_threshold}\nseed {seed}\nheight_filter {height_filter}\nsave_counts {save_counts}\n")
    
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
    with open(f"{out_dir}/experiment.txt", "a") as file:
        file.write(f"load_iteration {scene.loaded_iter}\n") # actual iteration, -1 resolved to the max
    gaussians.training_setup(opt)
    print(f"Loaded point cloud size: {len(gaussians.get_xyz)}")

    # Only Gaussians above the mean height are lifted from the seed mask (excludes the ground; off with --no_height_filter)
    z_mean = torch.mean(gaussians.get_xyz.cpu()[:, 2])
    print(f"All Gaussians z_min: {torch.min(gaussians.get_xyz.cpu()[:, 2])} zmax: {torch.max(gaussians.get_xyz.cpu()[:, 2])}")
    pts_filter = (gaussians.get_xyz.cpu()[:, 2] < z_mean) if height_filter else None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    check_yolosam(viewpoint_stack, dataset.source_path)
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
    for exp_id, this_mask_path in tqdm(enumerate(all_mask_paths), total=len(all_mask_paths), desc="Processing Masks"):
        print("-" * 50)
        this_mask_name = os.path.splitext(os.path.basename(this_mask_path))[0]
        
        if this_mask_name in processed_masks:
            print(f"{this_mask_name} already processed and saved")
            continue
        elif this_mask_name in buffered_masks:
            buffered_masks.remove(this_mask_name)
            processed_masks.add(this_mask_name)
            print(f"{this_mask_name} has been iterated once. will be moved to processed set.")

        this_image_name = this_mask_name[:-4]
        mask_idx = int(this_mask_name[-3:])
        CONSOLE.print(f"==== Train 3D segmentation against {this_mask_name} ====")
        
        this_viewpoint_cam = next(cam for cam in viewpoint_stack if cam.image_name == this_image_name)
        
        # Optimize Gaussians' labels w.r.t ONE segmentation
        # NOTE: all_counts is additive
        all_counts = opt_label_w_seg(gaussians, [this_viewpoint_cam], [this_mask_path], pipe, background, pts_filter)
        all_obj_labels = counts_to_obj_labels(all_counts)
        if torch.sum(all_obj_labels, dim=1)[1] == 0:
            print(f"Can't identify any Gaussians {'above average height ' if height_filter else ''}for this mask: {this_mask_name}), PASS")
            continue
        obj_used_mask = (all_obj_labels[1]).bool()

        #### Render from other cameras
        # Initialize a list of consistent segmentation for future fine-tuning
        matched_viewpoint_stack = [this_viewpoint_cam]
        matched_mask_paths = [this_mask_path]
        
        new_viewpoint_stack, new_mask_paths = find_match(
            target_viewpoint_stack = [vpt for vpt in viewpoint_stack if vpt.image_name != this_image_name],
            gs_params = (gaussians, pipe, background),
            obj_used_mask = obj_used_mask,
            iou_threshold = iou_threshold,
            dir_name = os.path.dirname(this_mask_path)
        )
        matched_viewpoint_stack += new_viewpoint_stack # as a whole
        matched_mask_paths += new_mask_paths
        processed_masks = update_processed_masks(processed_masks, new_mask_paths)
        CONSOLE.print(f"==== Find {len(new_mask_paths)} newly matched masks. {len(matched_mask_paths)} matched in total ====")

        #### Only do Refine training w.r.t newly found segmentation when a pairf of matches is found ####
        if len(new_viewpoint_stack) > 0:
            num_wheat_head += 1 # Potential wheat head
            # if find a match, then it's processed and create a dir for it
            this_mask_dir = f"{img_dir}/{num_wheat_head:04}"
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
                # fine-tuning
                new_viewpoint_stack, new_mask_paths = find_match(
                    target_viewpoint_stack = [
                        vpt for vpt in viewpoint_stack if vpt.image_name not in {mpt.image_name for mpt in matched_viewpoint_stack}
                    ],
                    gs_params = (gaussians, pipe, background),
                    obj_used_mask = obj_used_mask,
                    iou_threshold = iou_threshold,
                    dir_name = os.path.dirname(this_mask_path)
                )
                if len(new_viewpoint_stack) == 0:
                    CONSOLE.print(f"No new matched found from {len(viewpoint_stack) - len(matched_viewpoint_stack)} vpts")
                    break
                else:
                    matched_viewpoint_stack += new_viewpoint_stack # as a whole
                    matched_mask_paths += new_mask_paths
                    processed_masks = update_processed_masks(processed_masks, new_mask_paths)
                    CONSOLE.print(f"-- Find {len(new_mask_paths)} newly matched masks. {len(matched_mask_paths)} matched in total --")

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
                letter_suffix = find_new_mask_dir(img_dir, which_wheat_head)
                gaussians_obj.save_ply(f"{ply_dir}/wh_{which_wheat_head:04}_{letter_suffix}.ply")
                this_mask_dir = f"{img_dir}/{which_wheat_head:04}_{letter_suffix}"
                os.makedirs(this_mask_dir, exist_ok=True)
                CONSOLE.print(f"Create new mask dir {this_mask_dir}")
                writer.writerow([f"{which_wheat_head:04}_{letter_suffix}", this_mask_name, str(len(matched_viewpoint_stack)), str(num_GS)])
                results.flush()
            else:
                CONSOLE.print(f"======== Identify {num_wheat_head}th new wheat head with {len(matched_viewpoint_stack)} matches ========")
                which_wheat_head = num_wheat_head
                num_GS = torch.sum(gaussians_obj.get_which_object.detach() == which_wheat_head).item()
                gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != which_wheat_head), during_training=False)
                gaussians_obj.save_ply(f"{ply_dir}/wh_{which_wheat_head:04}.ply")
                writer.writerow([f"{which_wheat_head:04}", this_mask_name, str(len(matched_viewpoint_stack)), str(num_GS)])
                results.flush()

            if save_counts: # per-Gaussian counts of this head (~17 MB each), not read by any later step
                print(f"For wh {which_wheat_head}, all_counts.shape {all_counts.shape}")
                counts_save_pth = f"{count_dir}/{which_wheat_head:04}.pt"
                if os.path.exists(counts_save_pth):
                    print(f"{counts_save_pth} exists...removing...")
                    os.remove(counts_save_pth)
                torch.save(all_counts.detach().cpu(), counts_save_pth)

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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the processing order of the 2D masks")
    parser.add_argument("--no_height_filter", action="store_true", help="lift seed masks to all Gaussians, not only those above the mean height (the default is meant for wheat heads)")
    parser.add_argument("--save_counts", action="store_true", help="also save each wheat head's per-Gaussian counts to count/<id>.pt (~17 MB each), not needed by any later step")
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.load_iteration, args.exp_name, args.iou_threshold, args.seed, not args.no_height_filter, args.save_counts)