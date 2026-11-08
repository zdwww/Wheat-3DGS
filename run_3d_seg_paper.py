"""
3D wheat-head segmentation as used for the paper's results (Table 2): the "pipeline 1" algorithm of
run_3d_seg_new.py at commit 75f8854, ported to the current helpers. run_3d_seg.py is a later version of it.

Each 2D mask (in a seeded random order) is lifted to 3D with FlashSplat (only Gaussians above the mean height,
unless --no_height_filter),
rendered into all other training views and matched to the YOLO/SAM mask with the largest IoU there. If at least
--num_match views (including the seed view) match with IoU > --iou_threshold, the head is solved once from all
matched masks and its Gaussians are assigned to a new head id, overwriting earlier assignments.
Differences to run_3d_seg.py: one matching pass (no iterative re-matching), a head needs --num_match views
(not 2), and no merging into a previously found head.

Outputs are laid out like run_3d_seg.py, so eval_wheatgs.py and render_360.py work on them:
<model_path>/wheat-head/<exp_name>/{experiment.txt, results.csv, gaussians.ply, 2DSeg/, ply/wh_XXXX.ply, img/XXXX/}
"""
import os
import sys
import csv
import random
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from rich.console import Console

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import flashsplat_render
from scene import Scene, GaussianModel
from utils.wheatgs_utils import PILtoTorch, binarize_mask, vis_image_w_overlay
from utils.seg_utils import opt_label_w_seg, counts_to_obj_labels, find_match, update_processed_masks, check_yolosam

CONSOLE = Console()

def training(dataset, opt, pipe, load_iteration, exp_name, iou_threshold, num_match, seed=0, height_filter=True):
    # The order of the seed masks (shuffled below) drives the greedy matching, so fix it
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # All 3DSeg results will be saved under 3dgs_model_path/wheat-head/(exp_name)
    out_dir = os.path.join(dataset.model_path, "wheat-head", exp_name)
    sub_dirs = ["ply", "img"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(out_dir, sub_dir), exist_ok=True)
    ply_dir, img_dir = [os.path.join(out_dir, sub_dir) for sub_dir in sub_dirs]

    with open(f"{out_dir}/experiment.txt", "w") as file:
        file.write(f"exp_name {exp_name}\nalgorithm paper\niou_threshold {iou_threshold}\nnum_match {num_match}\nseed {seed}\nheight_filter {height_filter}\n")

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
    for viewpoint_cam in viewpoint_stack + viewpoint_stack_eval:
        twoD_seg_results[viewpoint_cam.image_name] = torch.zeros(viewpoint_cam.original_image.shape[1:], dtype=torch.int)
        torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")

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

        # Optimize Gaussians' labels w.r.t ONE segmentation
        all_counts = opt_label_w_seg(gaussians, [this_viewpoint_cam], [this_mask_path], pipe, background, pts_filter)
        all_obj_labels = counts_to_obj_labels(all_counts)
        if torch.sum(all_obj_labels, dim=1)[1] == 0:
            print(f"Can't identify any Gaussians {'above average height ' if height_filter else ''}for this mask: {this_mask_name}), PASS")
            continue
        obj_used_mask = (all_obj_labels[1]).bool()

        #### Render into the other cameras and match once
        new_viewpoint_stack, new_mask_paths = find_match(
            target_viewpoint_stack=[vpt for vpt in viewpoint_stack if vpt.image_name != this_image_name],
            gs_params=(gaussians, pipe, background),
            obj_used_mask=obj_used_mask,
            iou_threshold=iou_threshold,
            dir_name=os.path.dirname(this_mask_path))
        matched_viewpoint_stack = [this_viewpoint_cam] + new_viewpoint_stack
        matched_mask_paths = [this_mask_path] + new_mask_paths

        if len(matched_viewpoint_stack) < num_match:
            print(f"Not enough matchings are found ({len(matched_viewpoint_stack)} < {num_match})")
        else:
            num_wheat_head += 1
            processed_masks = update_processed_masks(processed_masks, matched_mask_paths)

            # Solve once w.r.t all matched masks (no height filter here, as in the paper's version)
            CONSOLE.print(f"======== Identify {num_wheat_head}th new wheat head with {len(matched_viewpoint_stack)} matches ========")
            all_counts = opt_label_w_seg(gaussians, matched_viewpoint_stack, matched_mask_paths, pipe, background)
            obj_used_mask = (counts_to_obj_labels(all_counts)[1]).bool()

            # Plain overwrite of earlier assignments, no merging into a previous wheat head
            gaussians.reset_label(obj_used_mask=obj_used_mask, set_which_object_to=num_wheat_head, merge=False)
            gaussians_obj = deepcopy(gaussians)
            num_GS = torch.sum(gaussians_obj.get_which_object.detach() == num_wheat_head).item()
            gaussians_obj.prune_points(mask=torch.flatten(gaussians_obj.get_which_object.detach() != num_wheat_head), during_training=False)
            gaussians_obj.save_ply(f"{ply_dir}/wh_{num_wheat_head:04}.ply")
            writer.writerow([f"{num_wheat_head:04}", this_mask_name, str(len(matched_viewpoint_stack)), str(num_GS)])
            results.flush()

            #### Save overlays of the seed mask and of the head rendered into every view, update the 2D seg ####
            this_mask_dir = f"{img_dir}/{num_wheat_head:04}"
            os.makedirs(this_mask_dir, exist_ok=True)
            with Image.open(this_mask_path) as temp:
                this_mask = binarize_mask(PILtoTorch(temp.copy(), this_viewpoint_cam.resolution))
            vis_image_w_overlay(img_tensor=this_viewpoint_cam.original_image,
                                save_dir=this_mask_dir,
                                save_name=this_mask_name,
                                pred_seg=this_mask.squeeze().numpy() > 0)
            for viewpoint_cam in viewpoint_stack + viewpoint_stack_eval:
                with torch.no_grad():
                    render_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask)
                    render_alpha = render_pkg["alpha"].squeeze().detach().cpu()
                    pred_seg = render_alpha.numpy() > 0.5
                    vis_image_w_overlay(img_tensor=viewpoint_cam.original_image,
                                        save_dir=this_mask_dir,
                                        save_name=viewpoint_cam.image_name,
                                        pred_seg=pred_seg,
                                        resize_factor=4)
                    # Update the 2D seg&count results
                    assert twoD_seg_results[viewpoint_cam.image_name].shape == render_alpha.shape
                    twoD_seg_results[viewpoint_cam.image_name][render_alpha > 0.5] = num_wheat_head
                    torch.save(twoD_seg_results[viewpoint_cam.image_name], f"{out_dir}/2DSeg/{viewpoint_cam.image_name}.pt")

        if exp_id % 5 == 0: # Save Gaussians every 5 distinct masks
            gaussians.save_ply(f"{out_dir}/gaussians.ply")
            print("Gaussians saved!")
        print(f"Processed masks {len(processed_masks)} / {len(all_mask_paths)}")

    gaussians.save_ply(f"{out_dir}/gaussians.ply")
    results.close()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3D segmentation as used for the paper's results")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--load_iteration', type=str, default="-1")
    parser.add_argument("--exp_name", type=str, help="Exp name")
    parser.add_argument("--iou_threshold", type=float, default=0.6, help="IOU threshold for matching")
    parser.add_argument("--num_match", type=int, default=5, help="Num of matched views (incl. the seed view) required for a wheat head")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the processing order of the 2D masks")
    parser.add_argument("--no_height_filter", action="store_true", help="lift seed masks to all Gaussians, not only those above the mean height (the default is meant for wheat heads)")
    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.load_iteration, args.exp_name, args.iou_threshold, args.num_match, args.seed, not args.no_height_filter)
