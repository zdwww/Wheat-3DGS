"""
2D segmentation metrics against the manual labels (Table 2 of the paper): IoU, precision, recall, F1, MSE and SSIM
of the binary wheat-head masks, per plot and averaged over plots.

For each plot, every <dataset_path>/manual_label/<view>_gt_mask.png is compared with the predicted mask of the same
view, <model_path>/wheat-head/<exp_name>/eval/test/segmentation/<view>.png (from eval_wheatgs.py), or
<pred_dir>/<view>.png for masks of another method. Both are binarised at --threshold. The metrics and a
TP/FP/FN overlay per view are saved next to the predictions folder (<model_path>/wheat-head/<exp_name>/eval/test/, or the parent of <pred_dir>).

Usage:
    python evaluate_2D.py -s data/D1/plot_46{1..7} -m output/D1/plot_46{1..7} --exp_name run1
    python evaluate_2D.py -s <dataset_path> --pred_dir <folder with <view>.png masks>
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from utils.loss_utils import ssim

METRICS = ["IoU", "Precision", "Recall", "F1", "MSE", "SSIM"]

def compute_metrics(gt, pred):
    """Pixel metrics of two binary masks (bool arrays of the same shape)"""
    tp = np.logical_and(gt, pred).sum()
    fp = pred.sum() - tp # in the prediction, not in the GT
    fn = gt.sum() - tp   # in the GT, not in the prediction
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    mse = np.mean(gt != pred) # masks are 0/1, so the squared error is the fraction of pixels that disagree
    to_tensor = lambda mask: torch.from_numpy(mask.astype(np.float32))[None, None]
    ssim_value = ssim(to_tensor(gt), to_tensor(pred)).item()
    return dict(zip(METRICS, [iou, precision, recall, f1, mse, ssim_value]))

def save_overlay(gt, pred, image_path, out_path, alpha=0.6):
    """Overlap red, GT only gray, prediction only light red, blended over the image"""
    vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
    vis[gt & pred] = [0, 0, 255]        # BGR
    vis[gt & ~pred] = [128, 128, 128]
    vis[~gt & pred] = [128, 213, 255]
    image = cv2.imread(image_path)
    overlay = cv2.addWeighted(vis.astype(np.float32) / 255, alpha, image.astype(np.float32) / 255, 1 - alpha, 0)
    cv2.imwrite(out_path, (overlay * 255).astype(np.uint8))

def evaluate_plot(dataset_path, pred_dir, threshold):
    """Metrics per labelled view of one plot, written to <pred_dir>/../metrics.txt"""
    gt_paths = sorted(glob.glob(os.path.join(dataset_path, "manual_label", "*_gt_mask.png")))
    assert gt_paths, f"No manual labels (*_gt_mask.png) in {os.path.join(dataset_path, 'manual_label')}"
    out_dir = os.path.dirname(os.path.normpath(pred_dir))
    results = {}
    for gt_path in gt_paths:
        view = os.path.basename(gt_path)[:-len("_gt_mask.png")]
        pred_path = os.path.join(pred_dir, f"{view}.png")
        assert os.path.exists(pred_path), f"No predicted mask for view {view}: {pred_path}"
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) >= threshold
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) >= threshold
        assert gt.shape == pred.shape, f"{view}: GT {gt.shape} and prediction {pred.shape} differ in size"
        results[view] = compute_metrics(gt, pred)
        image_paths = glob.glob(os.path.join(dataset_path, "images", f"{view}.*"))
        if image_paths:
            save_overlay(gt, pred, image_paths[0], os.path.join(out_dir, f"GT_overlay_{view}.png"))
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("view " + " ".join(METRICS) + "\n")
        for view, m in results.items():
            f.write(f"{view} " + " ".join(f"{m[k]:.4f}" for k in METRICS) + "\n")
    return results

def format_row(name, m):
    return f"{name:<16}" + "".join(f"{m[k]:>10.4f}" for k in METRICS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D segmentation metrics against the manual labels (Table 2)")
    parser.add_argument("-s", "--source_path", nargs="+", required=True, help="dataset path(s) of the plot(s), with manual_label/")
    parser.add_argument("-m", "--model_path", nargs="+", help="model path(s), in the same order as -s")
    parser.add_argument("--exp_name", type=str, help="segmentation run, as in run_3d_seg*.py and eval_wheatgs.py")
    parser.add_argument("--pred_dir", nargs="+", help="instead of -m/--exp_name: folder(s) with the predicted masks <view>.png, in the same order as -s")
    parser.add_argument("--threshold", type=int, default=128, help="grey value from which a pixel counts as wheat head")
    args = parser.parse_args()

    if args.pred_dir:
        pred_dirs = args.pred_dir
    else:
        assert args.model_path and args.exp_name, "Give -m and --exp_name, or --pred_dir"
        pred_dirs = [os.path.join(m, "wheat-head", args.exp_name, "eval", "test", "segmentation") for m in args.model_path]
    assert len(pred_dirs) == len(args.source_path), f"{len(args.source_path)} dataset paths but {len(pred_dirs)} model paths / pred dirs"

    print(f"{'':<16}" + "".join(f"{k:>10}" for k in METRICS))
    plot_means = []
    for source_path, pred_dir in zip(args.source_path, pred_dirs):
        results = evaluate_plot(source_path, pred_dir, args.threshold)
        plot_mean = {k: np.mean([m[k] for m in results.values()]) for k in METRICS}
        plot_means.append(plot_mean)
        print(format_row(os.path.basename(os.path.normpath(source_path)), plot_mean))
    if len(plot_means) > 1:
        print(format_row(f"Mean ({len(plot_means)})", {k: np.mean([m[k] for m in plot_means]) for k in METRICS}))
