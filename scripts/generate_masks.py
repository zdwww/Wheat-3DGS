"""
Generate the 2D wheat-head masks used by the 3D segmentation (the contents of yolosam.zip):
YOLOv5 wheat-head boxes per image, then one SAM mask per box.

The defaults (--conf 0.25, --img_size 640, SAM vit_h) reproduce the masks provided with our dataset (yolosam.zip).
For each plot folder it writes:
    bboxes/<image>.pt         N x 4 boxes (x1, y1, x2, y2) in image pixels
    masks/<image>_<NNN>.png   one binary mask (0/255) per box, in box order
    yolo_vis/, sam_vis/       overlays for visual checks (with --vis)

Usage:
    python scripts/generate_masks.py -s <dataset_path> [<dataset_path> ...] \
        --yolo_weights weights/wheat_head_detection_model.pt --sam_checkpoint weights/sam_vit_h_4b8939.pth
"""
import os
import sys
import glob
import gc
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# YOLOv5 v7.0 is a submodule; its top-level `models`/`utils` packages are also needed to unpickle the weights
sys.path.insert(0, os.path.join(REPO, "submodules", "yolov5"))
from models.common import DetectMultiBackend, AutoShape
from segment_anything import sam_model_registry, SamPredictor


def detect(image_paths, bbox_dir, args, vis_dir=None):
    """YOLOv5 boxes for every image, saved as bboxes/<image>.pt"""
    # Same model as torch.hub.load('ultralytics/yolov5', 'custom', path) in the original notebook
    model = AutoShape(DetectMultiBackend(args.yolo_weights, device=torch.device(args.device), fuse=True))
    model.conf, model.iou = args.conf, args.iou
    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        bbox_path = os.path.join(bbox_dir, f"{name}.pt")
        if args.skip_existing and os.path.exists(bbox_path):
            continue
        boxes = model(image_path, size=args.img_size).xyxy[0][:, :4].double().cpu()
        torch.save(boxes, bbox_path)
        print(f"{name}: {len(boxes)} boxes")
        if vis_dir is not None:
            image = cv2.imread(image_path)
            for x1, y1, x2, y2 in boxes.round().int().tolist():
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 8)
            cv2.imwrite(os.path.join(vis_dir, f"{name}.jpg"), image)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def segment(image_paths, bbox_dir, mask_dir, args, vis_dir=None):
    """One SAM mask per box, saved as masks/<image>_<NNN>.png"""
    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint).to(device=args.device)
    predictor = SamPredictor(sam)
    rng = np.random.default_rng(0)
    for image_path in image_paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        if args.skip_existing and glob.glob(os.path.join(mask_dir, f"{name}_*.png")):
            continue
        boxes = torch.load(os.path.join(bbox_dir, f"{name}.pt"))
        if len(boxes) == 0:
            print(f"{name}: no boxes, no masks")
            continue
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            predictor.set_image(image)
            masks = []
            for chunk in torch.split(boxes, args.sam_batch):
                transformed = predictor.transform.apply_boxes_torch(chunk, image.shape[:2]).to(args.device)
                chunk_masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None,
                                                            boxes=transformed, multimask_output=False)
                masks.append(chunk_masks[:, 0].cpu().numpy())
            masks = np.concatenate(masks)
            predictor.reset_image()
        assert len(masks) == len(boxes)
        for i, mask in enumerate(masks):
            Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(mask_dir, f"{name}_{i:03}.png"))
        print(f"{name}: {len(masks)} masks")
        if vis_dir is not None:
            overlay = image.astype(np.float32)
            for mask in masks:
                overlay[mask] = 0.4 * overlay[mask] + 0.6 * rng.integers(0, 256, 3)
            cv2.imwrite(os.path.join(vis_dir, f"{name}.jpg"), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    del predictor, sam
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="YOLOv5 + SAM wheat-head masks for Wheat3DGS")
    parser.add_argument("-s", "--source_paths", nargs="+", required=True, help="plot folder(s), each with an images/ subfolder")
    parser.add_argument("--images", type=str, default="images", help="image subfolder name")
    parser.add_argument("--out", type=str, default=None, help="output folder (default: the plot folder); with several plots, one subfolder per plot")
    parser.add_argument("--yolo_weights", type=str, default=os.path.join(REPO, "weights", "wheat_head_detection_model.pt"))
    parser.add_argument("--sam_checkpoint", type=str, default=os.path.join(REPO, "weights", "sam_vit_h_4b8939.pth"))
    parser.add_argument("--sam_model", type=str, default="vit_h", help="SAM model type matching the checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLOv5 confidence threshold (0.25 for the paper)")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLOv5 NMS IoU threshold")
    parser.add_argument("--img_size", type=int, default=640, help="YOLOv5 inference size (640 for the paper)")
    parser.add_argument("--sam_batch", type=int, default=64, help="boxes per SAM forward pass (lower it if the GPU runs out of memory)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vis", action="store_true", help="also save yolo_vis/ and sam_vis/ overlays")
    parser.add_argument("--skip_existing", action="store_true", help="skip images that already have boxes / masks")
    args = parser.parse_args()

    for source_path in args.source_paths:
        image_paths = sorted(glob.glob(os.path.join(source_path, args.images, "*.png")) +
                             glob.glob(os.path.join(source_path, args.images, "*.jpg")))
        assert len(image_paths) > 0, f"No images found in {os.path.join(source_path, args.images)}"
        if args.out is None:
            out_dir = source_path
        elif len(args.source_paths) > 1:
            out_dir = os.path.join(args.out, os.path.basename(os.path.normpath(source_path)))
        else:
            out_dir = args.out
        dirs = {d: os.path.join(out_dir, d) for d in ["bboxes", "masks"] + (["yolo_vis", "sam_vis"] if args.vis else [])}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        print(f"{source_path}: {len(image_paths)} images -> {out_dir}")
        detect(image_paths, dirs["bboxes"], args, dirs.get("yolo_vis"))
        segment(image_paths, dirs["bboxes"], dirs["masks"], args, dirs.get("sam_vis"))
