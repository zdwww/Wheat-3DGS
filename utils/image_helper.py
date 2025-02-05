import torch
import numpy as np
from PIL import Image, ImageDraw
import colorsys

########### Begin of Segmentation visualization functions from FlashSplat ###########
def visualize_obj(objects):
    if len(objects.shape) != 2:
        objects = objects.squeeze()
        assert len(objects.shape) == 2
    # Modify from numpy to torch
    rgb_mask = torch.zeros((*objects.shape[-2:], 3), dtype=torch.uint8)
    all_obj_ids = torch.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        # print("id", id, "colored_mask", colored_mask)
        rgb_mask[objects == id] = colored_mask
    rgb_mask = rgb_mask.permute(2, 0, 1)
    return rgb_mask

def id2rgb(idx, max_num_obj=999):
    if isinstance(idx, torch.Tensor):
        idx = idx.item()
    if not 0 <= idx <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((idx * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (idx % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = torch.zeros((3, ), dtype=torch.uint8)
    if idx == 0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0] = torch.tensor(r*255, dtype=torch.uint8)
    rgb[1] = torch.tensor(g*255, dtype=torch.uint8)
    rgb[2] = torch.tensor(b*255, dtype=torch.uint8)

    return rgb
########### End of Segmentation visualization functions from FlashSplat ###########

def overlay_image(image, mask, alpha=0.5):
    non_black_pixels = torch.any(mask > 0, dim=0)
    overlayed_image = image.clone()
    overlayed_image[:, non_black_pixels] = (alpha * mask[:, non_black_pixels] + (1 - alpha) * image[:, non_black_pixels])
    return overlayed_image

def normalize_to_0_1(img_tensor):
    # Normalize PIL loaded image tensor from 0-255 to 0-1
    if torch.max(img_tensor) > 1.0:
        return (img_tensor / 255.0).clamp(0.0, 1.0)
    else:
        return img_tensor

def PILtoTorch(pil_image, resolution=None, normalize=True):
    if resolution is not None:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
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

def calculate_precision(pred, gt):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    assert pred.shape == gt.shape
    intersection = np.logical_and(pred, gt)
    precision = np.sum(intersection) / np.sum(pred) if np.sum(pred) > 0 else 0.0
    return precision

def calculate_recall(pred, gt):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    assert pred.shape == gt.shape
    intersection = np.logical_and(pred, gt)
    recall = np.sum(intersection) / np.sum(gt) if np.sum(gt) > 0 else 0.0
    return recall

def read_mask(mask_path):
    with Image.open(mask_path) as temp:
        mask = binarize_mask(PILtoTorch(temp.copy())) > 0
    return mask
