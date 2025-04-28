
import torch
import numpy as np
from PIL import Image

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