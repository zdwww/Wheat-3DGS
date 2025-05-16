#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import colorsys
import numpy as np
from PIL import Image, ImageFilter

def dilate_mask(mask: torch.Tensor, border_width: int) -> torch.Tensor:
    """Dilate a binary mask using Pillow's MaxFilter."""
    mask_pil = Image.fromarray(mask.cpu().numpy().astype('uint8') * 255)
    dilated_pil = mask_pil.filter(ImageFilter.MaxFilter(border_width * 2 + 1))
    dilated_mask = torch.from_numpy(np.array(dilated_pil) > 0).to(mask.device)
    return dilated_mask

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def visualize_obj(objects):
    if len(objects.shape) != 2:
        objects = objects.squeeze()
        assert len(objects.shape) == 2
    # Modify from numpy to torch
    rgb_mask = torch.zeros((*objects.shape[-2:], 3), dtype=torch.uint8)
    all_obj_ids = torch.unique(objects)
    border_width = 2

    idx_lst = [4, 232]

    for id in all_obj_ids:
        color = id2rgb(id)
        obj_mask = (objects == id)
        rgb_mask[obj_mask] = color
        if id in idx_lst:
            dilated_mask = dilate_mask(obj_mask, border_width*2)
            border_region = dilated_mask & (~obj_mask)
            rgb_mask[border_region] = torch.tensor([255, 0, 0], dtype=rgb_mask.dtype, device=rgb_mask.device)
        else:
            dilated_mask = dilate_mask(obj_mask, border_width)
            border_region = dilated_mask & (~obj_mask)
            rgb_mask[border_region] = torch.tensor([1, 1, 1], dtype=rgb_mask.dtype, device=rgb_mask.device)
        # rgb_mask[objects == id] = color
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

def overlay_image(image, mask, alpha=0.5):
    non_black_pixels = torch.any(mask > 0, dim=0)
    overlayed_image = image.clone()
    overlayed_image[:, non_black_pixels] = (alpha * mask[:, non_black_pixels] + (1 - alpha) * image[:, non_black_pixels])
    return overlayed_image