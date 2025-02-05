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
import math
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 bbox_path, mask_paths, resolution, resolution_scale,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale).astype(np.float32)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.bbox_path = bbox_path
        self.mask_paths = mask_paths
        self.resolution = resolution
        self.resolution_scale = resolution_scale

        self.rect_camera = self.compute_image_plane_rect()
        self.rect_world, self.rectangle = self.get_near_plane_rect_world()
    
    def compute_image_plane_rect(self):
        # Compute rectangle in camera space
        half_width = self.znear * math.tan(self.FoVx / 2.0)
        half_height = self.znear * math.tan(self.FoVy / 2.0)
        zc = -self.zfar

        rect = {
            "top_left": (-half_width,  half_height, zc),
            "top_right": (half_width,  half_height, zc),
            "bottom_left": (-half_width, -half_height, zc),
            "bottom_right": (half_width, -half_height, zc)
        }
        return rect

    def get_near_plane_rect_world(self):
        # Transform the camera-space rectangle corners into world coordinates
        view_to_world = self.world_view_transform.inverse()

        # Convert each corner to homogeneous coords and transform
        rect_world = {}
        rectangle = []
        for key, (x, y, z) in self.rect_camera.items():
            corner_cam = torch.tensor([x, y, z, 1.0], dtype=torch.float32, device=self.data_device)
            corner_world = view_to_world @ corner_cam
            # Normalize by w just in case (should be 1.0 if it's a proper rigid transform)
            corner_world = corner_world / corner_world[3]
            rect_world[key] = (corner_world[0].item(), corner_world[1].item(), corner_world[2].item())
            rectangle.append(np.array([corner_world[0].item(), corner_world[1].item()]))
        rectangle = np.vstack(rectangle)
        return rect_world, rectangle

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

