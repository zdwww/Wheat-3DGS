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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import matplotlib.pyplot as plt

from flashsplat_rasterization import GaussianRasterizationSettings as FlashSplat_GaussianRasterizationSettings
from flashsplat_rasterization import GaussianRasterizer as FlashSplat_GaussianRasterizer
import pdb

def render(viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    separate_sh = False, 
    override_color = None,
    target_values = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        # antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # print("Scales precomputed: ", scales.shape, torch.max(scales), torch.min(scales))
    scales_norm = torch.norm(scales, p=2, dim=1, keepdim=False)
    # print(scales_norm.shape, torch.max(scales_norm), torch.min(scales_norm))
    # print("Opacity precomputed: ", opacity.shape, torch.max(opacity), torch.min(opacity))
    # plt.hist(opacity.detach().cpu().numpy(), bins=50, density=True, alpha=0.7, edgecolor='black')
    # plt.xlim(0.02, 0.6)
    # plt.show()
    # filtered_gs = (opacity < 0.1).squeeze()
    filtered_gs = (scales > 0.1).any(dim=1)
    # filtered_gs =  (pc.get_which_object.squeeze() == target_values[0])
    # print("Filtered Gaussians: ", torch.sum(filtered_gs))
    
    # if target_values is not None:
    #     palette = plt.get_cmap('tab20')
    #     for i, target_val in enumerate(target_values):
    #         mask = (pc.get_which_object.squeeze() == target_val)
    #         rgb_color = palette(i)[:3]
    #         colors_precomp[mask] = torch.tensor(rgb_color, device=colors_precomp.device, dtype=colors_precomp.dtype)

    # Render wheat head
    # rendered_wheats = []
    # if target_values is not None:
    #     for i, target_val in enumerate(target_values):
    #         mask = (pc.get_which_object.squeeze() == target_val)
    #         rendered_wheat, _, _, _ = rasterizer(
    #             means3D = means3D[mask],
    #             means2D = means2D[mask], 
    #             shs = shs,
    #             colors_precomp = colors_precomp[mask], 
    #             opacities = opacity[mask], 
    #             scales = scales[mask], 
    #             rotations = rotations[mask],
    #             cov3D_precomp = cov3D_precomp)
    #         rendered_wheats.append(rendered_wheat)
    #         print("Rendered wheat shape: ", rendered_wheat.shape, torch.max(rendered_wheat), torch.min(rendered_wheat))
    
    means3D = means3D
    means2D = means2D
    shs = shs
    colors_precomp = colors_precomp
    opacity = opacity
    scales = scales
    rotations = rotations
    cov3D_precomp = cov3D_precomp

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image, alpha_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else: # Set false by default
        rendered_image, radii, depth_image, alpha_image = rasterizer(
            means3D = means3D[~filtered_gs],
            means2D = means2D[~filtered_gs], 
            shs = shs,
            colors_precomp = colors_precomp[~filtered_gs], 
            opacities = opacity[~filtered_gs], 
            scales = scales[~filtered_gs], 
            rotations = rotations[~filtered_gs],
            cov3D_precomp = cov3D_precomp)
    # print("Rendered image shape: ", rendered_image.shape, "Depth image shape: ", depth_image.shape)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)

    # alpha = 0.5
    # palette = plt.get_cmap('Set1')
    # for i, rendered_wheat in enumerate(rendered_wheats):
    #     mask = (rendered_wheat < 1).any(dim=0, keepdim=True)
    #     target_color = torch.tensor(palette(i)[:3], device=rendered_image.device).view(3, 1, 1)
    #     rendered_image = torch.where(mask, 
    #         alpha * target_color + (1 - alpha) * rendered_image,
    #         rendered_image)

    # rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out

def flashsplat_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
                  override_color = None, gt_mask = None, used_mask = None, unique_label = None, setpdb=False,
                  obj_num = 2,):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # if unique_label is not None:
    #     pdb.set_trace()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = FlashSplat_GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        mask_grad=False,
        num_obj=obj_num,
    )

    rasterizer = FlashSplat_GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    if used_mask is not None:
        means3D = means3D[used_mask]
        opacity = opacity[used_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if used_mask is not None:
            scales = scales[used_mask]
            rotations = rotations[used_mask]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # pdb.set_trace()
            shs = pc.get_features
            if used_mask is not None:
                shs = shs[used_mask]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if setpdb:
        pdb.set_trace()
    rendered_image, radii, depth, alpha, contrib_num, used_count, proj_xy, gs_depth = rasterizer(
        gt_mask = gt_mask,
        unique_label = unique_label,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "alpha": alpha, 
            "depth": depth, 
            "contrib_num": contrib_num,
            "used_count": used_count,
            "proj_xy": proj_xy,
            "gs_depth": gs_depth}