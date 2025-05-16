import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer as Renderer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def get_raster_settings(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Helper function to get raster settings instead of performing rendering
    """
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
        debug=False
    )
    return raster_settings

def get_pts_label_as_rgb(gaussians_label : torch.Tensor):
    label_as_rgb = torch.zeros([gaussians_label.shape[0], 3]).float().cuda()
    label_as_rgb[:, 0] = gaussians_label.squeeze()
    label_as_rgb[:, 1] = gaussians_label.squeeze()
    label_as_rgb[:, 2] = gaussians_label.squeeze()
    return label_as_rgb

def gaussians_to_label_rendervar(gaussians : GaussianModel):
    means3D = gaussians.get_xyz.detach()
    means2D = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    opacity = gaussians.get_opacity.detach()
    scales = gaussians.get_scaling.detach()
    rotations = gaussians.get_rotation.detach()
    colors_precomp = get_pts_label_as_rgb(gaussians.get_label)

    rendervar = {
        'means3D': means3D,
        'colors_precomp': colors_precomp,
        'rotations': rotations,
        'opacities': opacity,
        'scales': scales,
        'means2D': means2D
    }
    return rendervar

def get_render_label(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor):
    """
    Helper function to get rendered label with another renderer initialized inside the function
    """
    curr_raster_settings = get_raster_settings(viewpoint_camera, pc, bg_color)
    label_renderer = Renderer(curr_raster_settings)
    label_rendervar = gaussians_to_label_rendervar(pc)
    render_label, _, _, _ = label_renderer(**label_rendervar)
    return render_label



def eval_obj_labels(all_obj_labels, viewpoint_cam, gaussians, pipe, background):
    from gaussian_renderer import flashsplat_render
    render_num = all_obj_labels.size(0)
    print("render_num", render_num)
    # render_num = 20 # OVERRIDE
    pred_mask = None
    max_alpha = None
    min_depth = None
    idx_lst = [1, 4, 8, 12 ,13, 15, 19, 20, 27, 32, 33, 34, 42, 45, 52, 53, 56, 57, 58, 59, 65, 67, 68, 70, \
        71, 72, 73, 85, 87, 91, 92, 93, 95, 97, 98, 100, 101, 103, 104, 105, 107, 109 , \
        112, 113, 114, 122, 123, 126, 131, 138, 142, 143, 146, 147, 149, 150, 151, 152, \
        164, 179, 180, 183, 189, 190, 200, 201, 205, 206, 208, 211, 215, 219, 232, 239, 241, 242, 243, 248, 251, 252,
        253, 260, 261, 268, 269, 271, 276, 278, 299, 300, 306, 313] 
    idx_lst += [9, 11, 16, 18, 22, 30, 38, 48, 49, 50, 51, 60, 66]
    idx_lst += [74, 76, 79, 94, 110, 116, 117, 118, 133, 144, 145]
    idx_lst += [156, 159, 160, 164, 167, 168, 173, 175, 181, 186]
    idx_lst += [231, 90, 254, 182]
        
    for obj_idx in range(render_num):
        if obj_idx not in idx_lst:
            continue
        obj_used_mask = (all_obj_labels[obj_idx]).bool()
        if obj_used_mask.sum().item() == 0 or obj_idx == 0: # obj 0 is background
            continue
        flashsplat_pkg = flashsplat_render(viewpoint_cam, gaussians, pipe, background, used_mask=obj_used_mask.to("cuda"))
        render_alpha = flashsplat_pkg["alpha"].detach().cpu()
        render_depth = flashsplat_pkg["depth"].detach().cpu()
        if pred_mask is None:
            pred_mask = torch.zeros_like(render_alpha)
            max_alpha = torch.zeros_like(render_alpha)
            min_depth = torch.ones_like(render_alpha)
        _pix_mask = (render_alpha > 0.5)
        pix_mask = _pix_mask.clone() 
        overlap_mask = (_pix_mask & (pred_mask > 0))
        if overlap_mask.sum().item() > 0:
            if (min_depth[overlap_mask].mean() < render_depth[overlap_mask].mean()):
                pix_mask[_pix_mask] = (~(pred_mask[_pix_mask] > 0)) 
        pred_mask[pix_mask] = obj_idx
        min_depth[pix_mask] = render_depth[pix_mask]
        max_alpha[pix_mask] = render_alpha[pix_mask]
    return pred_mask
