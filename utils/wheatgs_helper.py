import os
import gc
import math
import torch
import shutil
import ffmpeg
import subprocess
import numpy as np
from tqdm import tqdm
import torchvision
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
from scipy.spatial.transform import Slerp, Rotation
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam
from utils.image_helper import *
from tqdm import tqdm
import torch.nn.functional as F

torch.set_printoptions(precision=10)

# def nearest_k_viewpts(cam_centers, target_idx, k):
#     target_point = cam_centers[target_idx]
#     distances = torch.norm(cam_centers - target_point, dim=1)
#     nearest_indices = torch.topk(distances, k + 1, largest=False).indices[1:] 
#     return nearest_indices.tolist()

from shapely.geometry import Polygon
import numpy as np

def polygon_from_points(points):
    # Ensure points are in a proper order (if needed)
    # For rectangles, points are typically in order (e.g. clockwise)
    return Polygon(points)

def find_best_match(query_rect_points, list_of_rect_points):
    query_polygon = polygon_from_points(query_rect_points)

    max_intersection_area = 0
    best_match = None
    matched_idx = None

    # First pass: find the rectangle with maximum overlap
    for i, candidate_points in enumerate(list_of_rect_points):
        candidate_polygon = polygon_from_points(candidate_points)
        intersection_area = query_polygon.intersection(candidate_polygon).area
        if intersection_area > max_intersection_area:
            max_intersection_area = intersection_area
            best_match = candidate_polygon
            matched_idx = i

    # If no overlap found, find the closest rectangle
    if max_intersection_area == 0:
        min_distance = float('inf')
        closest_rect = None
        closest_idx = None
        for i, candidate_points in enumerate(list_of_rect_points):
            candidate_polygon = polygon_from_points(candidate_points)
            dist = query_polygon.distance(candidate_polygon)
            if dist < min_distance:
                min_distance = dist
                closest_rect = candidate_polygon
                closest_idx = i
        best_match = closest_rect
        matched_idx = closest_idx

    return best_match, matched_idx

def multi_instance_opt(all_contrib, gamma=0.):
    """
    Input:
    all_contrib: A_{e} with shape (obj_num, gs_num) 
    gamma: softening factor range from [-1, 1]
    
    Output: 
    all_obj_labels: results S with shape (obj_num, gs_num)
    where S_{i,j} denotes j-th gaussian belong i-th object
    """
    all_contrib_sum = all_contrib.sum(dim=0)
    all_obj_labels = torch.zeros_like(all_contrib).bool()
    for obj_idx, obj_contrib in tqdm(enumerate(all_contrib), desc="multi-view optimize"):
        obj_contrib = torch.stack([all_contrib_sum - obj_contrib, obj_contrib], dim=0)
        obj_contrib = F.normalize(obj_contrib, dim=0)
        obj_contrib[0, :] += gamma
        obj_label = torch.argmax(obj_contrib, dim=0)
        all_obj_labels[obj_idx] = obj_label
    return all_obj_labels

def short_image_name(image_name):
    parts = image_name.split("_")
    filtered_parts = [part for part in parts[2:] if not part.startswith("FIP")]
    result_string = "_".join(filtered_parts)
    return result_string

def get_center_and_diag(cam_centers):
    # Recalculate the scene radius based on Gaussians centers instead of camera centers
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal

def nearest_k_viewpts(vpt_stack, target_cam, k):
    cam_centers = [cam.camera_center.cpu() for cam in vpt_stack]
    cam_centers = torch.stack(cam_centers, dim=0)
    
    distances = torch.norm(cam_centers - target_cam, dim=1)
    if (distances < 1e-6).any():
        print("Sam cam in the stack")
        nearest_indices = torch.topk(distances, k + 1, largest=False).indices
        nearest_set = set(nearest_indices.tolist()) # for moving the target cam in remaining vpts
        nearest_indices = nearest_indices[1:]
    else:
        nearest_indices = torch.topk(distances, k, largest=False).indices
        nearest_set = set(nearest_indices.tolist())
    nearest_vpts = [vpt_stack[idx] for idx in nearest_indices.tolist()]
    remain_vpts = [vpt_stack[i] for i in range(len(vpt_stack)) if i not in nearest_set]
    return nearest_vpts, remain_vpts
    
def get_interpolated_viewpts_old(viewpoint_stack, N=100):
    """
    Input: original viewpoints stack
    """
    cam_centers = []
    qvecs = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        cam_rot = viewpoint_cam.R.T
        qvec = rotmat2qvec(np.transpose(viewpoint_cam.R))
        qvecs.append(qvec)
        cam_center =  - viewpoint_cam.R.dot(viewpoint_cam.T)
        cam_centers.append(cam_center)

        # These variables should be fixed for all cameras
        FoVx = viewpoint_cam.FoVx
        FoVy = viewpoint_cam.FoVy
        zfar = viewpoint_cam.zfar
        znear = viewpoint_cam.znear
        width = viewpoint_cam.image_width
        height = viewpoint_cam.image_height

        # Take trivial cameras for now
        if viewpoint_cam.image_name == "FPWW036_SR0461_6_FIP2_cam_09":
            bottom_left_idx = i
        elif viewpoint_cam.image_name == "FPWW036_SR0461_6_FIP2_cam_10":
            top_left_idx = i
        elif viewpoint_cam.image_name == "FPWW036_SR0461_1_FIP2_cam_01":
            bottom_right_idx = i
        elif viewpoint_cam.image_name == "FPWW036_SR0461_1_FIP2_cam_02":
            top_right_idx = i 

    cam1_idx = bottom_right_idx
    cam2_idx = top_left_idx
    center1, center2 = cam_centers[cam1_idx], cam_centers[cam2_idx]
    qvec1, qvec2 = qvecs[cam1_idx], qvecs[cam2_idx]

    rotations = Rotation.from_quat([qvec1, qvec2])
    slerp = Slerp([0,1], rotations)
    interp_times = np.linspace(0, 1, N)
    interpolated_rotations = slerp(interp_times)
    interpolated_qvecs = interpolated_rotations.as_quat()
    interpolated_centers = np.array([(1 - t) * center1 + t * center2 for t in interp_times])
    assert len(interpolated_qvecs) == len(interpolated_qvecs)

    new_viewpoint_stack = []
    for i, cam_center in enumerate(interpolated_centers):
        qvec = interpolated_qvecs[i]
        R = np.transpose(qvec2rotmat(qvec))
        T = - np.transpose(R).dot(cam_center)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0).astype(np.float64)).transpose(0, 1)
        camera_center = world_view_transform.inverse()[3, :3]
        world_view_transform = world_view_transform.to(torch.float32).cuda()
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        new_viewpoint_stack.append(MiniCam(width, height, FoVy, FoVx, znear, zfar, world_view_transform, full_proj_transform))
    return new_viewpoint_stack

def get_interpolated_viewpts(vpt1, vpt2, N=100):
    """
    Input: two viewpoints 
    """
    cam_rot1, cam_rot2 = vpt1.R.T, vpt2.R.T
    qvec1, qvec2 = rotmat2qvec(np.transpose(vpt1.R)), rotmat2qvec(np.transpose(vpt2.R))
    center1, center2  =  - vpt1.R.dot(vpt1.T),  - vpt2.R.dot(vpt2.T)

    # These variables should be fixed for all cameras
    FoVx, FoVy = vpt1.FoVx, vpt1.FoVy
    zfar, znear = vpt1.zfar, vpt1.znear
    width, height = vpt1.image_width, vpt1.image_height

    rotations = Rotation.from_quat([qvec1, qvec2])
    slerp = Slerp([0,1], rotations)
    interp_times = np.linspace(0, 1, N)
    interpolated_rotations = slerp(interp_times)
    interpolated_qvecs = interpolated_rotations.as_quat()
    interpolated_centers = np.array([(1 - t) * center1 + t * center2 for t in interp_times])
    assert len(interpolated_qvecs) == len(interpolated_qvecs)

    new_viewpoint_stack = []
    for i, cam_center in enumerate(interpolated_centers):
        qvec = interpolated_qvecs[i]
        R = np.transpose(qvec2rotmat(qvec))
        T = - np.transpose(R).dot(cam_center)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0).astype(np.float64)).transpose(0, 1)
        camera_center = world_view_transform.inverse()[3, :3]
        world_view_transform = world_view_transform.to(torch.float32).cuda()
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        new_viewpoint_stack.append(MiniCam(width, height, FoVy, FoVx, znear, zfar, world_view_transform, full_proj_transform))
    return new_viewpoint_stack

#### Begin of 360-degree camera trajectory copied from gsgen ####
def get_c2w_from_up_and_look_at(up, look_at, pos):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos
    return c2w

def get_camera_path_fixed_elevation(n_frames, n_circles=1, camera_distance=2, cam_center=[0, 0, 0], elevation=0):
    azimuth = np.linspace(0, 2 * np.pi * n_circles, n_frames)
    elevation_rad = np.deg2rad(elevation)
    x = camera_distance * np.cos(azimuth) * np.cos(elevation_rad)
    y = camera_distance * np.sin(azimuth) * np.cos(elevation_rad)
    z = camera_distance * np.sin(elevation_rad) * np.ones_like(x)

    up = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.array(cam_center, dtype=np.float32)
    pos = np.stack([x, y, z], axis=1)

    c2ws = []
    for i in range(n_frames):
        c2ws.append(get_c2w_from_up_and_look_at(up, look_at, pos[i]))
    c2ws = np.stack(c2ws, axis=0)
    return c2ws

#### End of 360-degree camera trajectory copied from gsgen ####

def eval_obj_labels(all_obj_labels, viewpoint_cam, gaussians, pipe, background):
    from gaussian_renderer import flashsplat_render
    render_num = all_obj_labels.size(0)
    pred_mask = None
    max_alpha = None
    min_depth = None
    for obj_idx in range(render_num):
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

def render_360(og_view, scene_radius, render_path, n_frames, framerate, gaussians, pipeline, background, all_obj_labels=None):
    from gaussian_renderer import render
    os.makedirs(render_path, exist_ok=True)
    gs_centroid = torch.mean(gaussians.get_xyz.detach(), dim=0).cpu().tolist()
    width, height = math.floor(og_view.image_width / 2), math.floor(og_view.image_height / 2)
    fovy, fovx = og_view.FoVy, og_view.FoVx
    znear, zfar = og_view.znear, og_view.zfar
    camera_distance = scene_radius * 2
    c2ws = get_camera_path_fixed_elevation(n_frames=n_frames, n_circles=1, camera_distance=camera_distance, cam_center=gs_centroid, elevation=45)
    for idx in tqdm(range(len(c2ws)), desc="render360"):
        c2w = c2ws[idx]
        c2w = np.vstack([c2w, [0.0, 0.0, 0.0, 1.0]])
        w2c = np.linalg.inv(np.float64(c2w))
        world_view_transform = torch.tensor(w2c.astype(np.float32)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        view = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"].detach().cpu()
        if all_obj_labels is not None:
            pred_seg = eval_obj_labels(all_obj_labels, view, gaussians, pipeline, background)
            # print("torch.max(pred_seg)", torch.max(pred_seg))
            rgb_mask = visualize_obj(pred_seg) / 255.0
            # print("torch.max(rgb_mask)", torch.max(rgb_mask))
            rgb_image = overlay_image(rendering, rgb_mask)
        else:
            rgb_image = rendering
        torchvision.utils.save_image(rgb_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        gc.collect()
        torch.cuda.empty_cache()
        
    output_video = os.path.join(os.path.dirname(render_path), "360.mp4")
    try:
        (
            ffmpeg
            .input(f"{render_path}/%05d.png", framerate=framerate, start_number=0)
            .filter("scale", "iw-mod(iw,2)", "ih-mod(ih,2)")  # Scale filter
            .output(output_video, r=framerate, vcodec="libx264", pix_fmt="yuv420p")
            .global_args("-loglevel", "error")  # Set global log level
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Video created successfully at {output_video}!")
    except ffmpeg.Error as e:
        print("Error during FFmpeg execution:")
        print("STDERR:", e.stderr.decode())
    
    # subprocess.run([
    #     "module load ffmpeg/6.0 && ",
    #     "ffmpeg",
    #     "-loglevel", "error",
    #     "-framerate", str(framerate),
    #     "-start_number", "0",
    #     "-i", "%05d.png",  
    #     "-vf", "scale=iw-mod(iw\,2):ih-mod(ih\,2)",
    #     "-r", str(framerate),
    #     "-vcodec", "libx264",
    #     "-pix_fmt", "yuv420p",
    #     output_video
    #     ], cwd=render_path, shell=True, check=True)
    
    # shutil.rmtree(render_path)
    return output_video
    