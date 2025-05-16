import cv2
import argparse
import math
from tqdm.auto import tqdm
import imageio as iio
import random
import os
import os.path as osp
from typing import List
import time
from typing import Tuple
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from gsplat._helper import load_test_data
from gsplat.rendering import rasterization
import nerfview
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import PipelineParams, ModelParams
from scene import GaussianModel
from scene.cameras import Camera
from gaussian_renderer.render_helper import eval_obj_labels
from utils.image_utils import id2rgb, visualize_obj, overlay_image

parser = argparse.ArgumentParser()
lp = ModelParams(parser) # dataset
pp = PipelineParams(parser)
parser.add_argument(
    "--output_dir", type=str, default="results/", help="where to dump outputs"
)
parser.add_argument(
    "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
)
parser.add_argument("--input_ply", type=str, default=None, help="path to the .ply file")
parser.add_argument("--labels_path", type=str, default=None, help="path to the .ply file")
parser.add_argument("--colmap_path", type=str, default=None, help="")
parser.add_argument("--images_path", type=str, default=None, help="")
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
parser.add_argument(
    "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
)
args = parser.parse_args()
assert args.scene_grid % 2 == 1, "scene_grid must be odd"
pipe = pp.extract(args)
pipe.convert_SHs_python = True
dataset = lp.extract(args)
dataset.sh_degree = 0
gaussians = GaussianModel(dataset.sh_degree)
gaussians.load_ply(args.input_ply, remove_features_rest = True)
print(f"Num of Gaussians loaded from {args.input_ply}: {len(gaussians.get_xyz)}")
print(gaussians.get_which_object)
torch.manual_seed(42)
device = "cuda"

all_obj_labels = torch.load(args.labels_path).cuda()

# Define gaussians and pipe

@torch.no_grad()
def viewer_render_fn(camera_state: nerfview.CameraState, 
    img_wh: Tuple[int, int],
    # gaussians: GaussianModel,
    # pipe: dict,
) -> np.ndarray:  # Expected shape: (H, W, 3) with dtype uint8
    with torch.no_grad():
        W, H = img_wh
        K = camera_state.get_K(img_wh)
        W2C = np.linalg.inv(camera_state.c2w)
        R = W2C[:3, :3].transpose()
        T = W2C[:3, 3]        
        fx = K[0, 0]
        fy = K[1, 1]
        FoVx = 2 * np.arctan(W / (2 * fx))
        FoVy = 2 * np.arctan(H / (2 * fy))
        camera = Camera(
            colmap_id=-1,
            R=R, T=T,
            FoVx=FoVx, FoVy=FoVy,
            image=None,
            image_name="render_view",
            uid=0,
            resolution=(W, H),
            data_device="cuda",
            # cx=K[0, 2], cy=K[1, 2],
            # fl_x=fx, fl_y=fy,
            # meta_only=True
        )
        background = torch.zeros(3, dtype=torch.float32, device="cuda") # to change background color
        background = torch.ones(3, dtype=torch.float32, device="cuda")
        rendered_output = render(
            viewpoint_camera=camera.cuda(),
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            scaling_modifier=1.0,
            # target_values = [17]
            # target_values = [27, 94, 14, 9, 24, 72, 8, 9, 32, 35, 41, 44, 50, 68]
        )
        img = rendered_output["render"].detach().cpu()
        pred_seg = eval_obj_labels(all_obj_labels, camera.cuda(), gaussians, pipe, background).detach().cpu()
        rgb_mask = visualize_obj(pred_seg) / 255.0
        img = overlay_image(img, rgb_mask, alpha=0.3)
        img = (img.numpy() * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        return img

server = viser.ViserServer(port=args.port, verbose=False)
# server.scene.world_axes.visible = True

######## Begin of Colmap part ########

colmap_path = args.colmap_path
images_path = args.images_path
downsample_factor = 10
reorient_scene = True
cameras = read_cameras_binary(os.path.join(colmap_path, "cameras.bin"))
images = read_images_binary(os.path.join(colmap_path, "images.bin"))
points3d = read_points3d_binary(os.path.join(colmap_path, "points3D.bin"))
points = np.array([points3d[p_id].xyz for p_id in points3d])
print(f"Points centroid: {np.mean(points, axis=0)}")

img_ids = [im.id for im in images.values()]

def set_camera_frustums(
        server,
        scale_factor=0.1,
        downsample_factor=10,
        frustum_scale=0.05,
        frustum_axes_length=0.005,
        frustum_axes_radius=0.001
    ):
    """Set up camera frames and frustums following COLMAP convention."""
    camera_handles = {}
    frames = []
    image = None
    
    for img_id in tqdm(img_ids):
        print(img_id)
        img = images[img_id]
        cam = cameras[img.camera_id]
        W, H = cam.width, cam.height
        fx, fy, cx, cy = cam.params
        print(f"Camera {img_id}: {W}x{H}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        # Skip images that don't exist.
        image_filename = os.path.join(images_path, img.name)
        
        elements = os.path.splitext(img.name)[0].split("_")

        if len(elements) == 5:
            group = 1
        else:
            if int(elements[2]) == 6:
                group = 2
            else:
                group = 3
        print("elements:", elements, "group:", group)

        if int(elements[-1]) > 10:
            split = 'test'
        else:
            split = 'train'
        color = (255, 71, 77) if split == 'train' else (0, 255, 0) # red or green
        # color = (255, 0, 0) # red
        
        # Only visualize if in a specific group
        # if True: # group == 1: # and int(elements[-1]) == 6:
        if (group == 2 and int(elements[-1]) == 9) or (group == 3 and int(elements[-1]) == 2):
            pass
        else:
            T_world_camera = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(img.qvec), img.tvec
            ).inverse()
            wxyz = T_world_camera.rotation().wxyz
            position=T_world_camera.translation()
            
            # Move the camera down
            position[2] = position[2] - 2.5

            # Add coordinate frame (useful for debugging)
            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=wxyz,
                position=position,
                axes_length= 0.0, # 0.1, # frustum_axes_length
                axes_radius= 0.0, # 0.005, # frustum_axes_length
                visible=True
            )
            frames.append(frame)

            # Load image if we find it
            if os.path.exists(image_filename):
                # image = cv2.cvtColor(cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
                image = iio.imread(image_filename)
                image = image[::downsample_factor, ::downsample_factor]
            else:
                print(f"Image {image_filename} not found.")
                continue  
            
            fov = 2 * np.arctan2(H / 2, fy) 
            aspect = W / H
            fov *= 4 # 2.5 
            print(f"fov {fov}, aspect {aspect}")
            # Add frustum
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=fov,
                aspect=aspect,
                scale=0.04,
                color=color,
                image=image,
                visible=True,
            )

            # Attach callback to go to the camera when clicked
            def create_camera_callback(frame_position, frame_wxyz, cam_name):
                def callback(event):
                    with event.client.atomic():
                        event.client.camera.position = frame_position
                        event.client.camera.wxyz = frame_wxyz
                return callback
            
            frustum.on_click(create_camera_callback(position, wxyz, img.name))
            camera_handles[img_id] = frustum
    
    return camera_handles, frames

_ = nerfview.Viewer(
    server=server,
    render_fn=viewer_render_fn,
    mode="rendering",
)
camera_handles, frames = set_camera_frustums(server)
print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)
