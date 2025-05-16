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
        rendered_output = render(
            viewpoint_camera=camera.cuda(),
            pc=gaussians,
            pipe=pipe,
            bg_color=torch.zeros(3, dtype=torch.float32, device="cuda"),
            scaling_modifier=1.0,
            target_values = [1, 20, 27]
        )
        img = (rendered_output["render"].detach().cpu().numpy() * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        return img

server = viser.ViserServer(port=args.port, verbose=False)

######## Begin of Colmap part ########

colmap_path = args.colmap_path
images_path = args.images_path
downsample_factor = 10
reorient_scene = True

server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

cameras = read_cameras_binary(os.path.join(colmap_path, "cameras.bin"))
images = read_images_binary(os.path.join(colmap_path, "images.bin"))
points3d = read_points3d_binary(os.path.join(colmap_path, "points3D.bin"))
points = np.array([points3d[p_id].xyz for p_id in points3d])

gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.")

# Let's rotate the scene so the average camera direction is pointing up.
if reorient_scene:
    average_up = (
        vtf.SO3(np.array([img.qvec for img in images.values()]))
        @ np.array([0.0, -1.0, 0.0])  # -y is up in the local frame!
    ).mean(axis=0)
    average_up /= np.linalg.norm(average_up)

    rotate_axis = np.cross(average_up, np.array([0.0, 0.0, 1.0]))
    rotate_axis /= np.linalg.norm(rotate_axis)
    rotate_angle = np.arccos(np.dot(average_up, np.array([0.0, 0.0, 1.0])))
    R_scene_colmap = vtf.SO3.exp(rotate_axis * rotate_angle)
    server.scene.add_frame(
        "/colmap",
        show_axes=False,
        wxyz=R_scene_colmap.wxyz,
    )
else:
    R_scene_colmap = vtf.SO3.identity()

# Get transformed z-coordinates and place grid at 5th percentile height.
transformed_z = (R_scene_colmap @ points)[..., 2]
# grid_height = float(np.percentile(transformed_z, 5))
# server.scene.add_grid(name="/grid", position=(0.0, 0.0, grid_height))

@gui_reset_up.on_click
def _(event: viser.GuiEvent) -> None:
    client = event.client
    assert client is not None
    client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array(
        [0.0, -1.0, 0.0])
gui_frames = server.gui.add_slider(
    "Max frames",
    min=1,
    max=len(images),
    step=1,
    initial_value=min(len(images), 100))
frames: List[viser.FrameHandle] = []

def visualize_frames() -> None:
    """Send all COLMAP elements to viser for visualization. This could be optimized
    a ton!"""

    # Remove existing image frames.
    for frame in frames:
        frame.remove()
    frames.clear()

    # Interpret the images and cameras.
    img_ids = [im.id for im in images.values()]
    random.shuffle(img_ids)
    img_ids = sorted(img_ids[: gui_frames.value])

    def attach_callback(
        frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
    ) -> None:
        @frustum.on_click
        def _(_) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position

    for img_id in tqdm(img_ids):
        img = images[img_id]
        cam = cameras[img.camera_id]

        # Skip images that don't exist.
        image_filename = os.path.join(images_path, img.name)
        if not os.path.exists(image_filename):
            print(f"Image {image_filename} not found.")
            continue

        T_world_camera = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(img.qvec), img.tvec
        ).inverse()
        frame = server.scene.add_frame(
            f"/colmap/frame_{img_id}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.1,
            axes_radius=0.005,
        )
        frames.append(frame)

        # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
        if cam.model != "PINHOLE":
            print(f"Expected pinhole camera, but got {cam.model}")

        H, W = cam.height, cam.width
        fy = cam.params[1]
        image = iio.imread(image_filename)
        image = image[::downsample_factor, ::downsample_factor]
        frustum = server.scene.add_camera_frustum(
            f"/colmap/frame_{img_id}/frustum",
            fov=2 * np.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.15,
            image=image,
        )
        attach_callback(frustum, frame)

need_update = True
@gui_frames.on_update
def _(_) -> None:
    # nonlocal need_update
    need_update = True

while True:
    if need_update:
        need_update = False
        visualize_frames()

_ = nerfview.Viewer(
    server=server,
    render_fn=viewer_render_fn,
    mode="rendering",
)
print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)
