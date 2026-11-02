"""
Interactive viewer (viser + nerfview) for a Wheat3DGS 3D segmentation: the reconstructed plot with every
segmented wheat head in its own colour, and the training/test cameras (red/green, click one to jump to it).

Head colours come from the `which_object` id stored in the segmentation's gaussians.ply. They are baked into
the Gaussians once at startup, so each frame is a single render. The "Show head colours" checkbox switches
back to the original appearance.
Navigation: left-drag orbit, right-drag pan, scroll zoom; W/S forward/back, A/D left/right, Q/E down/up, arrows turn.

Usage (from this folder):
    python wheatgs_rendering.py --input_ply <model_path>/wheat-head/<exp_name>/gaussians.ply \
        --colmap_path <dataset_path>/sparse/0 --images_path <dataset_path>/images --port 8080
"""
import os
import time
import colorsys
import argparse

import numpy as np
import torch
import imageio.v2 as iio
import viser
import viser.transforms as vtf
from viser.extras.colmap import read_cameras_binary, read_images_binary, read_cameras_text, read_images_text
import nerfview

from arguments import PipelineParams, ModelParams
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera

parser = argparse.ArgumentParser()
lp = ModelParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--input_ply", type=str, required=True, help="gaussians.ply of a 3D segmentation (with which_object)")
parser.add_argument("--colmap_path", type=str, required=True, help="sparse/0 of the plot (cameras + images, .bin or .txt)")
parser.add_argument("--images_path", type=str, required=True, help="images/ of the plot (camera thumbnails)")
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
args = parser.parse_args()
pipe = pp.extract(args)
dataset = lp.extract(args)

with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree) # 0 by default: view-independent colours only, faster
    gaussians.load_ply(args.input_ply, remove_features_rest=(dataset.sh_degree == 0))
    which_object = gaussians.get_which_object.squeeze(-1).long()
    head_ids = torch.unique(which_object[which_object > 0])
    print(f"Loaded {len(which_object)} Gaussians, {len(head_ids)} wheat heads from {args.input_ply}")

    # Bake one flat colour per head (golden-ratio hue spacing, so neighbouring ids differ) into the DC term
    SH_C0 = 0.28209479177387814
    orig_features = (gaussians._features_dc.detach().clone(), gaussians._features_rest.detach().clone())
    head_features = (orig_features[0].clone(), orig_features[1].clone())
    lut = torch.zeros(int(which_object.max()) + 1, 3, device="cuda")
    for head_id in head_ids.tolist():
        rgb = colorsys.hsv_to_rgb((head_id * 0.618033988749895) % 1.0, 0.85, 0.95)
        lut[head_id] = (torch.tensor(rgb, device="cuda") - 0.5) / SH_C0
    in_head = which_object > 0
    head_features[0][in_head, 0, :] = lut[which_object[in_head]]
    head_features[1][in_head] = 0.0

def set_head_colours(show):
    features_dc, features_rest = head_features if show else orig_features
    gaussians._features_dc.data.copy_(features_dc)
    gaussians._features_rest.data.copy_(features_rest)

set_head_colours(True)
background = torch.ones(3, dtype=torch.float32, device="cuda")

@torch.no_grad()
def viewer_render_fn(camera_state: nerfview.CameraState, render_tab_state: nerfview.RenderTabState) -> np.ndarray:
    W, H = render_tab_state.viewer_width, render_tab_state.viewer_height
    K = camera_state.get_K((W, H))
    W2C = np.linalg.inv(camera_state.c2w)
    camera = Camera(resolution=(W, H), colmap_id=-1, R=W2C[:3, :3].transpose(), T=W2C[:3, 3],
                    FoVx=2 * np.arctan(W / (2 * K[0, 0])), FoVy=2 * np.arctan(H / (2 * K[1, 1])),
                    image=None, image_name="render_view", uid=0, data_device="cuda")
    img = render(viewpoint_camera=camera, pc=gaussians, pipe=pipe, bg_color=background)["render"]
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

server = viser.ViserServer(port=args.port, verbose=False)
viewer = nerfview.Viewer(server=server, render_fn=viewer_render_fn, mode="rendering")

show_heads = server.gui.add_checkbox("Show head colours", initial_value=True)
@show_heads.on_update
def _(_):
    with viewer.lock:
        set_head_colours(show_heads.value)
    viewer.rerender(None)

# Cameras at their COLMAP poses: training views red, test views (cam_11/12) green; click one to jump to it
if os.path.exists(os.path.join(args.colmap_path, "cameras.bin")):
    cameras = read_cameras_binary(os.path.join(args.colmap_path, "cameras.bin"))
    images = read_images_binary(os.path.join(args.colmap_path, "images.bin"))
else:
    cameras = read_cameras_text(os.path.join(args.colmap_path, "cameras.txt"))
    images = read_images_text(os.path.join(args.colmap_path, "images.txt"))

for img in images.values():
    cam = cameras[img.camera_id]
    fy = cam.params[1] if cam.model == "PINHOLE" else cam.params[0] # SIMPLE_PINHOLE: f, cx, cy
    T_world_camera = vtf.SE3.from_rotation_and_translation(vtf.SO3(img.qvec), img.tvec).inverse()
    wxyz, position = T_world_camera.rotation().wxyz, T_world_camera.translation()
    image_name = os.path.splitext(img.name)[0]
    is_test = image_name.split("_")[-1].isdigit() and int(image_name.split("_")[-1]) > 10
    image_path = os.path.join(args.images_path, img.name)
    thumbnail = iio.imread(image_path)[::10, ::10] if os.path.exists(image_path) else None
    frustum = server.scene.add_camera_frustum(f"/cameras/{image_name}", fov=2 * np.arctan2(cam.height / 2, fy),
                                              aspect=cam.width / cam.height, scale=0.04, image=thumbnail,
                                              wxyz=wxyz, position=position,
                                              color=(0, 255, 0) if is_test else (255, 71, 77))
    def jump_to(event, wxyz=wxyz, position=position):
        with event.client.atomic():
            event.client.camera.wxyz = wxyz
            event.client.camera.position = position
    frustum.on_click(jump_to)

print(f"Viewer running at http://localhost:{args.port} ... Ctrl+C to exit.")
while True:
    time.sleep(10)
