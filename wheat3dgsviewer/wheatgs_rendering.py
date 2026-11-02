"""
Interactive viewer (viser + nerfview) for a Wheat3DGS 3D segmentation: the reconstructed plot with the segmented
wheat heads coloured, and the training/test cameras (red/green, click one to jump to it).

"Colour by" (colours are baked into the Gaussians, so each frame is a single render):
- Final heads: one colour per head id (`which_object` in the segmentation's gaussians.ply).
- Saved heads: copies (needs --heads_dir <exp>/ply): the saved heads (one wh_XXXX.ply per head found, as used for
  phenotyping) can contain the same Gaussians when a head was saved more than once, which the final heads hide.
  Gaussians in 1 / 2 / >= 3 saved heads are green / yellow / red.
- Original: the reconstruction's own colours.
"Saved head" picks one saved head and colours it and the saved heads sharing >= 50% of its Gaussians.
Navigation: left-drag orbit, right-drag pan, scroll zoom; W/S forward/back, A/D left/right, Q/E down/up, arrows turn.

Usage (from this folder):
    python wheatgs_rendering.py --input_ply <model_path>/wheat-head/<exp_name>/gaussians.ply \
        --colmap_path <dataset_path>/sparse/0 --images_path <dataset_path>/images --port 8080 \
        [--heads_dir <model_path>/wheat-head/<exp_name>/ply]
"""
import argparse
import colorsys
import glob
import os
import time

import imageio.v2 as iio
import nerfview
import numpy as np
import torch
import viser
import viser.transforms as vtf
from plyfile import PlyData
from viser.extras.colmap import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
)

from arguments import ModelParams, PipelineParams
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
parser.add_argument("--heads_dir", type=str, default=None, help="ply/ of the same segmentation: its saved heads (wh_XXXX.ply)")
args = parser.parse_args()
pipe = pp.extract(args)
dataset = lp.extract(args)

SH_C0 = 0.28209479177387814
def rgb_to_dc(rgb):
    return (torch.as_tensor(rgb, dtype=torch.float32, device="cuda") - 0.5) / SH_C0

def id_colour(i):
    # Golden-ratio hue spacing, so neighbouring ids get different colours
    return colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 0.95)

with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree) # 0 by default: view-independent colours only, faster
    gaussians.load_ply(args.input_ply, remove_features_rest=(dataset.sh_degree == 0))
    which_object = gaussians.get_which_object.squeeze(-1).long()
    print(f"Loaded {len(which_object)} Gaussians, {len(torch.unique(which_object[which_object > 0]))} wheat heads from {args.input_ply}")
    orig_features = (gaussians._features_dc.detach().clone(), gaussians._features_rest.detach().clone())

# Saved heads: each wh_XXXX.ply is a pruned copy of the same Gaussians, so they are matched by their coordinates
saved_heads = {} # saved head id -> indices of its Gaussians in gaussians.ply
if args.heads_dir is not None:
    index_of = {key: i for i, key in enumerate(map(bytes, gaussians.get_xyz.detach().cpu().numpy()))}
    for path in sorted(glob.glob(os.path.join(args.heads_dir, "wh_*.ply"))):
        vertex = PlyData.read(path)["vertex"]
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        saved_heads[int(os.path.basename(path)[3:-4])] = torch.tensor([index_of[key] for key in map(bytes, xyz)], device="cuda")
    del index_of
    copies = torch.bincount(torch.cat(list(saved_heads.values())), minlength=len(which_object))
    print(f"Loaded {len(saved_heads)} saved heads from {args.heads_dir}; Gaussians in 1 / 2 / >= 3 saved heads: "
          f"{(copies == 1).sum().item()} / {(copies == 2).sum().item()} / {(copies >= 3).sum().item()}")

def duplicates_of(i):
    """Saved heads sharing >= 50% of the smaller head's Gaussians with saved head i"""
    member = torch.zeros(len(which_object), dtype=torch.bool, device="cuda")
    member[saved_heads[i]] = True
    return [j for j, idx in saved_heads.items()
            if j != i and member[idx].sum().item() >= 0.5 * min(len(idx), len(saved_heads[i]))]

@torch.no_grad()
def set_colours(mode, saved_id=0):
    features_dc, features_rest = orig_features[0].clone(), orig_features[1].clone()
    if saved_id in saved_heads:
        # Duplicates first, then the selected head on top: what sticks out in another colour is a duplicate's extra part
        for j in duplicates_of(saved_id) + [saved_id]:
            features_dc[saved_heads[j], 0, :] = rgb_to_dc(id_colour(j))
            features_rest[saved_heads[j]] = 0.0
    elif mode == "Final heads":
        lut = torch.stack([rgb_to_dc(id_colour(i)) for i in range(int(which_object.max()) + 1)])
        in_head = which_object > 0
        features_dc[in_head, 0, :] = lut[which_object[in_head]]
        features_rest[in_head] = 0.0
    elif mode == "Saved heads: copies":
        for selected, rgb in [(copies == 1, (0.2, 0.8, 0.2)), (copies == 2, (1.0, 0.85, 0.0)), (copies >= 3, (0.9, 0.1, 0.1))]:
            features_dc[selected, 0, :] = rgb_to_dc(rgb)
            features_rest[selected] = 0.0
    gaussians._features_dc.data.copy_(features_dc)
    gaussians._features_rest.data.copy_(features_rest)

set_colours("Final heads")
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

colour_by = server.gui.add_dropdown("Colour by", ["Final heads", "Saved heads: copies", "Original"] if saved_heads
                                    else ["Final heads", "Original"], initial_value="Final heads")
LEGENDS = {"Final heads": "One colour per final head (`which_object`)",
           "Saved heads: duplicates": "Gaussians in:\n- 🟩 1 saved head\n- 🟨 2 saved heads\n- 🟥 3 or more saved heads",
           "Original": "Reconstruction colours"}
legend = server.gui.add_markdown(LEGENDS["Final heads"])
if saved_heads:
    saved_id = server.gui.add_slider("Saved head (0 = off)", min=0, max=max(saved_heads), step=1, initial_value=0)
    go_to = server.gui.add_button("Go to saved head")
    info = server.gui.add_markdown(f"{len(saved_heads)} saved heads")

def update_colours(_=None):
    i = int(saved_id.value) if saved_heads else 0
    legend.content = "Selected saved head and its duplicates, each in its own colour" if i in saved_heads else LEGENDS[colour_by.value]
    if saved_heads:
        if i == 0:
            info.content = f"{len(saved_heads)} saved heads"
        elif i not in saved_heads:
            info.content = f"No saved head {i}"
        else:
            info.content = f"Head {i}: {len(saved_heads[i])} Gaussians, duplicates (>= 50% shared): {duplicates_of(i) or 'none'}"
    with viewer.lock:
        set_colours(colour_by.value, i)
    viewer.rerender(None)

colour_by.on_update(update_colours)
if saved_heads:
    saved_id.on_update(update_colours)

    @go_to.on_click
    def _(event):
        # Look at the head's centre from the current viewing direction, 30 cm away
        if int(saved_id.value) not in saved_heads:
            return
        centre = gaussians.get_xyz[saved_heads[int(saved_id.value)]].mean(dim=0).detach().cpu().numpy()
        camera = event.client.camera
        direction = np.asarray(camera.position) - np.asarray(camera.look_at)
        direction = direction / np.linalg.norm(direction)
        with event.client.atomic():
            camera.look_at = centre
            camera.position = centre + 0.3 * direction

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
