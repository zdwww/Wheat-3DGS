"""Render FruitNeRF's wheat-head semantics on the test views, for evaluate_2D.py (paper Table 2).

Saves sigmoid(semantics) * 255 as a greyscale <view>.png per test view (and the RGB render as <view>_rgb.png).
evaluate_2D.py binarises at grey value 128 by default, i.e. sigmoid >= 0.5. Run in the `fruitnerf` env:

    conda run -n fruitnerf python scripts/baselines/fruitnerf_render_semantics.py \
        --load-config <run>/config.yml --output-dir <run>/semantics_test
    python evaluate_2D.py -s <plot> --pred_dir <run>/semantics_test
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load-config", type=Path, required=True, help="config.yml of a FruitNeRF run")
    parser.add_argument("--output-dir", type=Path, required=True, help="folder for the <view>.png masks")
    args = parser.parse_args()

    _, pipeline, checkpoint, _ = eval_setup(args.load_config, test_mode="test")
    print(f"Loaded {checkpoint}")
    pipeline.eval()
    filenames = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for camera_ray_bundle, _ in pipeline.datamanager.fixed_indices_eval_dataloader:
            view = Path(filenames[int(camera_ray_bundle.camera_indices[0, 0, 0])]).stem
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            semantics = torch.sigmoid(outputs["semantics"][..., 0]).cpu().numpy()
            rgb = torch.clamp(outputs["rgb"], 0, 1).cpu().numpy()
            Image.fromarray((semantics * 255).round().astype(np.uint8), mode="L").save(args.output_dir / f"{view}.png")
            Image.fromarray((rgb * 255).round().astype(np.uint8), mode="RGB").save(args.output_dir / f"{view}_rgb.png")
            print(f"{view}: {semantics.shape[1]}x{semantics.shape[0]}, wheat (>= 0.5) {100 * (semantics >= 0.5).mean():.1f}%")


if __name__ == "__main__":
    main()
