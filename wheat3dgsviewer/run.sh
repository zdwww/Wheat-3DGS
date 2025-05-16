#!/bin/bash
# python singlewheat_rendering.py \
#     --input_ply ../WheatGS-workspace/output/plot_461/wheat-head/run_vis/wh_it2.ply \
#     --colmap_path ../WheatGS-workspace/data/plot_461/sparse/0 \
#     --images_path ../WheatGS-workspace/data/plot_461/images \

python wheatgs_rendering.py \
    --input_ply ../WheatGS-workspace/output/plot_464/wheat-head/run_test/gaussians.ply \
    --colmap_path ../WheatGS-workspace/data/plot_464/sparse/0 \
    --images_path ../WheatGS-workspace/data/plot_464/images \
    --labels_path ../WheatGS-workspace/output/plot_464/wheat-head/run_test/all_obj_labels.pth
    # --ckpt examples/assets/ckpt_6999_crop.pt \
    # --input_ply ../WheatGS-workspace/output/plot_464/wheat-head/run_test/gaussians.ply \
    # --ckpt ../WheatGS-workspace/output/plot_465/chkpnt7000.pth