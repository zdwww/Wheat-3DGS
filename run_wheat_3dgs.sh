#!/bin/bash
# Wheat3DGS pipeline on one plot, with the paper's settings (README Steps 1, 3, 4 and 5):
# 3DGS reconstruction + novel view synthesis metrics, 3D segmentation, video of the field, Table 2 metrics.
# Usage: bash run_wheat_3dgs.sh <dataset_path> <model_path> [exp_name]
set -e
SRC_PATH=$1
MDL_PATH=$2
EXP_NAME=${3:-run1}
if [ -z "$SRC_PATH" ] || [ -z "$MDL_PATH" ]; then
    echo "Usage: bash run_wheat_3dgs.sh <dataset_path> <model_path> [exp_name]"
    exit 1
fi

# Step 1: 3DGS reconstruction, then render the test views and compute PSNR/SSIM/LPIPS
python -u train_vanilla_3dgs.py -s $SRC_PATH -m $MDL_PATH --resolution 1 --eval
python -u render.py -m $MDL_PATH --skip_train
python -u metrics.py -m $MDL_PATH

# Step 3: 3D segmentation of the wheat heads (the paper's algorithm and settings)
python -u run_3d_seg_paper.py -s $SRC_PATH -m $MDL_PATH --exp_name $EXP_NAME --seed 0

# Step 4: video around the segmented field (also saves all_counts.pth, used by Step 5)
python -u render_360.py -s $SRC_PATH -m $MDL_PATH --render_type field --exp_name $EXP_NAME \
    --n_frames 200 --framerate 20 --elevation 45
# One video per wheat head (takes long):
# python -u render_360.py -s $SRC_PATH -m $MDL_PATH --render_type head --exp_name $EXP_NAME --n_frames 100 --framerate 10 --elevation 30

# Step 5: render the 3D segmentation into the test views and compute the Table 2 metrics against the manual labels
python -u eval_wheatgs.py -s $SRC_PATH -m $MDL_PATH --exp_name $EXP_NAME --skip_train --load_counts
python -u evaluate_2D.py -s $SRC_PATH -m $MDL_PATH --exp_name $EXP_NAME
