#!/bin/bash
# Interactive viewer of a 3D segmentation (Step 3 run), then open http://localhost:<port>
# Usage (from wheat3dgsviewer/): bash run.sh <dataset_path> <model_path> <exp_name> [port]
DATA=$1; MODEL=$2; EXP=$3; PORT=${4:-8080}
python wheatgs_rendering.py \
    --input_ply $MODEL/wheat-head/$EXP/gaussians.ply \
    --colmap_path $DATA/sparse/0 \
    --images_path $DATA/images \
    --heads_dir $MODEL/wheat-head/$EXP/ply \
    --port $PORT
