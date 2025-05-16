#!/bin/bash
#SBATCH -J 3d_seg
#SBATCH --mail-type=END
#SBATCH --time=48:00:00
#SBATCH --gres=gpumem:24g
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_log/run_461.out
#SBATCH --error=sbatch_log/run_461.err

module load stack/2024-04
module load gcc/8.5.0 cuda/11.8.0 ninja/1.11.1 ffmpeg/4.4.1 eth_proxy
echo $(module list)

source ~/.bashrc
conda activate base
conda activate gs_w_depth

nvidia-smi

HOME_DIR=/cluster/scratch/daizhang
DATE=20240717
PLOT=plot_461
SRC_PATH=${HOME_DIR}/Wheat-GS-data/${DATE}/${PLOT}
MDL_PATH=${HOME_DIR}/Wheat-GS-output/${DATE}/${PLOT}

EXP_NAME=run1

# OG 3DGS reconstruction as initialization
python -u train.py \
    -s $SRC_PATH \
    -m $MDL_PATH \
    --resolution 1 \
    --eval

python -u render.py \
    -s $SRC_PATH \
    -m $MDL_PATH \
    --resolution 1 \
    --eval \
    --iteration 7000

python -u metrics.py \
    -m $MDL_PATH \

# Run 3D segmentation on reconstruction
python -u run_3d_seg.py \
    -s $SRC_PATH \
    -m $MDL_PATH \
    --resolution 1 \
    --eval \
    --iou_threshold 0.6 \
    --exp_name ${EXP_NAME}

# Evaluating 3D segmentation qualitively on both entire wheat field and individual wheat heads
python -u render_360.py \
    -s $SRC_PATH \
    -m $MDL_PATH \
    --render_type field \
    --exp_name ${EXP_NAME} \
    --n_frames 200 \
    --framerate 20 

python -u render_360.py \
    -s $SRC_PATH \
    -m $MDL_PATH \
    --render_type head \
    --exp_name ${EXP_NAME} \
    
python -u eval_wheatgs.py \
    -s $SRC_PATH \
    -m $MDL_PATH \
    --resolution 1 \
    --eval \
    --exp_name ${EXP_NAME} \
    --load_counts