#!/bin/bash
#SBATCH -J 3d_seg
#SBATCH --mail-type=END
#SBATCH --time=48:00:00
#SBATCH --gres=gpumem:24g
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_log/render_461.out
#SBATCH --error=sbatch_log/render_461.out

module load stack/2024-04
module load gcc/8.5.0 cuda/11.8.0 ninja/1.11.1 ffmpeg/4.4.1 eth_proxy
echo $(module list)

export WANDB_API_KEY=75a89a1a45f5525dc3717034484308953c5e267a

source ~/.bashrc
conda activate base

conda activate gs_w_depth

nvidia-smi

HOME_DIR=/cluster/scratch/daizhang
DATE=20240717
PLOT=plot_461

# for i in {461..467}; do
#     PLOT="plot_${i}"
#     python -u eval_wheatgs.py \
#         -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} \
#         -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT} \
#         --resolution 1 \
#         --eval \
#         --exp_name run1 \
#         --skip_train
# done

python -u render_360.py \
    -s ${HOME_DIR}/Wheat-GS-data-scaled/20240717/${PLOT} \
    -m ${HOME_DIR}/Wheat-GS-output-scaled/20240717/${PLOT} \
    --resolution 1 \
    --eval \
    --which_wheat_head all

# for i in {461..467}; do
#     PLOT="plot_${i}"
#     python -u metrics.py -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT}
#     # python -u render.py -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT} --resolution 1 --eval --iteration 7000
# done