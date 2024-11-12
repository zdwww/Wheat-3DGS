#!/bin/bash
#SBATCH -J 3d_seg
#SBATCH --mail-type=END,FAIL
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:16g
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_log/plot461_%j.out
#SBATCH --error=sbatch_log/plot461_%j.out

module load stack/2024-04
module load gcc/8.5.0 cuda/11.8.0 ninja/1.11.1 eth_proxy
echo $(module list)

export WANDB_API_KEY=75a89a1a45f5525dc3717034484308953c5e267a

source ~/.bashrc
conda activate base

conda activate gs_w_depth

nvidia-smi

HOME_DIR=/cluster/scratch/daizhang

# python -u run_3d_seg.py -s ../Wheat-GS-data/FPWW0340068_20230623 -m output/FPWW0340068_20230623
# python -u run_3d_seg_new.py -s ../Wheat-GS-data/20240717/plot_461 -m /cluster/scratch/daizhang/Wheat-GS-output/OG3DGS/plot_461_res1 --resolution 1 --iou_threshold 0.6 --num_match 5 --exp_name run2
# python -u separate_Gaussian.py -s ../Wheat-GS-data/20240717/plot_461

# python -u train.py -s ${HOME_DIR}/Wheat-GS-data-scaled/20240717/plot_461 -m ${HOME_DIR}/Wheat-GS-output-scaled/20240717/plot_461 --resolution 1 --eval

# python -u render.py -s ${HOME_DIR}/Wheat-GS-data-scaled/20240717/plot_461 -m ${HOME_DIR}/Wheat-GS-output-scaled/20240717/plot_461 --resolution 1 --eval

python -u run_3d_seg_new.py -s ${HOME_DIR}/Wheat-GS-data-scaled/20240717/plot_461 -m ${HOME_DIR}/Wheat-GS-output-scaled/20240717/plot_461 --resolution 1 --eval --iou_threshold 0.65 --num_match 3 --exp_name run2


