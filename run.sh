#!/bin/bash
#SBATCH -J 3d_seg
#SBATCH --mail-type=END,FAIL
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=sbatch_log/%j.out
#SBATCH --error=sbatch_log/%j.out

module load stack/2024-04
module load gcc/8.5.0 cuda/11.8.0 ninja/1.11.1 eth_proxy
echo $(module list)

source ~/.bashrc
conda activate base

conda activate gs_w_depth

nvidia-smi

python -u run_3d_seg.py -s ../Wheat-GS-data/20240717/plot_461 -m output/plot_461/
