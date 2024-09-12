#!/bin/bash
#SBATCH -J setup
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --output=./output/setup.out

module load stack/2024-04
module load gcc/8.5.0 cuda/11.8.0 ninja/1.11.1 eth_proxy
echo $(module list)

source ~/.bashrc
conda activate base

conda activate gs_w_depth

pip install --no-cache-dir submodules/diff-gaussian-rasterization
pip install --no-cache-dir submodules/simple-knn
