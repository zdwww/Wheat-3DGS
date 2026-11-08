#!/bin/bash
# Runs a command inside the Wheat3DGS Apptainer image (README: Apptainer), on a SLURM cluster or directly.
# The image: apptainer build wheat3dgs.sif docker://joaquingajardo/wheat3dgs:torch1.13-cu116
# Usage, from the repository root:
#   sbatch scripts/apptainer_job.sh bash run_wheat_3dgs.sh data/plot_461 output/plot_461
#   bash scripts/apptainer_job.sh python render.py -m output/plot_461      # without SLURM
# Environment variables:
#   SIF         the image (default: wheat3dgs.sif in the repository root)
#   BIND        extra folders to mount at the same path, comma-separated (e.g. data/outputs on a scratch disk)
#   TORCH_HOME  folder with the LPIPS weights for metrics.py, if the compute nodes have no internet access
# Add cluster-specific options to sbatch, e.g. on ETH Euler --gres=gpumem:24g for a GPU with at least 24 GB VRAM.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00

REPO=${SLURM_SUBMIT_DIR:-$PWD}
SIF=${SIF:-$REPO/wheat3dgs.sif}
BINDS="--bind $REPO:/workspace/Wheat-GS"
[ -n "$BIND" ] && BINDS="$BINDS --bind $BIND"
ENVS="LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu" # container libraries first: --nv adds the host's libGL, which breaks open3d
[ -n "$TORCH_HOME" ] && BINDS="$BINDS --bind $TORCH_HOME" && ENVS="$ENVS,TORCH_HOME=$TORCH_HOME"

echo "host: $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "repo: $REPO @ $(git -C $REPO log --oneline -1)"; echo "image: $SIF"; echo "command: $*"
apptainer exec --nv --cleanenv $BINDS --pwd /workspace/Wheat-GS --env $ENVS $SIF "$@"
