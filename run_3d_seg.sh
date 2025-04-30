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

# python -u run_3d_seg.py -s ../Wheat-GS-data/FPWW0340068_20230623 -m output/FPWW0340068_20230623
# python -u run_3d_seg_new.py -s ../Wheat-GS-data/20240717/plot_461 -m /cluster/scratch/daizhang/Wheat-GS-output/OG3DGS/plot_461_res1 --resolution 1 --iou_threshold 0.6 --num_match 5 --exp_name run2
# python -u separate_Gaussian.py -s ../Wheat-GS-data/20240717/plot_461

# python -u fine-tune-white.py \
#     -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} \
#     -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT}_white \
#     --resolution 1 \
#     --eval \
#     --white_background \
#     --iterations 3000

# python -u train.py \
#     -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} \
#     -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT}_rerun \
#     --resolution 1 \
#     --eval \

# python -u render.py -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} -m ${HOME_DIR}/Wheat-GS-output-undistorted/${DATE}/${PLOT} --resolution 1 --eval --iteration 7000
# python -u metrics.py -m ${HOME_DIR}/Wheat-GS-output-undistorted/${DATE}/${PLOT}

# python -u run_3d_seg_new.py -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT} --resolution 1 --eval --iou_threshold 0.6 --num_match 5 --exp_name run1
# python run_3d_seg_vis.py \
#     -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} \
#     -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT} \
#     --resolution 1 \
#     --eval \
#     --iou_threshold 0.7 \
#     --exp_name run_vis

# python process_count.py -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT} --which_run rerun

python -u render_360.py \
    -s ${HOME_DIR}/Wheat-GS-data-scaled/20240717/${PLOT} \
    -m ${HOME_DIR}/Wheat-GS-output-scaled/20240717/${PLOT} \
    --resolution 1 \
    --eval \
    --which_wheat_head all \
    --exp_name run_test

# --which_wheat_head all

# python ffmpeg_test.py

# python -u eval_wheatgs.py \
#     -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} \
#     -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT} \
#     --resolution 1 \
#     --eval \
#     --exp_name rerun \
#     --load_counts


# for i in {461..467}; do
#     PLOT="plot_${i}"
#     python -u metrics.py -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT}
#     # python -u render.py -s ${HOME_DIR}/Wheat-GS-data-scaled/${DATE}/${PLOT} -m ${HOME_DIR}/Wheat-GS-output-scaled/${DATE}/${PLOT} --resolution 1 --eval --iteration 7000
# done