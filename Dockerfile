# Docker setup for Wheat3DGS
# Build:  docker build -t wheat3dgs .
# Run:    docker run --gpus all -it --shm-size=16g \
#           -v /path/to/data:/workspace/data wheat3dgs

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget unzip ffmpeg libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (numpy<2 is required by torch 1.13)
RUN pip install --no-cache-dir \
        "numpy<2" scipy matplotlib pillow scikit-learn shapely pandas pyyaml \
        plyfile tqdm rich gdown "wandb<0.20" ffmpeg-python gitpython joblib \
        opencv-python-headless open3d imageio jupyterlab ninja

# Build the CUDA extensions at image build time. Setting the target
# architectures means no GPU is needed during `docker build`.
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

WORKDIR /workspace/Wheat-GS
COPY submodules ./submodules
RUN pip install --no-cache-dir \
        ./submodules/simple-knn \
        ./submodules/diff-gaussian-rasterization \
        ./submodules/flashsplat-rasterization

# 2D mask generation (scripts/generate_masks.py): YOLOv5 v7.0 (submodule) + SAM
RUN pip install --no-cache-dir seaborn thop \
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"

# Phenotyping against TLS (scripts/phenotyping)
RUN pip install --no-cache-dir hdbscan==0.8.38.post1 openpyxl==3.1.5

# Interactive viewer (wheat3dgsviewer/); nerfview also needs splines and jaxtyping, not declared as dependencies
RUN pip install --no-cache-dir viser==1.0.26 nerfview==0.1.3 splines==0.3.3 jaxtyping==0.3.7

# The base image's conda ffmpeg has no libx264 (render_360.py videos): use the apt ffmpeg installed above
RUN ln -sf /usr/bin/ffmpeg /opt/conda/bin/ffmpeg

COPY . .

CMD ["/bin/bash"]
