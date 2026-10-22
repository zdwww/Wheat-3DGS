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

COPY . .

CMD ["/bin/bash"]
