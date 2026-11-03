# Docker setup for Wheat3DGS
# Build:  docker build -t wheat3dgs .
# Run:    docker run --gpus all -it --shm-size=16g --user $(id -u):$(id -g) -e HOME=/tmp \
#           -v $PWD/data:/workspace/Wheat-GS/data -v $PWD/output:/workspace/Wheat-GS/output \
#           -v $PWD/weights:/workspace/Wheat-GS/weights wheat3dgs

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget unzip ffmpeg libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies, pinned to the tested versions (requirements.txt)
WORKDIR /workspace/Wheat-GS
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Build the CUDA extensions at image build time. Setting the target
# architectures means no GPU is needed during `docker build`.
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

COPY submodules ./submodules
RUN pip install --no-cache-dir \
        ./submodules/simple-knn \
        ./submodules/diff-gaussian-rasterization \
        ./submodules/flashsplat-rasterization

# The base image's conda ffmpeg has no libx264 (render_360.py videos): use the apt ffmpeg installed above
RUN ln -sf /usr/bin/ffmpeg /opt/conda/bin/ffmpeg

COPY . .

CMD ["/bin/bash"]
