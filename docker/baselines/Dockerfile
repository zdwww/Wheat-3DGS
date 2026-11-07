# Baselines for the Wheat3DGS paper:  docker build -t joaquingajardo/wheat3dgs-baselines:ns1.1.5 docker/baselines
# - nerfacto, splatfacto, instant-ngp: nerfstudio 1.1.5 of the base image, `ns-train <method> ...`
#   (the paper used 1.1.3 for nerfacto, 1.1.5 for the others)
# - FruitNeRF: conda env `fruitnerf` (nerfstudio 0.3.2), `conda run -n fruitnerf ns-train fruit_nerf ...`
FROM ghcr.io/nerfstudio-project/nerfstudio:1.1.5@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e

# No GPU during the build: compile tiny-cuda-nn and nerfacc for these GPUs (same as the base image)
ENV TCNN_CUDA_ARCHITECTURES="90;89;86;80;75;70;61" \
    TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0" \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=$PATH:/opt/conda/bin

# FruitNeRF README: nerfstudio install docs up to tiny-cuda-nn, then nerfstudio==0.3.2
# (+ viser/tyro pins from FruitNeRF issues #4/#33). Constraints = pip freeze of the env used for the paper.
COPY fruitnerf-constraints.txt fruitnerf-cde0974.patch /workspace/
RUN conda create -y -n fruitnerf python=3.8 && \
    conda install -y -n fruitnerf -c "nvidia/label/cuda-11.8.0" cuda-toolkit && \
    conda run -n fruitnerf pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118 && \
    conda run -n fruitnerf pip install ninja && \
    conda clean -afy
# libcuda.so comes with the GPU driver and isn't there during `docker build`: link tiny-cuda-nn against the toolkit's stub
RUN STUBS=$(dirname $(find /opt/conda/envs/fruitnerf -path "*stubs/libcuda.so" | head -1)) && test -f "$STUBS/libcuda.so" && echo "stubs: $STUBS" && \
    LIBRARY_PATH=$STUBS conda run -n fruitnerf pip install \
        "git+https://github.com/NVlabs/tiny-cuda-nn/@075158a70b87dba8729188a9cadc9411cfa4b71d#subdirectory=bindings/torch" && \
    conda run -n fruitnerf pip install viser==0.1.24 tyro==0.8.12 nerfstudio==0.3.2 \
        -c /workspace/fruitnerf-constraints.txt --extra-index-url https://download.pytorch.org/whl/cu118 && \
    conda clean -afy

# FruitNeRF at the commit used for the paper, with our patch (eval fixes, use the splits in transforms.json)
RUN git clone https://github.com/meyerls/FruitNeRF.git /workspace/FruitNeRF && cd /workspace/FruitNeRF && \
    git checkout cde09740299b8cf04a15c0cc1b21b261f2eb8b5d && git apply /workspace/fruitnerf-cde0974.patch && \
    conda run -n fruitnerf pip install -e . -c /workspace/fruitnerf-constraints.txt
# (`ns-train -h` can't be checked here: tiny-cuda-nn needs a GPU to import)

# instant-ngp: nerfacc in the base image has no compiled CUDA code and there's no nvcc to build it on first use;
# compile it with the env's CUDA toolkit (prebuilt nerfacc 0.5.2 wheels only exist for torch 2.0)
RUN export PATH=/opt/conda/envs/fruitnerf/bin:$PATH CUDA_HOME=/opt/conda/envs/fruitnerf && \
    /usr/bin/python3 -m pip install wheel && \
    /usr/bin/python3 -m pip install --no-build-isolation --no-deps --force-reinstall --no-binary nerfacc nerfacc==0.5.2 && \
    /usr/bin/python3 -c "from nerfacc import csrc" && \
    rm -rf /root/.cache/pip /root/.cache/conda

# torch.compile (splatfacto) calls getpass.getuser(), which fails for a `docker run --user <uid>` without a passwd entry
ENV USER=user
WORKDIR /workspace
