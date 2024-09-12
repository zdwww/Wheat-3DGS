source ~/.bashrc
conda activate base

# create environment without submodules and torch
conda env create -f ./environment_euler.yml

conda activate gs_w_depth

# manually install torch related packages
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# call a slurm file to install submodules
sbatch ./setup_submodules.sh
