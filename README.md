<p align="center">
  <h3 align="center">🌾 Wheat3DGS <br> In-field 3D Reconstruction, Instance Segmentation and Phenotyping of Wheat Heads with Gaussian Splatting</h3>
  <h5 align="center">CVPR 2025 Agriculture-Vision Workshop</h5>
</p>

<div align="center"> 

[Project Page](https://zdwww.github.io/wheat3dgs/) | [Paper](https://arxiv.org/abs/2504.06978) | [Data](https://drive.google.com/drive/folders/1DJPs_E8-93dCysYkQ0-uxHrAcZGTZiVh)

  <img src="assets/teaser.png">
</div>

## Updates
- <b>[5/2/2025]</b>  Code for wheat head morphology calculation released at `wheatheadsmorphology`
- <b>[4/30/2025]</b> Initial code release 

## 📝 TODO List
- \[ \] Insllation instruction
- \[ \] Viser-based 3D wheat head segmentation viewer

## 🛠️ Setup
The setup should be very similar to the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) except we used a modified version of [differential gaussian rasterization](https://github.com/ashawkey/diff-gaussian-rasterization/tree/8829d14f814fccdaf840b7b0f3021a616583c0a1) with support of depth & alpha rendering, and an additional [flashsplat-rasterization](https://github.com/florinshen/flashsplat-rasterization/tree/189c483ffa33dd6d5661343ce496df0c6eb80a0c) submodule. We will release the complete `requirements.txt` later.

## Using Wheat3DGS
The majority of the Wheat3DGS pipeline can be executed by running this script.
```
sbatch run_wheat_3dgs.sh
```

## Baseline
To reproduce the baseline results presented in the paper (i.e. [FruitNeRF](https://github.com/meyerls/FruitNeRF)), please refer to the original repository and the scripts in the `scripts` folder.

<!-- ## Results
<p align="center">
  <img src="assets/FPWW036_SR0461_1_FIP2_cam_02.jpg" alt="Image 1" width="45%" />
  <img src="assets/FPWW036_SR0461_1_FIP2_cam_04.jpg" alt="Image 2" width="45%" />
</p> -->

## Acknowledgement
Our implementation is based on the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [FlashSplat](https://github.com/florinshen/FlashSplat). We thank the authors for their revolutionary work and open-source contributions. 

## Citation
If you find our paper useful, please cite us:
```bib
@article{zhang2025wheat3dgs,
  title={Wheat3DGS: In-field 3D Reconstruction, Instance Segmentation and Phenotyping of Wheat Heads with Gaussian Splatting},
  author={Zhang, Daiwei and Gajardo, Joaquin and Medic, Tomislav and Katircioglu, Isinsu and Boss, Mike and Kirchgessner, Norbert and Walter, Achim and Roth, Lukas},
  journal={arXiv preprint arXiv:2504.06978},
  year={2025}
}