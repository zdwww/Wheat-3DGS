# Reproducibility notes

This page describes how to reproduce the paper's results, three issues we found after publication, and the settings we recommend for new work ("corrected track"). Running the repository as described in the [README](README.md) reproduces the paper; the corrections are opt-in.

## Reproducing the paper

Use the original dataset, the provided 2D masks (`yolosam.zip`) and `run_3d_seg_paper.py` with its defaults (`--iou_threshold 0.6 --num_match 5`, height filter on), e.g. with `run_wheat_3dgs.sh`. Our re-run (15k iterations, `--seed 0`) gives a 7-plot mean IoU / precision / recall / F1 of .506 / .806 / .576 / .669 (Table 2: .50 / .81 / .57 / .67) and PSNR / SSIM / LPIPS of 20.42 / .639 / .323 (supplementary NVS table, 3DGS at 15k iterations: 20.416 / .639 / .323). For the trait comparison with the laser scans (Tables 3 and 4), run Step 6 of the README with `--paper_heads_dir` on the paper's head point clouds of all three methods (included in the data release): this gives 3DGS vs TLS length ρ .49 and MAPE 15.0 (Table 3: .51 and 15.1), and ANOVA F for length / width / volume of 14.6 / 11.1 / 9.2, 4.6 / 35.0 / 7.9 and 6.3 / 10.6 / 7.2 for TLS / 3DGS / MVS (Table 4: 15.2 / 11.2 / 10.1, 5.2 / 35.0 / 8.5 and 6.9 / 10.8 / 7.1). The exact values cannot be recovered, because the paper-era scripts subsampled large heads without a fixed seed (seeds give ρ .485-.500). Laser-scan heads newly cut by the released scripts are smaller than the paper's, so new runs give higher agreement than Table 3 (see the table below).

## Issues found after publication

**1. Principal point (the "pixel shift" of Sec. 5.1).** The 3DGS camera model assumes the principal point at the image centre and ignores `cx`, `cy` in `sparse/`. In the original dataset the principal point is up to ~90 px off-centre (plot 461: 91.5 px, 18 px), so the 3DGS renders are slightly misaligned with the images. This is the shift reported in the paper, whose cause was unknown at the time; it lowers PSNR/SSIM and blurs the reconstruction. The corrected dataset was exported with the principal point at the image centre (Metashape's pinhole conversion; see Step 0 of the README, `scripts/preprocessing/`). The camera positions are the same within 1.6 mm, and the 3D heads match the dense photogrammetry (MVS) point clouds without the ICP correction the paper needed (`--icp_dir`).

**2. YOLOv5 input size.** The wheat-head detector was run at its default input size of 640 px, on 4096 x 3000 images, where many heads are too small to be detected. At 1280 px it finds 2.3x more heads on 17 July 2024 (63,184 vs 27,193 masks over 7 plots) and 8x more on 19 July 2024 (a second date in our data release, not used in the paper), whose darker images lose most heads at 640 px (48,123 vs 5,933). Generate masks with `scripts/generate_masks.py --img_size 1280 --out <dataset_path>/yolosam_1280` and select them with `--masks_dir yolosam_1280`.

**3. Height filter.** The 3D segmentation lifts each 2D mask only to the Gaussians above the plot's mean height, to exclude the ground. In plots with short genotypes this also removes whole rows of wheat heads (plot 466: 38% of the labelled head area lies below the mean height and is never segmented), and biases the heads towards the tall genotypes. Disable it with `--no_height_filter`.

## Corrected track (recommended for new work)

```bash
python scripts/generate_masks.py -s <dataset_path> --img_size 1280 --out <dataset_path>/yolosam_1280
python train_vanilla_3dgs.py -s <dataset_path> -m <model_path> --resolution 1 --eval
python run_3d_seg_paper.py -s <dataset_path> -m <model_path> --masks_dir yolosam_1280 --no_height_filter --exp_name corrected --seed 0
python eval_wheatgs.py -s <dataset_path> -m <model_path> --exp_name corrected --skip_train
```
with `<dataset_path>` a plot of the corrected dataset. For Step 6, leave out `--icp_dir`.

## Effect of each fix

17 July 2024, 7 plots, `run_3d_seg_paper.py`, seed 0. Table 2 metrics against the manual labels (one labelled view per plot; for the corrected dataset the labels were mapped to its images with a per-view homography, median residual 0.05 px). Traits against the laser scans as in Step 6, on the saved heads with at least 50 points (a few fewer than in the Saved heads column; ρ = Pearson correlation of head length with the laser scans, MAPE = mean absolute percentage error of length / width / volume).

| Data | Masks | Height filter | Saved heads | IoU | Precision | Recall | F1 | ρ length | MAPE L / W / V (%) |
|---|---|---|---|---|---|---|---|---|---|
| original, published¹ | 640 px | on | 2392 | .50 | .81 | .57 | .67 | .51 | 15.1 / 18.3 / 40.2 |
| original, published heads, newly cut laser-scan heads² | 640 px | on | 2392 | | | | | .79 | 9.8 / 17.6 / 36.2 |
| original, our re-run | 640 px | on | 2390 | .506 | .806 | .576 | .669 | .78 | 9.8 / 17.5 / 36.2 |
| corrected | 640 px | on | 2284 | .506 | .875 | .547 | .666 | .79 | 9.6 / 18.5 / 41.9 |
| corrected | 640 px | off | 2404 | .559 | .857 | .617 | .716 | .77 | 9.8 / 18.9 / 43.6 |
| corrected | 1280 px | on | 4791 | .539 | .748 | .639 | .687 | .80 | 12.2 / 21.1 / 45.3 |
| **corrected** | **1280 px** | **off** | **5518** | **.617** | .736 | **.794** | **.760** | .78 | 12.9 / 21.5 / 50.2 |

¹ Tables 2 and 3 of the paper (after removing failure cases, supplementary: ρ length .78, MAPE 8.2 / 16.7 / 32.2). The paper's laser-scan heads were cut with larger boxes than the released scripts give (see Step 6 of the README).

² The paper's own wheat-head point clouds (included in the data release) run through Step 6 like the rows below, i.e. with laser-scan heads newly cut by the released scripts. The rows below compare with this row, not with the published one.

Novel view synthesis (Step 1, 15k iterations, 7-plot mean PSNR / SSIM / LPIPS): 20.42 / .639 / .323 on the original dataset, 28.28 / .883 / .200 on the corrected one.

Reading the table:
- The corrected cameras alone mostly sharpen the reconstruction (+7.9 dB PSNR); segmentation and traits barely change.
- 1280 px masks and no height filter roughly double the number of heads and give the best segmentation. The additional heads include small, low and partly occluded ones, whose traits are measured less accurately: length MAPE 12.9% vs 9.8% at 640 px.
- For the most accurate traits under even illumination, 640 px masks without the height filter are a good alternative (F1 .716, length MAPE 9.8%). On darker images, 640 px misses most heads.
- `run_3d_seg.py` finds more heads per mask set and fewer duplicates, but merges and fragments more heads, and its traits are less accurate at every setting (length MAPE 12.9–16.9%, vs 9.6–12.9% for `run_3d_seg_paper.py`).
- The table uses the saved heads (`ply/`, as in the paper). The final heads (`export_heads.py --heads final`) don't overlap, but each Gaussian belongs only to the head that claimed it last, so heads lose parts to later ones: on the original dataset, length ρ .72 vs .78 and MAPE 13.2 vs 9.8%, with 1854 instead of 2390 heads. They are the better choice when each Gaussian must belong to one head (e.g. counting, rendering).
- At 1280 px the detector also finds some of the white marker plates, which can become false heads (here on 4 of 7 plots).

Caveats: Table 2 uses a single labelled view per plot. In the trait comparison, the laser-scan heads are cut with each Wheat3DGS head's bounding box, so the two lengths are not fully independent.

## Second date: 19 July 2024

The data release also contains 19 July 2024 (not in the paper), with darker illumination. Corrected track (`run_3d_seg_paper.py`, 1280 px masks, height filter off, seed 0), traits against the laser scans as in Step 6, without `--icp_dir`, with `--date 20240719` (the MVS point clouds are our own reconstruction of that date). There are no manual labels for Table 2, and these results have not been checked visually.

| Date | Method vs TLS | Saved heads | ρ length | MAE L / W (cm) / V (cm³) | MAPE L / W / V (%) | ANOVA F L / W / V | Cohen's f L / W / V |
|---|---|---|---|---|---|---|---|
| 17 July 2024 | Wheat3DGS | 5518 | .78 | 0.98 / 0.28 / 9.9 | 12.9 / 21.5 / 50.2 | 14.2 / 66.7 / 28.4 | .34 / .72 / .48 |
| | MVS | | .72 | 1.11 / 0.36 / 9.7 | 14.9 / 29.6 / 54.0 | 13.6 / 13.8 / 14.8 | .33 / .33 / .35 |
| | TLS | | | | | 21.2 / 16.0 / 12.9 | .41 / .37 / .33 |
| 19 July 2024 | Wheat3DGS | 4119 | .72 | 0.97 / 0.26 / 7.9 | 13.6 / 22.7 / 78.2 | 8.2 / 22.2 / 11.4 | .30 / .49 / .35 |
| | MVS | | .65 | 1.14 / 0.40 / 8.9 | 16.4 / 35.8 / 95.2 | 8.4 / 10.2 / 9.7 | .30 / .33 / .33 |
| | TLS | | | | | 15.4 / 9.5 / 9.9 | .40 / .32 / .33 |

MVS and TLS heads are cut with the Wheat3DGS bounding boxes, so all three methods measure the same heads. ANOVA F (as in Table 4 of the paper) and Cohen's f measure how well a trait separates the genotypes. F grows with the number of heads and with duplicate heads (see the note in Step 3 of the README), so compare it between methods within a date only; Cohen's f is the effect size (between-genotype vs within-genotype spread), independent of the number of heads, and can be compared between dates. Both are computed by step 9 (`compute_anova_stats`); Table 4 reports only F.
