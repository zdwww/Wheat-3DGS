0)Previous steps:
	A) SfM, dense MVS, Gaussian splatting / Wheat3DGS -> export results as point clouds (default in .ply file format) in PCS (project coordinate system)
	B) Aligning/registering individual scans into a common arbitrary local coordinate system LCS -> export in .e57 (if handled as separate point clouds, clip to ROI in cloudcompare/python and merge together)
	C) OPTIONAL: at some point during the process split point clouds into individual plots (both for A and B) for computational efficiency

Steps relying on the scripts provided in this folder:

1) Estimating transformation parameters between LCS and PCS
	- script: s1_transformations_estimate.py
	- input: manually detected centers of photogrammetric targets in dense MVS and TLS point clouds (+ assigned corresponding IDs)
	- output: transformation parameters from LCS_2_PCS
	- remark: herein done per plot, but can be done in a bulk (if rigid body over the whole region of interest can be assumed)
	
2) Apply transformations to transform TLS point clouds from LCS to PCS:
	- script: s2_transformations_apply.py
	- input: TLS point cloud(s), per plot, in LCS, transformation matrices from step 1)
	- output: TLS point cloud(s) in PCS, per plot

3) OPTIONAL: adjusting for a small (cm-level) shift between coordinate system of Wheat3DGS output point clouds and PCS (in which dense MVS and TLS point clouds are now in)
	- script: s3_transform_from_gscs_to_pcs.py
	- input: per wheat head (and per plot) gaussian centroids in .ply in GSCS; 4x4 transformation matrices GSCS-to-PCS (currently estimated separately per-plot in cloudcompare)
	- output: per wheat head (and per plot) gaussian centroids in .ply in PCS
	- remarks: GSCS = "gaussian splatting coordinate system"; this step is a correction of an error in processing, Wheat3DGS output should directly be in PCS
	- remarks2: transformation parameters estimated over ICP in a separate, semi manual procedure in cloud compare (I wanted to have my eyes on the process), can be easily automated by running any ICP implementation (e.g. cloudcompy or open3D)
	
4) OPTIONAL: filter-out ("trim") all points in query point clouds (tls, mvs) that are more than a threshold (0.015m in paper) away from Wheat3DGS wheat head point clouds,...
		   ...mostly for later computational efficiency (questionably relevant), I did it mostly for visual inspection of how well the point clouds are aligned and it stayed in the manuscript
	- script: s4_trim_point_clouds_by_proximity.py
	- input: query point clouds in pcs, Wheat3DGS output (wheat head instance) point clouds in pcs
	- output: trimmed query point clouds (cca 1/4 of the initial size) in pcs
	
5) Get per wheat head bounding boxes (and traits) in PCS using "wheatheadsmorphology" toolbox on Wheat3DGS output (needed for segmenting TLS and dense MVS point clouds)
	- script: s5_wheatheadmorphology.py
	- input: individual wheat head point clouds in .ply from Step 3)
	- output: bounding boxes in .json file format (and traits in .xlsx); optional: joint point cloud of processed/cleaned wheat heads
	- remark: see https://github.com/tomedic/wheatheadsmorphology-pub
	
6) Extract point clouds of individual wheat heads from plot-level TLS and/or dense MVS point clouds
	- script: s6_extract_wheat_head_instances.py
	- input: bounding boxes in .json file format from step 5), plot-level point clouds in PCS (see "0) Previous steps") - TLS and/or MVS
	- output: per wheat head (per plot) point clouds in .ply in PCS for TLS and/or MVS
	
5b) Run "wheatheadsmorphology" toolbox to get per wheat head traits (morphology) estimates for TLS and/or MVS (for Wheat3DGS already estimated in Step 5)
	- script: s5_wheatheadmorphology.py
	- input: individual wheat head point clouds in .ply (separated per instrument in different folders and per plot in sub-folders), result of Step 6)
	- output: .xlsx with traits; optional: joint point cloud of processed/cleaned wheat heads, bounding boxes
	- remark: see https://github.com/tomedic/wheatheadsmorphology-pub

7) Assign genotypes to each wheat head based on the D-USYS polygons for each row/genotype in LV95 (run for Wheat3DGS, TLS, MVS)
	- script: s7_assign_genotype.py
	- input: traits in .xlsx - outputs of Step 5) and 5b); genotype/row related bounding boxes in .json - provided by D-USYS; transformation matrices from lcs_to_pcs (per-plot, result of Step 1) );
			 extra transformation matrix from pcs_to_lcs for D-USYS bounding boxes (estimated using all target center coordinates except ones for the corrupted plot 466) 
	- output: traits + genotype/row_id for each wheat head per plot (one .csv file per plot for each instrument)
	- remark: analysis made in LCS (local coordinate system of TLS data) because one of the plots processed with COLMAP had a completely messed-up coordinate system and this was at the time easy hack to deal with it 
	          if such errors avoided, all data should already be in PCS (or LV95, hence needing only a shift of [-2693000, -1256000, 0]), and all transformations can be removed from the script

8) Concatenate all per-plot .csv files with traits and genotypes (result of Step 7) into a single .csv file (run for Wheat3DGS, TLS, MVS)
	- script: s8_combine_plot_files.py

9) Compute traits statistics for length, width, volume - including stats required for Table 3 and 4 in the paper
	- script: s9_compute_traits_statistics.py
	- input: .csv files (one per instrument) with traits and genotype_id per wheat head for all plots combined; output of Step 8)
	- output: .csv tables with statistics (Table 3 and 4 from the paper)


