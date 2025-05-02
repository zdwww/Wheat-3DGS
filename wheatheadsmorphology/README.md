# 🌾 Wheat Head Morphology Calculation

This Python application runs a configurable point cloud processing pipeline for extracting structural traits
(morphology) of individual wheat head point clouds. The pipeline is configured via a YAML file and performs I/O,
preprocessing, and geometric analysis of 3D point clouds.

---

## 📥 Input

- folder of point cloud files (individual wheat heads) in .txt or .ply file format (see ./data_example/)
- ⚙️ Configuration: YAML file defining necessary parameters (see default: config.yaml)

---

## 📤 Output

- A table of structural traits per wheat head (`.xlsx`) in a predefined output directory
(see config.yaml, default: ./results/)
- Optional outputs (see config.yaml):
  - Processed point clouds in a single .ply file;
  used for visual inspection of the preprocessing results (e.g. in CloudCompare - CC)
  **Note:** When importing .ply in CC, under "Scalar fields" select "Add" and under "None" select "file_id"
  - Per wheat head bounding boxes (obb - oriented bounding boxes and aabb - axis aligned bounding boxes) 
  in a single .json file

---

## 📐 Extracted Traits

- **Position [m]**: Centroid of the point cloud (X, Y, Z)
- **Length [m]**: 2D spline length projected onto the PCA P1–P2 plane
- **Width [m]**: Robust point-to-plane distance with respect to the PCA P1–P2 plane
- **Volume [m³]**: Volume of a convex hull (via Open3D/Qhull)
- **Curvature [a.u.]**: Ratio of spline length to chord length
- **Inclination [deg]**: Angle between 1st PCA component and vertical (Z) axis

---

## 🧪 Project Structure

```
main.py                 # Entry point
config.yaml             # Default configuration
wheatheadsmorphology/   # Package with all necessary functions
├── pipeline.py         # Main script (I/0, point cloud processing, morphology)
data_example/           # example dataset for testing the application
results/                # Default results directory
```

---

## ▶️ How to Run

```bash
python main.py -c path/to/config.yaml
```

If no path is provided, it defaults to `./config.yaml`.

---

## 📦 Requirements

- see requirements.txt

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Authors

Tomislav Medic (tomedic.tm@gmail.com) & ChatGPT  
Updated for public release: **2 May 2025**

---

## 📚 Citation

This code was developed for:

Zhang, Daiwei, Joaquin Gajardo, Tomislav Medic, Isinsu Katircioglu, Mike Boss, Norbert Kirchgessner, Achim Walter, and Lukas Roth.  
**"Wheat3DGS: In-field 3D Reconstruction, Instance Segmentation and Phenotyping of Wheat Heads with Gaussian Splatting."**  
*arXiv preprint* arXiv:2504.06978 (2025).  
[https://arxiv.org/abs/2504.06978](https://arxiv.org/abs/2504.06978)