#  Image Clustering with Hierarchical (Agglomerative) Clustering

Unsupervised image organisation using **HOG / CNN features → PCA → Agglomerative Clustering**.  
No labels required. Automatically groups visually similar images and produces a full HTML report.

---

##  Features

| Feature | Details |
|---|---|
| **Feature extraction** | HOG (default), raw pixels, or MobileNetV2 CNN embeddings |
| **Dimensionality reduction** | PCA (required) + optional UMAP |
| **Clustering** | Agglomerative Clustering (ward / complete / average / single linkage) |
| **Auto-k detection** | Silhouette score sweep across user-defined range |
| **Evaluation metrics** | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| **Visualisations** | Dendrogram · 2-D scatter · k-sweep chart · per-cluster image grids |
| **Report** | Self-contained `results/report.html` with dark-mode UI |
| **Demo mode** | Generates synthetic shape images — zero setup needed |

---

##  Project Structure

```
image_clustering/
├── cluster.py               # Main pipeline
├── requirements.txt         # Python dependencies
├── README.md
├── data/                    # Put your images here (any subfolders OK)
│   └── demo_images/         # Auto-created in demo mode
├── results/                 # All outputs land here
│   ├── report.html          # ← Open this in a browser
│   ├── dendrogram.png
│   ├── scatter.png
│   ├── k_sweep.png
│   ├── cluster_0_grid.png
│   ├── cluster_1_grid.png   # … one grid per cluster
│   └── cluster_0/           # Copies of images per cluster
│       └── *.jpg
└── notebooks/
    └── exploration.ipynb    # Step-by-step interactive notebook
```

---

##  Quick Start

### 1 — Clone / Download

```bash
git clone https://github.com/yourname/image-clustering.git
cd image_clustering
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Optional upgrades** — for best results:
> ```bash
> pip install torch torchvision   # CNN feature extraction
> pip install umap-learn          # UMAP 2-D projection
> ```

### 3 — Run with demo data (no images needed)

```bash
python cluster.py --demo
```

Results are written to `results/`. Open **`results/report.html`** in any browser.

### 4 — Run on your own images

```bash
python cluster.py --data path/to/your/images
```

Supported formats: `.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff`  
Subfolders are searched recursively.

---

##  All Options

```
python cluster.py [OPTIONS]

  --data       DIR    Folder containing images  [default: data]
  --out        DIR    Output folder             [default: results]
  --feature    STR    hog | pixel | cnn         [default: hog]
  --linkage    STR    ward|complete|average|single [default: ward]
  --k          INT    Fix number of clusters (skip auto-detect)
  --k_min      INT    Min k for sweep           [default: 2]
  --k_max      INT    Max k for sweep           [default: 8]
  --pca        INT    PCA components            [default: 50]
  --umap             Use UMAP for 2-D projection (needs umap-learn)
  --demo             Generate & use synthetic demo images
```

### Examples

```bash
# Auto-detect clusters, HOG features (recommended default)
python cluster.py --data ./my_photos

# Force 5 clusters using CNN embeddings
python cluster.py --data ./my_photos --feature cnn --k 5

# Ward linkage + UMAP + sweep up to 12 clusters
python cluster.py --data ./my_photos --k_max 12 --umap --linkage ward

# Use complete linkage (useful for non-elliptical clusters)
python cluster.py --data ./my_photos --linkage complete --k_min 3 --k_max 10
```

---

##  How It Works

```
Images
  │
  ▼
Feature Extraction  (HOG / pixel / CNN)
  │  Per image → feature vector
  ▼
Standardisation     (zero-mean, unit-variance)
  │
  ▼
PCA                 (reduce to n_pca dimensions)
  │  keeps ~95% variance, removes noise
  ▼
[Optional UMAP]     (2-D only for visualisation)
  │
  ▼
Silhouette Sweep    (k_min … k_max)
  │  pick k with highest silhouette score
  ▼
Agglomerative Clustering  (chosen k, ward linkage)
  │
  ▼
Results & Report
```

### Algorithm Details — Agglomerative Clustering

Hierarchical clustering builds a tree (dendrogram) bottom-up:

1. Each image starts as its own cluster.
2. The two closest clusters are merged.
3. Step 2 repeats until `k` clusters remain.

**Linkage methods** determine "distance between clusters":

| Linkage | Merges clusters by | Best for |
|---|---|---|
| `ward` | Minimising within-cluster variance | Compact, similarly-sized clusters |
| `complete` | Maximum pairwise distance | Well-separated clusters |
| `average` | Average pairwise distance | Balanced cases |
| `single` | Minimum pairwise distance | Elongated / chain-like clusters |

---

## 📊 Evaluation Metrics

| Metric | Range | Better when |
|---|---|---|
| **Silhouette Score** | −1 to 1 | Higher (close to 1 = tight, well-separated) |
| **Davies-Bouldin** | 0 → ∞ | Lower (0 = perfect separation) |
| **Calinski-Harabasz** | 0 → ∞ | Higher (dense, well-separated clusters) |

The HTML report displays all three metrics for each value of k.

---

##  Interactive Notebook

```bash
pip install jupyter
jupyter notebook notebooks/exploration.ipynb
```

The notebook walks through every step interactively with inline plots.

---

## 🐍 Python API

```python
from cluster import extract_features, reduce_dimensions, cluster_images

# 1. Features
features = extract_features(image_paths, method="hog")

# 2. Reduce
X_pca, X_2d, _, _ = reduce_dimensions(features, n_components_pca=30)

# 3. Cluster
labels, model = cluster_images(X_pca, n_clusters=4)

# labels[i] == cluster index for image_paths[i]
```

---

##  Troubleshooting

| Problem | Fix |
|---|---|
| `No images found` | Check `--data` path; use `--demo` to test |
| `ModuleNotFoundError: umap` | `pip install umap-learn` |
| `ModuleNotFoundError: torch` | `pip install torch torchvision` or use `--feature hog` |
| Too few clusters found | Increase `--k_max` or try `--feature cnn` |
| Report images look grey | Increase `--pca` or switch feature method |
| Slow on large datasets | Reduce `--pca`, use `--feature pixel`, or subsample images |

---


