#======================================================================
#  Image Clustering using Hierarchical (Agglomerative) Clustering
#=====================================================================
# Extracts deep features from images using a pretrained CNN,
# reduces dimensionality with PCA + UMAP, then clusters with
# Agglomerative Clustering. Saves a full HTML report and
# per-cluster image grids.


import os
import sys
import argparse
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


# ─── Feature Extractor ───────────────────────────────────────────────────────

def extract_features(image_paths, method="hog", img_size=(128, 128)):
    """
    Extract image features using one of three methods:
      - 'pixel'  : raw flattened pixels (grayscale, fast)
      - 'hog'    : Histogram of Oriented Gradients (recommended)
      - 'cnn'    : MobileNetV2 embeddings (best quality, needs torch)
    """
    features = []

    if method == "cnn":
        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            device = "cuda" if torch.cuda.is_available() else "cpu"
            weights = MobileNet_V2_Weights.DEFAULT
            model = mobilenet_v2(weights=weights)
            model.classifier = torch.nn.Identity()
            model.eval().to(device)
            transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            with torch.no_grad():
                for p in tqdm(image_paths, desc="Extracting CNN features"):
                    img = Image.open(p).convert("RGB")
                    t = transform(img).unsqueeze(0).to(device)
                    feat = model(t).squeeze().cpu().numpy()
                    features.append(feat)
        except ImportError:
            print("PyTorch not found, falling back to HOG features.")
            return extract_features(image_paths, method="hog", img_size=img_size)

    elif method == "hog":
        from skimage.feature import hog
        from skimage.color import rgb2gray
        from skimage.transform import resize
        for p in tqdm(image_paths, desc="Extracting HOG features"):
            img = np.array(Image.open(p).convert("RGB"))
            img = resize(img, img_size, anti_aliasing=True)
            gray = rgb2gray(img)
            feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm="L2-Hys")
            features.append(feat)

    else:  # pixel
        for p in tqdm(image_paths, desc="Extracting pixel features"):
            img = Image.open(p).convert("L").resize(img_size)
            features.append(np.array(img).flatten() / 255.0)

    return np.array(features)


# ─── Dimensionality Reduction ─────────────────────────────────────────────────

def reduce_dimensions(features, n_components_pca=50, use_umap=False, n_components_2d=2):
    """PCA → (optional UMAP) → 2-D for visualisation."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_components_pca = min(n_components_pca, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components_pca, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA: {n_components_pca} components explain {explained*100:.1f}% variance")

    if use_umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components_2d, random_state=42)
            X_2d = reducer.fit_transform(X_pca)
            print("  UMAP: done")
        except ImportError:
            print("  UMAP not installed; using PCA-2D instead.")
            X_2d = X_pca[:, :n_components_2d]
    else:
        pca2 = PCA(n_components=n_components_2d, random_state=42)
        X_2d = pca2.fit_transform(X_pca)

    return X_pca, X_2d, scaler, pca


# ─── Optimal K via Silhouette ─────────────────────────────────────────────────

def find_optimal_k(X, k_range=(2, 10), linkage_method="ward"):
    """Sweep k and return scores."""
    scores = {}
    for k in tqdm(range(k_range[0], k_range[1] + 1), desc="Finding optimal k"):
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        scores[k] = {"silhouette": sil,
                     "davies_bouldin": davies_bouldin_score(X, labels),
                     "calinski_harabasz": calinski_harabasz_score(X, labels)}
    return scores


# ─── Clustering ───────────────────────────────────────────────────────────────

def cluster_images(X, n_clusters, linkage_method="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    return labels, model


# ─── Visualisations ──────────────────────────────────────────────────────────

COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
          "#8338ec", "#06d6a0", "#fb8500", "#023e8a", "#c77dff"]


def plot_dendrogram(X, linkage_method, out_path, n_samples=200):
    """Truncated dendrogram of a random sample."""
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    Z = linkage(pdist(X_sample), method=linkage_method)

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color="#555566",
               leaf_rotation=90, leaf_font_size=8)
    ax.set_title("Hierarchical Clustering Dendrogram", color="white",
                 fontsize=14, pad=12, fontfamily="monospace")
    ax.set_xlabel("Sample index / cluster size", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Distance", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def plot_scatter(X_2d, labels, out_path):
    """2-D scatter of clusters."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d0d0d")
    ax.set_facecolor("#111111")
    unique = np.unique(labels)
    for i, cl in enumerate(unique):
        mask = labels == cl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[i % len(COLORS)], s=60, alpha=0.75,
                   edgecolors="white", linewidths=0.3, label=f"Cluster {cl}")
    ax.legend(framealpha=0.2, labelcolor="white", edgecolor="#444444")
    ax.set_title("Cluster Scatter (2-D projection)", color="white",
                 fontsize=13, fontfamily="monospace")
    ax.tick_params(colors="#777777")
    for spine in ax.spines.values():
        spine.set_color("#2a2a2a")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def plot_k_sweep(scores, best_k, out_path):
    ks = list(scores.keys())
    sils = [scores[k]["silhouette"] for k in ks]
    dbs  = [scores[k]["davies_bouldin"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d0d0d")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#2a2a2a")

    ax1.plot(ks, sils, "o-", color="#e63946", lw=2)
    ax1.axvline(best_k, color="#f4a261", ls="--", lw=1.5, label=f"Best k={best_k}")
    ax1.set_title("Silhouette Score ↑", color="white", fontfamily="monospace")
    ax1.set_xlabel("k", color="#aaa"); ax1.set_ylabel("Score", color="#aaa")
    ax1.legend(labelcolor="white", edgecolor="#444")

    ax2.plot(ks, dbs, "o-", color="#457b9d", lw=2)
    ax2.axvline(best_k, color="#f4a261", ls="--", lw=1.5, label=f"Best k={best_k}")
    ax2.set_title("Davies-Bouldin Score ↓", color="white", fontfamily="monospace")
    ax2.set_xlabel("k", color="#aaa"); ax2.set_ylabel("Score", color="#aaa")
    ax2.legend(labelcolor="white", edgecolor="#444")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()


def save_cluster_grids(image_paths, labels, out_dir, max_per_cluster=16):
    """Save a thumbnail grid for each cluster."""
    for cl in np.unique(labels):
        idxs = np.where(labels == cl)[0][:max_per_cluster]
        n = len(idxs)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5),
                                  facecolor="#0d0d0d")
        axes = np.array(axes).flatten()
        for ax in axes:
            ax.axis("off")

        for i, idx in enumerate(idxs):
            img = Image.open(image_paths[idx]).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(Path(image_paths[idx]).name[:14],
                               color="#aaaaaa", fontsize=7, fontfamily="monospace")

        fig.suptitle(f"Cluster {cl}  ({len(np.where(labels==cl)[0])} images)",
                     color=COLORS[cl % len(COLORS)], fontsize=13,
                     fontfamily="monospace", y=1.01)
        plt.tight_layout()
        plt.savefig(out_dir / f"cluster_{cl}_grid.png", dpi=120,
                    bbox_inches="tight", facecolor="#0d0d0d")
        plt.close()


# ─── HTML Report ──────────────────────────────────────────────────────────────

def build_html_report(image_paths, labels, scores, best_k, out_dir):
    n = len(image_paths)
    unique_clusters = np.unique(labels)
    cluster_counts = {int(cl): int(np.sum(labels == cl)) for cl in unique_clusters}

    cluster_cards = ""
    for cl in unique_clusters:
        color = COLORS[int(cl) % len(COLORS)]
        count = cluster_counts[int(cl)]
        grid_img = f"cluster_{cl}_grid.png"
        cluster_cards += f"""
        <div class="card">
          <div class="card-header" style="border-left:4px solid {color}">
            <span class="chip" style="background:{color}22;color:{color}">Cluster {cl}</span>
            <span class="count">{count} images</span>
          </div>
          <img src="{grid_img}" alt="Cluster {cl} grid" loading="lazy">
        </div>"""

    metric_rows = ""
    for k, m in scores.items():
        best_row = "best-row" if k == best_k else ""
        metric_rows += f"""
        <tr class="{best_row}">
          <td>{k}{' ★' if k==best_k else ''}</td>
          <td>{m['silhouette']:.4f}</td>
          <td>{m['davies_bouldin']:.4f}</td>
          <td>{m['calinski_harabasz']:.1f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Clustering Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  :root{{--bg:#090909;--surface:#111;--border:#222;--text:#e8e8e8;--muted:#666;--accent:#e63946;}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;padding:2rem 1rem;}}
  header{{text-align:center;margin-bottom:3rem;}}
  header h1{{font-size:clamp(2rem,5vw,3.5rem);font-weight:800;letter-spacing:-2px;}}
  header h1 span{{color:var(--accent);}}
  header p{{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:.85rem;margin-top:.5rem;}}
  .stats-row{{display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;margin-bottom:3rem;}}
  .stat-box{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem 2rem;text-align:center;}}
  .stat-box .val{{font-size:2rem;font-weight:800;color:var(--accent);}}
  .stat-box .lbl{{font-size:.75rem;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-top:.2rem;}}
  h2{{font-size:1.4rem;margin:2.5rem 0 1rem;border-bottom:1px solid var(--border);padding-bottom:.5rem;}}
  .viz-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1.5rem;margin-bottom:2rem;}}
  .viz-grid img{{width:100%;border-radius:10px;border:1px solid var(--border);}}
  .cluster-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1.2rem;}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
  .card-header{{display:flex;justify-content:space-between;align-items:center;padding:.75rem 1rem;border-bottom:1px solid var(--border);}}
  .chip{{padding:.25rem .7rem;border-radius:20px;font-family:'JetBrains Mono',monospace;font-size:.8rem;font-weight:700;}}
  .count{{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:.8rem;}}
  .card img{{width:100%;display:block;}}
  table{{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:.85rem;}}
  th,td{{padding:.7rem 1rem;border-bottom:1px solid var(--border);text-align:right;}}
  th{{color:var(--muted);font-weight:400;text-align:right;}}
  th:first-child,td:first-child{{text-align:left;}}
  .best-row td{{color:#e9c46a;font-weight:700;}}
  .best-row td:first-child::after{{content:' ';}}
  footer{{text-align:center;color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:.75rem;margin-top:4rem;}}
</style>
</head>
<body>
<header>
  <h1>Image <span>Clustering</span> Report</h1>
  <p>Hierarchical (Agglomerative) Clustering · HOG Features · PCA Reduction</p>
</header>

<div class="stats-row">
  <div class="stat-box"><div class="val">{n}</div><div class="lbl">total images</div></div>
  <div class="stat-box"><div class="val">{best_k}</div><div class="lbl">optimal clusters</div></div>
  <div class="stat-box"><div class="val">{scores[best_k]['silhouette']:.3f}</div><div class="lbl">silhouette score</div></div>
  <div class="stat-box"><div class="val">{scores[best_k]['davies_bouldin']:.3f}</div><div class="lbl">davies-bouldin ↓</div></div>
</div>

<h2>Visualisations</h2>
<div class="viz-grid">
  <img src="dendrogram.png" alt="Dendrogram">
  <img src="scatter.png" alt="Cluster Scatter">
  <img src="k_sweep.png" alt="K Sweep">
</div>

<h2>Clusters</h2>
<div class="cluster-grid">{cluster_cards}</div>

<h2>Metric Sweep (k = {min(scores)} → {max(scores)})</h2>
<table>
  <thead><tr><th>k</th><th>Silhouette ↑</th><th>Davies-Bouldin ↓</th><th>Calinski-Harabasz ↑</th></tr></thead>
  <tbody>{metric_rows}</tbody>
</table>

<footer><p>Generated by image_clustering · Hierarchical Agglomerative Clustering</p></footer>
</body>
</html>"""

    with open(out_dir / "report.html", "w") as f:
        f.write(html)


# ─── Demo Data Generator ──────────────────────────────────────────────────────

def generate_demo_images(out_dir, n_per_class=20):
    """
    Creates synthetic coloured-shape images so the project runs
    without any real dataset. Three visual categories:
      0 → red circles   1 → blue squares   2 → green triangles
    """
    import random
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = [
        ("red_circle",   (220, 50, 50),  "circle"),
        ("blue_square",  (50, 80, 220),  "square"),
        ("green_tri",    (30, 180, 80),  "triangle"),
    ]
    count = 0
    for name, color, shape in classes:
        for i in range(n_per_class):
            img = Image.new("RGB", (128, 128), (20, 20, 20))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            x0 = random.randint(20, 50); y0 = random.randint(20, 50)
            x1 = x0 + random.randint(30, 60); y1 = y0 + random.randint(30, 60)
            jitter = lambda c: max(0, min(255, c + random.randint(-30, 30)))
            jc = tuple(jitter(c) for c in color)
            if shape == "circle":
                draw.ellipse([x0, y0, x1, y1], fill=jc)
            elif shape == "square":
                draw.rectangle([x0, y0, x1, y1], fill=jc)
            else:
                cx = (x0 + x1) // 2
                draw.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=jc)
            img.save(out_dir / f"{name}_{i:03d}.png")
            count += 1
    print(f"  Generated {count} demo images in '{out_dir}'")
    return list(out_dir.glob("*.png"))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Image Clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",     default="data",     help="Folder with images (or 'demo')")
    parser.add_argument("--out",      default="results",  help="Output folder")
    parser.add_argument("--feature",  default="hog",      choices=["hog","pixel","cnn"],
                        help="Feature extraction method")
    parser.add_argument("--linkage",  default="ward",
                        choices=["ward","complete","average","single"],
                        help="Linkage method for agglomerative clustering")
    parser.add_argument("--k",        default=None, type=int,
                        help="Number of clusters (auto-detect if omitted)")
    parser.add_argument("--k_min",    default=2, type=int)
    parser.add_argument("--k_max",    default=8, type=int)
    parser.add_argument("--pca",      default=50, type=int, help="PCA components")
    parser.add_argument("--umap",     action="store_true",  help="Use UMAP for 2-D viz")
    parser.add_argument("--demo",     action="store_true",  help="Generate demo images and run")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading images …")
    if args.demo or args.data == "demo":
        image_paths = generate_demo_images("data/demo_images", n_per_class=25)
    else:
        data_dir = Path(args.data)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        image_paths = sorted([p for p in data_dir.rglob("*") if p.suffix.lower() in exts])
        if not image_paths:
            print(f"No images found in '{args.data}'. Run with --demo to use synthetic data.")
            sys.exit(1)

    print(f"  Found {len(image_paths)} images")
    if len(image_paths) < 4:
        print("Need at least 4 images. Exiting."); sys.exit(1)

    # ── Extract features ─────────────────────────────────────────────────────
    print(f"\n[2/5] Extracting features (method={args.feature}) …")
    features = extract_features(image_paths, method=args.feature)
    print(f"  Feature matrix: {features.shape}")

    # ── Reduce dimensions ────────────────────────────────────────────────────
    print("\n[3/5] Reducing dimensions …")
    X_pca, X_2d, _, _ = reduce_dimensions(features, n_components_pca=args.pca,
                                           use_umap=args.umap)

    # ── Find optimal k ───────────────────────────────────────────────────────
    k_max_safe = min(args.k_max, len(image_paths) - 1)
    if args.k:
        best_k = args.k
        scores = find_optimal_k(X_pca, k_range=(max(2, best_k-2), min(k_max_safe, best_k+2)),
                                linkage_method=args.linkage)
        if best_k not in scores:
            scores[best_k] = {}
            model = AgglomerativeClustering(n_clusters=best_k, linkage=args.linkage)
            lbl = model.fit_predict(X_pca)
            scores[best_k] = {
                "silhouette": silhouette_score(X_pca, lbl),
                "davies_bouldin": davies_bouldin_score(X_pca, lbl),
                "calinski_harabasz": calinski_harabasz_score(X_pca, lbl),
            }
    else:
        print(f"\n[4/5] Sweeping k from {args.k_min} to {k_max_safe} …")
        scores = find_optimal_k(X_pca, k_range=(args.k_min, k_max_safe),
                                linkage_method=args.linkage)
        best_k = max(scores, key=lambda k: scores[k]["silhouette"])
        print(f"  Best k = {best_k}  (silhouette = {scores[best_k]['silhouette']:.4f})")

    # ── Cluster ───────────────────────────────────────────────────────────────
    print(f"\n[4/5] Clustering with k={best_k}, linkage={args.linkage} …")
    labels, _ = cluster_images(X_pca, best_k, linkage_method=args.linkage)
    for cl in np.unique(labels):
        print(f"  Cluster {cl}: {np.sum(labels==cl)} images")

    # ── Save cluster folders ──────────────────────────────────────────────────
    for cl in np.unique(labels):
        cl_dir = out_dir / f"cluster_{cl}"
        cl_dir.mkdir(exist_ok=True)
        for idx in np.where(labels == cl)[0]:
            shutil.copy(image_paths[idx], cl_dir / Path(image_paths[idx]).name)

    # ── Visualise ─────────────────────────────────────────────────────────────
    print("\n[5/5] Generating visualisations …")
    plot_dendrogram(X_pca, args.linkage,   out_dir / "dendrogram.png")
    plot_scatter(X_2d, labels,             out_dir / "scatter.png")
    plot_k_sweep(scores, best_k,           out_dir / "k_sweep.png")
    save_cluster_grids(image_paths, labels, out_dir)
    build_html_report(image_paths, labels, scores, best_k, out_dir)

    print(f"\n  Done!  Open  results/report.html  to explore the clusters.")


if __name__ == "__main__":
    main()
