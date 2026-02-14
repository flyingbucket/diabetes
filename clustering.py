# -*- coding: utf-8 -*-
"""
Cluster pharmacist intervention features, plot PCA scatter, silhouette curve, and feature heatmap.

Run:
  python cluster_intervention.py \
    --input "/mnt/data/糖肾MDT数据.xlsx" \
    --sheet "均一化" \
    --kmin 2 --kmax 6 \
    --out_png "/mnt/data/pharm_intervention_clustering.png" \
    --out_csv "/mnt/data/pharm_intervention_with_clusters.csv"
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---- Column mapping (Chinese -> English) for pharmacist intervention block ----
INTERVENTION_COL_MAP = {
    "用药合理性评分（低中高）": "med_rationality_score",
    "患者依从性评分（低中高）": "adherence_score",
    "治疗方案优化干预数（新增、停用或剂量调整）": "regimen_opt_interventions",
    "GDMT（新四联）使用数量": "gdmt_count",
    "药物相互作用或不良反应干预数": "ddi_adr_interventions",
    "用药教育药物数量": "med_education_count",
    "医生接受建议并执行数": "physician_acceptance_count",
}

# If you previously renamed columns to English, we also accept these names directly:
INTERVENTION_ENGLISH = list(INTERVENTION_COL_MAP.values())


def _kmeans_fit_predict(X, k, random_state=42):
    """Compatibility wrapper for different sklearn versions."""
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(X), km


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Excel path")
    parser.add_argument(
        "--sheet", default="均一化", help="Sheet name (default: 均一化)"
    )
    parser.add_argument(
        "--kmin", type=int, default=2, help="Min k for silhouette (default: 2)"
    )
    parser.add_argument(
        "--kmax", type=int, default=6, help="Max k for silhouette (default: 6)"
    )
    parser.add_argument(
        "--out_png", default="pharm_intervention_clustering.png", help="Output PNG path"
    )
    parser.add_argument(
        "--out_csv", default="", help="Optional: save CSV with best-k cluster labels"
    )
    args = parser.parse_args()

    # In this workbook, header row is the 2nd row (first row is section banners)
    df = (
        pd.read_excel(args.input, sheet_name=args.sheet, header=1)
        .dropna(how="all")
        .copy()
    )

    # Rename Chinese columns to English where present (only intervention block)
    present_map = {
        cn: en for cn, en in INTERVENTION_COL_MAP.items() if cn in df.columns
    }
    df = df.rename(columns=present_map)

    # Determine which intervention columns are available
    available_cols = [c for c in INTERVENTION_ENGLISH if c in df.columns]
    if len(available_cols) < 2:
        raise ValueError(
            f"Not enough intervention columns found. "
            f"Available columns: {available_cols}. "
            f"Please check sheet '{args.sheet}' and column names."
        )

    # Build feature matrix
    X_raw = df[available_cols].copy()

    # Impute missing values with median (robust for counts/scores)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    # Standardize to z-scores
    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp)

    # --- Silhouette over k ---
    k_values = list(range(args.kmin, args.kmax + 1))
    sil_scores = []
    kms = {}

    for k in k_values:
        labels, km = _kmeans_fit_predict(X, k, random_state=42)
        # silhouette requires at least 2 clusters and no empty cluster
        score = silhouette_score(X, labels, metric="euclidean")
        sil_scores.append(score)
        kms[k] = (labels, km)

    best_idx = int(np.argmax(sil_scores))
    best_k = k_values[best_idx]
    best_labels, best_km = kms[best_k]

    # PCA (3 components so we can do PC1-PC2 and PC2-PC3)
    pca = PCA(n_components=3, random_state=42)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100  # %

    # Prepare two-color map (purple/yellow) for k=2; otherwise use tab10
    if best_k == 2:
        cmap_scatter = ListedColormap(["#F1C40F", "#3B0F70"])  # yellow, purple-ish
    else:
        cmap_scatter = plt.cm.get_cmap("tab10", best_k)

    # Heatmap values: mean z-score per cluster for each feature
    cluster_means = []
    for c in range(best_k):
        cluster_means.append(X[best_labels == c].mean(axis=0))
    cluster_means = np.vstack(cluster_means).T  # shape: n_features x n_clusters

    # ---- Plot 2x2 figure (A-D) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # A: PC1 vs PC2
    scA = axA.scatter(
        Z[:, 0], Z[:, 1], c=best_labels, cmap=cmap_scatter, s=18, alpha=0.9
    )
    axA.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
    axA.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
    axA.set_title("A", loc="left", fontweight="bold")
    axA.grid(True, alpha=0.3)

    # B: PC2 vs PC3
    scB = axB.scatter(
        Z[:, 1], Z[:, 2], c=best_labels, cmap=cmap_scatter, s=18, alpha=0.9
    )
    axB.set_xlabel(f"PC2 ({var[1]:.1f}% variance)")
    axB.set_ylabel(f"PC3 ({var[2]:.1f}% variance)")
    axB.set_title("B", loc="left", fontweight="bold")
    axB.grid(True, alpha=0.3)

    # C: silhouette curve
    axC.plot(k_values, sil_scores, marker="o", linewidth=2)
    axC.axvline(best_k, linestyle="--", linewidth=1.5)
    axC.set_xlabel("Number of Clusters")
    axC.set_ylabel("Silhouette Score")
    axC.set_title("C", loc="left", fontweight="bold")
    axC.grid(True, alpha=0.3)
    axC.text(best_k + 0.05, max(sil_scores), f"Optimal k={best_k}", va="top")

    # D: feature heatmap (z-scores)
    im = axD.imshow(cluster_means, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axD.set_title("D", loc="left", fontweight="bold")
    axD.set_yticks(range(len(available_cols)))
    axD.set_yticklabels(available_cols)
    axD.set_xticks(range(best_k))
    axD.set_xticklabels([f"Cluster {i}" for i in range(best_k)])

    # annotate values on heatmap
    for i in range(cluster_means.shape[0]):
        for j in range(cluster_means.shape[1]):
            axD.text(
                j, i, f"{cluster_means[i, j]:.2f}", ha="center", va="center", fontsize=8
            )

    cbar = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
    cbar.set_label("Z-score")

    # overall title
    fig.suptitle(
        "Patient stratification using pharmacist intervention features (KMeans + PCA)",
        y=0.98,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    fig.savefig(args.out_png, bbox_inches="tight")
    print(f"[OK] Saved figure: {args.out_png}")
    print(f"[OK] Best k = {best_k}, silhouette = {sil_scores[best_idx]:.4f}")

    # Optionally save CSV with cluster label
    if args.out_csv:
        out_df = df.copy()
        out_df["cluster_kmeans_bestk"] = best_labels
        out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved CSV with clusters: {args.out_csv}")


if __name__ == "__main__":
    main()
