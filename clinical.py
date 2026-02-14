# -*- coding: utf-8 -*-
"""
1) Cluster using pharmacist intervention features (KMeans, best-k by silhouette).
2) For each clinical metric (exclude BP), plot grouped bars for Baseline/3m/6m
   with two sub-bars by cluster, and overlay a line connecting bar tops per cluster.

Run:
  python plot_clinical_by_cluster.py \
    --input "/mnt/data/糖肾MDT数据.xlsx" \
    --sheet "均一化" \
    --kmin 2 --kmax 6 \
    --out_dir "/mnt/data/clinical_barplots"
"""

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


# -----------------------------
# Column maps
# -----------------------------
# Intervention columns (for clustering)
INTERVENTION_COL_MAP = {
    "用药合理性评分（低中高）": "med_rationality_score",
    "患者依从性评分（低中高）": "adherence_score",
    "治疗方案优化干预数（新增、停用或剂量调整）": "regimen_opt_interventions",
    "GDMT（新四联）使用数量": "gdmt_count",
    "药物相互作用或不良反应干预数": "ddi_adr_interventions",
    "用药教育药物数量": "med_education_count",
    "医生接受建议并执行数": "physician_acceptance_count",
}
INTERVENTION_ENGLISH = list(INTERVENTION_COL_MAP.values())

# Clinical columns (for plotting)
CLINICAL_COL_MAP = {
    "eGFR(ml/min/1.73m2)": "egfr_baseline",
    "UACR(g/g)": "uacr_baseline",
    "24hUTP(g)": "utp24h_baseline",
    "HbA1c(%)": "hba1c_baseline",
    "LDL-C(mmol/L)": "ldlc_baseline",
    "BP(mmHg)": "bp_baseline",
    "eGFR": "egfr_3m",
    "UACR": "uacr_3m",
    "24hUTP": "utp24h_3m",
    "HbA1c": "hba1c_3m",
    "LDL-C": "ldlc_3m",
    "BP": "bp_3m",
    "eGFR.1": "egfr_6m",
    "UACR.1": "uacr_6m",
    "24hUTP.1": "utp24h_6m",
    "HbA1c.1": "hba1c_6m",
    "LDL-C.1": "ldlc_6m",
    "BP.1": "bp_6m",
}

# Plot-friendly labels
METRIC_LABELS = {
    "egfr": "eGFR (ml/min/1.73m²)",
    "uacr": "UACR (g/g)",
    "utp24h": "24h UTP (g)",
    "hba1c": "HbA1c (%)",
    "ldlc": "LDL-C (mmol/L)",
}


def _kmeans_fit_predict(X, k, random_state=42):
    """Compatibility wrapper for different sklearn versions."""
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def pick_best_k_by_silhouette(X, kmin=2, kmax=6, random_state=42):
    k_values = list(range(kmin, kmax + 1))
    scores = []
    models = {}
    for k in k_values:
        labels, km = _kmeans_fit_predict(X, k, random_state=random_state)
        score = silhouette_score(X, labels, metric="euclidean")
        scores.append(score)
        models[k] = (labels, km)
    best_idx = int(np.argmax(scores))
    best_k = k_values[best_idx]
    return best_k, scores, k_values, models[best_k][0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_metric_grouped_bars_with_lines(
    df: pd.DataFrame,
    cluster_col: str,
    metric_base: str,
    out_path: str,
    title_prefix: str = "",
    cluster_order=None,
):
    """
    metric_base: e.g. 'egfr' -> uses egfr_baseline, egfr_3m, egfr_6m
    Produces grouped bars with sub-bars by cluster, plus per-cluster line connecting bar tops.
    """
    timepoints = [("baseline", "Baseline"), ("3m", "3 months"), ("6m", "6 months")]
    cols = [f"{metric_base}_baseline", f"{metric_base}_3m", f"{metric_base}_6m"]

    # Only plot if all columns exist
    if not all(c in df.columns for c in cols):
        return False

    # Determine cluster order
    if cluster_order is None:
        cluster_order = sorted([c for c in df[cluster_col].dropna().unique().tolist()])

    # Compute means (you可以改成 median，如果更稳健)
    means = np.zeros((len(cluster_order), len(cols)), dtype=float)
    for i, cl in enumerate(cluster_order):
        sub = df[df[cluster_col] == cl]
        for j, col in enumerate(cols):
            means[i, j] = pd.to_numeric(sub[col], errors="coerce").mean()

    # Style: match your example (yellow/purple for 2 clusters)
    # If clusters != 2, fall back to tab10.
    if len(cluster_order) == 2:
        colors = ["#F1C40F", "#3B0F70"]  # yellow, purple-ish
    else:
        cmap = plt.cm.get_cmap("tab10", len(cluster_order))
        colors = [cmap(i) for i in range(len(cluster_order))]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=500)

    x = np.arange(len(timepoints))  # group centers
    total_width = 0.78
    bar_width = total_width / max(len(cluster_order), 1)

    # bar offsets so sub-bars are centered within each group
    offsets = (
        np.arange(len(cluster_order)) - (len(cluster_order) - 1) / 2.0
    ) * bar_width

    # Bars + per-cluster line (connect bar centers)
    for i, cl in enumerate(cluster_order):
        x_sub = x + offsets[i]
        ax.bar(
            x_sub,
            means[i, :],
            width=bar_width * 0.95,
            label=f"Cluster {cl}",
            color=colors[i],
            alpha=0.9,
        )
        # line on top of bars for this cluster
        ax.plot(
            x_sub,
            means[i, :],
            marker="o",
            linewidth=2,
            color=colors[i],
            alpha=0.95,
        )

    # Labels & grid
    ax.set_xticks(x)
    ax.set_xticklabels([tp[1] for tp in timepoints])
    ylabel = METRIC_LABELS.get(metric_base, metric_base)
    ax.set_ylabel(ylabel)

    title = f"{title_prefix}{ylabel} by cluster over time".strip()
    ax.set_title(title)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2 if len(cluster_order) == 2 else 1, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Excel path")
    parser.add_argument(
        "--sheet", default="均一化", help="Sheet name (default: 均一化)"
    )
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=6)
    parser.add_argument("--out_dir", required=True, help="Directory to save plots")
    parser.add_argument(
        "--save_csv", default="", help="Optional: save CSV with cluster labels"
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Read: in your file, column names are on row 2
    df = (
        pd.read_excel(args.input, sheet_name=args.sheet, header=1)
        .dropna(how="all")
        .copy()
    )

    # Rename intervention + clinical columns (only what exists)
    rename_map = {}
    rename_map.update(
        {k: v for k, v in INTERVENTION_COL_MAP.items() if k in df.columns}
    )
    rename_map.update({k: v for k, v in CLINICAL_COL_MAP.items() if k in df.columns})
    df = df.rename(columns=rename_map)

    # Build intervention feature matrix
    available_interv = [c for c in INTERVENTION_ENGLISH if c in df.columns]
    if len(available_interv) < 2:
        raise ValueError(f"Not enough intervention columns found: {available_interv}")

    X_raw = df[available_interv].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp)

    # Best k by silhouette
    best_k, sil_scores, k_values, labels = pick_best_k_by_silhouette(
        X, args.kmin, args.kmax
    )
    df["cluster_kmeans_bestk"] = labels

    print(f"[OK] Best k = {best_k}")
    for k, s in zip(k_values, sil_scores):
        print(f"  k={k}: silhouette={s:.4f}")

    # Optional save CSV with labels
    if args.save_csv:
        df.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved labeled CSV: {args.save_csv}")

    # Plot clinical metrics (exclude BP)
    metric_bases = ["egfr", "uacr", "utp24h", "hba1c", "ldlc"]
    made_any = False
    for m in metric_bases:
        out_path = os.path.join(args.out_dir, f"{m}_barplot.png")
        ok = plot_metric_grouped_bars_with_lines(
            df=df,
            cluster_col="cluster_kmeans_bestk",
            metric_base=m,
            out_path=out_path,
            title_prefix="",
            cluster_order=sorted(df["cluster_kmeans_bestk"].unique().tolist()),
        )
        if ok:
            made_any = True
            print(f"[OK] Saved plot: {out_path}")
        else:
            print(
                f"[SKIP] Missing columns for metric '{m}' (need {m}_baseline/{m}_3m/{m}_6m)."
            )

    if not made_any:
        print(
            "[WARN] No metric plots were generated (likely missing follow-up columns)."
        )


if __name__ == "__main__":
    main()
