# -*- coding: utf-8 -*-
"""
Cluster pharmacist intervention features -> KMeans(best k by silhouette)
Then plot clinical metrics as grouped boxplots (Baseline/3m/6m) split by cluster,
overlay mean-lines, and add p-value for 6m cluster difference.
Includes BP as two metrics: SBP and DBP (two separate figures).

Run:
  python plot_clinical_boxplot_by_cluster.py \
    --input "/mnt/data/糖肾MDT数据.xlsx" \
    --sheet "均一化" \
    --kmin 2 --kmax 6 \
    --out_dir "/mnt/data/clinical_boxplots" \
    --save_csv "/mnt/data/with_cluster_labels.csv" \
    --ptest mw
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

from scipy.stats import mannwhitneyu, ttest_ind


# -----------------------------
# Column maps
# -----------------------------
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

METRIC_LABELS = {
    "egfr": "eGFR (ml/min/1.73m²)",
    "uacr": "UACR (g/g)",
    "utp24h": "24h UTP (g)",
    "hba1c": "HbA1c (%)",
    "ldlc": "LDL-C (mmol/L)",
    "sbp": "SBP (mmHg)",  # 高压/收缩压
    "dbp": "DBP (mmHg)",  # 低压/舒张压
}


def _kmeans_fit_predict(X, k, random_state=42):
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def pick_best_k_by_silhouette(X, kmin=2, kmax=6, random_state=42):
    k_values = list(range(kmin, kmax + 1))
    scores = []
    best = None
    best_k = None
    best_labels = None
    for k in k_values:
        labels, _ = _kmeans_fit_predict(X, k, random_state=random_state)
        score = silhouette_score(X, labels, metric="euclidean")
        scores.append(score)
        if best is None or score > best:
            best = score
            best_k = k
            best_labels = labels
    return best_k, scores, k_values, best_labels


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def split_bp_series_to_sbp_dbp(bp: pd.Series):
    """
    Parse BP string like '130/90' -> SBP=130, DBP=90.
    Robust to spaces, missing, non-standard strings (returns NaN when cannot parse).
    """
    s = bp.astype(str).str.strip()
    parts = s.str.split("/", n=1, expand=True)
    if parts.shape[1] != 2:
        return pd.Series([np.nan] * len(bp), index=bp.index), pd.Series(
            [np.nan] * len(bp), index=bp.index
        )
    sbp = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    dbp = pd.to_numeric(parts[1].str.strip(), errors="coerce")
    return sbp, dbp


def test_pvalue_6m(df: pd.DataFrame, cluster_col: str, col_6m: str, method: str = "mw"):
    """
    p-value comparing cluster0 vs cluster1 at 6m.
    method:
      - "mw": Mann–Whitney U (non-parametric, default)
      - "ttest": Welch t-test
    """
    clusters = sorted(df[cluster_col].dropna().unique().tolist())
    if len(clusters) != 2:
        return np.nan

    a = _safe_numeric_series(df.loc[df[cluster_col] == clusters[0], col_6m]).dropna()
    b = _safe_numeric_series(df.loc[df[cluster_col] == clusters[1], col_6m]).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan

    if method == "ttest":
        _, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(p)

    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return np.nan
    return float(p)


def plot_metric_boxplots_with_mean_lines_and_p(
    df: pd.DataFrame,
    cluster_col: str,
    metric_base: str,
    out_path: str,
    p_test: str = "mw",
):
    time_order = [("baseline", "Baseline"), ("3m", "3 months"), ("6m", "6 months")]
    cols = [f"{metric_base}_baseline", f"{metric_base}_3m", f"{metric_base}_6m"]
    if not all(c in df.columns for c in cols):
        return False

    clusters = sorted(df[cluster_col].dropna().unique().tolist())
    if len(clusters) < 2:
        return False

    x = np.arange(len(time_order))
    total_width = 0.72
    box_width = total_width / len(clusters)
    offsets = (np.arange(len(clusters)) - (len(clusters) - 1) / 2.0) * box_width

    # colors: yellow/purple for 2 clusters
    if len(clusters) == 2:
        colors = ["#F1C40F", "#3B0F70"]
    else:
        cmap = plt.cm.get_cmap("tab10", len(clusters))
        colors = [cmap(i) for i in range(len(clusters))]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=500)

    mean_points = np.zeros((len(clusters), len(cols)), dtype=float)

    # Draw boxplots per cluster so we can control positions/colors
    for i, cl in enumerate(clusters):
        data = []
        for j, col in enumerate(cols):
            vals = _safe_numeric_series(df.loc[df[cluster_col] == cl, col]).dropna()
            data.append(vals.values)
            mean_points[i, j] = float(vals.mean()) if len(vals) else np.nan

        pos = x + offsets[i]
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=box_width * 0.90,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.55)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_alpha(0.9)

        # mean line at mean positions
        ax.plot(
            pos,
            mean_points[i, :],
            marker="o",
            linewidth=2,
            color=colors[i],
            alpha=0.95,
            label=f"Cluster {cl}",
        )

    # p-value at 6m (cluster 0 vs 1)
    p = test_pvalue_6m(df, cluster_col, f"{metric_base}_6m", method=p_test)
    p_text = "p = NA" if not np.isfinite(p) else f"6m difference p = {p:.3g}"

    ylabel = METRIC_LABELS.get(metric_base, metric_base)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by cluster over time\n({p_text})")

    ax.set_xticks(x)
    ax.set_xticklabels([t[1] for t in time_order])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=2 if len(clusters) == 2 else 1, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sheet", default="均一化")
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=6)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_csv", default="")
    parser.add_argument("--ptest", default="mw", choices=["mw", "ttest"])
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    # Read: in your file, column names are on row 2
    df = (
        pd.read_excel(args.input, sheet_name=args.sheet, header=1)
        .dropna(how="all")
        .copy()
    )

    # Rename columns
    rename_map = {}
    rename_map.update(
        {k: v for k, v in INTERVENTION_COL_MAP.items() if k in df.columns}
    )
    rename_map.update({k: v for k, v in CLINICAL_COL_MAP.items() if k in df.columns})
    df = df.rename(columns=rename_map)

    # ---- Split BP into SBP/DBP numeric columns (baseline/3m/6m) ----
    for tp in ["baseline", "3m", "6m"]:
        bp_col = f"bp_{tp}"
        if bp_col in df.columns:
            sbp, dbp = split_bp_series_to_sbp_dbp(df[bp_col])
            df[f"sbp_{tp}"] = sbp
            df[f"dbp_{tp}"] = dbp

    # ---- Clustering intervention features ----
    available_interv = [c for c in INTERVENTION_ENGLISH if c in df.columns]
    if len(available_interv) < 2:
        raise ValueError(f"Not enough intervention columns found: {available_interv}")

    X_raw = df[available_interv].copy()
    X_imp = SimpleImputer(strategy="median").fit_transform(X_raw)
    X = StandardScaler().fit_transform(X_imp)

    best_k, sil_scores, k_values, labels = pick_best_k_by_silhouette(
        X, args.kmin, args.kmax
    )
    df["cluster_kmeans_bestk"] = labels

    print(f"[OK] Best k = {best_k}")
    for k, s in zip(k_values, sil_scores):
        print(f"  k={k}: silhouette={s:.4f}")

    if args.save_csv:
        df.to_csv(args.save_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved labeled CSV: {args.save_csv}")

    # ---- Plot metrics (now includes SBP/DBP) ----
    metric_bases = ["egfr", "uacr", "utp24h", "hba1c", "ldlc", "sbp", "dbp"]

    made_any = False
    for m in metric_bases:
        out_path = os.path.join(args.out_dir, f"{m}_boxplot.png")
        ok = plot_metric_boxplots_with_mean_lines_and_p(
            df=df,
            cluster_col="cluster_kmeans_bestk",
            metric_base=m,
            out_path=out_path,
            p_test=args.ptest,
        )
        if ok:
            made_any = True
            print(f"[OK] Saved plot: {out_path}")
        else:
            print(
                f"[SKIP] Missing columns for '{m}' (need {m}_baseline/{m}_3m/{m}_6m)."
            )

    if not made_any:
        print("[WARN] No plots were generated.")


if __name__ == "__main__":
    main()
