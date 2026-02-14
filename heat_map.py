# -*- coding: utf-8 -*-
"""
1) Correlation bar plot: each feature vs response ("改善")
2) Correlation matrix heatmap: features vs features (exclude response)

- English-only labels via FEATURE_NAME_MAP (unmapped columns are dropped from plots)
- Supports Pearson / Spearman correlations
- Outputs:
    --out_bar  : bar plot PNG
    --out_heat : heatmap PNG

Run:
  python corr_bar_and_heatmap.py --input ./相关性.xlsx --sheet Sheet1 --response_col 改善
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# --- Map ordinal text to numeric (adjust if needed) ---
ORDINAL_MAP = {"低": 0.0, "中": 0.5, "高": 1.0, "差": 0.0, "无": np.nan}

# --- Chinese -> English short names (edit to match your headers) ---
FEATURE_NAME_MAP = {
    "用药合理性评分（低中高）": "Rationality",
    "患者依从性评分（低中高）": "Adherence",
    "治疗方案优化干预数（新增、停用或剂量调整）": "Regimen changes",
    "GDMT（新四联）使用数量": "GDMT count",
    "药物相互作用或不良反应干预数": "DDI/ADR actions",
    "用药教育药物数量": "Education meds",
    "医生接受建议并执行数": "Accepted actions",
    # If you also want these included in the heatmap/bar plot, uncomment:
    # "临床指标基线eGFR(ml/min/1.73m2)": "eGFR baseline",
    # "临床指标随访6个月eGFR": "eGFR 6m",
}
RESPONSE_NAME_EN = "Improvement"


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a column to numeric; map ordinal Chinese labels; coerce others."""
    if s.dtype == "O":
        s2 = s.astype(str).str.strip()
        if s2.isin(list(ORDINAL_MAP.keys())).mean() > 0.2:
            return s2.map(ORDINAL_MAP).astype(float)
        return pd.to_numeric(s2, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def stars_from_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def corr_and_p(x: np.ndarray, y: np.ndarray, method: str):
    """Return (corr, pvalue)."""
    if method == "spearman":
        r, p = spearmanr(x, y, nan_policy="omit")
    else:
        r, p = pearsonr(x, y)
    return float(r), float(p)


def compute_feature_response_table(
    df: pd.DataFrame, feature_cols, y: pd.Series, method: str
):
    rows = []
    for col in feature_cols:
        x = to_numeric_series(df[col])
        mask = np.isfinite(x.values) & np.isfinite(y.values)
        if mask.sum() < 5:
            rows.append((col, np.nan, np.nan))
            continue
        xv = x.values[mask]
        yv = y.values[mask]
        if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            rows.append((col, np.nan, np.nan))
            continue
        r, p = corr_and_p(xv, yv, method)
        rows.append((col, r, p))
    return pd.DataFrame(rows, columns=["feature", "corr", "p"]).dropna(subset=["corr"])


def plot_bar(res: pd.DataFrame, method: str, out_png: str, title: str):
    # sort like paper figures
    res = res.sort_values("corr", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=500)
    idx = np.arange(len(res))
    ax.bar(idx, res["corr"].values, alpha=0.9)
    ax.axhline(0, linewidth=1)

    ylab = "Correlation (ρ)" if method == "spearman" else "Correlation (r)"
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12)

    ax.set_xticks(idx)
    ax.set_xticklabels(res["feature_label"].tolist(), rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=9)

    # stars
    y_vals = res["corr"].values
    scale = np.nanmax(np.abs(y_vals)) if len(y_vals) else 1.0
    offset = 0.03 * (scale if np.isfinite(scale) and scale > 0 else 1.0)

    for i, (r, p) in enumerate(zip(res["corr"].values, res["p"].values)):
        st = stars_from_p(p)
        if not st:
            continue
        if r >= 0:
            ax.text(i, r + offset, st, ha="center", va="bottom", fontsize=12)
        else:
            ax.text(i, r - offset, st, ha="center", va="top", fontsize=12)

    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(corr_mat: pd.DataFrame, out_png: str, title: str):
    labels = corr_mat.columns.tolist()
    M = corr_mat.values.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=500)
    im = ax.imshow(M, vmin=-1, vmax=1, aspect="equal", cmap="RdBu_r")
    # --- write correlation values into each cell ---
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    ax.tick_params(axis="both", labelsize=9)

    # minor grid to mimic neat matrix look
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./相关性.xlsx")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--response_col", default="改善")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--out_bar", default="./corr_bar.png")
    parser.add_argument("--out_heat", default="./corr_heatmap.png")
    args = parser.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)

    if args.response_col not in df.columns:
        raise ValueError(f"Response column '{args.response_col}' not found.")

    # Use ONLY mapped feature columns (ensures English-only labels)
    feature_cols = [c for c in FEATURE_NAME_MAP.keys() if c in df.columns]
    if len(feature_cols) < 2:
        raise ValueError(
            "Too few mapped features found in the sheet. "
            "Check FEATURE_NAME_MAP keys vs your Excel headers."
        )

    # Response
    y = to_numeric_series(df[args.response_col])

    # --- 1) Bar plot: feature vs response ---
    res = compute_feature_response_table(df, feature_cols, y, args.method)
    res["feature_label"] = res["feature"].map(FEATURE_NAME_MAP)

    # Drop anything that failed mapping or correlation
    res = res.dropna(subset=["feature_label"]).copy()
    if res.empty:
        raise ValueError("No valid features left for bar plot after cleaning.")

    plot_bar(
        res=res,
        method=args.method,
        out_png=args.out_bar,
        title="Correlates of improvement",
    )
    print(f"[OK] Saved bar plot: {args.out_bar}")

    # --- 2) Heatmap: feature-feature correlation matrix (exclude response) ---
    X = pd.DataFrame(
        {FEATURE_NAME_MAP[c]: to_numeric_series(df[c]) for c in feature_cols}
    )
    corr_mat = X.corr(method=args.method)

    plot_heatmap(
        corr_mat=corr_mat,
        out_png=args.out_heat,
        title="Feature correlation matrix",
    )
    print(f"[OK] Saved heatmap: {args.out_heat}")

    # Optional console output
    print("\nFeature vs response correlations:")
    print(res[["feature_label", "corr", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
