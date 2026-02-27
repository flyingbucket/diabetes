# -*- coding: utf-8 -*-
"""
Correlation bar plot (English labels):
- Bar height = correlation with treatment response (Pearson r or Spearman rho)
- Asterisks = significance (* p<0.05, ** p<0.01, *** p<0.001)
- X-axis uses English short names via FEATURE_NAME_MAP (no Chinese shown)

Input: ./相关性.xlsx (Sheet1)
Default response column: 改善
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr


# --- Map ordinal text to numeric (adjust if needed) ---
ORDINAL_MAP = {"低": 0.0, "中": 0.5, "高": 1.0, "差": 0.0, "无": np.nan}

RESPONSE_NAME_EN = "Treatment response"

FEATURE_NAME_MAP = {
    "用药合理性评分（低中高）": "Rationality",
    "患者依从性评分（低中高）": "Adherence",
    "治疗方案优化干预数（新增、停用或剂量调整）": "Regimen changes",
    "GDMT（新四联）使用数量": "GDMT count",
    "药物相互作用或不良反应干预数": "DDI/ADR actions",
    "用药教育药物数量": "Education meds",
    "医生接受建议并执行数": "Accepted actions",
    # "临床指标基线eGFR(ml/min/1.73m2)": "eGFR baseline",
    # "临床指标随访6个月eGFR": "eGFR 6m",
}
RESPONSE_NAME_EN = "Improvement"


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a column to numeric; map known ordinal Chinese labels; coerce others."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./相关性.xlsx")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--response_col", default="改善")
    parser.add_argument("--binary_response", action="store_true")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="spearman")
    parser.add_argument("--out_png", default="./correlation_plot.png")
    args = parser.parse_args()

    df = pd.read_excel(args.input, sheet_name=args.sheet)

    if args.response_col not in df.columns:
        raise ValueError(
            f"Response column '{args.response_col}' not found. Columns: {list(df.columns)}"
        )

    # Build response
    y = to_numeric_series(df[args.response_col])
    if args.binary_response:
        med = np.nanmedian(y.values)
        y = (y > med).astype(float)  # 1 = better response
        y_name = f"{RESPONSE_NAME_EN} (binary)"
    else:
        y_name = RESPONSE_NAME_EN

    # Predictor columns: all except response
    pred_cols = [c for c in df.columns if c != args.response_col]

    results = []
    for col in pred_cols:
        x = to_numeric_series(df[col])
        mask = np.isfinite(x.values) & np.isfinite(y.values)

        if mask.sum() < 5:
            results.append((col, np.nan, np.nan))
            continue

        xv = x.values[mask]
        yv = y.values[mask]

        if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            results.append((col, np.nan, np.nan))
            continue

        r, p = corr_and_p(xv, yv, args.method)
        results.append((col, r, p))

    res = (
        pd.DataFrame(results, columns=["feature", "corr", "p"])
        .dropna(subset=["corr"])
        .copy()
    )

    # Map to English labels; DROP anything not mapped to avoid Chinese labels showing
    res["feature_label"] = res["feature"].map(FEATURE_NAME_MAP)
    res = res.dropna(subset=["feature_label"]).copy()

    if res.empty:
        raise ValueError(
            "No features left after applying FEATURE_NAME_MAP. "
            "Please verify your Excel headers match the keys in FEATURE_NAME_MAP."
        )

    res = res.sort_values("corr", ascending=False).reset_index(drop=True)

    # ---- Plot (English) ----
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=500)

    idx = np.arange(len(res))
    ax.bar(idx, res["corr"].values, alpha=0.9)
    ax.axhline(0, linewidth=1)

    # Simple English labels (compact)
    ylab = "Correlation (ρ)" if args.method == "spearman" else "Correlation (r)"
    ax.set_ylabel(ylab)
    ax.set_title("Correlates of treatment response", fontsize=12)

    ax.set_xticks(idx)
    ax.set_xticklabels(res["feature_label"].tolist(), rotation=35, ha="center")
    ax.tick_params(axis="x", labelsize=9)

    # Stars + numeric values
    # ---- 调整坐标轴范围 ----
    y_vals = res["corr"].values
    ymax = np.nanmax(y_vals)
    ymin = np.nanmin(y_vals)

    y_range = ymax - ymin
    if y_range == 0:
        y_range = 1.0

    margin = 0.20 * y_range
    ax.set_ylim(ymin - margin, ymax + margin)

    # ---- offset ----
    value_offset = 0.03 * y_range
    star_offset = 0.08 * y_range

    for i, (r, p) in enumerate(zip(res["corr"].values, res["p"].values)):
        # ---- 数值 ----
        value_text = f"{r:.2f}"  # 保留两位小数
        if r >= 0:
            ax.text(
                i, r + value_offset, value_text, ha="center", va="bottom", fontsize=9
            )
        else:
            ax.text(i, r - value_offset, value_text, ha="center", va="top", fontsize=9)

        # ---- 星号 ----
        st = stars_from_p(p)
        if st:
            if r >= 0:
                ax.text(i, r + star_offset, st, ha="center", va="bottom", fontsize=12)
            else:
                ax.text(i, r - star_offset, st, ha="center", va="top", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight")
    print(f"[OK] Saved: {args.out_png}")

    # Console table (English label included)
    print(res[["feature", "feature_label", "corr", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
