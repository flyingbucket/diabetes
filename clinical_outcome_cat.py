import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 环境准备 ---
out_dir = "fisher_exact_results"
save_dir = f"./out_new/{out_dir}"
os.makedirs(save_dir, exist_ok=True)

# 设置绘图风格
sns.set_theme(style="whitegrid", font_scale=1.1)
# plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# 加载数据
df = pd.read_excel("预后.bak.xlsx", sheet_name="Sheet1")
df = df.rename(
    columns={
        "Change in eGFR, mL/min/1.73 m2": "eGFR_num",
        "Initial dialysis": "Initial_dialysis",
    }
)

# --- 2. 特征工程 ---
eGFR_threshold = -10
df["eGFR_status"] = np.where(df["eGFR_num"] < eGFR_threshold, "Decline", "Stable")
cluster_map = {0: "L-intensity", 1: "H-intensity"}
df["Cluster_Label"] = df["cluster_kmeans_bestk"].map(cluster_map)

y_cols = ["MACE", "Readmission", "Initial_dialysis"]
for col in y_cols:
    df[col] = (df[col] > 0).astype(int)

# --- 3. 执行 Fisher 精确检验并收集数据 ---
all_results = []
plot_list = []

for y_col in y_cols:
    print(f"\n正在分析结局: {y_col}")

    for group in ["L-intensity", "H-intensity"]:
        subset = df[df["Cluster_Label"] == group]
        table = pd.crosstab(subset["eGFR_status"], subset[y_col])

        # 补全 2x2，确保逻辑严密
        for status in ["Decline", "Stable"]:
            if status not in table.index:
                table.loc[status] = [0, 0]
        for outcome in [0, 1]:
            if outcome not in table.columns:
                table[outcome] = 0

        formatted_table = [
            [table.loc["Decline", 1], table.loc["Decline", 0]],
            [table.loc["Stable", 1], table.loc["Stable", 0]],
        ]

        odds_ratio, p_value = fisher_exact(formatted_table)

        # 存入绘图列表
        for status in ["Decline", "Stable"]:
            event = table.loc[status, 1]
            total = table.loc[status].sum()
            rate = (event / total * 100) if total > 0 else 0
            plot_list.append(
                {
                    "Outcome": y_col,
                    "Group": group,
                    "Status": status,
                    "Rate": rate,
                    "Label": f"{int(event)}/{int(total)}",
                    "P_value": p_value,
                }
            )

        all_results.append(
            {"Outcome": y_col, "Group": group, "P_value": p_value, "OR": odds_ratio}
        )


# --- 3. 六组 Fisher 精确检验 ---
all_results = []

egfr_groups = ["Decline", "Stable"]
intensity_groups = ["L-intensity", "H-intensity"]

for status in egfr_groups:
    print(f"\\n{'=' * 60}")
    print(f"eGFR status: {status} (threshold = {eGFR_threshold})")
    print(f"{'=' * 60}")

    # 先按 eGFR 状态分层，再比较两种用药强度
    subset = df[
        (df["eGFR_status"] == status) & (df["Cluster_Label"].isin(intensity_groups))
    ].copy()

    for y_col in y_cols:
        # 行为用药强度，列为结局：0=未发生，1=发生
        table = pd.crosstab(
            subset["Cluster_Label"],
            subset[y_col],
        )

        # 确保表格始终为 2×2，行列顺序固定
        table = table.reindex(
            index=intensity_groups,
            columns=[0, 1],
            fill_value=0,
        )

        l_no = int(table.loc["L-intensity", 0])
        l_yes = int(table.loc["L-intensity", 1])
        h_no = int(table.loc["H-intensity", 0])
        h_yes = int(table.loc["H-intensity", 1])

        l_total = l_no + l_yes
        h_total = h_no + h_yes

        l_rate = l_yes / l_total * 100 if l_total > 0 else np.nan
        h_rate = h_yes / h_total * 100 if h_total > 0 else np.nan
        risk_diff = (
            h_rate - l_rate
            if (not np.isnan(h_rate) and not np.isnan(l_rate))
            else np.nan
        )

        # Fisher 精确检验：H-intensity vs L-intensity
        # OR > 1 表示 H-intensity 组的结局发生优势较高
        contingency_table = [
            [h_yes, h_no],
            [l_yes, l_no],
        ]

        if l_total == 0 or h_total == 0:
            odds_ratio = np.nan
            p_value = np.nan
        else:
            odds_ratio, p_value = fisher_exact(
                contingency_table,
                alternative="two-sided",
            )

        result = {
            "eGFR_status": status,
            "eGFR_threshold": eGFR_threshold,
            "Outcome": y_col,
            "L_events": l_yes,
            "L_total": l_total,
            "L_rate_percent": l_rate,
            "H_events": h_yes,
            "H_total": h_total,
            "H_rate_percent": h_rate,
            "Risk_difference_H_minus_L_percentage_points": risk_diff,
            "OR_H_vs_L": odds_ratio,
            "P_value": p_value,
        }
        all_results.append(result)

        print(f"\\nOutcome: {y_col}")
        print(
            f"  L-intensity: {l_yes}/{l_total} ({l_rate:.2f}%)"
            if l_total > 0
            else "  L-intensity: 0 patients"
        )
        print(
            f"  H-intensity: {h_yes}/{h_total} ({h_rate:.2f}%)"
            if h_total > 0
            else "  H-intensity: 0 patients"
        )
        print(
            f"  OR (H vs L): {odds_ratio:.4f}"
            if not np.isnan(odds_ratio)
            else "  OR (H vs L): NA"
        )
        print(
            f"  Fisher P-value: {p_value:.4f}"
            if not np.isnan(p_value)
            else "  Fisher P-value: NA"
        )

# --- 4. 保存结果 ---
results_df = pd.DataFrame(all_results)

csv_path = os.path.join(
    save_dir,
    "fisher_between_group_comparison_stratified_by_egfr.csv",
)
results_df.to_csv(csv_path, index=False)

print(f"\\n六组检验完成，结果已保存至：{csv_path}")
print(results_df.to_string(index=False))

# --- 4. Visualization ---
df_plot = pd.DataFrame(plot_list)

# Explicitly use modern FacetGrid/figure API
g = sns.catplot(
    data=df_plot,
    kind="bar",
    x="Status",
    y="Rate",
    hue="Group",
    col="Outcome",
    col_order=y_cols,
    palette="muted",
    alpha=0.8,
    height=5,
    aspect=0.8,
)

# Robust labeling logic using modern bar_label API
for ax in g.axes.flat:
    # Extract outcome name from the facet title
    title_text = ax.get_title()
    outcome_name = title_text.split("=")[-1].strip()
    curr_df = df_plot[df_plot["Outcome"] == outcome_name]

    # Annotate n/N on top of bars
    for container in ax.containers:
        group_label = container.get_label()
        labels = []

        # Match each bar's position with the corresponding label in dataframe
        for i in range(len(container)):
            status_val = ax.get_xticklabels()[i].get_text()
            match = curr_df[
                (curr_df["Group"] == group_label) & (curr_df["Status"] == status_val)
            ]
            if not match.empty:
                # Use .iloc[0] on the matched DataFrame safely
                labels.append(str(match["Label"].iloc[0]))
            else:
                labels.append("")

        ax.bar_label(container, labels=labels, padding=3, fontsize=9, fontweight="bold")

    # Annotate P-values in the subtitle
    try:
        # Accessing underlying values to avoid Pyright attribute issues
        p_l = curr_df.loc[curr_df["Group"] == "L-intensity", "P_value"].values[0]
        p_h = curr_df.loc[curr_df["Group"] == "H-intensity", "P_value"].values[0]
        ax.set_title(f"{outcome_name}\nL-P={p_l:.3f} | H-P={p_h:.3f}", fontsize=11)
    except (IndexError, KeyError):
        # Fallback if specific group data is missing
        ax.set_title(f"{outcome_name}")

    ax.set_ylabel("Incidence Rate (%)")
    ax.set_xlabel("eGFR Status")
    ax.set_ylim(0, 115)

# Replace deprecated .fig with .figure
g.figure.subplots_adjust(top=0.8)
g.figure.suptitle(
    f"Outcome Risk by eGFR Status (Threshold: {eGFR_threshold} mL/min)", fontsize=14
)

# Export as PDF for high-quality publication
pdf_path = os.path.join(save_dir, "fisher_outcomes_comparison.pdf")
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Visualization complete! PDF saved to: {pdf_path}")
# --- 5. 保存结果 ---
pd.DataFrame(all_results).to_csv(f"{save_dir}/fisher_analysis_summary.csv", index=False)
