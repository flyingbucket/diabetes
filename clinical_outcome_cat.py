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
