import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("干预措施.xlsx", sheet_name="Sheet1").dropna().copy()
cluster_map = {0: "H-intensity", 1: "L-intensity"}
df = df.rename(columns={"cluster_kmeans_bestk": "Cluster"})
df["Cluster"] = df["Cluster"].map(cluster_map)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# --- 第一张图：Rationality 各个档位的数量 ---
sns.countplot(
    data=df,
    x="Rationality",
    hue="Cluster",
    order=["low", "mid", "high"],  # 确保顺序一致
    ax=axes[0],
)
axes[0].set_title("Distribution of Rationality Levels by Cluster", fontsize=14)
axes[0].set_ylabel("Count")

# --- 第二张图：后续数值型指标的均值 ---
value_cols = [
    "Regimen optimization changes",
    "GDMT counts",
    "DDI/ADR actions",
    "Medication education counts",
    "Accepted actions",
]

df_melted = df.melt(
    id_vars=["Cluster"],
    value_vars=value_cols,
    var_name="Intervention_Metric",
    value_name="Mean_Value",
)

sns.barplot(
    data=df_melted,
    x="Intervention_Metric",
    y="Mean_Value",
    hue="Cluster",
    ax=axes[1],
    errorbar=None,
)
axes[1].set_title("Mean Value of Intervention Metrics by Cluster", fontsize=14)
axes[1].set_ylabel("Average Count")
# plt.xticks(rotation=15)  # 旋转标签防止重叠

plt.tight_layout()
plt.show()
fig.savefig("./out_new/inter.png", dpi=500)
