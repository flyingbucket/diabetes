import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 数据预处理
df = pd.read_excel("干预措施.xlsx", sheet_name="Sheet1").dropna().copy()
cluster_map = {0: "H-intensity", 1: "L-intensity"}
df = df.rename(columns={"cluster_kmeans_bestk": "Cluster"})
df["Cluster"] = df["Cluster"].map(cluster_map)

# 统一首字母大写
df["Rationality"] = df["Rationality"].str.capitalize()
df["Adherence"] = df["Adherence"].str.capitalize()


sns.set_theme(style="whitegrid")

# --- 图 1：Rationality 分布图 ---
plt.figure(figsize=(8, 6))
sns.countplot(
    data=df,
    x="Rationality",
    hue="Cluster",
    hue_order=["L-intensity", "H-intensity"],
    order=["Low", "Mid", "High"],
    width=0.6,  # 收窄柱子
)
plt.title("Distribution of Rationality Levels by Cluster", fontsize=14)
plt.ylabel("Counts")
plt.xlabel("Rationality")
plt.tight_layout()
plt.savefig("./out_new/inter/rationality_dist.png", dpi=500)
plt.close()

# --- 图 2：Adherence 分布图 (新增) ---
plt.figure(figsize=(8, 6))
sns.countplot(
    data=df,
    x="Adherence",
    hue="Cluster",
    hue_order=["L-intensity", "H-intensity"],
    order=["Low", "Mid", "High"],
    width=0.6,
)
plt.title("Distribution of Adherence Levels by Cluster", fontsize=14)
plt.ylabel("Counts")
plt.xlabel("Adherence")
plt.tight_layout()
plt.savefig("./out_new/inter/adherence_dist.png", dpi=500)
plt.close()

# --- 图 3：Intervention 均值对比图 ---
# 修改列名为要求的名词
rename_dict = {
    "Regimen optimization changes": "Regimen optimization suggestions",
    "Medication education counts": "Medication education quantity",
}
df_renamed = df.rename(columns=rename_dict)

value_cols = [
    "Regimen optimization suggestions",
    "GDMT counts",
    "DDI/ADR actions",
    "Medication education quantity",
    "Accepted actions",
]

df_melted = df_renamed.melt(
    id_vars=["Cluster"],
    value_vars=value_cols,
    var_name="Intervention",
    value_name="Mean_Value",
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_melted,
    x="Intervention",
    y="Mean_Value",
    hue="Cluster",
    hue_order=["L-intensity", "H-intensity"],
    errorbar=None,
    width=0.7,  # 收窄柱子
)
plt.title("Mean Value of Interventions by Cluster", fontsize=14)
plt.ylabel("Average Counts")
plt.xticks(rotation=15)  # 建议保留轻微旋转，否则长标签会重叠
plt.tight_layout()
plt.savefig("./out_new/inter/intervention_means.png", dpi=500)
plt.close()
