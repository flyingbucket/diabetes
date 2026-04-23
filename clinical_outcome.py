import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from patsy import cr  # 用于生成样条基函数
import os

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
out_dir = "outcome_rcs_percentage_pdf"
save_dir = f"./out_new/{out_dir}"
os.makedirs(save_dir, exist_ok=True)

# 加载数据
df0 = pd.read_excel("预后.bak.xlsx", sheet_name="Sheet1")
df = df0.rename(
    columns={
        "Change in eGFR, mL/min/1.73 m2": "eGFR_num",
        "Initial dialysis": "Initial_dialysis",
    }
)
df = df0.rename(
    columns={
        "Change in eGFR, %": "eGFR_percentage",
        "Initial dialysis": "Initial_dialysis",
    }
)

cluster_map = {0: "L-intensity", 1: "H-intensity"}
df["Cluster_Label"] = df["cluster_kmeans_bestk"].map(cluster_map)

# x_col = "eGFR_num"
x_col = "eGFR_percentage"
y_cols = ["MACE", "Readmission", "Initial_dialysis"]
cluster_col = "cluster_kmeans_bestk"

for col in y_cols:
    if col in df.columns:
        df[col] = (df[col] > 0).astype(int)

palette_colors = {0: "#56B4E9", 1: "#E69F00"}
cluster_labels = {0: "L-intensity", 1: "H-intensity"}

sns.set_theme(style="whitegrid")

for y_col in y_cols:
    plt.figure(figsize=(9, 6))

    # 构建 RCS 模型
    # cr(x, df=4) 表示使用 4 个结点的限制性立方样条
    # 交互项写为 cr(...) : cluster_col
    formula = f"{y_col} ~ cr({x_col}, df=3) * {cluster_col}"

    try:
        model = smf.logit(formula, data=df).fit(disp=0)

        # 提取交互项的联合 P 值
        p_vals = model.pvalues
        inter_p = p_vals[p_vals.index.str.contains(":")].min()
        p_str = (
            f"P-interaction = {inter_p:.3f}"
            if inter_p >= 0.001
            else "P-interaction < 0.001"
        )
    except Exception as e:
        print(f"Error fitting {y_col}: {e}")
        continue

    # 生成预测数据 (用于绘制平滑曲线)
    # 创建一个精细的 X 轴序列
    x_smooth = np.linspace(df[x_col].min(), df[x_col].max(), 300)

    for val in [0, 1]:
        # 构造预测用的 DataFrame
        pred_df = pd.DataFrame({x_col: x_smooth, cluster_col: val})

        # 获取预测结果（含置信区间）
        # get_prediction 返回的是在概率尺度上的均值和区间
        prediction = model.get_prediction(pred_df).summary_frame()
        # breakpoint()
        # 提取 p_i 和 CI
        y_prob = prediction["predicted"]
        y_low = prediction["ci_lower"]
        y_high = prediction["ci_upper"]

        # 绘图
        label = cluster_labels[val]
        color = palette_colors[val]

        plt.plot(x_smooth, y_prob, label=label, color=color, linewidth=2.5)
        plt.fill_between(x_smooth, y_low, y_high, color=color, alpha=0.15)

    # 美化与保存
    plt.title(f"RCS Predicted Risk of {y_col}\n({p_str})", fontsize=14, pad=15)
    # plt.xlabel("Change in eGFR (mL/min/1.73 m²)", fontsize=12)
    plt.xlabel("Change in eGFR, %", fontsize=12)
    plt.ylabel("Predicted Probability of Event ($p_i$)", fontsize=12)
    plt.ylim(0, 1)  # 概率范围 0-1
    plt.legend(title="Groups", frameon=True)

    # 导出
    plt.tight_layout()
    # plt.savefig(f"{save_dir}/rcs_plot_{y_col.lower()}.png", dpi=500)
    plt.savefig(f"{save_dir}/rcs_plot_{y_col.lower()}.pdf")
    plt.show()
