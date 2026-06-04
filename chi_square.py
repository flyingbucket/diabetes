import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("outcome_new.csv")

target_cols = ["Readmission", "Initial dialysis", "MACE"]

for col in target_cols:
    df[col] = df[col].clip(upper=1)

print("================ 数据处理验证 ================")
for col in target_cols:
    print(f"[{col}] 转换后的分布:\n{df[col].value_counts()}\n")


print("================ 卡方检验结果 ================\n")

for col in target_cols:
    contingency_table = pd.crosstab(df["cluster_kmeans_bestk"], df[col])

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"指标: {col}")
    print("列联表 (0=未发生, 1=发生):")
    print(contingency_table)
    print(f"-> 卡方统计量 (Chi2) : {chi2:.4f}")
    print(f"-> p-value : {p_value:.5f}")
