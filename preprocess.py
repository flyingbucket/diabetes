# -*- coding: utf-8 -*-
"""
Read normalized sheet from 糖肾MDT数据.xlsx, rename columns to English, and export to CSV.

Usage:
  python export_normalized_to_csv.py \
    --input "/mnt/data/糖肾MDT数据.xlsx" \
    --sheet "均一化" \
    --output "/mnt/data/dkd_mdt_normalized_renamed.csv"
"""

import argparse
import pandas as pd


RENAME_MAP = {
    # --- Basic info ---
    "病历号": "patient_id",
    "患者姓名": "patient_name",
    "性别（男0,女1）": "sex",  # 0=male, 1=female
    "年龄": "age",
    "糖尿病病史（年）": "diabetes_duration_years",
    "糖肾分期": "dkd_stage",
    "查尔森合并症指数（CCI）": "charlson_index",
    # --- Baseline clinical ---
    "eGFR(ml/min/1.73m2)": "egfr_baseline",
    "UACR(g/g)": "uacr_baseline",
    "24hUTP(g)": "utp24h_baseline",
    "HbA1c(%)": "hba1c_baseline",
    "LDL-C(mmol/L)": "ldlc_baseline",
    "BP(mmHg)": "bp_baseline",
    # --- Pharmacist intervention ---
    "用药合理性评分（低中高）": "med_rationality_score",
    "患者依从性评分（低中高）": "adherence_score",
    "治疗方案优化干预数（新增、停用或剂量调整）": "regimen_opt_interventions",
    "GDMT（新四联）使用数量": "gdmt_count",
    "药物相互作用或不良反应干预数": "ddi_adr_interventions",
    "用药教育药物数量": "med_education_count",
    "医生接受建议并执行数": "physician_acceptance_count",
    # --- 3-month follow-up clinical ---
    "eGFR": "egfr_3m",
    "UACR": "uacr_3m",
    "24hUTP": "utp24h_3m",
    "HbA1c": "hba1c_3m",
    "LDL-C": "ldlc_3m",
    "BP": "bp_3m",
    "GDMT是否中断治疗（是1否0）": "gdmt_discontinued_3m",
    # --- 6-month follow-up clinical (pandas auto adds .1 for duplicates) ---
    "eGFR.1": "egfr_6m",
    "UACR.1": "uacr_6m",
    "24hUTP.1": "utp24h_6m",
    "HbA1c.1": "hba1c_6m",
    "LDL-C.1": "ldlc_6m",
    "BP.1": "bp_6m",
    "GDMT是否中断治疗": "gdmt_discontinued_6m",
    # --- Outcomes ---
    "再入院次数": "readmission_count",
    "是否透析": "dialysis",
    "重大心血管不良事件（MACE）发生数": "mace_count",
}


def split_bp(series: pd.Series):
    """
    Split BP like '130/90' -> sbp=130, dbp=90. Returns two numeric Series.
    """
    s = series.astype(str).str.strip()
    parts = s.str.split("/", n=1, expand=True)
    if parts.shape[1] != 2:
        return pd.Series([pd.NA] * len(series)), pd.Series([pd.NA] * len(series))
    sbp = pd.to_numeric(parts[0], errors="coerce")
    dbp = pd.to_numeric(parts[1], errors="coerce")
    return sbp, dbp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the Excel file")
    parser.add_argument(
        "--sheet", default="均一化", help="Sheet name (default: 均一化)"
    )
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    # header=1: use the 2nd row as column names (the 1st row is section headers in this file)
    df = pd.read_excel(args.input, sheet_name=args.sheet, header=1)

    # drop completely empty rows
    df = df.dropna(how="all").copy()

    # rename columns (only those present)
    present_map = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=present_map)

    # optional: split BP columns into SBP/DBP
    for bp_col in ["bp_baseline", "bp_3m", "bp_6m"]:
        if bp_col in df.columns:
            sbp, dbp = split_bp(df[bp_col])
            df[f"sbp_{bp_col.replace('bp_', '')}"] = sbp
            df[f"dbp_{bp_col.replace('bp_', '')}"] = dbp

    # save (utf-8-sig is friendlier for Excel opening)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}, Cols: {df.shape[1]}")


if __name__ == "__main__":
    main()
