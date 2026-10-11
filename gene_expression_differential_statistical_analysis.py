#!/usr/bin/env python3
"""Gene Expression Data Explorer: differential expression via Welch's t-test.

Based on the statistical-analysis implementation in the accompanying project report.
Input: gene-by-sample CSV (first column: Gene), with nonnegative, linear-scale
normalized expression values. Do NOT use log2-transformed values to compute the
ratio-based log2 fold changes in this script.

Example:
  python gene_expression_differential_statistical_analysis.py \
      --input preprocessed_gene_expression.csv \
      --controls Control_1 Control_2 Control_3 \
      --treatments Treatment_1 Treatment_2 Treatment_3

This is a simple two-group analysis of continuous expression data. For raw RNA-seq
counts, use dedicated count models such as DESeq2 or edgeR instead.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def differential_expression_analysis(
    expression_df: pd.DataFrame,
    control_columns: list[str],
    treatment_columns: list[str],
    gene_column: str = "Gene",
    alpha: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> pd.DataFrame:
    """Return per-gene group means, Welch statistics, FDR and classifications.

    Requires at least two numeric observations per group for each analyzed gene.
    Missing/non-numeric observations are dropped; unusable genes are skipped.
    Uninformative/undefined p-values are replaced with 1 for FDR adjustment.
    """
    required = [gene_column, *control_columns, *treatment_columns]
    missing = [column for column in required if column not in expression_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(control_columns) < 2 or len(treatment_columns) < 2:
        raise ValueError("At least two sample columns are required per group.")
    if set(control_columns) & set(treatment_columns):
        raise ValueError("Control and treatment columns must not overlap.")
    if not (0 < alpha < 1) or log2fc_threshold < 0:
        raise ValueError("Require 0 < alpha < 1 and log2fc_threshold >= 0.")

    results = []
    for _, row in expression_df.iterrows():
        control = pd.to_numeric(row[control_columns], errors="coerce").dropna().to_numpy(dtype=float)
        treatment = pd.to_numeric(row[treatment_columns], errors="coerce").dropna().to_numpy(dtype=float)
        control = control[np.isfinite(control)]
        treatment = treatment[np.isfinite(treatment)]
        if len(control) < 2 or len(treatment) < 2:
            continue
        if (control < 0).any() or (treatment < 0).any():
            raise ValueError(f"Negative expression value found in gene {row[gene_column]!r}.")

        control_mean = float(np.mean(control))
        treatment_mean = float(np.mean(treatment))
        # The small pseudocount follows the implementation documented in the report.
        pseudocount = 1e-9
        log2fc = float(np.log2((treatment_mean + pseudocount) / (control_mean + pseudocount)))
        # Constant identical groups have no evidence of a difference; avoid SciPy warnings.
        if np.all(control == control[0]) and np.all(treatment == treatment[0]) and control[0] == treatment[0]:
            t_stat, p_value = 0.0, 1.0
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                t_stat, p_value = ttest_ind(treatment, control, equal_var=False, nan_policy="omit")
            t_stat, p_value = float(t_stat), float(p_value)
        results.append({
            "Gene": row[gene_column],
            "Control_Mean": control_mean,
            "Treatment_Mean": treatment_mean,
            "log2FC": log2fc,
            "T_statistic": t_stat,
            "P_value": p_value,
        })

    results_df = pd.DataFrame(results, columns=[
        "Gene", "Control_Mean", "Treatment_Mean", "log2FC", "T_statistic", "P_value"
    ])
    if results_df.empty:
        return results_df.assign(Adjusted_P_value=pd.Series(dtype=float), Regulation=pd.Series(dtype=str))

    valid_p = results_df["P_value"].where(
        np.isfinite(results_df["P_value"]) & results_df["P_value"].between(0, 1),
        1.0,
    )
    results_df["Adjusted_P_value"] = multipletests(valid_p.to_numpy(), alpha=alpha, method="fdr_bh")[1]
    conditions = [
        (results_df["Adjusted_P_value"] < alpha) & (results_df["log2FC"] >= log2fc_threshold),
        (results_df["Adjusted_P_value"] < alpha) & (results_df["log2FC"] <= -log2fc_threshold),
    ]
    results_df["Regulation"] = np.select(
        conditions, ["Upregulated", "Downregulated"], default="Not Significant"
    )
    return results_df.sort_values("Adjusted_P_value").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="preprocessed_gene_expression.csv", help="Gene-by-sample CSV")
    parser.add_argument("--output", default="differential_expression_results.csv", help="Output CSV")
    parser.add_argument("--controls", nargs="+", default=["Control_1", "Control_2", "Control_3"])
    parser.add_argument("--treatments", nargs="+", default=["Treatment_1", "Treatment_2", "Treatment_3"])
    parser.add_argument("--gene-column", default="Gene")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--log2fc-threshold", type=float, default=1.0)
    args = parser.parse_args()

    expression_data = pd.read_csv(args.input)
    dge_results = differential_expression_analysis(
        expression_data, args.controls, args.treatments,
        gene_column=args.gene_column, alpha=args.alpha,
        log2fc_threshold=args.log2fc_threshold,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dge_results.to_csv(args.output, index=False)
    print(dge_results.head(10).to_string(index=False))
    print("Total genes analyzed:", len(dge_results))
    for category in ["Upregulated", "Downregulated"]:
        print(category + " genes:", int((dge_results["Regulation"] == category).sum()))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
