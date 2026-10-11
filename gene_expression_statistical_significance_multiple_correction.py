#!/usr/bin/env python3
"""Statistical significance and multiple-hypothesis correction.

Extracted from the Gene Expression Data Explorer report.
Requires differential_expression_results.csv with Gene, log2FC and p_value.
"""

# ==========================================================
# Gene Expression Data Explorer
# Task: Statistical Significance Testing and Multiple
#       Hypothesis Correction
# ==========================================================

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

# ----------------------------------------------------------
# 1. Load differential-expression results
# ----------------------------------------------------------
# Expected columns:
#   Gene      : gene identifier
#   log2FC    : log2 fold change
#   p_value   : raw p-value from the selected DE test
#
# Example file name:
#   differential_expression_results.csv

results = pd.read_csv("differential_expression_results.csv")

required_columns = {"Gene", "log2FC", "p_value"}
missing_columns = required_columns - set(results.columns)
if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

# Ensure numerical fields are numeric.
results["log2FC"] = pd.to_numeric(
    results["log2FC"], errors="coerce"
)
results["p_value"] = pd.to_numeric(
    results["p_value"], errors="coerce"
)

# Remove rows where the raw p-value or fold change is missing.
results = results.dropna(
    subset=["Gene", "log2FC", "p_value"]
).copy()

# Raw p-values must lie between 0 and 1.
if ((results["p_value"] < 0) | (results["p_value"] > 1)).any():
    raise ValueError("All p-values must be between 0 and 1.")

print("Number of genes tested:", len(results))

# ----------------------------------------------------------
# 2. Raw statistical-significance flag
# ----------------------------------------------------------
# The differential-expression method should already have
# produced one raw p-value per gene. Here we apply an
# uncorrected alpha = 0.05 flag before multiple testing.

alpha = 0.05
results["Significant_Raw_0.05"] = results["p_value"] < alpha

# ----------------------------------------------------------
# 3. Benjamini-Hochberg False Discovery Rate (FDR)
# ----------------------------------------------------------
# BH controls the expected proportion of false discoveries
# among rejected hypotheses and is commonly used for large
# gene-expression studies.

reject_fdr, p_fdr, _, _ = multipletests(
    results["p_value"].values,
    alpha=alpha,
    method="fdr_bh"
)

results["FDR_BH"] = p_fdr
results["Significant_FDR_0.05"] = reject_fdr

# ----------------------------------------------------------
# 4. Bonferroni correction
# ----------------------------------------------------------
# Bonferroni controls the family-wise error rate and is more
# conservative than FDR for large numbers of tests.

reject_bonf, p_bonf, _, _ = multipletests(
    results["p_value"].values,
    alpha=alpha,
    method="bonferroni"
)

results["Bonferroni"] = p_bonf
results["Significant_Bonferroni_0.05"] = reject_bonf

# ----------------------------------------------------------
# 5. Add direction of expression change
# ----------------------------------------------------------
results["Direction"] = np.where(
    results["log2FC"] > 0,
    "Upregulated",
    "Downregulated"
)

# ----------------------------------------------------------
# 6. Generate statistically significant gene lists
# ----------------------------------------------------------
# Primary list: BH FDR < 0.05
significant_fdr = results.loc[
    results["Significant_FDR_0.05"]
].copy()

# Optional stricter list: Bonferroni-adjusted p < 0.05
significant_bonferroni = results.loc[
    results["Significant_Bonferroni_0.05"]
].copy()

# Sort genes by adjusted p-value.
significant_fdr = significant_fdr.sort_values(
    ["FDR_BH", "p_value"]
)
significant_bonferroni = significant_bonferroni.sort_values(
    ["Bonferroni", "p_value"]
)

# ----------------------------------------------------------
# 7. Export complete corrected results and significant genes
# ----------------------------------------------------------
results = results.sort_values("p_value")

results.to_csv(
    "differential_expression_with_adjusted_pvalues.csv",
    index=False
)

significant_fdr.to_csv(
    "significant_genes_FDR.csv",
    index=False
)

significant_bonferroni.to_csv(
    "significant_genes_Bonferroni.csv",
    index=False
)

# ----------------------------------------------------------
# 8. Print summary
# ----------------------------------------------------------
print("Raw p < 0.05:", results["Significant_Raw_0.05"].sum())
print("BH FDR < 0.05:", results["Significant_FDR_0.05"].sum())
print(
    "Bonferroni-adjusted p < 0.05:",
    results["Significant_Bonferroni_0.05"].sum()
)

print("\nTop FDR-significant genes:")
print(
    significant_fdr[
        ["Gene", "log2FC", "p_value", "FDR_BH", "Bonferroni", "Direction"]
    ].head(20)
)

print("\nMultiple-hypothesis correction completed successfully.")
