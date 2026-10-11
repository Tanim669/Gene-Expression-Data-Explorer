#!/usr/bin/env python3
"""Gene Expression Data Explorer: statistical significance and multiple-testing correction.

Extracted from the project report's complete Python implementation.
Requires: numpy, pandas, statsmodels.
Input (in working directory): differential_expression_results.csv
Required columns: Gene, log2FC, P_value.
Run: python gene_expression_significance_testing.py
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def apply_multiple_testing_correction(
    results_df,
    pvalue_column="P_value",
    log2fc_column="log2FC",
    gene_column="Gene",
    alpha=0.05,
    log2fc_threshold=1.0
):
    """Apply FDR and Bonferroni correction to differential-expression results."""

    results = results_df.copy()

    # Convert raw p-values to numeric and validate the range 0-1.
    results[pvalue_column] = pd.to_numeric(
        results[pvalue_column], errors="coerce"
    )
    invalid = (
        results[pvalue_column].isna() |
        (results[pvalue_column] < 0) |
        (results[pvalue_column] > 1)
    )

    # Missing/invalid p-values are treated conservatively as p = 1.
    p_values = results[pvalue_column].mask(invalid, 1.0)

    # Benjamini-Hochberg false discovery rate correction.
    fdr_reject, fdr_adjusted, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method="fdr_bh"
    )

    # Bonferroni family-wise error correction.
    bonf_reject, bonf_adjusted, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method="bonferroni"
    )

    results["FDR_Adjusted_P_value"] = fdr_adjusted
    results["FDR_Significant"] = fdr_reject
    results["Bonferroni_Adjusted_P_value"] = bonf_adjusted
    results["Bonferroni_Significant"] = bonf_reject

    # Apply the primary FDR threshold and biological effect-size threshold.
    significant = results[
        (results["FDR_Adjusted_P_value"] < alpha) &
        (results[log2fc_column].abs() >= log2fc_threshold)
    ].copy()

    significant["Regulation"] = np.where(
        significant[log2fc_column] >= log2fc_threshold,
        "Upregulated",
        "Downregulated"
    )

    # Rank the significant genes by adjusted p-value.
    significant = significant.sort_values(
        ["FDR_Adjusted_P_value", pvalue_column]
    ).reset_index(drop=True)

    # Rank the full table as well.
    results = results.sort_values(
        "FDR_Adjusted_P_value"
    ).reset_index(drop=True)

    return results, significant


# ---------------------------------------------------------
# Example integration into the existing analysis pipeline
# ---------------------------------------------------------

# Load results produced by the differential-expression step.
dge_results = pd.read_csv("differential_expression_results.csv")

full_results, significant_genes = apply_multiple_testing_correction(
    dge_results,
    pvalue_column="P_value",
    log2fc_column="log2FC",
    gene_column="Gene",
    alpha=0.05,
    log2fc_threshold=1.0
)

# Export all genes with adjusted p-values.
full_results.to_csv(
    "differential_expression_with_adjusted_pvalues.csv",
    index=False
)

# Export the final statistically significant gene list.
significant_genes.to_csv(
    "statistically_significant_genes.csv",
    index=False
)

print("Total genes tested:", len(full_results))
print(
    "FDR-significant genes:",
    int(full_results["FDR_Significant"].sum())
)
print(
    "Bonferroni-significant genes:",
    int(full_results["Bonferroni_Significant"].sum())
)
print(
    "Genes passing FDR and fold-change thresholds:",
    len(significant_genes)
)
print(significant_genes.head(20))
