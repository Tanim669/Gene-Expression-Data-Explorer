# ==========================================================
# Gene Expression Data Explorer
# Task: Initial Data Cleaning and Quality Control
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------
# Expected expression matrix format:
#   - Rows: genes
#   - Columns: samples
#   - First column: gene identifier
# Expected metadata columns:
#   - Sample
#   - Group
#   - Optional: Batch

expr = pd.read_csv("gene_expression_matrix.csv", index_col=0)
metadata = pd.read_csv("sample_metadata.csv").set_index("Sample")

# Reorder metadata to match the expression matrix columns.
metadata = metadata.loc[expr.columns]

# Convert all expression values to numeric.
expr = expr.apply(pd.to_numeric, errors="coerce")

if (expr.dropna() < 0).any().any():
    raise ValueError("Expression values must not be negative for this workflow.")

print("Original dataset dimensions:", expr.shape)
print("Total missing values:", int(expr.isna().sum().sum()))

# ----------------------------------------------------------
# 2. Missing-value assessment
# ----------------------------------------------------------
missing_per_sample = expr.isna().sum()
missing_per_gene = expr.isna().sum(axis=1)

missing_summary = pd.DataFrame({
    "Missing_Count": missing_per_sample,
    "Missing_Percent": 100 * missing_per_sample / expr.shape[0]
})
missing_summary.to_csv("missing_value_summary.csv")

# Remove genes with more than 20% missing values.
max_missing_fraction = 0.20
keep_genes = (missing_per_gene / expr.shape[1]) <= max_missing_fraction
clean_expr = expr.loc[keep_genes].copy()

print("Genes removed for excessive missingness:", int((~keep_genes).sum()))

# Impute remaining missing values using the median of each gene
# across available samples. Median imputation is robust to occasional
# extreme values and is used here only for sparse missingness.
for gene in clean_expr.index:
    if clean_expr.loc[gene].isna().any():
        gene_median = clean_expr.loc[gene].median(skipna=True)
        clean_expr.loc[gene] = clean_expr.loc[gene].fillna(gene_median)

print("Missing values after cleaning:", int(clean_expr.isna().sum().sum()))

# ----------------------------------------------------------
# 3. Optional low-expression filtering
# ----------------------------------------------------------
# For count-like data, remove genes with fewer than 10 counts
# in all but one sample.
low_expression_keep = (clean_expr >= 10).sum(axis=1) >= 2
filtered_expr = clean_expr.loc[low_expression_keep].copy()

print("Genes before low-expression filtering:", clean_expr.shape[0])
print("Genes after low-expression filtering:", filtered_expr.shape[0])

# ----------------------------------------------------------
# 4. Log transformation for exploratory QC
# ----------------------------------------------------------
log_expr = np.log2(filtered_expr + 1.0)

# ----------------------------------------------------------
# 5. Sample-level QC metrics
# ----------------------------------------------------------
qc = pd.DataFrame(index=filtered_expr.columns)
qc["Total_Expression"] = filtered_expr.sum(axis=0)
qc["Median_log2Expr"] = log_expr.median(axis=0)
qc["Mean_log2Expr"] = log_expr.mean(axis=0)
qc["SD_log2Expr"] = log_expr.std(axis=0)
qc["Detected_Genes"] = (filtered_expr > 0).sum(axis=0)
qc["Missing_Before_Cleaning"] = missing_per_sample

# ----------------------------------------------------------
# 6. Sample correlation QC
# ----------------------------------------------------------
correlation_matrix = log_expr.corr(method="pearson")
correlation_matrix.to_csv("sample_correlation_matrix.csv")

qc["Mean_Pairwise_Correlation"] = [
    correlation_matrix.loc[sample].drop(sample).mean()
    for sample in filtered_expr.columns
]

# Correlation heatmap
plt.figure(figsize=(8, 7))
plt.imshow(correlation_matrix.values, aspect="auto", vmin=0, vmax=1)
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90,
    fontsize=7
)
plt.yticks(
    range(len(correlation_matrix.index)),
    correlation_matrix.index,
    fontsize=7
)
plt.colorbar(label="Pearson r")
plt.title("Pearson Sample Correlation Heatmap")
plt.tight_layout()
plt.savefig("sample_correlation_heatmap.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 7. Sample distribution QC plot
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.boxplot(
    [log_expr[c].values for c in log_expr.columns],
    tick_labels=log_expr.columns,
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Samples")
plt.ylabel("log2(Expression + 1)")
plt.title("Post-cleaning Sample Expression Distributions")
plt.tight_layout()
plt.savefig("sample_distribution_boxplot.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 8. PCA-based quality control
# ----------------------------------------------------------
X = StandardScaler().fit_transform(log_expr.T)
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X)

qc["PC1"] = pcs[:, 0]
qc["PC2"] = pcs[:, 1]
qc["PCA_Distance"] = np.sqrt(
    (qc["PC1"] - qc["PC1"].mean()) ** 2 +
    (qc["PC2"] - qc["PC2"].mean()) ** 2
)

plt.figure(figsize=(7, 5))
for group in metadata["Group"].unique():
    mask = metadata["Group"] == group
    plt.scatter(
        qc.loc[mask.index[mask], "PC1"],
        qc.loc[mask.index[mask], "PC2"],
        label=group,
        s=60,
        alpha=0.8
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("PCA-based Sample Quality Control")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("pca_quality_control.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 9. Outlier identification
# ----------------------------------------------------------
# Flag samples that are unusual in one or more QC metrics.
# These thresholds are screening criteria, not automatic reasons
# for removal. Every flagged sample should be reviewed in context.

metrics = [
    "Total_Expression",
    "Median_log2Expr",
    "Detected_Genes",
    "Mean_Pairwise_Correlation",
    "PCA_Distance"
]

outlier_flags = pd.DataFrame(False, index=qc.index, columns=metrics)

for metric in metrics:
    metric_z = pd.Series(
        zscore(qc[metric], nan_policy="omit"),
        index=qc.index
    )

    if metric == "Mean_Pairwise_Correlation":
        # Only unusually LOW correlation is considered problematic.
        outlier_flags[metric] = metric_z < -2
    elif metric == "PCA_Distance":
        # Only unusually LARGE PCA distance is considered problematic.
        outlier_flags[metric] = metric_z > 2
    else:
        outlier_flags[metric] = metric_z.abs() > 2

qc["Outlier_Flag_Count"] = outlier_flags.sum(axis=1)
qc["Overall_Outlier"] = qc["Outlier_Flag_Count"] >= 1

# ----------------------------------------------------------
# 10. Handling plan for flagged outliers
# ----------------------------------------------------------
# Do NOT remove samples automatically.
# Recommended review steps:
#   1. Confirm sample identity and metadata.
#   2. Review library size / total expression.
#   3. Review distribution shape and median expression.
#   4. Review sample correlation and PCA position.
#   5. Check known batch, sequencing, or processing information.
#   6. Remove a sample only when technical evidence supports exclusion.
#   7. If an outlier is biologically plausible, retain it and document it.

flagged_samples = qc.index[qc["Overall_Outlier"]].tolist()
print("Flagged samples requiring review:", flagged_samples)

# ----------------------------------------------------------
# 11. Export cleaned data and QC results
# ----------------------------------------------------------
filtered_expr.to_csv("cleaned_expression_matrix.csv")
qc.to_csv("sample_qc_metrics.csv")
outlier_flags.to_csv("sample_outlier_flags.csv")

print("Data cleaning and quality control successfully completed.")
