import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list

# ==========================================================
# Gene Expression Data Explorer
# Task: Initial Exploratory Data Analysis (EDA) and Visualization
# Inputs: normalized expression matrix + sample metadata
# ==========================================================

# ----------------------------------------------------------
# 1. Load normalized expression data and metadata
# ----------------------------------------------------------
# Expected expression format:
# - Rows: genes
# - Columns: samples
# - First column: gene identifier
# Recommended input: TMM-normalized log2(CPM + 1) matrix

expr = pd.read_csv("tmm_log2_cpm.csv", index_col=0)
metadata = pd.read_csv("sample_metadata.csv")

# Required metadata columns: Sample and Group
# Optional metadata column: Batch
required = {"Sample", "Group"}
missing = required.difference(metadata.columns)
if missing:
    raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

# Ensure metadata and expression samples match and use the same order.
metadata = metadata.set_index("Sample")
missing_meta = [s for s in expr.columns if s not in metadata.index]
if missing_meta:
    raise ValueError(f"Metadata missing samples: {missing_meta}")
metadata = metadata.loc[expr.columns]

# Coerce expression values to numeric and check for missing values.
expr = expr.apply(pd.to_numeric, errors="coerce")
if expr.isna().any().any():
    raise ValueError("Expression matrix contains missing or non-numeric values.")

print("Expression matrix dimensions:", expr.shape)
print("Samples:", list(expr.columns))
print(metadata.head())

# ----------------------------------------------------------
# 2. Basic descriptive statistics
# ----------------------------------------------------------
sample_summary = pd.DataFrame({
    "Mean": expr.mean(axis=0),
    "Median": expr.median(axis=0),
    "StdDev": expr.std(axis=0),
    "Min": expr.min(axis=0),
    "Max": expr.max(axis=0)
})
sample_summary.to_csv("eda_sample_summary.csv")
print("Sample-level expression summary:")
print(sample_summary)

# ----------------------------------------------------------
# 3. Sample distribution box plot
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.boxplot(
    [expr[c].values for c in expr.columns],
    tick_labels=expr.columns,
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Samples")
plt.ylabel("Normalized expression")
plt.title("Sample Expression Distributions")
plt.tight_layout()
plt.savefig("eda_boxplot.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 4. Select highly variable genes for dimension reduction
# ----------------------------------------------------------
# Using highly variable genes reduces noise from nearly invariant features.
gene_variance = expr.var(axis=1)
n_top = min(500, expr.shape[0])
top_genes = gene_variance.nlargest(n_top).index
expr_hvg = expr.loc[top_genes]

# Samples must be rows and genes must be columns for scikit-learn.
X = expr_hvg.T

# Standardize each selected gene so highly expressed genes do not dominate.
X_scaled = StandardScaler().fit_transform(X)

# ----------------------------------------------------------
# 5. Principal Component Analysis (PCA)
# ----------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
pc_scores = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "Sample": expr.columns,
    "PC1": pc_scores[:, 0],
    "PC2": pc_scores[:, 1],
    "Group": metadata["Group"].values
})

if "Batch" in metadata.columns:
    pca_df["Batch"] = metadata["Batch"].values

pca_df.to_csv("eda_pca_coordinates.csv", index=False)

plt.figure(figsize=(8, 6))
for group in pca_df["Group"].unique():
    mask = pca_df["Group"] == group
    plt.scatter(
        pca_df.loc[mask, "PC1"],
        pca_df.loc[mask, "PC2"],
        s=80,
        label=group
    )

for _, row in pca_df.iterrows():
    plt.annotate(
        row["Sample"],
        (row["PC1"], row["PC2"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.axhline(0, linewidth=0.6)
plt.axvline(0, linewidth=0.6)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.title("PCA of Gene-Expression Samples")
plt.legend()
plt.tight_layout()
plt.savefig("eda_pca.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 6. PCA batch-effect check, when batch metadata are available
# ----------------------------------------------------------
if "Batch" in pca_df.columns:
    plt.figure(figsize=(8, 6))
    for batch in pca_df["Batch"].unique():
        mask = pca_df["Batch"] == batch
        plt.scatter(
            pca_df.loc[mask, "PC1"],
            pca_df.loc[mask, "PC2"],
            s=80,
            label=batch
        )
    for _, row in pca_df.iterrows():
        plt.annotate(
            row["Sample"],
            (row["PC1"], row["PC2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("PCA Colored by Batch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_pca_by_batch.png", dpi=300)
    plt.close()

# ----------------------------------------------------------
# 7. t-SNE
# ----------------------------------------------------------
# Perplexity must be lower than the number of samples.
n_samples = X_scaled.shape[0]
perplexity = min(30, max(2, (n_samples - 1) // 3))

if perplexity >= n_samples:
    perplexity = max(1, n_samples - 1)

if n_samples >= 4:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=1500
    )
    tsne_scores = tsne.fit_transform(X_scaled)

    tsne_df = pd.DataFrame({
        "Sample": expr.columns,
        "tSNE1": tsne_scores[:, 0],
        "tSNE2": tsne_scores[:, 1],
        "Group": metadata["Group"].values
    })
    tsne_df.to_csv("eda_tsne_coordinates.csv", index=False)

    plt.figure(figsize=(8, 6))
    for group in tsne_df["Group"].unique():
        mask = tsne_df["Group"] == group
        plt.scatter(
            tsne_df.loc[mask, "tSNE1"],
            tsne_df.loc[mask, "tSNE2"],
            s=80,
            label=group
        )
    for _, row in tsne_df.iterrows():
        plt.annotate(
            row["Sample"],
            (row["tSNE1"], row["tSNE2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE of Gene-Expression Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eda_tsne.png", dpi=300)
    plt.close()
else:
    print("t-SNE skipped because too few samples are available.")

# ----------------------------------------------------------
# 8. Heatmap of the most variable genes
# ----------------------------------------------------------
n_heatmap_genes = min(40, expr.shape[0])
heatmap_genes = gene_variance.nlargest(n_heatmap_genes).index
heat = expr.loc[heatmap_genes]

# Gene-wise z-score for pattern visualization.
heat_z = heat.sub(heat.mean(axis=1), axis=0).div(
    heat.std(axis=1).replace(0, 1),
    axis=0
)

# Hierarchical clustering of genes and samples.
row_order = leaves_list(
    linkage(heat_z.values, method="average", metric="euclidean")
)
col_order = leaves_list(
    linkage(heat_z.T.values, method="average", metric="euclidean")
)
heat_ordered = heat_z.iloc[row_order, col_order]

plt.figure(figsize=(10, 8))
plt.imshow(heat_ordered.values, aspect="auto", interpolation="nearest")
plt.xticks(
    range(len(heat_ordered.columns)),
    heat_ordered.columns,
    rotation=45,
    ha="right",
    fontsize=8
)
plt.yticks(
    range(len(heat_ordered.index)),
    heat_ordered.index,
    fontsize=6
)
plt.colorbar(label="Gene-wise z-score")
plt.xlabel("Samples")
plt.ylabel("Genes")
plt.title("Heatmap of Top Variable Genes")
plt.tight_layout()
plt.savefig("eda_heatmap_top_variable_genes.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 9. Correlation matrix as an additional sample-QC output
# ----------------------------------------------------------
sample_corr = expr.corr(method="pearson")
sample_corr.to_csv("eda_sample_correlation.csv")

plt.figure(figsize=(7, 6))
plt.imshow(sample_corr.values, aspect="auto", vmin=-1, vmax=1)
plt.xticks(range(len(sample_corr.columns)), sample_corr.columns, rotation=45, ha="right")
plt.yticks(range(len(sample_corr.index)), sample_corr.index)
plt.colorbar(label="Pearson correlation")
plt.title("Sample Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_sample_correlation_heatmap.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 10. Print EDA summary
# ----------------------------------------------------------
print("PCA explained variance:")
print({
    "PC1": pca.explained_variance_ratio_[0],
    "PC2": pca.explained_variance_ratio_[1]
})

if "Batch" in metadata.columns:
    batch_pca_summary = pca_df.groupby("Batch")[["PC1", "PC2"]].mean()
    print("Mean PCA coordinates by batch:")
    print(batch_pca_summary)

print("EDA successfully completed.")
print("Generated outputs: box plot, PCA, optional batch PCA, t-SNE, heatmap, correlation matrix, and coordinate tables.")
