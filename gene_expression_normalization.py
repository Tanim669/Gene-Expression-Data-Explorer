import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Gene Expression Data Explorer
# Task: Data Normalization and Transformation
# Method: TMM-style normalization + log2(CPM + 1)
# ==========================================================

# ----------------------------------------------------------
# 1. Load raw gene-expression count matrix
# ----------------------------------------------------------
# Expected format:
#   - Rows: genes
#   - Columns: samples
#   - First column: gene identifier (used as row index)
#   - Remaining values: non-negative raw integer counts

counts_df = pd.read_csv(
    "gene_expression_counts.csv",
    index_col=0
)

# Convert all expression columns to numeric values.
counts_df = counts_df.apply(pd.to_numeric, errors="coerce")

if counts_df.isna().any().any():
    raise ValueError("The count matrix contains missing or non-numeric values.")

if (counts_df < 0).any().any():
    raise ValueError("Raw RNA-seq counts cannot contain negative values.")

print("Dataset dimensions:", counts_df.shape)
print(counts_df.head())

# ----------------------------------------------------------
# 2. Optional low-expression filtering
# ----------------------------------------------------------
# Keep genes with at least 10 counts in at least 2 samples.
# Thresholds can be modified for the project design.

keep = (counts_df >= 10).sum(axis=1) >= 2
filtered_counts = counts_df.loc[keep].copy()

print("Genes before filtering:", counts_df.shape[0])
print("Genes after filtering:", filtered_counts.shape[0])

# ----------------------------------------------------------
# 3. Helper function: weighted mean
# ----------------------------------------------------------

def weighted_mean(values, weights):
    return np.sum(values * weights) / np.sum(weights)

# ----------------------------------------------------------
# 4. Calculate one TMM-style scaling factor
# ----------------------------------------------------------

def calc_tmm_factor(
    test,
    ref,
    logratio_trim=0.30,
    sum_trim=0.05
):
    """
    Calculate a TMM-style scaling factor for one test sample
    relative to a reference sample.

    Parameters
    ----------
    test : array-like
        Raw counts for the test sample.
    ref : array-like
        Raw counts for the reference sample.
    logratio_trim : float
        Fraction of M-values trimmed from the two tails.
    sum_trim : float
        Fraction of A-values trimmed from the two tails.

    Returns
    -------
    float
        TMM-style normalization factor.
    """

    test = np.asarray(test, dtype=float)
    ref = np.asarray(ref, dtype=float)

    n_test = test.sum()
    n_ref = ref.sum()

    if n_test <= 0 or n_ref <= 0:
        raise ValueError("Library size must be greater than zero.")

    # TMM ratio calculations require positive counts in both samples.
    mask = (test > 0) & (ref > 0)
    y = test[mask]
    r = ref[mask]

    # M-values: log2 expression ratios after library-size scaling.
    log_r = np.log2((y / n_test) / (r / n_ref))

    # A-values: average log abundance.
    abs_e = 0.5 * np.log2((y / n_test) * (r / n_ref))

    # Approximate delta-method precision weights.
    variance = (
        ((n_test - y) / (n_test * y)) +
        ((n_ref - r) / (n_ref * r))
    )
    weights = 1.0 / variance

    finite = (
        np.isfinite(log_r) &
        np.isfinite(abs_e) &
        np.isfinite(weights) &
        (weights > 0)
    )

    log_r = log_r[finite]
    abs_e = abs_e[finite]
    weights = weights[finite]

    if len(log_r) < 10:
        return 1.0

    # Trim extreme M-values and A-values.
    m_low, m_high = np.quantile(
        log_r,
        [logratio_trim / 2, 1 - logratio_trim / 2]
    )

    a_low, a_high = np.quantile(
        abs_e,
        [sum_trim / 2, 1 - sum_trim / 2]
    )

    keep = (
        (log_r >= m_low) &
        (log_r <= m_high) &
        (abs_e >= a_low) &
        (abs_e <= a_high)
    )

    if keep.sum() == 0:
        return 1.0

    mean_m = weighted_mean(log_r[keep], weights[keep])

    return 2 ** mean_m

# ----------------------------------------------------------
# 5. Normalize the complete count matrix
# ----------------------------------------------------------

def tmm_normalize(count_matrix):
    """
    Apply TMM-style between-sample normalization and return
    normalized CPM plus log2(CPM + 1) values.
    """

    library_sizes = count_matrix.sum(axis=0).astype(float)

    if (library_sizes <= 0).any():
        raise ValueError("Every sample must contain at least one count.")

    # Initial CPM values are used only for choosing a reference sample.
    initial_cpm = count_matrix.div(library_sizes, axis=1) * 1e6

    # Choose the sample whose nonzero 75th percentile is closest
    # to the mean upper-quartile value across all samples.
    upper_quartiles = (
        initial_cpm
        .replace(0, np.nan)
        .quantile(0.75, axis=0)
    )

    reference_sample = (
        upper_quartiles - upper_quartiles.mean()
    ).abs().idxmin()

    reference_counts = count_matrix[reference_sample].values

    factors = {}

    for sample in count_matrix.columns:
        if sample == reference_sample:
            factors[sample] = 1.0
        else:
            factors[sample] = calc_tmm_factor(
                count_matrix[sample].values,
                reference_counts
            )

    factors = pd.Series(factors, dtype=float)

    # Center factors so their geometric mean is exactly 1.
    factors = factors / np.exp(np.mean(np.log(factors)))

    effective_library_sizes = library_sizes * factors

    # TMM-normalized CPM.
    normalized_cpm = (
        count_matrix.div(effective_library_sizes, axis=1) * 1e6
    )

    # Log transformation for visualization/exploratory analysis.
    log2_cpm = np.log2(normalized_cpm + 1.0)

    return (
        reference_sample,
        library_sizes,
        factors,
        effective_library_sizes,
        normalized_cpm,
        log2_cpm
    )

(
    reference_sample,
    library_sizes,
    tmm_factors,
    effective_library_sizes,
    normalized_cpm,
    log2_cpm
) = tmm_normalize(filtered_counts)

print("Reference sample:", reference_sample)
print("TMM normalization factors:")
print(tmm_factors)

# ----------------------------------------------------------
# 6. Generate pre-normalization values for visualization
# ----------------------------------------------------------

raw_cpm = filtered_counts.div(library_sizes, axis=1) * 1e6
raw_log2_cpm = np.log2(raw_cpm + 1.0)

# ----------------------------------------------------------
# 7. Export normalized data and normalization metadata
# ----------------------------------------------------------

normalized_cpm.to_csv("tmm_normalized_cpm.csv")
log2_cpm.to_csv("tmm_log2_cpm.csv")

normalization_metadata = pd.DataFrame({
    "Library_Size": library_sizes,
    "TMM_Factor": tmm_factors,
    "Effective_Library_Size": effective_library_sizes
})

normalization_metadata.to_csv("tmm_normalization_factors.csv")

# ----------------------------------------------------------
# 8. Pre-normalization boxplot
# ----------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.boxplot(
    [raw_log2_cpm[c].values for c in raw_log2_cpm.columns],
    tick_labels=raw_log2_cpm.columns,
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Samples")
plt.ylabel("log2(CPM + 1)")
plt.title("Pre-normalization Expression Distributions")
plt.tight_layout()
plt.savefig("pre_normalization_boxplot.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 9. Post-normalization boxplot
# ----------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.boxplot(
    [log2_cpm[c].values for c in log2_cpm.columns],
    tick_labels=log2_cpm.columns,
    showfliers=False
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Samples")
plt.ylabel("TMM-normalized log2(CPM + 1)")
plt.title("Post-normalization Expression Distributions")
plt.tight_layout()
plt.savefig("post_normalization_boxplot.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 10. Pre-normalization density plot
# ----------------------------------------------------------

plt.figure(figsize=(10, 5))

for sample in raw_log2_cpm.columns:
    values = raw_log2_cpm[sample].dropna().values
    density, edges = np.histogram(
        values,
        bins=80,
        density=True
    )
    centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(centers, density, label=sample)

plt.xlabel("log2(CPM + 1)")
plt.ylabel("Density")
plt.title("Pre-normalization Density Distributions")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("pre_normalization_density.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 11. Post-normalization density plot
# ----------------------------------------------------------

plt.figure(figsize=(10, 5))

for sample in log2_cpm.columns:
    values = log2_cpm[sample].dropna().values
    density, edges = np.histogram(
        values,
        bins=80,
        density=True
    )
    centers = (edges[:-1] + edges[1:]) / 2
    plt.plot(centers, density, label=sample)

plt.xlabel("TMM-normalized log2(CPM + 1)")
plt.ylabel("Density")
plt.title("Post-normalization Density Distributions")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("post_normalization_density.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 12. Print summary statistics
# ----------------------------------------------------------

summary_table = pd.DataFrame({
    "Library_Size": library_sizes,
    "TMM_Factor": tmm_factors,
    "Effective_Library_Size": effective_library_sizes,
    "Median_Pre_log2CPM": raw_log2_cpm.median(axis=0),
    "Median_Post_log2CPM": log2_cpm.median(axis=0)
})

print("Normalization summary:")
print(summary_table)

print("Normalization and transformation successfully completed.")
