#!/usr/bin/env python3
"""Visualize differential gene expression with a volcano plot and gene heatmap.

Companion implementation for the Gene Expression Data Explorer visualization report.
Inputs:
  --results: CSV with gene identifiers, log2 fold change, adjusted p-values.
  --expression: CSV with gene IDs in first column, samples in remaining columns;
                values must be normalized expression suitable for a heatmap.
Outputs: volcano_plot.png, significant_genes_heatmap.png, significant_genes.csv.

Example:
  python gene_expression_differential_visualization.py \
    --results differential_expression_results.csv \
    --expression normalized_expression_matrix.csv --output-dir figures
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_column(df, candidates, description):
    lookup = {str(c).lower(): c for c in df.columns}
    for item in candidates:
        if item.lower() in lookup:
            return lookup[item.lower()]
    raise ValueError(f'Missing {description} column. Accepted names: {", ".join(candidates)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='differential_expression_results.csv')
    parser.add_argument('--expression', default='normalized_expression_matrix.csv')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--fdr', type=float, default=0.05)
    parser.add_argument('--log2fc', type=float, default=1.0)
    parser.add_argument('--top-genes', type=int, default=25)
    parser.add_argument('--label-genes', type=int, default=8)
    args = parser.parse_args()
    if not (0 < args.fdr < 1) or args.log2fc < 0 or args.top_genes < 1 or args.label_genes < 0:
        parser.error('Require 0 < --fdr < 1, --log2fc >= 0, --top-genes >= 1, --label-genes >= 0')

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    de = pd.read_csv(args.results)
    gene_col = find_column(de, ['Gene', 'gene', 'gene_id'], 'gene identifier')
    fc_col = find_column(de, ['log2FoldChange', 'log2FC', 'log2_fold_change'], 'log2 fold change')
    padj_col = find_column(de, ['padj', 'Adjusted_P_value', 'FDR_Adjusted_P_value', 'FDR_BH', 'adj_p_val'], 'FDR-adjusted p-value')
    data = de[[gene_col, fc_col, padj_col]].copy()
    data.columns = ['Gene', 'log2FC', 'padj']
    if data['Gene'].isna().any() or data['Gene'].astype(str).duplicated().any():
        raise ValueError('Gene IDs in differential-expression results must be nonmissing and unique.')
    for col in ('log2FC', 'padj'):
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['log2FC', 'padj'])
    data = data[np.isfinite(data['log2FC']) & np.isfinite(data['padj'])].copy()
    if data.empty or ((data['padj'] < 0) | (data['padj'] > 1)).any():
        raise ValueError('No usable values or adjusted p-values outside [0,1].')
    data['Significant'] = (data['padj'] < args.fdr) & (data['log2FC'].abs() >= args.log2fc)
    significant = data[data['Significant']].sort_values('padj').copy()
    significant.to_csv(output / 'significant_genes.csv', index=False)
    print(f'Tested genes: {len(data)}; significant (FDR < {args.fdr}, abs(log2FC) >= {args.log2fc}): {len(significant)}')

    # Clip exact zero p-values for safe logarithms without changing significance decisions.
    y = -np.log10(data['padj'].clip(lower=np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(10, 6))
    for mask, label in [(~data['Significant'], 'Not significant'), (data['Significant'], 'Significant')]:
        ax.scatter(data.loc[mask, 'log2FC'], y.loc[mask], s=17, alpha=0.7, label=label)
    ax.axvline(-args.log2fc, linestyle='--', linewidth=1)
    ax.axvline(args.log2fc, linestyle='--', linewidth=1)
    ax.axhline(-np.log10(args.fdr), linestyle='--', linewidth=1)
    for index in significant.head(args.label_genes).index:
        ax.annotate(str(data.at[index, 'Gene']), (data.at[index, 'log2FC'], y.at[index]), fontsize=7, xytext=(3, 4), textcoords='offset points')
    ax.set(xlabel='log2 fold change', ylabel='-log10(FDR-adjusted p-value)', title=f'Volcano Plot of Differential Gene Expression (n={len(significant)} significant)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'volcano_plot.png', dpi=300)
    plt.close(fig)

    expr = pd.read_csv(args.expression, index_col=0)
    if expr.index.isna().any() or expr.index.astype(str).duplicated().any():
        raise ValueError('Expression matrix gene identifiers must be nonmissing and unique.')
    expr.index = expr.index.astype(str)
    expr = expr.apply(pd.to_numeric, errors='coerce')
    if expr.isna().any().any() or not np.isfinite(expr.to_numpy()).all():
        raise ValueError('Expression matrix contains missing or nonfinite values.')
    selected = [str(g) for g in significant['Gene'] if str(g) in expr.index][:args.top_genes]
    if not selected:
        print('No significant genes overlap expression matrix; heatmap skipped.')
        return
    matrix = expr.loc[selected].astype(float)
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1, ddof=0).replace(0, 1)
    heat = matrix.sub(means, axis=0).div(stds, axis=0)
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(heat.columns) + 3), max(5, 0.28 * len(heat) + 2)))
    im = ax.imshow(heat.to_numpy(), aspect='auto', cmap='coolwarm')
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set(xlabel='Samples', ylabel='Genes', title='Top Significant Differentially Expressed Genes (row Z-scores)')
    fig.colorbar(im, ax=ax, label='Row Z-score')
    fig.tight_layout()
    fig.savefig(output / 'significant_genes_heatmap.png', dpi=300)
    plt.close(fig)
    print(f'Heatmap generated for {len(selected)} significant genes.')


if __name__ == '__main__':
    main()
