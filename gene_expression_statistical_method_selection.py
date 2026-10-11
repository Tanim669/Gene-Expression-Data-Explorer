#!/usr/bin/env python3
"""Gene Expression Data Explorer: statistical method selection.

Companion script to the Statistical Method Research and Selection Report.
This is a new executable implementation of the report's decision rules;
the source report contains comparative research rather than executable code.

Examples:
    python gene_expression_statistical_method_selection.py --data-type normalized --design two-group
    python gene_expression_statistical_method_selection.py --data-type counts --design two-group --output selection.json
    python gene_expression_statistical_method_selection.py --list-methods

This tool recommends methods; it does not perform differential-expression
analysis or infer data type from a dataset. Inspect actual assay/input types.
"""

import argparse
import json
from pathlib import Path

METHODS = {
    "Welch's two-sample t-test": {
        "data": "Normalized continuous expression; independent two-group comparisons",
        "strength": "Simple Python implementation; unequal variances allowed",
        "limitation": "No cross-gene variance moderation; not recommended for raw RNA-seq counts",
        "implementation": "SciPy: scipy.stats.ttest_ind(equal_var=False)",
    },
    "limma / limma-voom": {
        "data": "Transformed continuous expression, or RNA-seq counts with voom",
        "strength": "Empirical Bayes moderation and flexible experimental designs",
        "limitation": "Requires appropriate design, preprocessing, and typically R/Bioconductor",
        "implementation": "R/Bioconductor: limma; voom for RNA-seq counts",
    },
    "DESeq2": {
        "data": "Raw RNA-seq integer counts",
        "strength": "Negative-binomial models with dispersion estimation and covariates",
        "limitation": "Not suitable for arbitrary normalized continuous input",
        "implementation": "R/Bioconductor: DESeq2",
    },
    "edgeR": {
        "data": "Raw RNA-seq integer counts",
        "strength": "Negative-binomial models, normalization, and flexible designs",
        "limitation": "Requires count-data preprocessing and design specification",
        "implementation": "R/Bioconductor: edgeR",
    },
    "Mann-Whitney U test": {
        "data": "Normalized rankable measurements, two independent groups",
        "strength": "Does not assume normal distributions",
        "limitation": "Low power/resolution for very small samples; not count-specific",
        "implementation": "SciPy: scipy.stats.mannwhitneyu",
    },
}


def recommend_method(data_type: str, design: str = "two-group", nonnormal: bool = False) -> dict:
    """Apply the report's qualitative recommendations to explicit input metadata.

    The recommendations assume independent samples; paired/repeated measures
    require a design-specific model and are not supported by this selector.
    """
    if data_type == "counts":
        primary = ["DESeq2", "edgeR"]
        reason = "Raw RNA-seq counts warrant count-based negative-binomial models."
    elif design == "complex":
        primary = ["limma / limma-voom"]
        reason = "Continuous expression with covariates/complex designs benefits from linear modeling."
    else:
        primary = ["Welch's two-sample t-test"]
        reason = "Normalized continuous measurements in two independent groups fit the report's initial demonstration workflow."
    secondary = []
    if data_type == "normalized" and nonnormal and design == "two-group":
        secondary.append("Mann-Whitney U test")
    return {
        "input_type": data_type,
        "design": design,
        "strongly_nonnormal": nonnormal,
        "recommended_primary": primary,
        "secondary_sensitivity_methods": secondary,
        "rationale": reason,
        "multiple_testing": "Benjamini-Hochberg FDR; exploratory adjusted p < 0.05",
        "effect_size": "Optional |log2 fold change| >= 1; compute on a suitable scale",
        "important_note": (
            "This is selection guidance, not differential-expression analysis. "
            "Verify sample independence, normalization, assay type, and model assumptions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-type", choices=["normalized", "counts"],
                        help="normalized continuous expression or raw RNA-seq counts")
    parser.add_argument("--design", choices=["two-group", "complex"], default="two-group",
                        help="two independent groups or a design with covariates")
    parser.add_argument("--strongly-nonnormal", action="store_true",
                        help="consider Mann-Whitney U as sensitivity analysis for normalized two-group data")
    parser.add_argument("--list-methods", action="store_true", help="print all five compared methods")
    parser.add_argument("--output", type=Path, help="optional path for JSON report")
    args = parser.parse_args()

    if not args.list_methods and not args.data_type:
        parser.error("provide --data-type, or use --list-methods")
    result = {"methods_evaluated": METHODS} if args.list_methods else recommend_method(
        args.data_type, args.design, args.strongly_nonnormal
    )
    formatted = json.dumps(result, indent=2, ensure_ascii=False)
    print(formatted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
