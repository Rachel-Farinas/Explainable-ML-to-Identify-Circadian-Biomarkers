# exploratory_analysis.py
# Sanity checks, outlier detection, and data-quality reports.

import numpy as np
import pandas as pd
from scipy import stats


def run_nan_check(num_nan: dict) -> None:
    """Reports whether any participant CSVs contain NaN activity values."""
    if all(v == 0 for v in num_nan.values()):
        print("No NaN value(s) found in OBF dataset.")
    else:
        print("NaN value(s) found in OBF dataset.")


def check_constant_errors(metadata: pd.DataFrame) -> pd.DataFrame:
    """Prints and returns participants with constant-value (frozen) errors."""
    affected = metadata[metadata["contains_constant_error_values"] == True]
    print(f"Number of participants with constant error values: {len(affected)}")
    print(affected["number"])
    return affected


def check_high_nonwear(metadata: pd.DataFrame) -> pd.DataFrame:
    """Prints and returns participants whose overall non-wear ratio exceeds the threshold."""
    affected = metadata[metadata["meets_activity_threshold"] == False]
    print(f"Number of participants with high non-wear time: {len(affected)}")
    print(affected["number"])
    return affected


def check_insufficient_data(metadata: pd.DataFrame, min_valid_days: int = 3) -> pd.DataFrame:
    """Prints participants with fewer than `min_valid_days` usable days."""
    affected = metadata[metadata["salvaged_valid_days"] < min_valid_days]
    print(f"Number of participants with insufficient data: {len(affected)}")
    print(affected["number"])
    print(affected["salvaged_valid_days"])
    return affected


def check_metric_ranges(metadata: pd.DataFrame) -> None:
    """
    Asserts that IS, IV, RA, L5, and M10 are within clinically acceptable
    ranges and prints the results.
    """
    print("\n--- Metric Range Checks ---")
    print(f"IS < 0:  {(metadata['IS'] < 0).any()}")
    print(f"IS > 1:  {(metadata['IS'] > 1).any()}")
    print(f"IV < 0:  {(metadata['IV'] < 0).any()}")
    print(f"IV > 2:  {(metadata['IV'] > 2).any()}")
    print(f"RA < 0:  {(metadata['relative_amplitude'] < 0).any()}")
    print(f"RA > 1:  {(metadata['relative_amplitude'] > 1).any()}")
    print(f"L5 > M10: {(metadata['L5'] > metadata['M10']).any()}")
    print(f"IS max:  {metadata['IS'].max():.4f}")
    print(f"IV max:  {metadata['IV'].max():.4f}")
    print(f"RA max:  {metadata['relative_amplitude'].max():.4f}")


def remove_sample_entropy_outliers(
    metadata: pd.DataFrame,
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Iteratively removes sample-entropy outliers (|z| > `outlier_threshold`).
    Uses the known bad participant 'clinical_82' as the first removal, then
    re-evaluates twice to catch secondary outliers, matching the original
    notebook logic.
    """
    def _zscore_col(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            stats.zscore(df["sample_entropy"], nan_policy="omit"),
            index=df.index,
        )

    # Pass 1 – compute Z-scores and report outliers
    metadata = metadata.copy()
    metadata["sample_entropy_Zscore"] = _zscore_col(metadata)

    mask     = metadata["sample_entropy_Zscore"].abs() > outlier_threshold
    outliers = metadata[mask]
    print(f"\nTotal participants flagged as outliers (pass 1): {len(outliers)}")
    print(outliers["sample_entropy_Zscore"])

    # Pass 2 – remove clinical_82 explicitly (matches original notebook)
    metadata = metadata[metadata["number"] != "clinical_82"].copy()
    metadata["sample_entropy_Zscore"] = _zscore_col(metadata)

    # Pass 3 – remove any remaining outliers
    metadata = metadata[
        metadata["sample_entropy_Zscore"].abs() <= outlier_threshold
    ].copy()

    # Final Z-score re-calibration
    mu    = metadata["sample_entropy"].mean()
    sigma = metadata["sample_entropy"].std()
    metadata["sample_entropy_Zscore"] = (metadata["sample_entropy"] - mu) / sigma

    print(f"Final dataset N: {len(metadata)}")
    print(
        f"New Z-score range: "
        f"{metadata['sample_entropy_Zscore'].min():.2f} to "
        f"{metadata['sample_entropy_Zscore'].max():.2f}"
    )

    return metadata


def check_low_sample_entropy(metadata: pd.DataFrame, threshold: float = 0.2) -> None:
    """Prints participants with very low sample entropy (potential data quality issue)."""
    low = metadata[metadata["sample_entropy"] < threshold]
    print(f"\nNumber of participants with very low sample entropy: {len(low)}")
    print(low["number"])


def find_strong_correlations(
    metadata: pd.DataFrame,
    features: list[str],
    threshold: float = 0.6,
) -> None:
    """Prints feature pairs whose |r| exceeds `threshold` for each group."""
    print(f"\n--- Identifying correlations where |r| > {threshold} ---\n")

    for group in metadata["group"].unique():
        print(f"Group: {group.upper()}")
        group_df    = metadata[metadata["group"] == group][features]
        corr_matrix = group_df.corr()
        pairs       = corr_matrix.unstack()
        strong_pairs = pairs[(abs(pairs) > threshold) & (pairs < 1.0)]

        seen = set()
        if not strong_pairs.empty:
            for (f1, f2), val in strong_pairs.items():
                pair_key = tuple(sorted((f1, f2)))
                if pair_key not in seen:
                    print(f"  - {f1} <--> {f2} : r = {val:.3f}")
                    seen.add(pair_key)
        else:
            print("  - No correlations above threshold.")
        print("-" * 30)
