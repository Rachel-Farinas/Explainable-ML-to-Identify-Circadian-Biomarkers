# diagnostics.py
# Post-model diagnostics: Z-score group-marker table and clinical audit table.

import numpy as np
import pandas as pd


def compute_group_zscore_table(
    metadata: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Computes per-group Z-scores relative to the population mean/std for each
    metric, identifies the primary marker for each group (the metric with the
    highest absolute Z-score), and returns the annotated table.
    """
    pop_mean    = metadata[metrics].mean()
    pop_std     = metadata[metrics].std()
    group_means = metadata.groupby("group")[metrics].mean()

    z_scores = (group_means - pop_mean) / pop_std

    def identify_marker(row: pd.Series) -> str:
        abs_row    = row.abs()
        top_metric = abs_row.idxmax()
        direction  = "Low" if row[top_metric] < 0 else "High"
        return f"{direction} {top_metric}"

    z_scores["Primary Marker"] = z_scores.apply(identify_marker, axis=1)

    print("\n--- Group Z-Score Table ---")
    print(z_scores)

    return z_scores


def build_clinical_audit_table(
    metadata: pd.DataFrame,
    y_true,
    y_pred,
    label_encoder,
    display_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Creates one representative Hit / Miss row per (Actual, Predicted) pair so
    clinicians can inspect which features are driving misclassifications.
    """
    if display_cols is None:
        display_cols = ["Actual", "Status", "Predicted", "M10", "sampEn_delta",
                        "daytime_volatility", "IV"]

    audit_df              = metadata.copy()
    audit_df["Actual"]    = label_encoder.inverse_transform(y_true)
    audit_df["Predicted"] = label_encoder.inverse_transform(y_pred)

    audit_df["Status"] = np.where(
        audit_df["Actual"] == audit_df["Predicted"],
        "CORRECT (Hit)",
        "MISSED (Interaction)",
    )

    final_table = (
        audit_df.groupby(["Actual", "Predicted"])
        .head(1)
        .sort_values(by=["Actual", "Predicted"])
    )

    # Keep only columns that actually exist in the dataframe
    existing_cols = [c for c in display_cols if c in final_table.columns]
    final_table   = final_table[existing_cols]

    print("\n--- Clinical Interaction Audit (Full Model) ---")
    print(final_table.round(3).to_string())

    return final_table
