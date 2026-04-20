#!/usr/bin/env python3
# main.py
# Entry point for the OBF Actigraphy Analysis Pipeline.
#
# Usage (from the project root directory):
#   python main.py
#
# Expected directory layout:
#   ├── main.py
#   ├── scripts/
#   │   ├── __init__.py
#   │   ├── config.py
#   │   ├── metadata_aggregation.py
#   │   ├── preprocessing.py
#   │   ├── feature_extraction.py
#   │   ├── exploratory_analysis.py
#   │   ├── plotting.py
#   │   ├── diagnostics.py
#   │   ├── performance.py
#   │   ├── transformer_setup.py
#   │   └── embeddings.py
#   ├── actigraphy/
#   │   ├── adhd/           *.csv
#   │   ├── control/        *.csv
#   │   ├── depression/     *.csv
#   │   ├── schizophrenia/  *.csv
#   │   └── clinical/       *.csv
#   ├── metadata/
#   │   ├── adhd-info.csv
#   │   ├── control-info.csv
#   │   └── ...
#   ├── Pretrained-Actigraphy-Transformer/
#   │   └── weights/
#   │       └── PAT-L_29k_weights.h5
#   ├── plots/              <- created automatically
#   └── embeddings/         <- created automatically

import os
import sys

# Ensure the project root is on sys.path so `scripts` is importable as a package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force TensorFlow to use CPU before any TF imports happen
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd

from scripts.config import (
    ACTIGRAPHY_DIR,
    CORRELATION_THRESHOLD,
    FEATURE_SETS,
    GROUPS,
)
from scripts.metadata_aggregation import load_master_metadata
from scripts.feature_extraction import (
    RunActigraphyPipeline,
    RunTemporalComplexityPipeline,
    num_nan,
)
from scripts.exploratory_analysis import (
    check_constant_errors,
    check_high_nonwear,
    check_insufficient_data,
    check_low_sample_entropy,
    check_metric_ranges,
    find_strong_correlations,
    remove_sample_entropy_outliers,
    run_nan_check,
)
from scripts.plotting import (
    plot_behavioral_network_vertical_dynamic,
    plot_confusion_matrix_grid,
    plot_correlation_heatmaps,
    plot_feature_importance,
    plot_redundancy_heatmap,
    plot_spider,
)
from scripts.diagnostics import build_clinical_audit_table, compute_group_zscore_table
from scripts.performance import encode_labels, run_feature_set_comparison, run_final_model, tune_xgboost
from scripts.embeddings import (
    print_embedding_preview,
    run_extraction_pipeline,
    save_embeddings,
)


def main():
    # ══════════════════════════════════════════════════════════════
    # 1.  METADATA INGESTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 1 - Ingesting metadata")
    print("=" * 60)

    master_metadata = load_master_metadata()
    print(master_metadata.head())

    # ══════════════════════════════════════════════════════════════
    # 2.  ACTIGRAPHY PROCESSING & FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 2 - Running actigraphy pipeline")
    print("=" * 60)

    master_metadata = RunActigraphyPipeline(GROUPS, ACTIGRAPHY_DIR, master_metadata)

    # ══════════════════════════════════════════════════════════════
    # 3.  EXPLORATORY ANALYSIS & QUALITY CHECKS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 3 - Exploratory analysis & sanity checks")
    print("=" * 60)

    run_nan_check(num_nan)
    print(master_metadata.head())

    check_constant_errors(master_metadata)
    check_high_nonwear(master_metadata)
    check_insufficient_data(master_metadata, min_valid_days=3)

    # Filter: keep only participants with >= 3 usable days
    final_master_metadata = master_metadata[
        master_metadata["salvaged_valid_days"] >= 3
    ].copy()

    check_metric_ranges(final_master_metadata)

    # Outlier removal for sample entropy
    final_master_metadata = remove_sample_entropy_outliers(final_master_metadata)
    check_low_sample_entropy(final_master_metadata)
    final_master_metadata.reset_index(drop=True, inplace=True)

    # ══════════════════════════════════════════════════════════════
    # 4.  CORRELATION ANALYSIS & PLOTTING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 4 - Correlation analysis & initial plots")
    print("=" * 60)

    features = ["IS", "IV", "sample_entropy", "relative_amplitude", "L5", "M10"]

    plot_correlation_heatmaps(final_master_metadata, features)
    find_strong_correlations(final_master_metadata, features, CORRELATION_THRESHOLD)
    plot_behavioral_network_vertical_dynamic(
        final_master_metadata, features, threshold=CORRELATION_THRESHOLD
    )

    # ══════════════════════════════════════════════════════════════
    # 5.  TEMPORAL COMPLEXITY (CIRCADIAN WINDOWS)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 5 - Temporal complexity (circadian windows)")
    print("=" * 60)

    final_master_metadata = RunTemporalComplexityPipeline(
        final_master_metadata, ACTIGRAPHY_DIR
    )

    print(
        final_master_metadata[
            ["number", "group", "sampEn_morning", "sampEn_night"]
        ].head()
    )

    print("\nNew Derived Features:")
    print(
        final_master_metadata[
            ["number", "group", "sampEn_delta", "sampEn_ratio", "daytime_volatility"]
        ].head()
    )

    # ══════════════════════════════════════════════════════════════
    # 6.  FEATURE REDUNDANCY CHECK & SPIDER PLOT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 6 - Feature redundancy check & spider plot")
    print("=" * 60)

    check_features = [
        "IS", "IV", "M10", "L5", "relative_amplitude",
        "sampEn_delta", "sampEn_ratio", "daytime_volatility",
    ]
    plot_redundancy_heatmap(final_master_metadata, check_features)

    spider_metrics = [
        "IS", "IV", "L5", "M10",
        "relative_amplitude", "sampEn_delta", "daytime_volatility",
    ]
    spider_labels = ["IS", "IV", "L5", "M10", "Rel_Amp", "SampEn_Delta", "Volatility"]
    plot_spider(final_master_metadata, spider_metrics, spider_labels)

    # ══════════════════════════════════════════════════════════════
    # 7.  GROUP Z-SCORE TABLE (PRIMARY MARKERS)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 7 - Group Z-score / primary marker table")
    print("=" * 60)

    metrics_only = [
        "IS", "IV", "relative_amplitude", "M10", "L5",
        "sampEn_delta", "daytime_volatility",
    ]
    compute_group_zscore_table(final_master_metadata, metrics_only)

    # ══════════════════════════════════════════════════════════════
    # 8.  XGBOOST - FEATURE SET COMPARISON
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 8 - XGBoost feature-set comparison")
    print("=" * 60)

    y, le = encode_labels(final_master_metadata)

    cv_results, y_preds, accuracies = run_feature_set_comparison(
        final_master_metadata, y, FEATURE_SETS
    )

    plot_confusion_matrix_grid(
        y_true=y,
        y_preds=y_preds,
        class_labels=le.classes_,
        accuracies=accuracies,
    )

    # ══════════════════════════════════════════════════════════════
    # 9.  CLINICAL AUDIT TABLE (MISIDENTIFICATION ANALYSIS)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 9 - Clinical audit (full-model misidentifications)")
    print("=" * 60)

    full_model_name = "Full model (all features)"
    y_pred_full     = y_preds[full_model_name]

    build_clinical_audit_table(
        metadata=final_master_metadata,
        y_true=y,
        y_pred=y_pred_full,
        label_encoder=le,
    )

    # ══════════════════════════════════════════════════════════════
    # 10. XGBOOST - HYPERPARAMETER TUNING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 10 - XGBoost hyperparameter tuning")
    print("=" * 60)

    grid_search, X, weights, cv = tune_xgboost(final_master_metadata, y)

    # ══════════════════════════════════════════════════════════════
    # 11. FEATURE IMPORTANCE PLOT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 11 - Feature importance")
    print("=" * 60)

    best_xgb = grid_search.best_estimator_
    best_xgb.fit(X, y, sample_weight=weights)

    plot_feature_importance(X.columns, best_xgb.feature_importances_)

    # ══════════════════════════════════════════════════════════════
    # 12. FINAL TUNED MODEL - PERFORMANCE REPORT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 12 - Final tuned-model performance report")
    print("=" * 60)

    final_best_model, y_pred_final = run_final_model(
        grid_search, X, y, weights, cv, le
    )

    # ══════════════════════════════════════════════════════════════
    # 13. PAT EMBEDDINGS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 13 - PAT embedding extraction")
    print("=" * 60)

    try:
        X_train, X_test, y_train_emb, y_test_emb = run_extraction_pipeline(
            data_root=ACTIGRAPHY_DIR
        )

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape:  {X_test.shape}")

        save_embeddings(X_train, X_test, y_train_emb, y_test_emb)
        print_embedding_preview(X_train)

    except Exception as e:
        print(f"[embeddings] Extraction failed: {e}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()