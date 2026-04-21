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
#   │   └── schizophrenia/  *.csv
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

import math
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
from sklearn.preprocessing import LabelEncoder
from scripts.embeddings import (
    print_embedding_preview,
    run_extraction_pipeline,
    save_embeddings,
)
from scripts.embedding_analysis import (
    load_embeddings,
    plot_tsne,
    plot_umap,
    run_embedding_classification,
    plot_embedding_vs_handcrafted,
    run_similarity_analysis,
    plot_similarity_distributions,
)


def main():
    # ══════════════════════════════════════════════════════════════
    # 1.  METADATA INGESTION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 1 - Ingesting metadata")
    print("=" * 60)

    master_metadata = load_master_metadata()

    # Convert age ranges (e.g. "25-34") to a rounded-up midpoint stored in
    # a new column "age_estimated" so the original "age" column is untouched.
    if "age" in master_metadata.columns:
        def parse_age(val):
            try:
                s = str(val).strip()
                if "-" in s:
                    lo, hi = s.split("-")
                    return math.ceil((float(lo) + float(hi)) / 2)
                return math.ceil(float(s))
            except (ValueError, TypeError):
                return float("nan")
        master_metadata["age_estimated"] = master_metadata["age"].apply(parse_age)
    else:
        print("[main] Warning: 'age' column missing from metadata — adding age_estimated as NaN.")
        master_metadata["age_estimated"] = float("nan")

    # Ensure gender is numeric (1 = female, 2 = male)
    if "gender" in master_metadata.columns:
        master_metadata["gender"] = pd.to_numeric(master_metadata["gender"], errors="coerce")
    else:
        print("[main] Warning: 'gender' column missing from metadata — adding as NaN.")
        master_metadata["gender"] = float("nan")

    # Impute any remaining NaNs with group median,
    # falling back to global median if an entire group is missing.
    for col in ["age_estimated", "gender"]:
        master_metadata[col] = master_metadata.groupby("group")[col].transform(
            lambda x: x.fillna(x.median())
        )
        master_metadata[col] = master_metadata[col].fillna(master_metadata[col].median())

    print(f"[main] age_estimated sample: {master_metadata['age_estimated'].head().tolist()}")

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
        "age_estimated", "gender",
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
        "sampEn_delta", "daytime_volatility", "age_estimated", "gender",
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
        display_cols=["Actual", "Status", "Predicted", "age_estimated", "gender",
                      "M10", "sampEn_delta", "daytime_volatility", "IV"],
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

    # ══════════════════════════════════════════════════════════════
    # 14. EMBEDDING ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("STEP 14 - Embedding analysis")
    print("=" * 60)

    try:
        X_train_emb, X_test_emb, y_train_emb, y_test_emb = load_embeddings()

        # Combine train + test for visualization and similarity
        X_all = np.concatenate([X_train_emb, X_test_emb], axis=0)
        y_all = np.concatenate([y_train_emb, y_test_emb], axis=0)

        # Map integer labels to group name strings
        category_map  = {0: "control", 1: "adhd", 2: "depression", 3: "schizophrenia"}
        y_all_named   = np.array([category_map[int(i)] for i in y_all])
        y_train_named = np.array([category_map[int(i)] for i in y_train_emb])
        y_test_named  = np.array([category_map[int(i)] for i in y_test_emb])

        le_emb = LabelEncoder()
        le_emb.fit(y_all_named)
        y_all_enc   = le_emb.transform(y_all_named)
        y_train_enc = le_emb.transform(y_train_named)
        y_test_enc  = le_emb.transform(y_test_named)

        # 1. Visualisation
        plot_tsne(X_all, y_all_enc, le_emb)
        plot_umap(X_all, y_all_enc, le_emb)

        # 2. Classification — compare PAT embeddings vs handcrafted features
        summary = run_embedding_classification(
            X_train_emb, X_test_emb, y_train_enc, y_test_enc, le_emb
        )
        best_handcrafted = max(accuracies.values())
        embedding_accs   = dict(zip(summary["Classifier"], summary["Test Accuracy"]))
        plot_embedding_vs_handcrafted(embedding_accs, best_handcrafted)

        # 3. Similarity — group cohesion and atypical participants
        sim_df = run_similarity_analysis(X_all, y_all_enc, le_emb)
        plot_similarity_distributions(sim_df, le_emb)

    except FileNotFoundError:
        print("[embedding_analysis] No embedding files found — run Step 13 first.")
    except Exception as e:
        print(f"[embedding_analysis] Analysis failed: {e}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()