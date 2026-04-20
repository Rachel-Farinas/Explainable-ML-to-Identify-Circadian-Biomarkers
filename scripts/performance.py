# performance.py
# XGBoost model training, cross-validation, hyperparameter tuning,
# and classification reporting.

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from .config import FEATURE_SETS, RANDOM_STATE, XGBOOST_PARAM_GRID


def encode_labels(metadata: pd.DataFrame) -> tuple:
    """
    Encodes the 'group' column into integer labels.

    Returns
    -------
    y  : numpy array of encoded labels
    le : fitted LabelEncoder (needed to decode predictions later)
    """
    le = LabelEncoder()
    y  = le.fit_transform(metadata["group"])
    return y, le


def run_feature_set_comparison(
    metadata: pd.DataFrame,
    y,
    feature_sets: dict | None = None,
) -> tuple[list[dict], dict, dict]:
    """
    Runs 5-fold stratified cross-validation for each feature set, collects
    accuracy stats, and generates cross-validated predictions for confusion
    matrices.

    Returns
    -------
    cv_results   : list of dicts with per-set stats
    y_preds      : {set_name: y_pred_array}
    accuracies   : {set_name: overall_accuracy}
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        tree_method="hist",
        device="cpu",
        eval_metric="mlogloss",
    )

    cv_results = []
    y_preds    = {}
    accuracies = {}

    for name, feats in feature_sets.items():
        X_subset   = metadata[feats]
        stats      = cross_validate(model, X_subset, y, cv=cv, scoring="accuracy")
        y_pred_cv  = cross_val_predict(model, X_subset, y, cv=cv)
        acc        = accuracy_score(y, y_pred_cv)

        cv_results.append({
            "Feature Set":       name,
            "Mean Accuracy":     stats["test_score"].mean(),
            "Standard Deviation": stats["test_score"].std(),
            "Best Fold":         stats["test_score"].max(),
            "Overall Accuracy":  acc,
        })

        y_preds[name]    = y_pred_cv
        accuracies[name] = acc

    tournament_table = (
        pd.DataFrame(cv_results)
        .sort_values(by="Mean Accuracy", ascending=False)
    )
    print("\n--- Feature Set Tournament ---")
    print(tournament_table.round(3).to_string())

    return cv_results, y_preds, accuracies


def tune_xgboost(
    metadata: pd.DataFrame,
    y,
    feature_set_name: str = "Full model (all features)",
    feature_sets: dict | None = None,
    param_grid: dict | None = None,
) -> tuple:
    """
    Applies class-balanced sample weights and runs GridSearchCV to tune
    XGBoost hyperparameters.

    Returns
    -------
    grid_search  : fitted GridSearchCV object
    X            : feature matrix used for training
    weights      : sample-weight array
    cv           : the StratifiedKFold object used
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS
    if param_grid is None:
        param_grid = XGBOOST_PARAM_GRID

    X = metadata[feature_sets[feature_set_name]]

    weights = class_weight.compute_sample_weight(class_weight="balanced", y=y)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        verbose=1,
    )

    grid_search.fit(X, y, sample_weight=weights)

    print(f"\nNew Tuned Accuracy: {grid_search.best_score_:.2%}")
    print(f"Best Parameters:   {grid_search.best_params_}")

    return grid_search, X, weights, cv


def run_final_model(
    grid_search,
    X,
    y,
    weights,
    cv,
    le: LabelEncoder,
) -> tuple:
    """
    Retrains the best estimator on the full data, generates cross-validated
    predictions, and prints the final classification report.

    Returns
    -------
    final_best_model : fitted XGBClassifier
    y_pred_final     : cross-validated predictions
    """
    final_best_model = xgb.XGBClassifier(
        max_depth=grid_search.best_params_["max_depth"],
        learning_rate=grid_search.best_params_["learning_rate"],
        n_estimators=grid_search.best_params_["n_estimators"],
        subsample=grid_search.best_params_["subsample"],
        colsample_bytree=grid_search.best_params_["colsample_bytree"],
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    y_pred_final = cross_val_predict(
        final_best_model,
        X,
        y,
        cv=cv,
        params={"sample_weight": weights},
    )

    report = classification_report(y, y_pred_final, target_names=le.classes_)
    print("\n--- Final Model Diagnostic Performance ---")
    print(report)

    # Refit on full data for feature-importance extraction
    final_best_model.fit(X, y, sample_weight=weights)

    return final_best_model, y_pred_final
