# config.py
# Central configuration: paths, constants, and pipeline settings.

import os

# BASE_DIR = the project root (the folder containing main.py).
# __file__ is  <project_root>/scripts/config.py
# dirname once  -> <project_root>/scripts
# dirname twice -> <project_root>
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Input directories ─────────────────────────────────────────────────────────
ACTIGRAPHY_DIR = os.path.join(BASE_DIR, "actigraphy")   # /actigraphy/<group>/
METADATA_DIR   = os.path.join(BASE_DIR, "metadata")     # /metadata/<group>-info.csv

# ── Output directories ────────────────────────────────────────────────────────
PLOTS_DIR      = os.path.join(BASE_DIR, "plots")        # /plots/
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")   # /embeddings/

# ── PAT model weights ─────────────────────────────────────────────────────────
PAT_REPO_DIR     = os.path.join(BASE_DIR, "Pretrained-Actigraphy-Transformer")
PAT_WEIGHTS_PATH = os.path.join(PAT_REPO_DIR, "weights", "PAT-L_29k_weights.h5")

# ── Actigraphy groups ─────────────────────────────────────────────────────────
GROUPS = ["adhd", "schizophrenia", "depression", "control", "clinical"]

# ── Non-wear detection thresholds ─────────────────────────────────────────────
NONWEAR_ZERO_THRESHOLD   = 60    # consecutive zero-activity minutes -> non-wear
NONWEAR_MAX_INACTIVITY   = 0.3   # participant-level non-wear ratio ceiling
NONWEAR_DAILY_LIMIT      = 0.2   # per-day non-wear ratio ceiling

# ── Data cleaning thresholds ──────────────────────────────────────────────────
CONSTANT_ERROR_DURATION  = 360   # minutes of constant non-zero -> technical fault
MIN_VALID_DAYS           = 3     # minimum usable days after cleaning

# ── Sample-entropy outlier removal ────────────────────────────────────────────
ZSCORE_OUTLIER_THRESHOLD = 3     # |z| > 3 -> outlier

# ── Circadian window definitions ──────────────────────────────────────────────
CIRCADIAN_BINS   = [0, 6, 12, 18, 24]
CIRCADIAN_LABELS = ["night", "morning", "afternoon", "evening"]

# ── PAT / embedding settings ──────────────────────────────────────────────────
PAT_INPUT_SIZE  = 10080   # 7 days x 1440 min/day
PAT_PATCH_SIZE  = 9
PAT_EMBED_DIM   = 96
PAT_NUM_LAYERS  = 4
PAT_TEST_SIZE   = 0.20
RANDOM_STATE    = 42

# ── XGBoost feature sets ──────────────────────────────────────────────────────
FEATURE_SETS = {
    "Base features (IS, IV, RA)":
        ["IS", "IV", "relative_amplitude"],

    "Base features + L5 and M10":
        ["IS", "IV", "relative_amplitude", "M10", "L5"],

    "Base features + sampEn_delta and daytime_volatility":
        ["IS", "IV", "relative_amplitude", "sampEn_delta", "daytime_volatility"],

    "Full model (all features)":
        ["IS", "IV", "relative_amplitude", "M10", "L5", "sampEn_delta", "daytime_volatility"],
}

# ── XGBoost hyperparameter search grid ───────────────────────────────────────
XGBOOST_PARAM_GRID = {
    "max_depth":        [3, 4, 5, 6, 7, 8, 10],
    "learning_rate":    [0.01, 0.05, 0.1, 0.001, 0.005],
    "n_estimators":     [100, 200],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ── Correlation analysis ──────────────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.6

# ── Ensure output directories exist ──────────────────────────────────────────
os.makedirs(PLOTS_DIR,      exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)