# Explainable ML to Identify Circadian Biomarkers

A modular Python pipeline for processing wrist actigraphy data across clinical groups (ADHD, depression, schizophrenia, and healthy controls) to extract circadian-rhythm features, detect behavioral signatures, and classify diagnostic groups using XGBoost and a Pretrained Actigraphy Transformer (PAT).

---

## Overview

Wrist actigraphy records minute-by-minute movement data over multiple days. This pipeline ingests raw actigraphy CSVs, cleans them for non-wear time and sensor faults, extracts a suite of circadian and complexity metrics, and feeds them into both a classical ML classifier (XGBoost) and a deep-learning transformer encoder (PAT) to generate participant embeddings.

**Clinical groups supported:** `adhd`, `schizophrenia`, `depression`, `control`, `clinical`

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Ingest per-group metadata CSVs into a master DataFrame |
| 2 | Run actigraphy preprocessing and feature extraction per participant |
| 3 | Exploratory analysis: NaN checks, data quality flags, outlier removal |
| 4 | Per-group correlation heatmaps and behavioral network graphs |
| 5 | Temporal complexity analysis across four circadian windows |
| 6 | Feature redundancy check and spider/radar plot |
| 7 | Group Z-score table to identify primary behavioral markers |
| 8 | XGBoost feature-set comparison with 5-fold cross-validation |
| 9 | Clinical audit table of model hits and misidentifications |
| 10 | XGBoost hyperparameter tuning via GridSearchCV |
| 11 | Feature importance visualization |
| 12 | Final tuned model performance report |
| 13 | PAT transformer embedding extraction and save |

---

## Features Extracted

**Circadian rhythm metrics:**
- **IS** — Interdaily Stability: day-to-day regularity of the rest-activity rhythm
- **IV** — Intradaily Variability: fragmentation of the rest-activity rhythm
- **L5** — Mean activity of the least-active 5-hour window
- **M10** — Mean activity of the most-active 10-hour window
- **RA** — Relative Amplitude: strength of the sleep/wake cycle

**Complexity metrics:**
- **Sample Entropy** — overall signal complexity
- **sampEn_delta** — morning vs. night entropy difference (circadian delta)
- **sampEn_ratio** — morning / night entropy ratio
- **daytime_volatility** — std of entropy across morning, afternoon, and evening windows

---

## Project Structure

```
OBF/
├── main.py                          # Entry point — run this
├── scripts/
│   ├── __init__.py
│   ├── config.py                    # All paths, thresholds, and constants
│   ├── metadata_aggregation.py      # Loads per-group metadata CSVs
│   ├── preprocessing.py             # Non-wear detection, cleaning, gap imputation
│   ├── feature_extraction.py        # Circadian metrics + temporal complexity pipeline
│   ├── exploratory_analysis.py      # Sanity checks, outlier removal
│   ├── plotting.py                  # All visualizations
│   ├── diagnostics.py               # Z-score table, clinical audit table
│   ├── performance.py               # XGBoost CV, tuning, classification report
│   ├── transformer_setup.py         # PAT TransformerBlock + encoder architecture
│   └── embeddings.py                # PAT embedding extraction and saving
├── actigraphy/
│   ├── adhd/                        # One CSV per participant
│   ├── control/
│   ├── depression/
│   ├── schizophrenia/
│   └── clinical/
├── metadata/
│   ├── adhd-info.csv                # One row per participant
│   ├── control-info.csv
│   ├── depression-info.csv
│   ├── schizophrenia-info.csv
│   └── clinical-info.csv
├── Pretrained-Actigraphy-Transformer/
│   └── weights/
│       └── PAT-L_29k_weights.h5     # Pretrained PAT weights
├── plots/                           # Generated automatically
└── embeddings/                      # Generated automatically
```

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn \
            xgboost antropy networkx tensorflow keras antropy
```

**3. Clone the PAT repository and download weights:**
```bash
git clone https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer.git
```
Place the downloaded `PAT-L_29k_weights.h5` file at:
```
Pretrained-Actigraphy-Transformer/weights/PAT-L_29k_weights.h5
```

---

## Data Format

**Actigraphy CSVs** (`actigraphy/<group>/<participant_id>.csv`):

Each file represents one participant. Required columns:
| Column | Description |
|--------|-------------|
| `activity` | Raw activity count per minute |
| `date` | Date of the recording |
| `timestamp` | Full datetime timestamp (used for circadian windowing) |

**Metadata CSVs** (`metadata/<group>-info.csv`):

One row per participant. Required column:
| Column | Description |
|--------|-------------|
| `number` | Participant ID matching the actigraphy filename (without `.csv`) |

Additional demographic or clinical columns are preserved through the pipeline.

---

## Usage

From the project root directory:

```bash
python main.py
```

All plots are saved to `plots/` and all embeddings are saved to `embeddings/` automatically.

---

## Configuration

All paths, thresholds, and model settings are centralized in `scripts/config.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `NONWEAR_ZERO_THRESHOLD` | `60` | Consecutive zero-activity minutes to flag as non-wear |
| `NONWEAR_MAX_INACTIVITY` | `0.3` | Max overall non-wear ratio before participant is flagged |
| `CONSTANT_ERROR_DURATION` | `360` | Minutes of constant non-zero activity to flag as sensor fault |
| `MIN_VALID_DAYS` | `3` | Minimum usable days required to include a participant |
| `ZSCORE_OUTLIER_THRESHOLD` | `3` | Z-score threshold for sample entropy outlier removal |
| `PAT_INPUT_SIZE` | `10080` | Input length in minutes (7 days × 1440 min/day) |
| `CORRELATION_THRESHOLD` | `0.6` | Minimum \|r\| to display in network graphs |

---

## Outputs

**Plots** (saved to `plots/`):
- `vertical_monochrome_signature_grid.png` — per-group correlation heatmaps
- `dynamic_behavioral_networks.png` — correlation network graphs per group
- `feature_redundancy_heatmap.png` — global vs. circadian feature redundancy check
- `spider_behavioral_groups.png` — radar chart comparing normalized group metrics
- `confusion_matrices.png` — 2×2 grid of XGBoost confusion matrices per feature set
- `feature_importance.png` — XGBoost feature importance bar chart

**Embeddings** (saved to `embeddings/`):
- `X_train_embeddings.npy` — PAT embedding matrix for training participants
- `X_test_embeddings.npy` — PAT embedding matrix for test participants
- `y_train.npy` — integer labels for training set
- `y_test.npy` — integer labels for test set

---

## Models

### XGBoost Classifier
Four feature sets are compared using stratified 5-fold cross-validation:
1. Base features only (IS, IV, RA)
2. Base + L5 and M10
3. Base + circadian features (sampEn_delta, daytime_volatility)
4. Full model (all features)

The best feature set is then tuned via `GridSearchCV` with class-balanced sample weights to handle class imbalance.

### Pretrained Actigraphy Transformer (PAT)
The PAT encoder patches 7 days of actigraphy (10,080 minutes) into 70 patches of 144 minutes each, projects them into a 128-dimensional embedding space, and passes them through 2 transformer blocks before global average pooling to produce a single 128-dimensional embedding per participant.

Pretrained weights from [njacobsonlab/Pretrained-Actigraphy-Transformer](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer) are loaded if available.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data handling |
| `matplotlib`, `seaborn` | Plotting |
| `scipy` | Z-score computation |
| `scikit-learn` | Cross-validation, label encoding, confusion matrices |
| `xgboost` | Classification |
| `antropy` | Sample entropy calculation |
| `networkx` | Behavioral network graphs |
| `tensorflow` / `keras` | PAT transformer encoder |
