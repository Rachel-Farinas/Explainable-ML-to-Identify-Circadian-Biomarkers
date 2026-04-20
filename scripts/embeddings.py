# embeddings.py
# PAT embedding extraction: data loading, padding, extraction loop, and saving.

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ACTIGRAPHY_DIR,
    EMBEDDINGS_DIR,
    GROUPS,
    PAT_EMBED_DIM,
    PAT_INPUT_SIZE,
    PAT_NUM_LAYERS,
    PAT_PATCH_SIZE,
    PAT_TEST_SIZE,
    PAT_WEIGHTS_PATH,
    RANDOM_STATE,
)
from .transformer_setup import build_encoder_for_extraction


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_and_pad_data(
    file_input,
    target_length: int = PAT_INPUT_SIZE,
) -> np.ndarray:
    """
    Reads a participant CSV (or accepts an already-loaded DataFrame), extracts
    the 'activity_gaps_filled' column, and pads/clips to exactly
    `target_length` minutes.

    Parameters
    ----------
    file_input    : str path to a CSV file  OR  a pandas DataFrame
    target_length : desired output length in minutes (default 10 080 = 7 days)

    Returns
    -------
    float32 numpy array of shape (target_length,)
    """
    if isinstance(file_input, str):
        df = pd.read_csv(file_input)
    else:
        df = file_input

    df.columns = df.columns.str.strip()  # remove hidden whitespace in headers

    column_name = "activity_gaps_filled"

    if column_name not in df.columns:
        print(
            f"❌ Error: Column '{column_name}' not found in {file_input}. "
            f"Available columns: {list(df.columns)}"
        )
        activity_data = df.iloc[:, 2]
    else:
        activity_data = df[column_name]

    activity = pd.to_numeric(activity_data, errors="coerce").fillna(0).values
    activity = activity.astype("float32")

    current_length = len(activity)
    if current_length >= target_length:
        return activity[:target_length]
    padding_needed = target_length - current_length
    return np.pad(activity, (0, padding_needed), "constant", constant_values=0)


# ── Extraction pipeline ───────────────────────────────────────────────────────

def run_extraction_pipeline(
    data_root: str = ACTIGRAPHY_DIR,
    weights_path: str = PAT_WEIGHTS_PATH,
    categories: dict | None = None,
) -> tuple:
    """
    Full PAT embedding extraction pipeline:
      1. Collects all participant file paths and labels from `data_root`.
      2. Performs a stratified 80/20 train/test split by participant.
      3. Builds the PAT encoder and loads pretrained weights.
      4. Extracts embeddings for train and test sets.

    Parameters
    ----------
    data_root    : root directory containing one sub-folder per group
    weights_path : path to the pretrained PAT weights file (.h5)
    categories   : {group_name: label_int} mapping
                   (defaults to control=0, adhd=1, depression=2, schizophrenia=3)

    Returns
    -------
    X_train, X_test : embedding arrays  (N, embed_dim)
    y_train, y_test : integer label arrays
    """
    if categories is None:
        categories = {
            "control":       0,
            "adhd":          1,
            "depression":    2,
            "schizophrenia": 3,
        }

    # -- Collect files and labels
    file_paths = []
    labels     = []

    for cat_name, label_id in categories.items():
        folder = os.path.join(data_root, cat_name)
        if not os.path.isdir(folder):
            print(f"  [embeddings] Folder not found, skipping: {folder}")
            continue
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".csv")
        ]
        file_paths.extend(files)
        labels.extend([label_id] * len(files))

    # -- Train / test split
    train_files, test_files, y_train, y_test = train_test_split(
        file_paths,
        labels,
        test_size=PAT_TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    print(f"Total participants : {len(file_paths)}")
    print(f"Training on        : {len(train_files)}")
    print(f"Testing on         : {len(test_files)}")

    # -- Build model and optionally load weights
    model = build_encoder_for_extraction(
        input_size=PAT_INPUT_SIZE,
        patch_size=PAT_PATCH_SIZE,
        embed_dim=PAT_EMBED_DIM,
        num_layers=PAT_NUM_LAYERS,
    )

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("✅ Pretrained weights loaded successfully. Embeddings are meaningful!")
    else:
        print(
            f"❌ Could not find weights file at '{weights_path}'. "
            "Embeddings will be random (untrained encoder)."
        )

    # -- Extract
    def get_features(file_list: list) -> np.ndarray:
        X = [load_and_pad_data(f) for f in file_list]
        return model.predict(np.array(X))

    X_train = get_features(train_files)
    X_test  = get_features(test_files)

    return X_train, X_test, y_train, y_test


def save_embeddings(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train,
    y_test,
    output_dir: str = EMBEDDINGS_DIR,
) -> None:
    """
    Saves the four embedding arrays as .npy files in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train_embeddings.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test_embeddings.npy"),  X_test)
    np.save(os.path.join(output_dir, "y_train.npy"),            np.array(y_train))
    np.save(os.path.join(output_dir, "y_test.npy"),             np.array(y_test))

    print(f"✅ Embeddings saved to '{output_dir}'")
    print(f"   X_train shape : {X_train.shape}")
    print(f"   X_test shape  : {X_test.shape}")


def print_embedding_preview(X_train: np.ndarray) -> None:
    """Prints the first few rows of the embedding matrix."""
    df_embeddings = pd.DataFrame(X_train)
    print("\nEmbedding Matrix Preview:")
    print(df_embeddings.head())