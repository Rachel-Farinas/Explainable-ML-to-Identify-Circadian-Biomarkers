# metadata_aggregation.py
# Ingests per-group metadata CSVs and builds the master metadata DataFrame.

import glob
import os
import pandas as pd

from .config import METADATA_DIR


def load_master_metadata(metadata_dir: str = METADATA_DIR) -> pd.DataFrame:
    """
    Reads every *-info.csv found in `metadata_dir`, tags each row with its
    group name, and returns the concatenated master metadata DataFrame.
    """
    info_files = glob.glob(os.path.join(metadata_dir, "*-info.csv"))

    metadata_list = []
    for file_path in info_files:
        df = pd.read_csv(file_path)
        file_name  = os.path.basename(file_path)
        group_name = file_name.replace("-info.csv", "")
        df["group"] = group_name
        metadata_list.append(df)

    if not metadata_list:
        raise FileNotFoundError(
            f"No '*-info.csv' files found in '{metadata_dir}'. "
            "Check your METADATA_DIR setting in config.py."
        )

    master_metadata = pd.concat(metadata_list, ignore_index=True)
    print(f"[metadata] Loaded {len(master_metadata)} participants across "
          f"{len(metadata_list)} groups.")
    return master_metadata
