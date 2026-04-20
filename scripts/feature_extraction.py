# feature_extraction.py
# Circadian-rhythm metrics (IS, IV, L5, M10, RA, SampEn) and the participant
# pipeline that calls preprocessing and writes cleaned CSVs back to disk.

import glob
import os

import antropy as ant
import numpy as np
import pandas as pd

from .config import ACTIGRAPHY_DIR, GROUPS, MIN_VALID_DAYS
from .preprocessing import CleanParticipantData, DetectNonWearTime, ImputeShortStillness


# ── Global NaN tracker (populated by ProcessSingleParticipant) ────────────────
num_nan: dict[str, int] = {}


# ── Circadian-metric helpers ──────────────────────────────────────────────────

def CalculateHourlyMean(df: pd.DataFrame) -> np.ndarray:
    """Returns a length-24 array of mean activity (one value per hour)."""
    df["hour_of_day"] = (df.index // 60) % 24
    hourly_means = (
        df.groupby("hour_of_day")["activity_cleaned"]
        .mean()
        .reindex(range(24))
    )
    return hourly_means.values


def CalculateIntradailyVariability(df: pd.DataFrame) -> float:
    """IV: measures fragmentation of the rest-activity rhythm."""
    hourly_data  = df.groupby(df.index // 60)["activity_cleaned"].mean()
    diffs        = hourly_data.diff().dropna()
    numerator    = (diffs.pow(2)).sum()
    overall_mean = df["activity_cleaned"].mean()
    denominator  = ((hourly_data.dropna() - overall_mean) ** 2).sum()
    n            = hourly_data.notna().sum()

    if n <= 1 or denominator == 0:
        return np.nan
    return (n * numerator) / ((n - 1) * denominator)


def CalculateInterdailyStability(df: pd.DataFrame) -> float:
    """IS: measures day-to-day regularity of the rest-activity rhythm."""
    hourly_bins  = 24
    total_mean   = df["activity_cleaned"].mean()
    hourly_mean_list = CalculateHourlyMean(df)

    routine_strength = np.nansum((hourly_mean_list - total_mean) ** 2)

    valid_minutes    = df["activity_cleaned"].dropna()
    overall_variance = ((valid_minutes - total_mean) ** 2).sum()
    valid_n          = len(valid_minutes)

    if overall_variance == 0 or valid_n == 0:
        return np.nan
    return (valid_n * routine_strength) / (hourly_bins * overall_variance)


def CalculateL5(df: pd.DataFrame) -> float:
    """L5: mean activity of the least-active 5-hour window."""
    series          = pd.Series(CalculateHourlyMean(df))
    extended_series = pd.concat([series, series])
    return extended_series.rolling(window=5).mean().dropna().min()


def CalculateM10(df: pd.DataFrame) -> float:
    """M10: mean activity of the most-active 10-hour window."""
    series          = pd.Series(CalculateHourlyMean(df))
    extended_series = pd.concat([series, series])
    return extended_series.rolling(window=10).mean().dropna().max()


def CalculateRelativeAmplitude(L5: float, M10: float) -> float:
    """RA: strength of the sleep/wake cycle; ranges 0–1."""
    denominator = M10 + L5
    if denominator == 0 or np.isnan(denominator):
        return 0.0
    return (M10 - L5) / denominator


def CalculateSampleEntropy(df: pd.DataFrame) -> float:
    """Sample entropy of 'activity_gaps_filled'; requires ≥ 1440 valid points."""
    valid_data = df["activity_gaps_filled"].dropna().values
    if len(valid_data) < 1440:
        return np.nan
    try:
        return float(ant.sample_entropy(valid_data))
    except Exception:
        return np.nan


# ── Per-participant pipeline ──────────────────────────────────────────────────

def ProcessSingleParticipant(file_path: str) -> dict:
    """
    Reads a participant CSV, applies the full cleaning pipeline, computes all
    circadian metrics, writes the cleaned CSV back to disk, and returns a
    results dict keyed by participant number.
    """
    df      = pd.read_csv(file_path)
    file_id = os.path.basename(file_path).replace(".csv", "")

    num_nan[file_id] = df["activity"].isna().sum()

    # Step 1 – non-wear detection
    df, meets_threshold, non_wear_ratio, initial_valid_days = DetectNonWearTime(df)

    # Step 2 – remove frozen-value days
    df, salvaged_valid_days = CleanParticipantData(df, duration_threshold=360)

    # Step 3 – build 'activity_cleaned' (NaN for bad rows)
    df["activity_cleaned"] = df["activity"].copy()
    invalid_mask = (df["non_wear"] == True) | (df["is_usable_day"] == False)
    df.loc[invalid_mask, "activity_cleaned"] = np.nan

    # Step 4 – gap filling
    df["activity_gaps_filled"] = df["activity_cleaned"].interpolate(
        method="linear", limit=5
    )
    df = ImputeShortStillness(df)

    # Step 5 – feature extraction (only if ≥ MIN_VALID_DAYS)
    has_enough_data = salvaged_valid_days >= MIN_VALID_DAYS

    df.to_csv(file_path, index=False)

    l5  = float(CalculateL5(df))  if has_enough_data else np.nan
    m10 = float(CalculateM10(df)) if has_enough_data else np.nan

    results = {
        "number":                       file_id,
        "meets_activity_threshold":     meets_threshold,
        "non_wear_ratio":               non_wear_ratio,
        "original_valid_days":          initial_valid_days,
        "salvaged_valid_days":          salvaged_valid_days,
        "contains_constant_error_values": (salvaged_valid_days < initial_valid_days),
        "IS":               float(CalculateInterdailyStability(df)) if has_enough_data else np.nan,
        "IV":               float(CalculateIntradailyVariability(df)) if has_enough_data else np.nan,
        "L5":               l5,
        "M10":              m10,
        "relative_amplitude": float(CalculateRelativeAmplitude(l5, m10)) if has_enough_data else np.nan,
        "sample_entropy":   float(CalculateSampleEntropy(df)) if has_enough_data else np.nan,
    }
    return results


def RunActigraphyPipeline(
    groups: list[str],
    base_dir: str,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Iterates through group folders, runs ProcessSingleParticipant on every
    CSV, and writes results back into `metadata_df`.
    """
    for group in groups:
        target_folder = os.path.join(base_dir, group)
        csv_files     = glob.glob(os.path.join(target_folder, "*.csv"))

        print(f"--- Processing Group: {group} ({len(csv_files)} files found) ---")

        for file_path in csv_files:
            try:
                stats = ProcessSingleParticipant(file_path)
                mask  = metadata_df["number"] == stats["number"]
                for key, value in stats.items():
                    if key != "number":
                        metadata_df.loc[mask, key] = value
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

    return metadata_df


# ── Temporal (circadian-window) complexity ────────────────────────────────────

def CalculateTemporalComplexity(file_path: str) -> dict | None:
    """
    Cleans data and calculates Sample Entropy for the four circadian windows
    (night / morning / afternoon / evening).
    """
    from .config import CIRCADIAN_BINS, CIRCADIAN_LABELS

    try:
        df = pd.read_csv(file_path)

        df, _, _, _ = DetectNonWearTime(df)
        df, _       = CleanParticipantData(df)

        invalid_mask = (df["non_wear"] == True) | (df["is_usable_day"] == False)
        df.loc[invalid_mask, "activity"] = np.nan
        df["activity_filled"] = df["activity"].interpolate(method="linear", limit=5)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]      = df["timestamp"].dt.hour

        df["window"] = pd.cut(
            df["hour"],
            bins=CIRCADIAN_BINS,
            labels=CIRCADIAN_LABELS,
            right=False,
        )

        temporal_results = {}
        for window in CIRCADIAN_LABELS:
            signal = df[df["window"] == window]["activity_filled"].dropna().values
            if len(signal) > 100:
                try:
                    temporal_results[f"sampEn_{window}"] = ant.sample_entropy(signal)
                except Exception:
                    temporal_results[f"sampEn_{window}"] = np.nan
            else:
                temporal_results[f"sampEn_{window}"] = np.nan

        return temporal_results

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return None


def RunTemporalComplexityPipeline(
    final_metadata: pd.DataFrame,
    base_actigraphy_dir: str = ACTIGRAPHY_DIR,
) -> pd.DataFrame:
    """
    Iterates over participants in `final_metadata`, computes circadian-window
    Sample Entropy, and merges results back. Also derives sampEn_delta,
    sampEn_ratio, and daytime_volatility.
    """
    temporal_data_list = []
    success_count      = 0

    print("Starting Temporal Complexity Analysis...")

    for _, row in final_metadata.iterrows():
        participant_id = str(row["number"])
        group_folder   = str(row["group"]).lower()

        file_path = os.path.join(base_actigraphy_dir, group_folder, f"{participant_id}.csv")

        if not os.path.exists(file_path):
            # Try original case as fallback
            file_path = os.path.join(base_actigraphy_dir, str(row["group"]), f"{participant_id}.csv")

        if os.path.exists(file_path):
            features = CalculateTemporalComplexity(file_path)
            if features:
                features["number"] = participant_id
                temporal_data_list.append(features)
                success_count += 1

    print(f"\nProcessing complete. Successfully analyzed {success_count} participants.")

    if success_count == 0:
        print("No temporal data processed. Check if actigraphy folder path is correct.")
        return final_metadata

    temporal_df = pd.DataFrame(temporal_data_list)

    if "number" in final_metadata.columns and "number" in temporal_df.columns:
        temporal_df   = temporal_df.drop_duplicates(subset=["number"])
        final_metadata = final_metadata.merge(temporal_df, on="number", how="left")
        print("Success! Temporal SampEn merged into metadata.")
    else:
        print("Merge failed: 'number' column missing. Check column names.")
        return final_metadata

    # Derived circadian features
    daytime_windows = ["sampEn_morning", "sampEn_afternoon", "sampEn_evening"]

    final_metadata["sampEn_delta"] = (
        final_metadata["sampEn_morning"] - final_metadata["sampEn_night"]
    )
    final_metadata["sampEn_ratio"] = np.where(
        final_metadata["sampEn_night"] > 0,
        final_metadata["sampEn_morning"] / final_metadata["sampEn_night"],
        np.nan,
    )
    final_metadata["daytime_volatility"] = final_metadata[daytime_windows].std(axis=1)
    final_metadata.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("New Features Generated:")
    print(
        final_metadata[
            ["number", "group", "sampEn_delta", "sampEn_ratio", "daytime_volatility"]
        ].head()
    )

    return final_metadata