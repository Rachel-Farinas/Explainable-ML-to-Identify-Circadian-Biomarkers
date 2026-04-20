# preprocessing.py
# Non-wear detection, constant-value cleaning, and gap imputation.

import numpy as np
import pandas as pd

from .config import (
    NONWEAR_ZERO_THRESHOLD,
    NONWEAR_MAX_INACTIVITY,
    NONWEAR_DAILY_LIMIT,
    CONSTANT_ERROR_DURATION,
)


def DetectNonWearTime(
    df: pd.DataFrame,
    zero_threshold: int  = NONWEAR_ZERO_THRESHOLD,
    max_inactivity: float = NONWEAR_MAX_INACTIVITY,
    daily_non_wear_limit: float = NONWEAR_DAILY_LIMIT,
):
    """
    Flags rolling windows of `zero_threshold` consecutive zero-activity
    minutes as non-wear.

    Returns
    -------
    df                     : DataFrame with a new 'non_wear' bool column
    meets_activity_threshold : True if overall non-wear ratio ≤ max_inactivity
    non_wear_ratio         : fraction of total rows flagged as non-wear
    valid_days_count       : number of days whose per-day non-wear ≤ daily_non_wear_limit
    """
    is_zero    = df["activity"] == 0
    zero_streaks = is_zero.rolling(window=zero_threshold, min_periods=1).sum()
    df["non_wear"] = zero_streaks == zero_threshold

    non_wear_ratio = df["non_wear"].sum() / len(df)
    meets_activity_threshold = non_wear_ratio <= max_inactivity

    daily_non_wear_ratio = df.groupby("date")["non_wear"].mean()
    valid_days_count     = int((daily_non_wear_ratio <= daily_non_wear_limit).sum())

    return df, meets_activity_threshold, non_wear_ratio, valid_days_count


def ImputeShortStillness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills short zero/NaN gaps in 'activity_cleaned' with the hourly median,
    but only for rows that are NOT already flagged as non-wear.
    Result is stored in 'activity_gaps_filled'.
    """
    df["hour"]       = (df.index // 60) % 24
    hourly_stats     = df.groupby("hour")["activity"].transform("median")

    df["activity_gaps_filled"] = df["activity_cleaned"].copy()

    is_not_non_wear = df["non_wear"] == False
    is_hole         = df["activity_gaps_filled"].isna() | (df["activity_gaps_filled"] == 0)

    df.loc[is_not_non_wear & is_hole, "activity_gaps_filled"] = hourly_stats

    return df


def CleanParticipantData(
    df: pd.DataFrame,
    duration_threshold: int = CONSTANT_ERROR_DURATION,
):
    """
    Detects 'frozen' (constant non-zero) blocks of ≥ `duration_threshold`
    minutes, marks those DAYS as unusable, and returns only usable rows.

    Returns
    -------
    clean_df_segment   : DataFrame restricted to usable days
    salvaged_valid_days : number of days with per-day non-wear ≤ 0.2
    """
    value_changes  = df["activity"].diff() != 0
    blocks         = value_changes.cumsum()
    block_lengths  = df.groupby(blocks)["activity"].transform("count")

    df["technical_fault"] = (block_lengths >= duration_threshold) & (df["activity"] != 0)

    bad_dates        = df[df["technical_fault"] == True]["date"].unique()
    df["is_usable_day"] = ~df["date"].isin(bad_dates)

    clean_df_segment = df[df["is_usable_day"] == True].copy()

    daily_nonwear        = clean_df_segment.groupby("date")["non_wear"].mean()
    salvaged_valid_days  = int((daily_nonwear <= 0.2).sum())

    return clean_df_segment, salvaged_valid_days
