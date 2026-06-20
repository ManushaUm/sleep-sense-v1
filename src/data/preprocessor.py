"""
preprocessor.py — Timestamp alignment, label encoding, and data cleaning.

Handles:
  - Unix timestamp → datetime + date string conversion
  - EMA quality label → numeric score (0–3)
  - Outlier timestamp removal
  - Missing value strategies per modality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── EMA label encoding ────────────────────────────────────────────────────────

EMA_RATE_MAP = {
    "Very good": 3,
    "Fairly good": 2,
    "Fairly bad": 1,
    "Very bad": 0,
}

SCORE_TO_LABEL = {
    3: "Very good",
    2: "Fairly good",
    1: "Fairly bad",
    0: "Very bad",
}


def encode_sleep_rate(rate_str: str) -> int | None:
    """Map EMA sleep rate string to integer score (0–3). Returns None if unknown."""
    return EMA_RATE_MAP.get(rate_str, None)


def score_to_label(score: float) -> str:
    """Convert a continuous prediction score (0.0–3.0) to a quality label."""
    if score >= 2.5:
        return "Very good"
    elif score >= 1.5:
        return "Fairly good"
    elif score >= 0.5:
        return "Fairly bad"
    else:
        return "Very bad"


# ── Timestamp utilities ───────────────────────────────────────────────────────

def ts_to_datetime(ts: int) -> datetime:
    """Convert Unix timestamp (seconds) to UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def ts_to_date_str(ts: int) -> str:
    """Convert Unix timestamp to 'YYYY-MM-DD' string (UTC)."""
    return ts_to_datetime(ts).strftime("%Y-%m-%d")


def ts_to_hour(ts: int) -> float:
    """Extract hour of day (0–23.999) from Unix timestamp."""
    dt = ts_to_datetime(ts)
    return dt.hour + dt.minute / 60.0


def filter_daytime(df: pd.DataFrame, ts_col: str = "timestamp",
                   start_hour: int = 6, end_hour: int = 22) -> pd.DataFrame:
    """Keep only rows where timestamp falls within [start_hour, end_hour)."""
    df = add_hour_column(df, ts_col)
    return df[(df["hour"] >= start_hour) & (df["hour"] < end_hour)].copy()


def filter_evening(df: pd.DataFrame, ts_col: str = "timestamp",
                   start_hour: int = 20, end_hour: int = 22) -> pd.DataFrame:
    """Keep only rows in the evening window [start_hour, end_hour)."""
    return filter_daytime(df, ts_col, start_hour, end_hour)


def filter_late_night(df: pd.DataFrame, ts_col: str = "timestamp",
                      start_hour: int = 22) -> pd.DataFrame:
    """Keep only rows after start_hour (default 22:00)."""
    df = add_hour_column(df, ts_col)
    return df[df["hour"] >= start_hour].copy()


# ── Data cleaning ─────────────────────────────────────────────────────────────

# StudentLife study window: ~March–June 2013
STUDY_START_TS = 1362096000   # 2013-03-01 00:00:00 UTC
STUDY_END_TS   = 1372636800   # 2013-07-01 00:00:00 UTC


def remove_outlier_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Remove rows with timestamps outside the StudentLife study window."""
    return df[
        (df[ts_col] >= STUDY_START_TS) & (df[ts_col] <= STUDY_END_TS)
    ].copy()


def add_date_column(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Add a 'date' column (YYYY-MM-DD string) derived from timestamp column."""
    if "date" in df.columns:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df[ts_col], unit='s', utc=True).dt.strftime('%Y-%m-%d')
    return df


def add_hour_column(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Add an 'hour' column (0.0–23.99 float) derived from timestamp column."""
    if "hour" in df.columns:
        return df
    df = df.copy()
    dt_series = pd.to_datetime(df[ts_col], unit='s', utc=True)
    df["hour"] = dt_series.dt.hour + dt_series.dt.minute / 60.0 + dt_series.dt.second / 3600.0
    return df
