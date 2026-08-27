"""
Domain-aware cleaning helpers for master_races missing-value handling and modeling prep.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def lap_time_to_seconds(lap_time) -> float:
    """Convert quali/lap time string (M:SS.mmm) to total seconds."""
    if pd.isna(lap_time) or lap_time in ("\\N", "", "nan", "NaT", "None"):
        return np.nan
    text = str(lap_time).strip()
    if not text or text.lower() in {"\\n", "nan", "nat", "none"}:
        return np.nan
    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return np.nan
        return float(text)
    except (ValueError, TypeError):
        return np.nan


def seconds_to_lap_time(seconds) -> Optional[str]:
    """Convert seconds back to M:SS.mmm string."""
    if pd.isna(seconds):
        return np.nan
    total = float(seconds)
    minutes = int(total // 60)
    secs = total % 60
    return f"{minutes}:{secs:06.3f}"


def add_is_sprint_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag races with a sprint session. Clear sprint result columns on non-sprint weekends.
    """
    out = df.copy()
    indicator = "sprint_results_grid"
    if indicator not in out.columns and "sprint_results_positionOrder" in out.columns:
        indicator = "sprint_results_positionOrder"

    if indicator not in out.columns:
        out["is_sprint_weekend"] = 0
        return out

    values = pd.to_numeric(out[indicator], errors="coerce")
    out["is_sprint_weekend"] = (
        out.assign(_sp=values).groupby("raceId")["_sp"].transform(lambda s: s.notna().any()).astype(int)
    )

    sprint_value_cols = [c for c in out.columns if c.startswith("sprint_results_")]
    non_sprint = out["is_sprint_weekend"] == 0
    for col in sprint_value_cols:
        out.loc[non_sprint, col] = np.nan

    return out


def add_is_season_opener(df: pd.DataFrame) -> pd.DataFrame:
    """Round 1 of each season — no PRE_RACE standings yet."""
    out = df.copy()
    if "round" in out.columns:
        out["is_season_opener"] = (pd.to_numeric(out["round"], errors="coerce") == 1).astype(int)
    elif "date" in out.columns and "year" in out.columns:
        first_dates = out.groupby("year")["date"].transform("min")
        out["is_season_opener"] = (out["date"] == first_dates).astype(int)
    else:
        out["is_season_opener"] = 0
    return out


def add_qualifying_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add q1/q2/q3 seconds and reached_q2/reached_q3 flags (preserve structural NaN)."""
    out = df.copy()
    grid = pd.to_numeric(out["grid"], errors="coerce") if "grid" in out.columns else pd.Series(np.nan, index=out.index)

    out["reached_q2"] = ((grid <= 15) & grid.notna()).astype(int)
    out["reached_q3"] = ((grid <= 10) & grid.notna()).astype(int)

    session_rules = {
        "q1": ("q1_seconds", None),
        "q2": ("q2_seconds", "reached_q2"),
        "q3": ("q3_seconds", "reached_q3"),
    }
    for src, (dest, reach_flag) in session_rules.items():
        if src not in out.columns:
            continue
        secs = out[src].apply(lap_time_to_seconds)
        if reach_flag is not None:
            secs = secs.where(out[reach_flag] == 1)
        out[dest] = secs

    return out


# Columns excluded from modeling (raw strings / outcomes handled in modeling notebook)
MODELING_EXCLUDE_STRING_QUALI = ["q1", "q2", "q3"]
MODELING_EXCLUDE_SPRINT_STRINGS = [
    "sprint_results_time",
    "sprint_results_fastestLapTime",
]


def summarize_modeling_missingness(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Return missing counts for modeling feature columns."""
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing": missing,
                "pct": round(100.0 * missing / len(df), 2) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("missing", ascending=False)


def modeling_feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric + flag columns intended for tree models (post-cleaning)."""
    exclude = {
        "podium",
        "raceId",
        "driverId",
        "date",
        "year",
        "resultId",
        "points",
        "position",
        "positionOrder",
        "milliseconds",
        "time",
        "laps",
        "fastestLap",
        "fastestLapTime",
        "fastestLapSpeed",
        "rank",
        "statusId",
        "status_category",
        "constructor_results_points",
        "driver_standings_points",
        "driver_standings_position",
        "constructor_standings_points",
        "constructor_standings_position",
        *MODELING_EXCLUDE_STRING_QUALI,
        *MODELING_EXCLUDE_SPRINT_STRINGS,
    }
    return [c for c in df.columns if c not in exclude]
