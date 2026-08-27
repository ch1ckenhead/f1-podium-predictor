"""
Production-efficient FastF1 feature engineering.

Computes per-race driver metrics from historical race sessions, then exposes only
pre-race-safe columns (shifted rolling averages + session-start weather).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FASTF1_YEARS = range(2018, 2026)

# Intermediate metrics computed from completed races (not merged directly).
BASE_RACE_METRICS: Tuple[str, ...] = (
    "drs_activation_rate",
    "position_change_rate",
    "lap_time_std",
    "first_pit_lap",
    "sector_speed_laptime_corr",
    "tyre_efficiency_index",
)

ROLLING_WINDOWS: Tuple[int, ...] = (3, 5, 10)

WEATHER_START_FEATURES: Tuple[str, ...] = (
    "weather_airtemp_start",
    "weather_tracktemp_start",
    "weather_humidity_start",
    "weather_pressure_start",
    "weather_windspeed_start",
    "weather_rainfall_start",
)

LAPS_USECOLS = [
    "Year",
    "Event",
    "Session",
    "Driver",
    "DriverNumber",
    "LapNumber",
    "LapTime",
    "Position",
    "Deleted",
    "Stint",
    "TyreLife",
    "SpeedI1",
    "SpeedI2",
    "PitInTime",
]

TELEMETRY_USECOLS = ["Year", "Event", "Session", "Driver", "DRS"]

WEATHER_USECOLS = [
    "Year",
    "Event",
    "Session",
    "Time",
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "Rainfall",
]


def normalize_event_name(name) -> Optional[str]:
    if pd.isna(name):
        return None
    return str(name).strip()


def build_race_map(master_df: pd.DataFrame) -> Dict[Tuple[int, str], int]:
    """Map (year, event name) -> raceId from master."""
    mapping_df = master_df[["year", "name", "raceId"]].drop_duplicates(["year", "name"])
    return {
        (int(row.year), normalize_event_name(row.name)): int(row.raceId)
        for row in mapping_df.itertuples(index=False)
        if pd.notna(row.name) and pd.notna(row.raceId)
    }


def attach_race_id(df: pd.DataFrame, race_map: Dict[Tuple[int, str], int]) -> pd.DataFrame:
    """Vectorized raceId enrichment."""
    out = df.copy()
    out["_event_norm"] = out["Event"].map(normalize_event_name)
    out["raceId"] = [
        race_map.get((int(y), e)) if pd.notna(y) and e is not None else None
        for y, e in zip(out["Year"], out["_event_norm"])
    ]
    out.drop(columns=["_event_norm"], inplace=True)
    return out


def lap_time_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def prerace_driver_feature_names() -> List[str]:
    names: List[str] = []
    for metric in BASE_RACE_METRICS:
        for window in ROLLING_WINDOWS:
            names.append(f"{metric}_avg_last_{window}")
    return names


def prerace_feature_names() -> List[str]:
    return prerace_driver_feature_names() + list(WEATHER_START_FEATURES)


def _sector_speed_laptime_corr(group: pd.DataFrame) -> float:
    if len(group) < 3:
        return np.nan
    if "SpeedI1" not in group.columns or "SpeedI2" not in group.columns:
        return np.nan
    sector_speeds = (group["SpeedI1"].fillna(0) + group["SpeedI2"].fillna(0)) / 2
    sector_speeds = sector_speeds.replace(0, np.nan)
    valid = pd.DataFrame({"speed": sector_speeds, "lap_time": group["LapTimeSec"]}).dropna()
    if len(valid) < 3:
        return np.nan
    corr = valid["speed"].corr(valid["lap_time"])
    return -corr if pd.notna(corr) else np.nan


def _tyre_efficiency_index(group: pd.DataFrame) -> float:
    if len(group) == 0:
        return np.nan
    if "Stint" in group.columns and "TyreLife" in group.columns:
        efficiencies = []
        for _, stint_data in group.groupby("Stint"):
            if len(stint_data) < 2:
                continue
            avg_lap = stint_data["LapTimeSec"].mean()
            max_life = stint_data["TyreLife"].max()
            if max_life and max_life > 0 and pd.notna(avg_lap):
                efficiencies.append(avg_lap / max_life)
        return float(np.mean(efficiencies)) if efficiencies else np.nan
    return float(group["LapTimeSec"].mean()) if len(group) else np.nan


def aggregate_lap_metrics_for_year(laps_path: Path, race_map: Dict[Tuple[int, str], int]) -> pd.DataFrame:
    """Aggregate race-session lap metrics per (raceId, driver code)."""
    if not laps_path.exists():
        logger.warning("LAPS file missing: %s", laps_path)
        return pd.DataFrame()

    laps = pd.read_csv(laps_path, usecols=lambda c: c in LAPS_USECOLS, low_memory=False)
    laps = attach_race_id(laps, race_map)
    race_laps = laps[(laps["Session"] == "R") & laps["raceId"].notna()].copy()
    if race_laps.empty:
        return pd.DataFrame()

    race_laps["raceId"] = race_laps["raceId"].astype(int)
    race_laps["LapTimeSec"] = lap_time_seconds(race_laps["LapTime"])
    valid = race_laps[
        (race_laps["Deleted"] != True)  # noqa: E712
        & race_laps["LapTimeSec"].notna()
    ].copy()

    keys = ["raceId", "Driver"]
    valid = valid.sort_values(keys + ["LapNumber"])

    # Lap time std
    lap_std = valid.groupby(keys, as_index=False)["LapTimeSec"].std().rename(columns={"LapTimeSec": "lap_time_std"})

    # Position change rate
    valid["pos_change"] = valid.groupby(keys)["Position"].diff().ne(0).astype(int)
    pos_stats = valid.groupby(keys, as_index=False).agg(
        position_changes=("pos_change", lambda s: max(0, int(s.sum()) - 1)),
        total_laps=("LapNumber", "count"),
    )
    pos_stats["position_change_rate"] = pos_stats["position_changes"] / pos_stats["total_laps"].replace(0, np.nan)

    # Pit stops
    if "Stint" in valid.columns:
        valid["stint_change"] = valid.groupby(keys)["Stint"].diff()
        pit_rows = valid[valid["stint_change"] > 0]
        first_pit = pit_rows.groupby(keys, as_index=False)["LapNumber"].first().rename(columns={"LapNumber": "first_pit_lap"})
    elif "PitInTime" in race_laps.columns:
        pit_rows = race_laps[race_laps["PitInTime"].notna()]
        first_pit = pit_rows.groupby(keys, as_index=False)["LapNumber"].first().rename(columns={"LapNumber": "first_pit_lap"})
    else:
        first_pit = pd.DataFrame(columns=keys + ["first_pit_lap"])

    # Sector correlation + tyre efficiency (small number of groups — apply is fine)
    sector = (
        valid.groupby(keys, group_keys=False)
        .apply(_sector_speed_laptime_corr, include_groups=False)
        .reset_index(name="sector_speed_laptime_corr")
    )
    tyre = (
        valid.groupby(keys, group_keys=False)
        .apply(_tyre_efficiency_index, include_groups=False)
        .reset_index(name="tyre_efficiency_index")
    )

    metrics = lap_std.merge(pos_stats[keys + ["position_change_rate"]], on=keys, how="outer")
    metrics = metrics.merge(first_pit, on=keys, how="left")
    metrics = metrics.merge(sector, on=keys, how="left")
    metrics = metrics.merge(tyre, on=keys, how="left")
    metrics = metrics.rename(columns={"Driver": "driver_code"})
    metrics["driver_code"] = metrics["driver_code"].astype(str)
    return metrics


def build_car_number_map(laps_path: Path, race_map: Dict[Tuple[int, str], int]) -> pd.DataFrame:
    """Map (raceId, car number) -> driver code from race laps."""
    if not laps_path.exists():
        return pd.DataFrame(columns=["raceId", "car_number", "driver_code"])

    laps = pd.read_csv(
        laps_path,
        usecols=lambda c: c in ["Year", "Event", "Session", "Driver", "DriverNumber"],
        low_memory=False,
    )
    laps = attach_race_id(laps, race_map)
    race = laps[(laps["Session"] == "R") & laps["raceId"].notna()].copy()
    if race.empty:
        return pd.DataFrame(columns=["raceId", "car_number", "driver_code"])

    race["raceId"] = race["raceId"].astype(int)
    mapping = race[["raceId", "DriverNumber", "Driver"]].drop_duplicates()
    mapping = mapping.rename(columns={"DriverNumber": "car_number", "Driver": "driver_code"})
    mapping["car_number"] = pd.to_numeric(mapping["car_number"], errors="coerce")
    mapping["driver_code"] = mapping["driver_code"].astype(str)
    return mapping.dropna(subset=["car_number"])


def aggregate_drs_for_year(
    telemetry_path: Path,
    race_map: Dict[Tuple[int, str], int],
    car_number_map: pd.DataFrame,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    """Stream telemetry and aggregate DRS activation rate without loading full file."""
    if not telemetry_path.exists():
        logger.warning("TELEMETRY file missing: %s", telemetry_path)
        return pd.DataFrame()

    partials: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        telemetry_path,
        usecols=lambda c: c in TELEMETRY_USECOLS,
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk = chunk[chunk["Session"] == "R"]
        if chunk.empty:
            continue
        chunk = attach_race_id(chunk, race_map)
        chunk = chunk[chunk["raceId"].notna()]
        if chunk.empty:
            continue
        chunk["raceId"] = chunk["raceId"].astype(int)
        chunk["car_number"] = pd.to_numeric(chunk["Driver"], errors="coerce")
        chunk["drs_on"] = (pd.to_numeric(chunk["DRS"], errors="coerce") == 1).astype(np.int8)
        partial = chunk.groupby(["raceId", "car_number"], as_index=False).agg(
            drs_sum=("drs_on", "sum"),
            drs_count=("drs_on", "count"),
        )
        partials.append(partial)

    if not partials:
        return pd.DataFrame()

    combined = pd.concat(partials, ignore_index=True).groupby(["raceId", "car_number"], as_index=False).agg(
        drs_sum=("drs_sum", "sum"),
        drs_count=("drs_count", "sum"),
    )
    combined["drs_activation_rate"] = combined["drs_sum"] / combined["drs_count"].replace(0, np.nan)

    if car_number_map.empty:
        combined["driver_code"] = combined["car_number"].astype(str)
        return combined[["raceId", "driver_code", "drs_activation_rate"]]

    mapped = combined.merge(car_number_map, on=["raceId", "car_number"], how="left")
    mapped["driver_code"] = mapped["driver_code"].fillna(mapped["car_number"].astype(str))
    return mapped[["raceId", "driver_code", "drs_activation_rate"]].drop_duplicates(["raceId", "driver_code"])


def compute_race_driver_metrics(raw_root: Path, years: Iterable[int], race_map: Dict[Tuple[int, str], int]) -> pd.DataFrame:
    """Process laps + telemetry year-by-year to bound memory use."""
    year_frames: List[pd.DataFrame] = []

    for year in years:
        laps_path = raw_root / f"ALL_LAPS_{year}.csv"
        telemetry_path = raw_root / f"ALL_TELEMETRY_{year}.csv"

        lap_metrics = aggregate_lap_metrics_for_year(laps_path, race_map)
        car_map = build_car_number_map(laps_path, race_map)
        drs_metrics = aggregate_drs_for_year(telemetry_path, race_map, car_map)

        if lap_metrics.empty and drs_metrics.empty:
            continue

        if lap_metrics.empty:
            year_metrics = drs_metrics
        elif drs_metrics.empty:
            year_metrics = lap_metrics
        else:
            year_metrics = lap_metrics.merge(drs_metrics, on=["raceId", "driver_code"], how="outer")

        year_metrics["year"] = year
        year_frames.append(year_metrics)
        logger.info("  %s: %s race-driver rows", year, len(year_metrics))

    if not year_frames:
        return pd.DataFrame()

    return pd.concat(year_frames, ignore_index=True)


def add_prerace_rolling_features(race_metrics: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shifted rolling averages per driver — only prior races contribute.

    Returns raceId, driverId, driver_code, and *_avg_last_* columns.
    """
    driver_lookup = (
        master_df[master_df["year"] >= 2018][["raceId", "driverId", "code"]]
        .drop_duplicates(["raceId", "driverId"])
        .rename(columns={"code": "driver_code"})
    )
    driver_lookup["driver_code"] = driver_lookup["driver_code"].astype(str)

    merged = race_metrics.merge(driver_lookup, on=["raceId", "driver_code"], how="inner")
    merged = merged.sort_values(["driverId", "raceId"]).reset_index(drop=True)

    out = merged[["raceId", "driverId", "driver_code"]].copy()
    for metric in BASE_RACE_METRICS:
        if metric not in merged.columns:
            continue
        for window in ROLLING_WINDOWS:
            out[f"{metric}_avg_last_{window}"] = merged.groupby("driverId", sort=False)[metric].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

    return out


def compute_prerace_weather(raw_root: Path, years: Iterable[int], race_map: Dict[Tuple[int, str], int]) -> pd.DataFrame:
    """Session-start weather (first race-session sample) — known before/at race start."""
    frames: List[pd.DataFrame] = []

    for year in years:
        weather_path = raw_root / f"ALL_WEATHER_{year}.csv"
        if not weather_path.exists():
            continue

        weather = pd.read_csv(weather_path, usecols=lambda c: c in WEATHER_USECOLS, low_memory=False)
        weather = attach_race_id(weather, race_map)
        race_weather = weather[(weather["Session"] == "R") & weather["raceId"].notna()].copy()
        if race_weather.empty:
            continue

        race_weather["raceId"] = race_weather["raceId"].astype(int)
        sort_col = "Time" if "Time" in race_weather.columns else None
        if sort_col:
            race_weather = race_weather.sort_values(["raceId", sort_col])
        else:
            race_weather = race_weather.sort_values("raceId")

        start = race_weather.groupby("raceId", as_index=False).first()
        rename_map = {
            "AirTemp": "weather_airtemp_start",
            "TrackTemp": "weather_tracktemp_start",
            "Humidity": "weather_humidity_start",
            "Pressure": "weather_pressure_start",
            "WindSpeed": "weather_windspeed_start",
            "Rainfall": "weather_rainfall_start",
        }
        weather_feats = start[["raceId"]].copy()
        for src, dst in rename_map.items():
            if src in start.columns:
                weather_feats[dst] = start[src]
                if dst == "weather_rainfall_start":
                    weather_feats[dst] = weather_feats[dst].astype(bool)

        frames.append(weather_feats)
        logger.info("  weather %s: %s races", year, len(weather_feats))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop_duplicates("raceId")


def merge_prerace_features(
    master_df: pd.DataFrame,
    driver_features: pd.DataFrame,
    weather_features: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge pre-race columns onto master."""
    result = master_df.copy()
    feature_cols = prerace_feature_names()

    if not driver_features.empty:
        driver_cols = ["raceId", "driverId"] + [c for c in prerace_driver_feature_names() if c in driver_features.columns]
        result = result.merge(driver_features[driver_cols], on=["raceId", "driverId"], how="left")

    if not weather_features.empty:
        weather_cols = ["raceId"] + [c for c in WEATHER_START_FEATURES if c in weather_features.columns]
        result = result.merge(weather_features[weather_cols], on="raceId", how="left")

    added = [c for c in feature_cols if c in result.columns and c not in master_df.columns]
    logger.info("Merged %s pre-race FastF1 features", len(added))
    return result


def summarize_missingness(df: pd.DataFrame, features: Sequence[str], year_min: int = 2018) -> pd.DataFrame:
    subset = df[df["year"] >= year_min]
    rows = []
    for feature in features:
        if feature not in subset.columns:
            continue
        missing = int(subset[feature].isna().sum())
        total = len(subset)
        rows.append(
            {
                "feature": feature,
                "missing": missing,
                "available": total - missing,
                "missing_pct": (missing / total * 100) if total else 100.0,
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct")


def run_fastf1_feature_pipeline(
    master_df: pd.DataFrame,
    raw_root: Path,
    years: Iterable[int] = FASTF1_YEARS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline.

    Returns:
        master_with_features, driver_prerace_features, weather_features, race_metrics (internal)
    """
    master_2018plus = master_df[master_df["year"] >= 2018].copy()
    race_map = build_race_map(master_2018plus)

    logger.info("Built raceId map for %s events", len(race_map))

    logger.info("Computing race-driver metrics (year-by-year)...")
    race_metrics = compute_race_driver_metrics(raw_root, years, race_map)
    logger.info("Race metrics shape: %s", race_metrics.shape)

    logger.info("Building pre-race rolling driver features...")
    driver_prerace = add_prerace_rolling_features(race_metrics, master_df)
    logger.info("Pre-race driver features shape: %s", driver_prerace.shape)

    logger.info("Computing pre-race weather features...")
    weather_features = compute_prerace_weather(raw_root, years, race_map)
    logger.info("Weather features shape: %s", weather_features.shape)

    master_with_features = merge_prerace_features(master_df, driver_prerace, weather_features)
    return master_with_features, driver_prerace, weather_features, race_metrics
