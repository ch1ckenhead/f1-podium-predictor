"""Build master_races.csv from Kaggle-shaped tables (mirrors notebooks/02_data_combining.ipynb)."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

MIN_MASTER_YEAR = 1994


def handle_duplicate_columns_after_merge(
    merged_df: pd.DataFrame, source_name: str, temp_suffix: str = "_temp"
) -> Tuple[pd.DataFrame, list, dict]:
    if merged_df is None or merged_df.empty:
        return merged_df, [], {}

    temp_cols = [col for col in merged_df.columns if col.endswith(temp_suffix)]
    original_cols = [col[: -len(temp_suffix)] for col in temp_cols]

    columns_to_drop = []
    rename_dict = {}

    for orig_col, temp_col in zip(original_cols, temp_cols):
        if orig_col in merged_df.columns:
            both_exist_mask = merged_df[orig_col].notna() & merged_df[temp_col].notna()
            orig_only_mask = merged_df[orig_col].notna() & merged_df[temp_col].isna()
            temp_only_mask = merged_df[orig_col].isna() & merged_df[temp_col].notna()

            if both_exist_mask.any():
                identical = (
                    merged_df.loc[both_exist_mask, orig_col] == merged_df.loc[both_exist_mask, temp_col]
                ).all()
            else:
                identical = True

            if identical and not orig_only_mask.any() and not temp_only_mask.any():
                columns_to_drop.append(temp_col)
            else:
                rename_dict[temp_col] = f"{source_name}_{orig_col}"
        else:
            rename_dict[temp_col] = f"{source_name}_{orig_col}"

    cleaned_df = merged_df.drop(columns=columns_to_drop)
    cleaned_df = cleaned_df.rename(columns=rename_dict)
    return cleaned_df, columns_to_drop, rename_dict


def build_master_races(tables: Dict[str, pd.DataFrame], min_year: int = MIN_MASTER_YEAR) -> pd.DataFrame:
    """One row per (raceId, driverId); same merge order as 02_data_combining."""
    if "results" not in tables:
        raise ValueError("results table required")

    master = tables["results"].copy()
    races = tables["races"].copy()
    races["date"] = pd.to_datetime(races["date"], errors="coerce")

    master = master.merge(
        races[["raceId", "year", "round", "circuitId", "date", "name"]],
        on="raceId",
        how="left",
    )
    master = master[master["year"] >= min_year].copy()

    if "circuits" in tables:
        circuits = tables["circuits"].copy().rename(columns={"name": "circuit_name"})
        master = master.merge(circuits, on="circuitId", how="left", suffixes=("", "_circuit"))

    if "drivers" in tables:
        master = master.merge(tables["drivers"], on="driverId", how="left", suffixes=("", "_driver"))

    if "constructors" in tables:
        master = master.merge(
            tables["constructors"], on="constructorId", how="left", suffixes=("", "_constructor")
        )

    for source, keys in [
        ("driver_standings", ["raceId", "driverId"]),
        ("constructor_standings", ["raceId", "constructorId"]),
        ("constructor_results", ["raceId", "constructorId"]),
        ("qualifying", ["raceId", "driverId"]),
    ]:
        if source not in tables:
            continue
        master = master.merge(tables[source], on=keys, how="left", suffixes=("", "_temp"))
        master, _, _ = handle_duplicate_columns_after_merge(master, source, temp_suffix="_temp")

    if "sprint_results" in tables and not tables["sprint_results"].empty:
        sprint = tables["sprint_results"].copy()
        merge_keys = ["raceId", "driverId"]
        sprint_rename = {
            col: f"sprint_results_{col}" for col in sprint.columns if col not in merge_keys
        }
        sprint = sprint.rename(columns=sprint_rename)
        master = master.merge(sprint, on=merge_keys, how="left", suffixes=("", "_sprint"))

    master["podium"] = (pd.to_numeric(master["positionOrder"], errors="coerce") <= 3).astype(int)

    for col in ["lap_time_variance", "throttle_variance", "overtake_attempts", "avg_pit_stops"]:
        master[col] = pd.NA

    return master
