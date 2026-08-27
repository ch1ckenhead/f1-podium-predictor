"""
Normalize Jolpica/Ergast CSV exports into Kaggle-shaped tables for master_races build.

Ergast per-year files use string refs (driverId='hamilton'); Kaggle uses integer IDs
linked via drivers.driverRef / constructors.constructorRef / circuits.circuitRef.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

ERGAST_YEARS_DEFAULT = range(2018, 2026)
MIN_MASTER_YEAR = 1994
ERGAST_START_YEAR = 2018

# Ergast status strings → Jolpica statusId (from data/raw/ergast/status.csv)
ERGAST_STATUS_ALIASES = {
    "Finished": 1,
    "Lapped": 143,
    "Retired": 31,
    "Did not start": 142,
    "Disqualified": 2,
    "+1 Lap": 11,
    "+2 Laps": 12,
    "+3 Laps": 13,
}


def _project_root_from(path: Path) -> Path:
    root = path.resolve()
    if root.name == "src":
        return root.parent
    return root


def _parse_timedelta_seconds(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nat", "nan", "none"}:
        return None
    try:
        return pd.to_timedelta(text).total_seconds()
    except (ValueError, TypeError):
        return None


def _format_qualifying_time(value) -> Optional[str]:
    secs = _parse_timedelta_seconds(value)
    if secs is None:
        return None
    minutes = int(secs // 60)
    rem = secs % 60
    return f"{minutes}:{rem:06.3f}"


def _format_race_time(total_race_time, millis, position) -> Tuple[Optional[str], Optional[float]]:
    """Return (time string, milliseconds) in Kaggle/Ergast style."""
    if millis is not None and not (isinstance(millis, float) and pd.isna(millis)):
        ms = float(millis)
    else:
        secs = _parse_timedelta_seconds(total_race_time)
        ms = secs * 1000.0 if secs is not None else None

    if ms is None:
        return None, None

    if position == 1:
        total_secs = ms / 1000.0
        hours = int(total_secs // 3600)
        rem = total_secs % 3600
        minutes = int(rem // 60)
        seconds = rem % 60
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:06.3f}".rstrip("0").rstrip("."), ms
        return f"{minutes}:{seconds:06.3f}".rstrip("0").rstrip("."), ms

    gap_secs = _parse_timedelta_seconds(total_race_time)
    if gap_secs is not None:
        return f"+{gap_secs:06.3f}".rstrip("0").rstrip("."), ms
    return None, ms


def _position_display(status: str, position, position_text) -> Tuple[Optional[float], str, int]:
    """Map Ergast position fields → Kaggle position, positionText, positionOrder."""
    pos_num: Optional[float]
    try:
        pos_num = float(position) if position is not None and str(position) not in {"nan", "NaN"} else None
    except (TypeError, ValueError):
        pos_num = None

    pt = str(position_text) if position_text is not None else ""
    status_l = (status or "").strip()

    if status_l in {"Retired", "Did not start", "Disqualified"} or pt in {"R", "W", "D"}:
        display_pos = None
    elif status_l == "Lapped" and pos_num is not None:
        display_pos = pos_num
    else:
        display_pos = pos_num

    if pt.isdigit():
        order = int(pt)
        pos_text = pt
    elif pos_num is not None:
        order = int(pos_num)
        pos_text = "R" if status_l == "Retired" else ("W" if status_l == "Did not start" else str(int(pos_num)))
    else:
        order = 99
        pos_text = pt or "R"

    return display_pos, pos_text, order


def _status_to_id(status: str, status_lookup: Dict[str, int]) -> int:
    if status in status_lookup:
        return status_lookup[status]
    if status in ERGAST_STATUS_ALIASES:
        return ERGAST_STATUS_ALIASES[status]
    return status_lookup.get("Retired", 31)


def _load_status_lookup(ergast_root: Path, kaggle_root: Path) -> Dict[str, int]:
    erg_path = ergast_root / "status.csv"
    if erg_path.exists():
        st = pd.read_csv(erg_path)
        return dict(zip(st["status"].astype(str), st["statusId"].astype(int)))
    st = pd.read_csv(kaggle_root / "status.csv")
    return dict(zip(st["status"].astype(str), st["statusId"].astype(int)))


def _build_ref_maps(kaggle_root: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    drivers = pd.read_csv(kaggle_root / "drivers.csv")
    constructors = pd.read_csv(kaggle_root / "constructors.csv")
    circuits = pd.read_csv(kaggle_root / "circuits.csv")

    driver_map = dict(zip(drivers["driverRef"].astype(str), drivers["driverId"].astype(int)))
    constructor_map = dict(
        zip(constructors["constructorRef"].astype(str), constructors["constructorId"].astype(int))
    )
    circuit_map = dict(zip(circuits["circuitRef"].astype(str), circuits["circuitId"].astype(int)))
    return driver_map, constructor_map, circuit_map


def _extend_ref_map(ref_map: Dict[str, int], refs: Iterable[str]) -> Dict[str, int]:
    out = dict(ref_map)
    next_id = max(out.values(), default=0) + 1
    for ref in sorted(set(refs)):
        if ref not in out:
            out[ref] = next_id
            next_id += 1
    return out


def _race_id_lookup(kaggle_races: pd.DataFrame, ergast_races: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    """year, round → raceId; reuse Kaggle IDs, assign new for 2025+ rounds."""
    lookup: Dict[Tuple[int, int], int] = {}
    for _, row in kaggle_races.iterrows():
        lookup[(int(row["year"]), int(row["round"]))] = int(row["raceId"])

    next_id = int(kaggle_races["raceId"].max()) + 1
    for _, row in ergast_races.iterrows():
        key = (int(row["year"]), int(row["round"]))
        if key not in lookup:
            lookup[key] = next_id
            next_id += 1
    return lookup


def _read_year_files(ergast_root: Path, stem: str, years: Iterable[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for year in years:
        path = ergast_root / f"{stem}_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ergast_to_races(
    ergast_root: Path,
    kaggle_root: Path,
    years: Iterable[int] = ERGAST_YEARS_DEFAULT,
) -> pd.DataFrame:
    _, _, circuit_map = _build_ref_maps(kaggle_root)
    kaggle_races = pd.read_csv(kaggle_root / "races.csv")
    schedules = _read_year_files(ergast_root, "race_schedule", years)
    if schedules.empty:
        raise FileNotFoundError(f"No race_schedule_{{year}}.csv under {ergast_root}")

    schedules = schedules.copy()
    schedules["circuitId"] = schedules["circuitId"].map(circuit_map)
    schedules["date"] = pd.to_datetime(schedules["raceDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    schedules["name"] = schedules["raceName"]
    schedules["year"] = schedules["season"].astype(int)
    schedules["round"] = schedules["round"].astype(int)
    schedules["url"] = schedules["raceUrl"]

    race_lookup = _race_id_lookup(kaggle_races, schedules)
    schedules["raceId"] = schedules.apply(
        lambda r: race_lookup[(int(r["year"]), int(r["round"]))], axis=1
    )

    cols = ["raceId", "year", "round", "circuitId", "name", "date", "url"]
    return schedules[cols].drop_duplicates(subset=["raceId"])


def ergast_to_results(
    ergast_root: Path,
    kaggle_root: Path,
    races: pd.DataFrame,
    years: Iterable[int] = ERGAST_YEARS_DEFAULT,
) -> pd.DataFrame:
    status_lookup = _load_status_lookup(ergast_root, kaggle_root)
    driver_map, constructor_map, _ = _build_ref_maps(kaggle_root)

    raw = _read_year_files(ergast_root, "race_results", years)
    if raw.empty:
        raise FileNotFoundError(f"No race_results_{{year}}.csv under {ergast_root}")

    driver_map = _extend_ref_map(driver_map, raw["driverId"].astype(str))
    constructor_map = _extend_ref_map(constructor_map, raw["constructorId"].astype(str))

    race_keys = races.set_index(["year", "round"])["raceId"].to_dict()
    raw = raw.copy()
    raw["year"] = raw["season"].astype(int)
    raw["round"] = raw["round"].astype(int)
    raw["raceId"] = raw.apply(lambda r: race_keys[(int(r["year"]), int(r["round"]))], axis=1)

    rows = []
    next_result_id = int(pd.read_csv(kaggle_root / "results.csv")["resultId"].max()) + 1

    for _, r in raw.iterrows():
        display_pos, pos_text, pos_order = _position_display(
            r.get("status"), r.get("position"), r.get("positionText")
        )
        time_str, ms = _format_race_time(
            r.get("totalRaceTime"), r.get("totalRaceTimeMillis"), r.get("position")
        )
        if r.get("status") == "Lapped" or (isinstance(r.get("status"), str) and "Lap" in r.get("status")):
            if display_pos and display_pos > 1:
                time_str, ms = None, None

        status_id = _status_to_id(str(r.get("status", "")), status_lookup)
        if status_id >= 11 and status_id <= 19:
            time_str, ms = None, None
        if status_id in {2, 3, 4, 5, 6, 7, 31, 142} or str(r.get("status")) in {
            "Retired",
            "Did not start",
            "Disqualified",
        }:
            if status_id not in {11, 12, 13, 14, 15, 16, 143}:
                time_str, ms = None, None

        fl_time = _format_qualifying_time(r.get("fastestLapTime"))
        rows.append(
            {
                "resultId": next_result_id,
                "raceId": int(r["raceId"]),
                "driverId": driver_map[str(r["driverId"])],
                "constructorId": constructor_map[str(r["constructorId"])],
                "number": r.get("number"),
                "grid": r.get("grid"),
                "position": display_pos,
                "positionText": pos_text,
                "positionOrder": pos_order,
                "points": r.get("points"),
                "laps": r.get("laps"),
                "time": time_str,
                "milliseconds": ms,
                "fastestLap": r.get("fastestLapNumber"),
                "rank": r.get("fastestLapRank"),
                "fastestLapTime": fl_time,
                "fastestLapSpeed": r.get("fastestLapAvgSpeed"),
                "statusId": status_id,
            }
        )
        next_result_id += 1

    return pd.DataFrame(rows)


def ergast_to_qualifying(
    ergast_root: Path,
    kaggle_root: Path,
    races: pd.DataFrame,
    years: Iterable[int] = ERGAST_YEARS_DEFAULT,
) -> pd.DataFrame:
    driver_map, constructor_map, _ = _build_ref_maps(kaggle_root)
    raw = _read_year_files(ergast_root, "qualifying_results", years)
    if raw.empty:
        return pd.DataFrame(columns=["qualifyId", "raceId", "driverId", "constructorId", "number", "position", "q1", "q2", "q3"])

    driver_map = _extend_ref_map(driver_map, raw["driverId"].astype(str))
    constructor_map = _extend_ref_map(constructor_map, raw["constructorId"].astype(str))
    race_keys = races.set_index(["year", "round"])["raceId"].to_dict()

    next_id = int(pd.read_csv(kaggle_root / "qualifying.csv")["qualifyId"].max()) + 1
    rows = []
    for _, r in raw.iterrows():
        key = (int(r["season"]), int(r["round"]))
        rows.append(
            {
                "qualifyId": next_id,
                "raceId": race_keys[key],
                "driverId": driver_map[str(r["driverId"])],
                "constructorId": constructor_map[str(r["constructorId"])],
                "number": r.get("number"),
                "position": r.get("position"),
                "q1": _format_qualifying_time(r.get("Q1")),
                "q2": _format_qualifying_time(r.get("Q2")),
                "q3": _format_qualifying_time(r.get("Q3")),
            }
        )
        next_id += 1
    return pd.DataFrame(rows)


def _first_list_value(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            return str(parsed[0]) if parsed else None
        except (ValueError, SyntaxError):
            pass
    return text


def ergast_to_driver_standings(
    ergast_root: Path,
    kaggle_root: Path,
    races: pd.DataFrame,
    years: Iterable[int] = ERGAST_YEARS_DEFAULT,
) -> pd.DataFrame:
    driver_map, _, _ = _build_ref_maps(kaggle_root)
    raw = _read_year_files(ergast_root, "driver_standings", years)
    if raw.empty:
        return pd.DataFrame(
            columns=["driverStandingsId", "raceId", "driverId", "points", "position", "positionText", "wins"]
        )

    driver_map = _extend_ref_map(driver_map, raw["driverId"].astype(str))
    race_keys = races.set_index(["year", "round"])["raceId"].to_dict()
    next_id = int(pd.read_csv(kaggle_root / "driver_standings.csv")["driverStandingsId"].max()) + 1

    rows = []
    for _, r in raw.iterrows():
        rnd = int(r.get("standingsRound", r["round"]))
        key = (int(r["season"]), rnd)
        if key not in race_keys:
            continue
        rows.append(
            {
                "driverStandingsId": next_id,
                "raceId": race_keys[key],
                "driverId": driver_map[str(r["driverId"])],
                "points": r.get("points"),
                "position": r.get("position"),
                "positionText": r.get("positionText"),
                "wins": r.get("wins"),
            }
        )
        next_id += 1
    return pd.DataFrame(rows)


def ergast_to_constructor_standings(
    ergast_root: Path,
    kaggle_root: Path,
    races: pd.DataFrame,
    years: Iterable[int] = ERGAST_YEARS_DEFAULT,
) -> pd.DataFrame:
    _, constructor_map, _ = _build_ref_maps(kaggle_root)
    raw = _read_year_files(ergast_root, "constructor_standings", years)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "constructorStandingsId",
                "raceId",
                "constructorId",
                "points",
                "position",
                "positionText",
                "wins",
            ]
        )

    constructor_map = _extend_ref_map(constructor_map, raw["constructorId"].astype(str))
    race_keys = races.set_index(["year", "round"])["raceId"].to_dict()
    next_id = int(pd.read_csv(kaggle_root / "constructor_standings.csv")["constructorStandingsId"].max()) + 1

    rows = []
    for _, r in raw.iterrows():
        rnd = int(r.get("standingsRound", r["round"]))
        key = (int(r["season"]), rnd)
        if key not in race_keys:
            continue
        rows.append(
            {
                "constructorStandingsId": next_id,
                "raceId": race_keys[key],
                "constructorId": constructor_map[str(r["constructorId"])],
                "points": r.get("points"),
                "position": r.get("position"),
                "positionText": r.get("positionText"),
                "wins": r.get("wins"),
            }
        )
        next_id += 1
    return pd.DataFrame(rows)


def ergast_to_sprint_results(
    ergast_root: Path,
    kaggle_root: Path,
    races: pd.DataFrame,
    years: Iterable[int] = ERGAST_YEARS_DEFAULT,
) -> pd.DataFrame:
    driver_map, constructor_map, _ = _build_ref_maps(kaggle_root)
    frames = []
    for year in years:
        if year < 2021:
            continue
        path = ergast_root / f"sprint_results_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame(
            columns=[
                "resultId",
                "raceId",
                "driverId",
                "constructorId",
                "number",
                "grid",
                "position",
                "positionText",
                "positionOrder",
                "points",
                "laps",
                "time",
                "milliseconds",
                "fastestLap",
                "fastestLapTime",
                "statusId",
            ]
        )

    raw = pd.concat(frames, ignore_index=True)
    status_lookup = _load_status_lookup(ergast_root, kaggle_root)
    driver_map = _extend_ref_map(driver_map, raw["driverId"].astype(str))
    constructor_map = _extend_ref_map(constructor_map, raw["constructorId"].astype(str))
    race_keys = races.set_index(["year", "round"])["raceId"].to_dict()

    next_id = 1
    if (kaggle_root / "sprint_results.csv").exists():
        k_sprint = pd.read_csv(kaggle_root / "sprint_results.csv")
        if not k_sprint.empty:
            next_id = int(k_sprint["resultId"].max()) + 1

    rows = []
    for _, r in raw.iterrows():
        display_pos, pos_text, pos_order = _position_display(
            r.get("status"), r.get("position"), r.get("positionText")
        )
        time_str, ms = _format_race_time(
            r.get("totalRaceTime"), r.get("totalRaceTimeMillis"), r.get("position")
        )
        rows.append(
            {
                "resultId": next_id,
                "raceId": race_keys[(int(r["season"]), int(r["round"]))],
                "driverId": driver_map[str(r["driverId"])],
                "constructorId": constructor_map[str(r["constructorId"])],
                "number": r.get("number"),
                "grid": r.get("grid"),
                "position": display_pos,
                "positionText": pos_text,
                "positionOrder": pos_order,
                "points": r.get("points"),
                "laps": r.get("laps"),
                "time": time_str,
                "milliseconds": ms,
                "fastestLap": r.get("fastestLapNumber"),
                "fastestLapTime": _format_qualifying_time(r.get("fastestLapTime")),
                "statusId": _status_to_id(str(r.get("status", "")), status_lookup),
            }
        )
        next_id += 1
    return pd.DataFrame(rows)


def ergast_to_constructor_results(results: pd.DataFrame, kaggle_root: Path) -> pd.DataFrame:
    grouped = (
        results.groupby(["raceId", "constructorId"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "points"})
    )
    next_id = int(pd.read_csv(kaggle_root / "constructor_results.csv")["constructorResultsId"].max()) + 1
    grouped.insert(0, "constructorResultsId", range(next_id, next_id + len(grouped)))
    grouped["status"] = None
    return grouped


def build_kaggle_tables_from_ergast(
    ergast_root: Path,
    kaggle_root: Path,
    years: Iterable[int] = ERGAST_YEARS_DEFAULT,
) -> Dict[str, pd.DataFrame]:
    """Return Kaggle-shaped tables for the Ergast year range."""
    races = ergast_to_races(ergast_root, kaggle_root, years)
    results = ergast_to_results(ergast_root, kaggle_root, races, years)
    return {
        "races": races,
        "results": results,
        "qualifying": ergast_to_qualifying(ergast_root, kaggle_root, races, years),
        "driver_standings": ergast_to_driver_standings(ergast_root, kaggle_root, races, years),
        "constructor_standings": ergast_to_constructor_standings(ergast_root, kaggle_root, races, years),
        "constructor_results": ergast_to_constructor_results(results, kaggle_root),
        "sprint_results": ergast_to_sprint_results(ergast_root, kaggle_root, races, years),
    }


def merge_historical_and_ergast_kaggle_tables(
    kaggle_root: Path,
    ergast_root: Path,
    ergast_years: Iterable[int] = ERGAST_YEARS_DEFAULT,
    min_year: int = MIN_MASTER_YEAR,
    ergast_start_year: int = ERGAST_START_YEAR,
) -> Dict[str, pd.DataFrame]:
    """
    Kaggle tables for min_year..ergast_start_year-1, Ergast-normalized tables from ergast_start_year+.
    """
    ergast = build_kaggle_tables_from_ergast(ergast_root, kaggle_root, ergast_years)

    kaggle_races = pd.read_csv(kaggle_root / "races.csv")
    hist_race_ids = set(
        kaggle_races[kaggle_races["year"] < ergast_start_year]["raceId"].astype(int)
    )

    out: Dict[str, pd.DataFrame] = {}
    for name in ["drivers", "constructors", "circuits"]:
        out[name] = pd.read_csv(kaggle_root / f"{name}.csv")

    # Races: historical + ergast (dedupe by raceId)
    hist_races = kaggle_races[kaggle_races["year"] >= min_year]
    hist_races = hist_races[hist_races["year"] < ergast_start_year]
    out["races"] = pd.concat([hist_races, ergast["races"]], ignore_index=True).drop_duplicates(
        subset=["raceId"]
    )

    def _hist_table(filename: str, id_col: str) -> pd.DataFrame:
        df = pd.read_csv(kaggle_root / filename)
        return df[df["raceId"].isin(hist_race_ids)]

    for table in [
        "results",
        "qualifying",
        "driver_standings",
        "constructor_standings",
        "constructor_results",
        "sprint_results",
    ]:
        hist = _hist_table(f"{table}.csv", "raceId")
        erg = ergast[table]
        out[table] = pd.concat([hist, erg], ignore_index=True)

    return out
