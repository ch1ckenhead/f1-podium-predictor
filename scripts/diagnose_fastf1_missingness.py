"""Diagnose which FastF1 source datasets cause pre-race feature missingness."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fastf1_features import (  # noqa: E402
    FASTF1_YEARS,
    add_prerace_rolling_features,
    build_race_map,
    compute_race_driver_metrics,
)

PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw" / "fastf1_2018plus"

REF_WEATHER = "weather_airtemp_start"
REF_LAP = "lap_time_std_avg_last_5"
REF_DRS = "drs_activation_rate_avg_last_5"
REF_PIT = "first_pit_lap_avg_last_5"


def file_inventory() -> pd.DataFrame:
    rows = []
    for year in range(2018, 2027):
        rows.append(
            {
                "year": year,
                "laps": (RAW / f"ALL_LAPS_{year}.csv").exists(),
                "telemetry": (RAW / f"ALL_TELEMETRY_{year}.csv").exists(),
                "weather": (RAW / f"ALL_WEATHER_{year}.csv").exists(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    master_feat = pd.read_csv(PROCESSED / "master_races_clean_with_fastf1.csv", low_memory=False)
    master_clean = pd.read_csv(PROCESSED / "master_races_clean_ergast.csv", low_memory=False)
    m = master_feat[master_feat["year"] >= 2018].copy()
    m18 = master_clean[master_clean["year"] >= 2018][
        ["raceId", "driverId", "code", "year", "round", "name", "date"]
    ].copy()

    print("=" * 70)
    print("RAW FASTF1 FILE INVENTORY")
    print("=" * 70)
    print(file_inventory().to_string(index=False))

    print("\n" + "=" * 70)
    print(f"2018+ DRIVER-RACE ROWS: {len(m):,}")
    print("=" * 70)

    # --- Weather ---
    print("\n" + "=" * 70)
    print("1) WEATHER (ALL_WEATHER_*.csv) — race-level merge")
    print("=" * 70)
    w_miss = m[m[REF_WEATHER].isna()]
    print(f"Missing rows: {len(w_miss):,} ({len(w_miss)/len(m)*100:.1f}%)")

    by_year = (
        m.groupby("year")
        .agg(total=("raceId", "size"), missing=(REF_WEATHER, lambda s: s.isna().sum()))
        .assign(miss_pct=lambda d: (d["missing"] / d["total"] * 100).round(1))
    )
    print("\nBy year:")
    print(by_year.to_string())

    miss_races = (
        w_miss.groupby(["year", "round", "raceId", "name"], as_index=False)
        .size()
        .rename(columns={"size": "drivers_missing"})
    )
    print(f"\nUnique races with NO weather: {miss_races['raceId'].nunique()}")
    print(miss_races.sort_values(["year", "round"]).to_string(index=False))

    # --- Recompute pipeline intermediates ---
    print("\n" + "=" * 70)
    print("2) LAPS (ALL_LAPS_*.csv) — race_metrics join + rolling")
    print("=" * 70)

    race_map = build_race_map(m18)
    race_metrics = compute_race_driver_metrics(RAW, FASTF1_YEARS, race_map)
    driver_prerace = add_prerace_rolling_features(race_metrics, master_clean)

    rm_keys = set(zip(race_metrics["raceId"], race_metrics["driver_code"].astype(str)))
    pr = driver_prerace[["raceId", "driverId", REF_LAP]].rename(columns={REF_LAP: "roll"})

    diag = m18.merge(pr, on=["raceId", "driverId"], how="left")
    diag["in_race_metrics"] = [
        (rid, str(code)) in rm_keys for rid, code in zip(diag["raceId"], diag["code"])
    ]
    diag["lap_roll_missing"] = diag["roll"].isna()

    no_rm = diag[~diag["in_race_metrics"]]
    cold_start = diag[diag["in_race_metrics"] & diag["roll"].isna()]
    has_roll = diag[diag["roll"].notna()]

    print(f"No laps match (not in race_metrics):     {len(no_rm):,} rows")
    print(f"Laps match but rolling NaN (cold start): {len(cold_start):,} rows")
    print(f"Rolling available:                       {len(has_roll):,} rows")

    print("\nNo laps match — by year:")
    print(no_rm.groupby("year").size().rename("rows").to_string())

    if len(no_rm):
        print("\nRaces with most laps join failures:")
        print(
            no_rm.groupby(["year", "round", "name"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(15)
            .to_string(index=False)
        )

    print("\nCold start rolling NaN — by year:")
    print(cold_start.groupby("year").size().rename("rows").to_string())

    m18_sorted = m18.sort_values(["driverId", "raceId"])
    first_races = m18_sorted.groupby("driverId", as_index=False).first()
    first_keys = set(zip(first_races["raceId"], first_races["driverId"]))
    cold_start = cold_start.copy()
    cold_start["is_first_2018plus_race"] = [
        (r.raceId, r.driverId) in first_keys for r in cold_start.itertuples()
    ]
    print(
        f"\nOf {len(cold_start)} cold-start NaNs, "
        f"{cold_start['is_first_2018plus_race'].sum()} are the driver's first 2018+ race"
    )
    print(
        f"Remaining cold-start NaNs (prior races lacked lap metric): "
        f"{(~cold_start['is_first_2018plus_race']).sum()}"
    )

    # --- DRS ---
    print("\n" + "=" * 70)
    print("3) TELEMETRY (ALL_TELEMETRY_*.csv) — DRS rolling only")
    print("=" * 70)
    lap_na = m[REF_LAP].isna()
    drs_na = m[REF_DRS].isna()
    print(f"Lap rolling missing:  {lap_na.sum():,}")
    print(f"DRS rolling missing:  {drs_na.sum():,}")
    print(f"DRS missing, lap OK: {(drs_na & ~lap_na).sum():,}  <- telemetry-specific gap")

    extra_drs = m[~lap_na & drs_na][["year", "round", "name", "code", "driverId"]]
    if len(extra_drs):
        print("\nRows with lap rolling but NO DRS rolling:")
        print(extra_drs.to_string(index=False))
        print("\nBy year:")
        print(extra_drs.groupby("year").size().rename("rows").to_string())

    # Check if those drivers have drs in race_metrics for PRIOR races
    if len(extra_drs):
        drs_in_rm = race_metrics[race_metrics["drs_activation_rate"].notna()]
        print(f"\nRace-metrics rows with DRS value: {len(drs_in_rm):,} / {len(race_metrics):,}")

    # --- Pit ---
    print("\n" + "=" * 70)
    print("4) FIRST_PIT_LAP — derived from LAPS, extra NaN when no pit in history")
    print("=" * 70)
    pit_na = m[REF_PIT].isna()
    print(f"Pit rolling missing:       {pit_na.sum():,}")
    print(f"Pit missing, lap roll OK:  {(pit_na & ~lap_na).sum():,}")

    extra_pit = m[~lap_na & pit_na][["year", "round", "name", "code", REF_LAP, REF_PIT]]
    if len(extra_pit):
        print("\nSample: lap rolling OK but pit rolling NaN:")
        print(extra_pit.head(12).to_string(index=False))

    # --- Cross-tab weather vs laps ---
    print("\n" + "=" * 70)
    print("5) CROSS-TAB: weather missing vs lap rolling missing")
    print("=" * 70)
    print(
        pd.crosstab(
            m[REF_WEATHER].isna(),
            m[REF_LAP].isna(),
            rownames=["weather_na"],
            colnames=["lap_roll_na"],
            margins=True,
        )
    )

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: MISSINGNESS ATTRIBUTION (2018+ rows)")
    print("=" * 70)
    summary = {
        "cause": [
            "ALL_WEATHER_*.csv missing for year/race (2025 etc.)",
            "ALL_LAPS_*.csv join fail (raceId+code not in race_metrics)",
            "Expected cold start (first 2018+ race per driver, shift(1))",
            "Prior races lacked computable lap metric (not first race)",
            "ALL_TELEMETRY_*.csv gap (lap OK, DRS rolling NaN)",
            "No pit stop in prior races (lap OK, pit rolling NaN)",
        ],
        "approx_rows": [
            len(w_miss),
            len(no_rm),
            int(cold_start["is_first_2018plus_race"].sum()),
            int((~cold_start["is_first_2018plus_race"]).sum()),
            int((drs_na & ~lap_na).sum()),
            int((pit_na & ~lap_na).sum()),
        ],
    }
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nNote: categories overlap (e.g. 2025 rows often miss both weather and laps).")

    # --- Export reports ---
    out_dir = PROCESSED
    miss_races.to_csv(out_dir / "fastf1_missingness_weather_gaps_by_race.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "fastf1_missingness_attribution.csv", index=False)

    fail_all = diag[~diag["in_race_metrics"]][
        ["raceId", "driverId", "code", "year", "round", "name"]
    ].sort_values(["year", "round"])
    fail_all.to_csv(out_dir / "fastf1_missingness_laps_join_failures.csv", index=False)

    cold_export = cold_start[
        ["raceId", "driverId", "code", "year", "round", "name", "is_first_2018plus_race"]
    ].sort_values(["year", "round"])
    cold_export.to_csv(out_dir / "fastf1_missingness_cold_start_rows.csv", index=False)

    if len(extra_drs):
        extra_drs.to_csv(out_dir / "fastf1_missingness_drs_gaps.csv", index=False)
    if len(extra_pit):
        extra_pit.to_csv(out_dir / "fastf1_missingness_pit_gaps.csv", index=False)

    print(f"\nReports saved under {out_dir}/")
    print("  - fastf1_missingness_attribution.csv")
    print("  - fastf1_missingness_weather_gaps_by_race.csv")
    print("  - fastf1_missingness_laps_join_failures.csv")
    print("  - fastf1_missingness_cold_start_rows.csv")


if __name__ == "__main__":
    main()
