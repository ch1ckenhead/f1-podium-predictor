---
name: FastF1 to Kaggle Data Reproduction
overview: Create a new notebook to reproduce Kaggle-equivalent data from FastF1 sources, starting with 2024 validation, then producing 2025 data. The notebook will map all required columns, calculate standings, and handle sprint results in the same row structure as master_races_clean.csv.
todos:
  - id: setup_notebook
    content: Create new notebook 04.0_fastf1_data_reproduction.ipynb with setup cell (imports, paths, helper functions)
    status: completed
  - id: load_fastf1
    content: Implement load_fastf1_year() function to load and filter FastF1 data by year and session
    status: completed
    dependencies:
      - setup_notebook
  - id: extract_race_data
    content: Extract core race data from RESULTS (grid, position, points, laps, time, milliseconds, q1/q2/q3, statusId)
    status: completed
    dependencies:
      - load_fastf1
  - id: map_circuit
    content: Map circuit information (circuitId, lat, lng, round) from circuits.csv using event name matching
    status: completed
    dependencies:
      - extract_race_data
  - id: fastest_lap_laps
    content: Calculate fastestLap and fastestLapTime from ALL_LAPS data (minimum valid lap time per driver/race)
    status: completed
    dependencies:
      - extract_race_data
  - id: fastest_lap_speed
    content: Calculate fastestLapSpeed from TELEMETRY data (average speed of fastest lap)
    status: completed
    dependencies:
      - fastest_lap_laps
  - id: driver_age
    content: Map driver age from drivers.csv using code → driverId → dob lookup
    status: completed
    dependencies:
      - extract_race_data
  - id: sprint_results
    content: Extract sprint results from RESULTS and LAPS (Session == Sprint), merge to same row structure
    status: completed
    dependencies:
      - load_fastf1
  - id: standings_calc
    content: Calculate driver and constructor standings (points, position) from cumulative points, including PRE_RACE versions
    status: completed
    dependencies:
      - extract_race_data
  - id: derived_features
    content: Add derived features (podium, status_category) and prepare structure for rolling features
    status: completed
    dependencies:
      - standings_calc
  - id: column_ordering
    content: Reorder columns to match master_races_clean.csv structure and select reproduced columns only
    status: completed
    dependencies:
      - derived_features
  - id: validate_2024
    content: Validate 2024 reproduced data against master_races_clean.csv (row counts, column matches, standings accuracy)
    status: completed
    dependencies:
      - column_ordering
  - id: produce_2025
    content: Produce 2025 data using same pipeline, output to fastf1_2025_reproduced.csv
    status: completed
    dependencies:
      - validate_2024
  - id: append_save
    content: "Append 2025 data to master_races_clean.csv and save (note: rolling features need recalculation)"
    status: completed
    dependencies:
      - produce_2025
---

# FastF1 to Kaggle Data Reproduction Plan

## Overview

Create a new notebook `04.0_fastf1_data_reproduction.ipynb` (or user-specified name) to reproduce Kaggle-equivalent data from FastF1 sources. Keep `03.5_fastf1_modelling_pre_validation.ipynb` for validation checks only.

## Architecture

### Data Flow

```
FastF1 Data Sources:
├── ALL_RESULTS_*.csv (Race + Sprint sessions)
├── ALL_LAPS_*.csv (Lap times, fastest lap calculations)
├── ALL_TELEMETRY_*.csv (Speed data for fastest lap)
└── circuits.csv, drivers.csv (Lookup tables)

↓ Processing Steps ↓

Output: Kaggle-equivalent dataset
├── 2024 validation (compare against master_races_clean.csv)
└── 2025 production (append to master for rolling calculations)
```

## Implementation Structure

### Notebook: `04.0_fastf1_data_reproduction.ipynb`

#### Cell 1: Setup and Imports

- Import pandas, numpy, pathlib, re
- Set up paths (PROJECT_ROOT, FASTF1_DIR, KAGGLE_DIR, PROCESSED_DATA_DIR)
- Load lookup tables: `circuits.csv`, `drivers.csv`
- Create helper functions for time parsing (reuse from validation notebook)

#### Cell 2: Load FastF1 Data (Year-Specific)

- Function: `load_fastf1_year(year)` 
- Load ALL_RESULTS, ALL_LAPS, ALL_TELEMETRY for specified year
- Filter to Session == 'R' for race data, Session == 'Sprint' for sprint data
- Return dictionaries: `{'results': df, 'laps': df, 'telemetry': df}`

#### Cell 3: Core Race Data Extraction

- Function: `extract_race_data(fastf1_results, year)`
- Extract from RESULTS (Session == 'R'):
  - `grid` → `GridPosition`
  - `position` → `Position` (handle 'R' for retired → nan)
  - `points` → `Points`
  - `laps` → `Laps`
  - `time` → `Time` (parse to standard format)
  - `milliseconds` → convert `Time` to milliseconds
  - `q1`, `q2`, `q3` → `Q1`, `Q2`, `Q3` (from Session == 'Q')
  - `statusId` → map `Status` to `statusId` using status.csv lookup
  - `status_category` use same classifications as we did on the original kaggle data
- Match by: (Year, Event, Abbreviation)
- Output: DataFrame with one row per (Year, Event, Driver)

#### Cell 4: Circuit Mapping

- Function: `map_circuit_info(race_df, circuits_df)`
- Map race `Event` name → `circuitId` using `circuits.csv`
- Extract `lat`, `lng` from circuits
- Handle name normalization (e.g., "Australian Grand Prix" → circuit lookup)
- Add `round`: Sequential number per year (1, 2, 3...) based on date ordering

#### Cell 5: Fastest Lap Calculations (from LAPS data)

- Function: `calculate_fastest_lap_features(race_df, laps_df)`
- For each (Year, Event, Driver):
  - Filter LAPS to Session == 'R', that driver
  - Find minimum `LapTime` (exclude 0 or invalid times)
  - `fastestLap` → lap number of fastest lap
  - `fastestLapTime` → time of fastest lap
- Use chunked reading if needed for large files
- Match back to race_df

#### Cell 6: Fastest Lap Speed (from TELEMETRY)

- Function: `calculate_fastest_lap_speed(race_df, telemetry_df, laps_df)`
- For each driver's fastest lap:
  - Telemetry data is timed at roughly 200-300ms intervals
  - Will need to use *LapStartTime,*LapStartDate and laptime from LAPS data to know which speeds to average in the TELEMETRY data.
  - Filter TELEMETRY to that lap (using lap start time and end time boundaries)
  - Calculate average `Speed` for that lap
  - `fastestLapSpeed` → average speed
- Use chunked reading for telemetry

#### Cell 7: Driver Age Mapping

- Function: `add_driver_age(race_df, drivers_df)`
- Map `code` → `driverId` → `dob` from drivers.csv
- Calculate `driver_age` = race date - dob
- Handle missing drivers gracefully

#### Cell 8: Sprint Results Extraction

- Remember the sprint results will be attached to (in the same row as) race data for the full race so match by Year, Event, Driver as usual
- Function: `extract_sprint_data(fastf1_results, laps_df, year)`
- Filter RESULTS to Session == 'Sprint'
- Extract: `sprint_results_grid`, `sprint_results_positionOrder`, `sprint_results_points`
- From LAPS (Session == 'Sprint'):
  - `sprint_results_laps` → count laps
  - `sprint_results_time` → total time
  - `sprint_results_milliseconds` → convert time to ms
  - `sprint_results_fastestLap` → lap number of fastest lap
  - `sprint_results_fastestLapTime` → time of fastest lap
- Map `Status` → `sprint_results_statusId`
- Merge back to race_df on (Year, Event, Driver)

#### Cell 9: Standings Calculations

- Function: `calculate_standings(race_df)`
- Sort by `year`, `round`, `date`
- `driver_standings_points`: `groupby('driverId')['points'].cumsum()`
- `driver_standings_position`: Rank by `driver_standings_points` per race
- `constructor_standings_points`: Sum of `driver_standings_points` per constructor per race
- `constructor_standings_position`: Rank by `constructor_standings_points` per race
- Calculate PRE_RACE versions using `.shift(1)` grouped by driver/constructor

#### Cell 10: Additional Features

- Function: `add_derived_features(race_df)`
- `podium`: 1 if `position` in [1,2,3], else 0
- `status_category`: Map `statusId` to category (Finished, DNF, etc.)
- Add placeholder columns for rolling features (will be calculated in separate step)

#### Cell 11: Column Ordering and Final Structure

- Function: `reorder_columns(race_df)`
- Ensure column order matches `master_races_clean.csv` structure
- Select only columns that are being reproduced (not all 61 columns)
- Add missing columns as NaN if not yet calculated

#### Cell 12: 2024 Validation

- Load `master_races_clean.csv` filtered to year == 2024
- Compare reproduced FastF1 data against master
- Validation checks:
  - Row count match
  - Column match rates (reuse validation functions)
  - Standings accuracy
  - Time format consistency
- Report discrepancies

#### Cell 13: 2025 Production

- Run all functions for year == 2025
- Output: `fastf1_2025_reproduced.csv`
- Note: Some features may be NaN (rolling averages need historical data)

#### Cell 14: Append and Save

- Load `master_races_clean.csv`
- Append 2025 data
- Save as `master_races_clean_with_2025.csv` (or update existing)
- Note: Rolling features will need recalculation after append

## Key Implementation Details

### Time Parsing

- Reuse `parse_time_to_ms()` from validation notebook
- Handle FastF1 format: `0 days 00:39:09.686000`
- Convert to milliseconds: `pd.to_timedelta().total_seconds() * 1000`

### Circuit Name Matching

- Normalize names: remove "Grand Prix", handle variations
- Use fuzzy matching if exact match fails
- Create lookup dictionary: `{event_name: circuitId}`

### Fastest Lap Logic

- Filter LAPS to valid lap times (exclude 0, NaN, deleted laps)
- Use `LapTime` column, find minimum per driver/race
- Match lap number back to telemetry for speed calculation

### Standings Calculation

- Must sort by (year, round, date) before cumsum
- Handle driver changes mid-season
- Constructor points = sum of both drivers' standings points

### Sprint Results

- Same row structure as master
- Merge sprint data to race rows on (Year, Event, Driver)
- Handle races without sprints (NaN for sprint columns)

## File Structure

```
notebooks/
├── 03.5_fastf1_modelling_pre_validation.ipynb (validation only)
└── 04.0_fastf1_data_reproduction.ipynb (NEW - reproduction logic)

data/
├── processed/
│   ├── master_races_clean.csv (reference)
│   └── fastf1_2025_reproduced.csv (output)
```

## Testing Strategy

1. **2024 Validation**: Compare every column against master
2. **Edge Cases**: Handle missing data, driver changes, sprint races
3. **Performance**: Use chunked reading for telemetry (5-6GB files)
4. **Data Quality**: Log all mismatches, handle gracefully

## Next Steps After Reproduction

- Calculate rolling features (using existing notebook logic)
- Append to master dataset
- Recalculate all historical rolling averages with new 2025 data
- Design incremental update process for ongoing 2025 races