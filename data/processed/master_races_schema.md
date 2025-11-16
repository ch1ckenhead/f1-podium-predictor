# Master Races Schema

- **Rows**: 12,358
- **Columns**: 82
- **Description**: Combined F1 race data (1994+) with one row per (raceId, driverId)

## Column Sources

### results
- `resultId` (int64): 0.0% null
- `number` (object): 0.0% null
- `grid` (int64): 0.0% null
- `position` (object): 0.0% null
- `positionText` (object): 0.0% null
- `positionOrder` (int64): 0.0% null
- `points` (float64): 0.0% null
- `laps` (int64): 0.0% null
- `time` (object): 0.0% null
- `milliseconds` (object): 0.0% null
- ... and 5 more

### races
- `year` (int64): 0.0% null
- `round` (int64): 0.0% null
- `date` (datetime64[ns]): 0.0% null
- `name` (object): 0.0% null

### circuits
- `circuit_name` (object): 0.0% null
- `location` (object): 0.0% null
- `country` (object): 0.0% null
- `lat` (float64): 0.0% null
- `lng` (float64): 0.0% null
- `alt` (int64): 0.0% null

### drivers
- `driverRef` (object): 0.0% null
- `code` (object): 0.0% null
- `forename` (object): 0.0% null
- `surname` (object): 0.0% null
- `dob` (object): 0.0% null
- `nationality` (object): 0.0% null

### constructors
- `constructorRef` (object): 0.0% null
- `name` (object): 0.0% null
- `nationality` (object): 0.0% null

### driver_standings
- `driverStandingsId` (float64): 2.5% null
- `points` (float64): 0.0% null
- `position` (object): 0.0% null
- `positionText` (object): 0.0% null
- `wins` (float64): 2.5% null

### constructor_standings
- `constructorStandingsId` (float64): 1.0% null
- `points` (float64): 0.0% null
- `position` (object): 0.0% null
- `positionText` (object): 0.0% null
- `wins` (float64): 2.5% null

### constructor_results
- `constructorResultsId` (float64): 0.0% null
- `points` (float64): 0.0% null
- `status` (object): 0.0% null

### qualifying
- `qualifyId` (float64): 15.1% null
- `position` (object): 0.0% null
- `q1` (object): 15.1% null
- `q2` (object): 15.3% null
- `q3` (object): 15.5% null

### sprint_results
- `sprint_results_resultId` (float64): 97.1% null
- `sprint_results_constructorId` (float64): 97.1% null
- `sprint_results_number` (float64): 97.1% null
- `sprint_results_grid` (float64): 97.1% null
- `sprint_results_position` (object): 97.1% null
- `sprint_results_positionText` (object): 97.1% null
- `sprint_results_positionOrder` (float64): 97.1% null
- `sprint_results_points` (float64): 97.1% null
- `sprint_results_laps` (float64): 97.1% null
- `sprint_results_time` (object): 97.1% null
- ... and 4 more

### placeholders
- `lap_time_variance` (float64): 100.0% null
- `throttle_variance` (float64): 100.0% null
- `overtake_attempts` (float64): 100.0% null
- `avg_pit_stops` (float64): 100.0% null

### target
- `podium` (int64): 0.0% null

