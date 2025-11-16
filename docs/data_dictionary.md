# Data Dictionary

| Dataset | Source | Description | Key Columns | Update Frequency |
|---------|--------|-------------|-------------|------------------|
| races.csv | Ergast API export | Historical race metadata including season, round, circuit, and date. | raceId, year, round | On demand |
| results.csv | Ergast API export | Race results per driver including finishing position and status. | resultId, raceId, driverId, constructorId | On demand |
| qualifying.csv | Ergast API export | Qualifying session results providing grid positions and lap times. | qualifyId, raceId, driverId | On demand |
| driver_standings.csv | Ergast API export | Driver championship standings with points and wins totals. | driverStandingsId, raceId, driverId | On demand |
| constructor_standings.csv | Ergast API export | Constructor championship standings with points and wins totals. | constructorStandingsId, raceId, constructorId | On demand |
| pit_stops.csv | Ergast API export | Pit stop times and lap numbers per driver. | raceId, driverId, stop | On demand |
| weather_historical.csv | Third-party weather dataset | Historical weather observations aligned to race start times. | raceId, timestamp | Seasonal |
| weather_forecast.csv | Forecast provider | Forecasted weather metrics for Las Vegas GP weekend. | session, forecast_time | Daily |
| practice_sessions.csv | Team telemetry summary | Aggregated practice session deltas and long-run pace estimates. | sessionId, raceId, driverId | Weekend |

> Update this table as new raw files are added to data/raw/*.
