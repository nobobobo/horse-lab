# Historical Dataset Replay Design

## Purpose

Build the first replayable historical-data loop for Horse Lab. The system should load small local CSV fixtures through repository implementations, run the existing market-implied baseline and Kelly backtest, and produce deterministic performance metrics.

This phase deliberately avoids direct JRA-VAN integration. JRA-VAN Data Lab. is the intended production data source, but the next implementation slice should first prove the repository contracts, data layout, and backtest orchestration with local files.

## Data Source Strategy

Primary production source:

- JRA-VAN Data Lab. for official JRA historical and live data.
- Target data includes races, entries, results, win odds snapshots, time-series odds, scratches, body weight, weather, track condition, and future feature inputs.
- A future `jra_van` adapter should map JV-Link/JRA-VAN records into the same domain schemas used by CSV repositories.

MVP development source:

- Local CSV fixtures committed to the repository.
- CSV files use the same field names and types expected by the domain schemas.
- Fixtures are small but intentionally include multiple races, multiple runners, multiple odds timestamps, and edge cases around future odds and same-race settlement.

Out of scope for this phase:

- JV-Link automation.
- Paid-data credential handling.
- Web scraping.
- Large historical storage.
- LightGBM training.
- DuckDB or Parquet storage.

## Scope

In scope:

- Define canonical CSV fixture formats for races, entries, results, odds, and feature rows.
- Implement CSV-backed repositories conforming to the existing repository protocols.
- Implement a replay orchestration function that loads data, runs `MarketImpliedProbabilityModel`, and passes predictions into `BacktestSimulator`.
- Add deterministic sample data under `sample_data/`.
- Add tests that prove the replay loop is point-in-time safe and deterministic.
- Document how JRA-VAN records will map into the same repository boundary in a future adapter phase.

Out of scope:

- Production-size ingestion.
- Non-win bet types.
- Feature engineering beyond minimal fixture features.
- Model training.
- CLI packaging, unless a tiny internal function needs a script-like test wrapper.

## Proposed File Structure

```text
sample_data/
  README.md
  races.csv
  entries.csv
  results.csv
  odds.csv
  features.csv

src/horse_lab/data/
  csv_repositories.py
  csv_parsing.py

src/horse_lab/pipelines/
  __init__.py
  replay.py

tests/data/
  test_csv_repositories.py
  test_csv_parsing.py

tests/pipelines/
  test_replay.py
```

## CSV Schemas

### `races.csv`

Required columns:

- `race_id`
- `race_date`
- `venue`
- `race_number`
- `name`
- `surface`
- `distance_m`
- `direction`
- `track_condition`
- `weather`
- `grade`
- `start_time`
- `field_size`

Date/time format:

- `race_date`: `YYYY-MM-DD`
- `start_time`: ISO datetime with timezone omitted for MVP, interpreted as local Japan time.

### `entries.csv`

Required columns:

- `runner_id`
- `race_id`
- `horse_id`
- `horse_number`
- `gate_number`
- `jockey_id`
- `trainer_id`
- `carried_weight_kg`
- `body_weight_kg`
- `body_weight_diff_kg`
- `age`
- `is_scratched`

### `results.csv`

Required columns:

- `race_id`
- `runner_id`
- `finish_position`
- `is_disqualified`
- `is_dead_heat`
- `final_time_seconds`
- `prize_jpy`

### `odds.csv`

Required columns:

- `race_id`
- `runner_id`
- `bet_type`
- `captured_at`
- `odds`
- `popularity_rank`
- `pool_size_jpy`
- `source`

Important fixture requirement:

- Include at least one future odds quote after replay `as_of`; tests must prove it is ignored.

### `features.csv`

Required columns:

- `race_id`
- `runner_id`
- `as_of`
- `feature_version`
- Any number of `feature__<name>` columns.

The CSV loader strips the `feature__` prefix and returns a `FeatureRow.values` mapping.

## Repository Behavior

CSV repositories are read-only.

- `CsvRaceRepository.list_races(start_date, end_date)` returns races whose `race_date` is within the inclusive range.
- `CsvOddsRepository.list_odds(race_ids, captured_at_or_before)` returns only odds with `captured_at <= captured_at_or_before`.
- `CsvResultRepository.list_results(race_ids)` returns results for the requested races.
- `CsvFeatureRepository.list_feature_rows(race_ids, feature_version, as_of)` returns features matching the version with `FeatureRow.as_of <= as_of`. If multiple feature rows exist for the same runner, it returns the latest one at or before `as_of`.

All repository methods preserve deterministic ordering:

- races by `race_date`, `venue`, `race_number`;
- odds by `race_id`, `runner_id`, `captured_at`;
- results by `race_id`, `runner_id`;
- feature rows by `race_id`, `runner_id`.

## Replay Pipeline

Add `run_market_replay`.

Inputs:

- repositories for races, odds, results, and features;
- `start_date`;
- `end_date`;
- `as_of`;
- `feature_version`;
- `initial_bankroll_jpy`;
- `KellyConfig`.

Flow:

1. Load races by date.
2. Load feature rows for those races as of replay `as_of`.
3. Load odds snapshots at or before replay `as_of`.
4. Load official results for those races.
5. Run `MarketImpliedProbabilityModel.predict`.
6. Run `BacktestSimulator.run`.
7. Return a `ReplayResult` containing races, predictions, backtest result, and summary.

The pipeline should not fetch network data. It only coordinates repository calls.

## Testing

Tests should cover:

- CSV parsers convert strings into domain schema objects.
- Boolean, int, float, enum, date, and datetime parsing.
- Odds repository filters out future odds.
- Feature repository returns the latest feature row at or before `as_of`.
- Replay pipeline returns deterministic metrics from `sample_data/`.
- Same-race settlement remains grouped by race through the replay path.

The full test suite should continue to run without external services.

## JRA-VAN Adapter Boundary

This phase should leave a clear adapter boundary:

```text
JRA-VAN / JV-Link records
        -> future JraVanRaceRepository / JraVanOddsRepository / ...
        -> existing domain schemas
        -> same replay pipeline
```

No production JRA-VAN code is built in this phase. The design goal is to make the future adapter a source replacement, not a rewrite of backtesting or modeling logic.

## Future Extensions

After this phase:

1. Add DuckDB-backed repositories for larger local historical data.
2. Add JRA-VAN adapter spike behind the existing repository protocols.
3. Add odds timing policies such as latest-before-start, fixed minutes before start, and closing odds.
4. Add per-race exposure caps and duplicate prediction policy.
5. Add first non-market tabular baseline.
