# Phase 3.5 Data Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the replay dataset before Phase 4 ensemble work by preserving odds time series, adding payout/pool settlement artifacts, and expanding point-in-time safe features.

**Architecture:** Keep `odds.csv` as the compatibility latest-odds file, add `odds_timeseries.csv` and `payouts.csv` as enriched artifacts, and extend `features.csv` with market movement plus historical horse/person stats. Existing CSV repository and model contracts remain usable.

**Tech Stack:** Python 3.9 standard library, existing Horse Lab schemas, JRA-VAN staging/replay pipeline, pytest.

---

## File Structure

- Modify: `src/horse_lab/data/jravan/replay_dataset.py`
  - Add odds time series and derived payout outputs.
  - Add market movement features and categorical-safe person ID rendering.
  - Extend replay report counts.
- Modify: `src/horse_lab/features/past_performance.py`
  - Add expanded past-performance and jockey/trainer stats.
- Modify: `src/horse_lab/data/jravan/__init__.py`
  - Export new replay constants if needed.
- Modify: `tests/data/test_jravan_replay_dataset.py`
  - Cover new output CSVs and feature values.
- Modify: `tests/features/test_past_performance.py`
  - Cover expanded time-safe historical features.
- Modify: `docs/architecture.md`
  - Add Phase 3.5 and tighten obsolete detail.
- Modify: `docs/baseline_analysis.md`
  - Point next work toward enriched data and Phase 4 readiness.

---

### Task 1: Replay Enriched Artifacts

- [x] Add failing tests proving `odds_timeseries.csv` keeps all selected win quotes and `payouts.csv` includes per-runner settlement proxy rows.
- [x] Implement replay export paths for `odds_timeseries` and `payouts`.
- [x] Extend replay report output counts with `odds_timeseries` and `payouts`.
- [x] Keep existing `odds.csv` latest-per-runner behavior unchanged.

### Task 2: Market Movement Features

- [x] Add failing tests for `odds_open`, `odds_latest`, `odds_min`, `odds_max`, `odds_snapshot_count`, `odds_change_open_to_latest`, `implied_probability_change_open_to_latest`, and `pool_size_latest_jpy`.
- [x] Build these values from selected runner win odds time series.
- [x] Include the feature names in `FEATURE_NAMES` and `features.csv`.

### Task 3: Expanded Historical Features

- [x] Add failing tests for new horse-level features: top-3 rate, average prize, average final time, same-distance count/win rate, distance delta from last, last surface/grade/body weight.
- [x] Add failing tests for jockey/trainer point-in-time run counts and win rates.
- [x] Implement the features using only races before the target race start time.
- [x] Keep existing past-performance feature names and values compatible.

### Task 4: Categorical-Safe Person IDs

- [x] Add test proving `jockey_id` and `trainer_id` features render as non-numeric strings.
- [x] Prefix person IDs in replay features, e.g. `jockey:01020` and `trainer:04050`.
- [x] Re-run LightGBM tests to confirm categorical inference still works.

### Task 5: Docs and Verification

- [x] Update `docs/architecture.md` with Phase 3.5 scope and current completed phases.
- [x] Trim redundant or stale operational detail where it obscures roadmap status.
- [x] Update `docs/baseline_analysis.md` next-step recommendations.
- [x] Run focused tests, then full `PYTHONPATH=src python3 -m pytest tests`.
- [x] Run one real-data smoke build if local data exists.
- [x] Commit and push the completed Phase 3.5 work.
