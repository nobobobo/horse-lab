# MVP Baseline Design

## Purpose

Build the first working research loop for Horse Lab: convert runner-level odds into a market-implied probability baseline, size bets with conservative Kelly staking, and evaluate historical performance with a small backtest engine.

This MVP is intentionally narrow. It supports win bets first, uses dependency-light Python contracts, and creates the project structure needed for future LightGBM, Bayesian, and stacking models.

## Scope

In scope:

- Generate a modular project structure for data, features, models, betting, backtesting, evaluation, and tests.
- Implement a `MarketImpliedProbabilityModel` that predicts win probability from odds quotes.
- Implement Kelly stake sizing with fractional Kelly, maximum stake fraction, and minimum edge controls.
- Implement a simple historical backtest over runner-level predictions, odds, and results.
- Add focused tests for market probability normalization, Kelly sizing, and backtest accounting.

Out of scope for this MVP:

- Real JRA data acquisition.
- LightGBM or pandas-based feature training.
- Multi-bet tickets such as exacta, quinella, or trifecta.
- Live race-day automation.
- Model registry or remote artifact storage.

## Architecture

The MVP keeps the current dependency-light domain schema and adds focused modules:

```text
src/horse_lab/
  betting/
    __init__.py
    kelly.py
  data/
    __init__.py
    repositories.py
  features/
    __init__.py
    builders.py
  models/
    market.py
  backtesting/
    __init__.py
    simulator.py
  evaluation/
    __init__.py
    metrics.py

tests/
  betting/
  models/
  backtesting/
```

Each module has one responsibility:

- `betting.kelly`: converts probability and odds into stake fractions.
- `models.market`: implements a Level 0 baseline model using odds-derived probabilities.
- `backtesting.simulator`: simulates bets, bankroll movement, and race-level accounting.
- `evaluation.metrics`: summarizes ROI, hit rate, drawdown, and turnover.
- `data.repositories`: defines repository protocols for future CSV, SQLite, DuckDB, or API-backed storage.
- `features.builders`: defines a feature-builder protocol for future point-in-time feature generation.

## Data Flow

The initial working loop is:

```text
FeatureRow + OddsQuote + Result
        -> MarketImpliedProbabilityModel
        -> ModelPrediction
        -> Kelly staking
        -> BacktestResult
        -> ROI / max drawdown / hit rate / turnover
```

For the market model, `FeatureRow` is accepted to match the existing model interface, but the actual signal comes from `InferenceContext.odds`. This keeps the model compatible with future Level 0 model orchestration.

## Market-Implied Baseline

The model supports `BetType.WIN` only.

For each race:

1. Select the latest usable win odds quote at or before `InferenceContext.as_of`.
2. Convert decimal odds to raw implied probability with `1 / odds`.
3. Normalize probabilities within each race so runner probabilities sum to 1.0.
4. Return `ModelPrediction` rows with target `PredictionTarget.WIN_PROBABILITY`.

This creates a strong baseline because public odds already encode substantial crowd information. Future specialist models should be compared against this model before being trusted.

## Kelly Staking

The stake calculator uses:

```text
edge = probability * odds - 1
full_kelly = edge / (odds - 1)
stake_fraction = min(max(full_kelly * fractional_kelly, 0), max_stake_fraction)
```

MVP defaults:

- `fractional_kelly = 0.25`
- `max_stake_fraction = 0.02`
- `minimum_edge = 0.02`

If the edge is below `minimum_edge`, stake is zero. This avoids the common mistake of overbetting noisy edges.

## Backtest Engine

The simulator processes races in chronological order. For each candidate bet:

1. Use the model probability and selected market odds to compute stake.
2. Round stake to yen units.
3. Subtract stake from bankroll.
4. Add payout only when the runner wins.
5. Record bankroll after each bet.

The first implementation uses fixed decimal odds and single-runner win bets. It does not model pari-mutuel pool impact, slippage, taxes, exchange limits, or late odds movement.

## Error Handling

The MVP fails fast for invalid contracts:

- odds must be greater than 1.0;
- probabilities must be in `[0.0, 1.0]`;
- bankroll must be positive;
- Kelly config values must be non-negative;
- market predictions require matching win odds for the requested runners;
- backtests skip zero-stake recommendations rather than recording fake bets.

Missing optional data should not crash the pipeline unless it is needed for the requested action.

## Testing

Tests cover the first research loop:

- Market model normalizes odds-implied probabilities within a race.
- Market model uses the latest quote at or before `as_of`.
- Kelly returns zero when edge is too small or negative.
- Kelly caps stake at `max_stake_fraction`.
- Backtest produces correct bankroll changes for one winning and one losing bet.
- Backtest metrics compute ROI, hit rate, turnover, and max drawdown.

The MVP uses pytest. External ML dependencies are not required to run these tests.

## Future Extensions

After this MVP works, the next implementation slices should be:

1. CSV or DuckDB repositories for historical race, entry, result, and odds data.
2. Runner-level feature builders for recent form, distance/surface fit, field size, gate, and weight.
3. A tabular baseline using scikit-learn or LightGBM.
4. Out-of-fold prediction storage for Level 0 models.
5. A Level 1 meta learner with calibration.
6. Paper-trading automation for race-day inference.
