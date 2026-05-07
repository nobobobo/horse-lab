# MVP Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working pure-Python baseline loop that turns win odds into market-implied probabilities, sizes bets with conservative Kelly staking, and evaluates a simple historical backtest.

**Architecture:** Keep the existing schema and model base contracts as the shared domain layer. Add small modules for betting, market baseline prediction, metrics, backtesting, repository protocols, and feature-builder protocols, each with focused pytest coverage.

**Tech Stack:** Python 3.9+, standard library dataclasses/ABC/Protocol, pytest for tests, no runtime ML dependency for the MVP baseline.

---

## File Structure

- Modify: `pyproject.toml`
  - Adds pytest as a development dependency and configures test discovery.
- Create: `tests/test_package_imports.py`
  - Verifies the package imports and public baseline symbols are reachable.
- Create: `src/horse_lab/betting/__init__.py`
  - Exports Kelly staking API.
- Create: `src/horse_lab/betting/kelly.py`
  - Converts probability, odds, and bankroll into a capped stake decision.
- Create: `tests/betting/test_kelly.py`
  - Covers negative edge, minimum edge, cap, and validation behavior.
- Create: `src/horse_lab/models/market.py`
  - Implements the odds-implied probability baseline model.
- Modify: `src/horse_lab/models/__init__.py`
  - Exports `MarketImpliedProbabilityModel`.
- Create: `tests/models/test_market_model.py`
  - Covers race-level normalization and latest quote selection.
- Create: `src/horse_lab/evaluation/__init__.py`
  - Exports performance metric API.
- Create: `src/horse_lab/evaluation/metrics.py`
  - Computes ROI, hit rate, turnover, and max drawdown.
- Create: `tests/evaluation/test_metrics.py`
  - Covers metric arithmetic and empty-result safety.
- Create: `src/horse_lab/backtesting/__init__.py`
  - Exports simulator API.
- Create: `src/horse_lab/backtesting/simulator.py`
  - Applies Kelly sizing to historical predictions, odds, and results.
- Create: `tests/backtesting/test_simulator.py`
  - Covers one winning and one losing bet with bankroll accounting.
- Create: `src/horse_lab/data/__init__.py`
  - Exports repository protocols.
- Create: `src/horse_lab/data/repositories.py`
  - Defines storage contracts for races, odds, results, and feature rows.
- Create: `src/horse_lab/features/__init__.py`
  - Exports feature builder protocol.
- Create: `src/horse_lab/features/builders.py`
  - Defines point-in-time feature builder contract.

---

### Task 1: Test Harness And Package Smoke Test

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_package_imports.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_package_imports.py`:

```python
from horse_lab.schemas import BetType, PredictionTarget


def test_core_package_imports():
    assert BetType.WIN.value == "win"
    assert PredictionTarget.WIN_PROBABILITY.value == "win_probability"


def test_baseline_modules_import():
    from horse_lab.betting import KellyConfig, calculate_kelly_stake
    from horse_lab.backtesting import BacktestConfig, BacktestSimulator
    from horse_lab.evaluation import PerformanceSummary
    from horse_lab.models import MarketImpliedProbabilityModel

    assert KellyConfig.__name__ == "KellyConfig"
    assert callable(calculate_kelly_stake)
    assert BacktestConfig.__name__ == "BacktestConfig"
    assert BacktestSimulator.__name__ == "BacktestSimulator"
    assert PerformanceSummary.__name__ == "PerformanceSummary"
    assert MarketImpliedProbabilityModel.__name__ == "MarketImpliedProbabilityModel"
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
PYTHONPATH=src pytest tests/test_package_imports.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'horse_lab.betting'`.

- [ ] **Step 3: Update project test configuration**

Replace `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "horse-lab"
version = "0.1.0"
description = "Japanese horse-racing ensemble modeling and simulation platform."
readme = "docs/architecture.md"
requires-python = ">=3.9"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
ml = [
    "lightgbm>=4.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Add temporary import targets**

Create `src/horse_lab/betting/__init__.py`:

```python
"""Bet sizing helpers."""

from horse_lab.betting.kelly import KellyConfig, StakeDecision, calculate_kelly_stake

__all__ = [
    "KellyConfig",
    "StakeDecision",
    "calculate_kelly_stake",
]
```

Create `src/horse_lab/betting/kelly.py`:

```python
"""Kelly staking primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KellyConfig:
    fractional_kelly: float = 0.25
    max_stake_fraction: float = 0.02
    minimum_edge: float = 0.02
    stake_unit_jpy: int = 100


@dataclass(frozen=True)
class StakeDecision:
    probability: float
    odds: float
    edge: float
    full_kelly_fraction: float
    stake_fraction: float
    stake_jpy: int


def calculate_kelly_stake(
    *,
    probability: float,
    odds: float,
    bankroll_jpy: int,
    config: KellyConfig = KellyConfig(),
) -> StakeDecision:
    return StakeDecision(
        probability=probability,
        odds=odds,
        edge=0.0,
        full_kelly_fraction=0.0,
        stake_fraction=0.0,
        stake_jpy=0,
    )
```

Create `src/horse_lab/backtesting/__init__.py`:

```python
"""Historical betting simulation."""

from horse_lab.backtesting.simulator import BacktestConfig, BacktestResult, BacktestSimulator

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestSimulator",
]
```

Create `src/horse_lab/backtesting/simulator.py`:

```python
"""Minimal backtest simulator classes for import wiring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll_jpy: int = 100_000


@dataclass(frozen=True)
class BacktestResult:
    final_bankroll_jpy: int


class BacktestSimulator:
    pass
```

Create `src/horse_lab/evaluation/__init__.py`:

```python
"""Evaluation metric helpers."""

from horse_lab.evaluation.metrics import PerformanceSummary

__all__ = ["PerformanceSummary"]
```

Create `src/horse_lab/evaluation/metrics.py`:

```python
"""Performance summary primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSummary:
    total_bets: int
```

Create `src/horse_lab/models/market.py`:

```python
"""Market-implied probability baseline model."""

from __future__ import annotations


class MarketImpliedProbabilityModel:
    pass
```

Modify `src/horse_lab/models/__init__.py` to include:

```python
"""Model interfaces and base classes."""

from horse_lab.models.base import (
    BaseLevel0Model,
    BaseMetaModel,
    BaseRaceModel,
    InferenceContext,
    ModelArtifact,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.models.market import MarketImpliedProbabilityModel

__all__ = [
    "BaseLevel0Model",
    "BaseMetaModel",
    "BaseRaceModel",
    "InferenceContext",
    "MarketImpliedProbabilityModel",
    "ModelArtifact",
    "TrainingContext",
    "TrainingDataset",
]
```

- [ ] **Step 5: Run the smoke test to verify it passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_package_imports.py -v
```

Expected: PASS with `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_package_imports.py src/horse_lab/betting src/horse_lab/backtesting src/horse_lab/evaluation src/horse_lab/models/market.py src/horse_lab/models/__init__.py
git commit -m "test: add baseline package smoke test"
```

---

### Task 2: Kelly Stake Sizing

**Files:**
- Modify: `src/horse_lab/betting/kelly.py`
- Modify: `src/horse_lab/betting/__init__.py`
- Create: `tests/betting/test_kelly.py`

- [ ] **Step 1: Write failing Kelly tests**

Create `tests/betting/test_kelly.py`:

```python
import pytest

from horse_lab.betting import KellyConfig, calculate_edge, calculate_kelly_stake


def test_calculate_edge_uses_probability_times_odds_minus_one():
    assert calculate_edge(probability=0.4, odds=3.0) == pytest.approx(0.2)


def test_kelly_returns_zero_for_negative_edge():
    decision = calculate_kelly_stake(
        probability=0.2,
        odds=3.0,
        bankroll_jpy=100_000,
        config=KellyConfig(minimum_edge=0.0),
    )

    assert decision.edge == pytest.approx(-0.4)
    assert decision.full_kelly_fraction == 0.0
    assert decision.stake_fraction == 0.0
    assert decision.stake_jpy == 0


def test_kelly_returns_zero_below_minimum_edge():
    decision = calculate_kelly_stake(
        probability=0.34,
        odds=3.0,
        bankroll_jpy=100_000,
        config=KellyConfig(minimum_edge=0.03),
    )

    assert decision.edge == pytest.approx(0.02)
    assert decision.stake_fraction == 0.0
    assert decision.stake_jpy == 0


def test_kelly_caps_fraction_and_rounds_to_stake_unit():
    decision = calculate_kelly_stake(
        probability=0.6,
        odds=3.0,
        bankroll_jpy=101_000,
        config=KellyConfig(
            fractional_kelly=1.0,
            max_stake_fraction=0.05,
            minimum_edge=0.0,
            stake_unit_jpy=100,
        ),
    )

    assert decision.edge == pytest.approx(0.8)
    assert decision.full_kelly_fraction == pytest.approx(0.4)
    assert decision.stake_fraction == pytest.approx(0.05)
    assert decision.stake_jpy == 5_000


def test_kelly_validates_probability_odds_bankroll_and_config():
    with pytest.raises(ValueError, match="probability"):
        calculate_kelly_stake(probability=1.1, odds=2.0, bankroll_jpy=10_000)

    with pytest.raises(ValueError, match="odds"):
        calculate_kelly_stake(probability=0.5, odds=1.0, bankroll_jpy=10_000)

    with pytest.raises(ValueError, match="bankroll"):
        calculate_kelly_stake(probability=0.5, odds=2.0, bankroll_jpy=0)

    with pytest.raises(ValueError, match="fractional_kelly"):
        KellyConfig(fractional_kelly=-0.1)

    with pytest.raises(ValueError, match="stake_unit_jpy"):
        KellyConfig(stake_unit_jpy=0)
```

- [ ] **Step 2: Run Kelly tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/betting/test_kelly.py -v
```

Expected: FAIL with `ImportError` for `calculate_edge` or assertion failures from the temporary implementation.

- [ ] **Step 3: Implement Kelly stake sizing**

Replace `src/horse_lab/betting/kelly.py` with:

```python
"""Kelly staking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class KellyConfig:
    fractional_kelly: float = 0.25
    max_stake_fraction: float = 0.02
    minimum_edge: float = 0.02
    stake_unit_jpy: int = 100

    def __post_init__(self) -> None:
        if self.fractional_kelly < 0.0:
            raise ValueError("fractional_kelly must be non-negative")
        if self.max_stake_fraction < 0.0:
            raise ValueError("max_stake_fraction must be non-negative")
        if self.minimum_edge < 0.0:
            raise ValueError("minimum_edge must be non-negative")
        if self.stake_unit_jpy <= 0:
            raise ValueError("stake_unit_jpy must be positive")


@dataclass(frozen=True)
class StakeDecision:
    probability: float
    odds: float
    edge: float
    full_kelly_fraction: float
    stake_fraction: float
    stake_jpy: int


def calculate_edge(*, probability: float, odds: float) -> float:
    _validate_probability(probability)
    _validate_odds(odds)
    return probability * odds - 1.0


def calculate_kelly_stake(
    *,
    probability: float,
    odds: float,
    bankroll_jpy: int,
    config: KellyConfig = KellyConfig(),
) -> StakeDecision:
    _validate_probability(probability)
    _validate_odds(odds)
    if bankroll_jpy <= 0:
        raise ValueError("bankroll_jpy must be positive")

    edge = calculate_edge(probability=probability, odds=odds)
    if edge < config.minimum_edge:
        return StakeDecision(
            probability=probability,
            odds=odds,
            edge=edge,
            full_kelly_fraction=0.0,
            stake_fraction=0.0,
            stake_jpy=0,
        )

    full_kelly_fraction = max(edge / (odds - 1.0), 0.0)
    stake_fraction = min(
        full_kelly_fraction * config.fractional_kelly,
        config.max_stake_fraction,
    )
    raw_stake_jpy = bankroll_jpy * stake_fraction
    stake_jpy = floor(raw_stake_jpy / config.stake_unit_jpy) * config.stake_unit_jpy

    return StakeDecision(
        probability=probability,
        odds=odds,
        edge=edge,
        full_kelly_fraction=full_kelly_fraction,
        stake_fraction=stake_fraction,
        stake_jpy=stake_jpy,
    )


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0.0 and 1.0")


def _validate_odds(odds: float) -> None:
    if odds <= 1.0:
        raise ValueError("odds must be greater than 1.0")
```

Replace `src/horse_lab/betting/__init__.py` with:

```python
"""Bet sizing helpers."""

from horse_lab.betting.kelly import (
    KellyConfig,
    StakeDecision,
    calculate_edge,
    calculate_kelly_stake,
)

__all__ = [
    "KellyConfig",
    "StakeDecision",
    "calculate_edge",
    "calculate_kelly_stake",
]
```

- [ ] **Step 4: Run Kelly tests to verify they pass**

Run:

```bash
PYTHONPATH=src pytest tests/betting/test_kelly.py -v
```

Expected: PASS with `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/horse_lab/betting tests/betting/test_kelly.py
git commit -m "feat: add kelly stake sizing"
```

---

### Task 3: Market-Implied Probability Model

**Files:**
- Modify: `src/horse_lab/models/market.py`
- Modify: `src/horse_lab/models/__init__.py`
- Create: `tests/models/test_market_model.py`

- [ ] **Step 1: Write failing market model tests**

Create `tests/models/test_market_model.py`:

```python
import datetime as dt

import pytest

from horse_lab.models import InferenceContext, MarketImpliedProbabilityModel
from horse_lab.schemas import (
    BetType,
    FeatureName,
    FeatureRow,
    OddsQuote,
    PredictionTarget,
    RaceId,
    RunnerId,
)


def _feature_row(race_id: str, runner_id: str, as_of: dt.datetime) -> FeatureRow:
    return FeatureRow(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        as_of=as_of,
        feature_version="test-v1",
        values={FeatureName("bias"): 0.0},
    )


def _quote(
    race_id: str,
    runner_id: str,
    captured_at: dt.datetime,
    odds: float,
) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        bet_type=BetType.WIN,
        captured_at=captured_at,
        odds=odds,
    )


def test_market_model_normalizes_implied_probabilities_within_race():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    model = MarketImpliedProbabilityModel(model_version="test")

    predictions = model.predict(
        [
            _feature_row("race-1", "runner-1", as_of),
            _feature_row("race-1", "runner-2", as_of),
        ],
        context=InferenceContext(
            as_of=as_of,
            feature_version="test-v1",
            odds=[
                _quote("race-1", "runner-1", as_of, 2.0),
                _quote("race-1", "runner-2", as_of, 4.0),
            ],
        ),
    )

    by_runner = {prediction.runner_id: prediction for prediction in predictions}

    assert sum(prediction.probability for prediction in predictions) == pytest.approx(1.0)
    assert by_runner[RunnerId("runner-1")].probability == pytest.approx(2 / 3)
    assert by_runner[RunnerId("runner-2")].probability == pytest.approx(1 / 3)
    assert all(
        prediction.target == PredictionTarget.WIN_PROBABILITY
        for prediction in predictions
    )


def test_market_model_uses_latest_quote_at_or_before_as_of():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    model = MarketImpliedProbabilityModel(model_version="test")

    predictions = model.predict(
        [
            _feature_row("race-1", "runner-1", as_of),
            _feature_row("race-1", "runner-2", as_of),
        ],
        context=InferenceContext(
            as_of=as_of,
            feature_version="test-v1",
            odds=[
                _quote("race-1", "runner-1", dt.datetime(2026, 5, 7, 14, 40), 10.0),
                _quote("race-1", "runner-1", dt.datetime(2026, 5, 7, 14, 50), 2.0),
                _quote("race-1", "runner-1", dt.datetime(2026, 5, 7, 15, 0), 100.0),
                _quote("race-1", "runner-2", dt.datetime(2026, 5, 7, 14, 50), 4.0),
            ],
        ),
    )

    by_runner = {prediction.runner_id: prediction for prediction in predictions}

    assert by_runner[RunnerId("runner-1")].metadata["market_odds"] == 2.0
    assert by_runner[RunnerId("runner-1")].probability == pytest.approx(2 / 3)


def test_market_model_requires_matching_win_odds():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    model = MarketImpliedProbabilityModel(model_version="test")

    with pytest.raises(ValueError, match="Missing win odds"):
        model.predict(
            [_feature_row("race-1", "runner-1", as_of)],
            context=InferenceContext(as_of=as_of, feature_version="test-v1", odds=[]),
        )
```

- [ ] **Step 2: Run market tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/models/test_market_model.py -v
```

Expected: FAIL because the temporary market model has no compatible constructor or `predict` implementation.

- [ ] **Step 3: Implement the market model**

Replace `src/horse_lab/models/market.py` with:

```python
"""Market-implied probability baseline model."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from horse_lab.models.base import (
    BaseLevel0Model,
    InferenceContext,
    ModelArtifact,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.schemas import (
    BetType,
    FeatureRow,
    ModelName,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    RaceId,
    RunnerId,
    TrainingLabel,
)


class MarketImpliedProbabilityModel(BaseLevel0Model):
    """Baseline model that converts win odds into normalized probabilities."""

    def __init__(self, *, model_version: str = "market-implied-v1") -> None:
        super().__init__(
            model_name=ModelName("market_implied_probability"),
            model_version=model_version,
            target=PredictionTarget.WIN_PROBABILITY,
        )

    def fit(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
    ) -> ModelArtifact:
        artifact = ModelArtifact(
            model_name=self.model_name,
            model_version=self.model_version,
            trained_at=datetime.utcnow(),
            feature_version=context.feature_version,
            target=self.target,
            metrics={},
            metadata={"rows": float(len(dataset.feature_rows))},
        )
        self._artifact = artifact
        return artifact

    def predict(
        self,
        feature_rows: Sequence[FeatureRow],
        *,
        context: InferenceContext,
    ) -> Sequence[ModelPrediction]:
        latest_quotes = _latest_win_quotes_by_runner(context.odds, context.as_of)
        grouped_raw_probabilities: dict[RaceId, list[tuple[FeatureRow, OddsQuote, float]]] = (
            defaultdict(list)
        )

        for row in feature_rows:
            key = (row.race_id, row.runner_id)
            quote = latest_quotes.get(key)
            if quote is None:
                raise ValueError(
                    "Missing win odds for "
                    f"race_id={row.race_id!r}, runner_id={row.runner_id!r}"
                )
            grouped_raw_probabilities[row.race_id].append((row, quote, 1.0 / quote.odds))

        predictions: list[ModelPrediction] = []
        for race_id, rows in grouped_raw_probabilities.items():
            raw_sum = sum(raw_probability for _, _, raw_probability in rows)
            if raw_sum <= 0.0:
                raise ValueError(f"Raw implied probability sum is zero for race_id={race_id!r}")

            for row, quote, raw_probability in rows:
                predictions.append(
                    ModelPrediction(
                        race_id=row.race_id,
                        runner_id=row.runner_id,
                        model_name=self.model_name,
                        model_version=self.model_version,
                        target=self.target,
                        probability=raw_probability / raw_sum,
                        as_of=context.as_of,
                        metadata={
                            "market_odds": quote.odds,
                            "odds_captured_at": quote.captured_at.isoformat(),
                            "raw_implied_probability": raw_probability,
                        },
                    )
                )

        return predictions

    def fit_predict_oof(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
        n_splits: int,
    ) -> Sequence[ModelPrediction]:
        raise NotImplementedError(
            "MarketImpliedProbabilityModel requires odds snapshots in "
            "InferenceContext for prediction; OOF support will be added with "
            "the prediction store."
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": str(self.model_name),
            "model_version": self.model_version,
            "target": self.target.value,
        }
        (path / "market_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MarketImpliedProbabilityModel":
        payload = json.loads((path / "market_model.json").read_text(encoding="utf-8"))
        return cls(model_version=payload["model_version"])


def _latest_win_quotes_by_runner(
    odds: Sequence[OddsQuote],
    as_of: datetime,
) -> Mapping[tuple[RaceId, RunnerId], OddsQuote]:
    latest: dict[tuple[RaceId, RunnerId], OddsQuote] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN or quote.captured_at > as_of:
            continue
        key = (quote.race_id, quote.runner_id)
        previous = latest.get(key)
        if previous is None or quote.captured_at > previous.captured_at:
            latest[key] = quote
    return latest
```

- [ ] **Step 4: Run market tests to verify they pass**

Run:

```bash
PYTHONPATH=src pytest tests/models/test_market_model.py -v
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/horse_lab/models/market.py src/horse_lab/models/__init__.py tests/models/test_market_model.py
git commit -m "feat: add market implied probability model"
```

---

### Task 4: Evaluation Metrics

**Files:**
- Modify: `src/horse_lab/evaluation/metrics.py`
- Modify: `src/horse_lab/evaluation/__init__.py`
- Create: `tests/evaluation/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/evaluation/test_metrics.py`:

```python
import pytest

from horse_lab.evaluation import PerformanceSummary, compute_max_drawdown, summarize_performance


def test_compute_max_drawdown_uses_peak_to_trough_fraction():
    assert compute_max_drawdown([10_000, 12_000, 10_800, 13_000]) == pytest.approx(0.1)


def test_compute_max_drawdown_returns_zero_without_drawdown():
    assert compute_max_drawdown([10_000, 11_000, 12_000]) == 0.0
    assert compute_max_drawdown([]) == 0.0


def test_summarize_performance_computes_roi_hit_rate_turnover_and_profit():
    summary = summarize_performance(
        initial_bankroll_jpy=10_000,
        final_bankroll_jpy=10_800,
        stakes_jpy=[1_000, 1_200],
        payouts_jpy=[3_000, 0],
        wins=[True, False],
        bankroll_curve_jpy=[10_000, 12_000, 10_800],
    )

    assert isinstance(summary, PerformanceSummary)
    assert summary.total_bets == 2
    assert summary.wins == 1
    assert summary.total_staked_jpy == 2_200
    assert summary.total_payout_jpy == 3_000
    assert summary.net_profit_jpy == 800
    assert summary.roi == pytest.approx(800 / 2_200)
    assert summary.hit_rate == pytest.approx(0.5)
    assert summary.turnover == pytest.approx(2_200 / 10_000)
    assert summary.max_drawdown == pytest.approx(0.1)
    assert summary.final_bankroll_jpy == 10_800


def test_summarize_performance_handles_zero_bets():
    summary = summarize_performance(
        initial_bankroll_jpy=10_000,
        final_bankroll_jpy=10_000,
        stakes_jpy=[],
        payouts_jpy=[],
        wins=[],
        bankroll_curve_jpy=[10_000],
    )

    assert summary.total_bets == 0
    assert summary.roi == 0.0
    assert summary.hit_rate == 0.0
    assert summary.turnover == 0.0
```

- [ ] **Step 2: Run metric tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_metrics.py -v
```

Expected: FAIL because `compute_max_drawdown` and `summarize_performance` are not implemented.

- [ ] **Step 3: Implement performance metrics**

Replace `src/horse_lab/evaluation/metrics.py` with:

```python
"""Performance metric helpers for betting backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PerformanceSummary:
    total_bets: int
    wins: int
    total_staked_jpy: int
    total_payout_jpy: int
    net_profit_jpy: int
    roi: float
    hit_rate: float
    turnover: float
    max_drawdown: float
    final_bankroll_jpy: int


def compute_max_drawdown(bankroll_curve_jpy: Sequence[int]) -> float:
    if not bankroll_curve_jpy:
        return 0.0

    peak = float(bankroll_curve_jpy[0])
    max_drawdown = 0.0
    for value in bankroll_curve_jpy:
        value_float = float(value)
        if value_float > peak:
            peak = value_float
        if peak <= 0.0:
            continue
        drawdown = (peak - value_float) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def summarize_performance(
    *,
    initial_bankroll_jpy: int,
    final_bankroll_jpy: int,
    stakes_jpy: Sequence[int],
    payouts_jpy: Sequence[int],
    wins: Sequence[bool],
    bankroll_curve_jpy: Sequence[int],
) -> PerformanceSummary:
    total_bets = len(stakes_jpy)
    total_staked_jpy = sum(stakes_jpy)
    total_payout_jpy = sum(payouts_jpy)
    net_profit_jpy = final_bankroll_jpy - initial_bankroll_jpy
    win_count = sum(1 for won in wins if won)

    roi = net_profit_jpy / total_staked_jpy if total_staked_jpy else 0.0
    hit_rate = win_count / total_bets if total_bets else 0.0
    turnover = total_staked_jpy / initial_bankroll_jpy if initial_bankroll_jpy else 0.0

    return PerformanceSummary(
        total_bets=total_bets,
        wins=win_count,
        total_staked_jpy=total_staked_jpy,
        total_payout_jpy=total_payout_jpy,
        net_profit_jpy=net_profit_jpy,
        roi=roi,
        hit_rate=hit_rate,
        turnover=turnover,
        max_drawdown=compute_max_drawdown(bankroll_curve_jpy),
        final_bankroll_jpy=final_bankroll_jpy,
    )
```

Replace `src/horse_lab/evaluation/__init__.py` with:

```python
"""Evaluation metric helpers."""

from horse_lab.evaluation.metrics import (
    PerformanceSummary,
    compute_max_drawdown,
    summarize_performance,
)

__all__ = [
    "PerformanceSummary",
    "compute_max_drawdown",
    "summarize_performance",
]
```

- [ ] **Step 4: Run metric tests to verify they pass**

Run:

```bash
PYTHONPATH=src pytest tests/evaluation/test_metrics.py -v
```

Expected: PASS with `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/horse_lab/evaluation tests/evaluation/test_metrics.py
git commit -m "feat: add backtest performance metrics"
```

---

### Task 5: Backtest Simulator

**Files:**
- Modify: `src/horse_lab/backtesting/simulator.py`
- Modify: `src/horse_lab/backtesting/__init__.py`
- Create: `tests/backtesting/test_simulator.py`

- [ ] **Step 1: Write failing simulator tests**

Create `tests/backtesting/test_simulator.py`:

```python
import datetime as dt

import pytest

from horse_lab.backtesting import BacktestConfig, BacktestSimulator
from horse_lab.betting import KellyConfig
from horse_lab.schemas import (
    BetType,
    ModelName,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    RaceId,
    Result,
    RunnerId,
)


def _prediction(runner_id: str, probability: float, as_of: dt.datetime) -> ModelPrediction:
    return ModelPrediction(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        model_name=ModelName("test_model"),
        model_version="test",
        target=PredictionTarget.WIN_PROBABILITY,
        probability=probability,
        as_of=as_of,
    )


def _quote(runner_id: str, odds: float, captured_at: dt.datetime) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        bet_type=BetType.WIN,
        captured_at=captured_at,
        odds=odds,
    )


def _result(runner_id: str, finish_position: int) -> Result:
    return Result(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        finish_position=finish_position,
    )


def test_backtest_simulator_accounts_for_one_win_and_one_loss():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(
                fractional_kelly=1.0,
                max_stake_fraction=0.10,
                minimum_edge=0.0,
                stake_unit_jpy=1,
            ),
        )
    )

    result = simulator.run(
        predictions=[
            _prediction("runner-1", probability=0.6, as_of=as_of),
            _prediction("runner-2", probability=0.6, as_of=as_of),
        ],
        odds=[
            _quote("runner-1", odds=3.0, captured_at=as_of),
            _quote("runner-2", odds=2.0, captured_at=as_of),
        ],
        results=[
            _result("runner-1", finish_position=1),
            _result("runner-2", finish_position=2),
        ],
    )

    assert len(result.records) == 2
    assert result.records[0].stake_jpy == 1_000
    assert result.records[0].payout_jpy == 3_000
    assert result.records[0].bankroll_after_jpy == 12_000
    assert result.records[1].stake_jpy == 1_200
    assert result.records[1].payout_jpy == 0
    assert result.records[1].bankroll_after_jpy == 10_800
    assert result.final_bankroll_jpy == 10_800
    assert result.summary.total_bets == 2
    assert result.summary.wins == 1
    assert result.summary.roi == pytest.approx(800 / 2_200)
    assert result.summary.max_drawdown == pytest.approx(0.1)


def test_backtest_simulator_skips_zero_stake_predictions():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(minimum_edge=0.05),
        )
    )

    result = simulator.run(
        predictions=[_prediction("runner-1", probability=0.5, as_of=as_of)],
        odds=[_quote("runner-1", odds=2.0, captured_at=as_of)],
        results=[_result("runner-1", finish_position=1)],
    )

    assert result.records == ()
    assert result.final_bankroll_jpy == 10_000
    assert result.summary.total_bets == 0


def test_backtest_simulator_requires_matching_odds_and_results():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(config=BacktestConfig(initial_bankroll_jpy=10_000))

    with pytest.raises(ValueError, match="Missing odds"):
        simulator.run(
            predictions=[_prediction("runner-1", probability=0.6, as_of=as_of)],
            odds=[],
            results=[_result("runner-1", finish_position=1)],
        )

    with pytest.raises(ValueError, match="Missing result"):
        simulator.run(
            predictions=[_prediction("runner-1", probability=0.6, as_of=as_of)],
            odds=[_quote("runner-1", odds=3.0, captured_at=as_of)],
            results=[],
        )
```

- [ ] **Step 2: Run simulator tests to verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/backtesting/test_simulator.py -v
```

Expected: FAIL because `BacktestConfig` does not accept `kelly_config` and `BacktestSimulator.run` is not implemented.

- [ ] **Step 3: Implement the simulator**

Replace `src/horse_lab/backtesting/simulator.py` with:

```python
"""Historical betting simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from horse_lab.betting import KellyConfig, calculate_kelly_stake
from horse_lab.evaluation import PerformanceSummary, summarize_performance
from horse_lab.schemas import (
    BetType,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    RaceId,
    Result,
    RunnerId,
)


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll_jpy: int = 100_000
    kelly_config: KellyConfig = field(default_factory=KellyConfig)

    def __post_init__(self) -> None:
        if self.initial_bankroll_jpy <= 0:
            raise ValueError("initial_bankroll_jpy must be positive")


@dataclass(frozen=True)
class BetRecord:
    race_id: RaceId
    runner_id: RunnerId
    bet_type: BetType
    probability: float
    odds: float
    edge: float
    kelly_fraction: float
    stake_fraction: float
    stake_jpy: int
    payout_jpy: int
    profit_jpy: int
    bankroll_after_jpy: int
    is_win: bool


@dataclass(frozen=True)
class BacktestResult:
    records: tuple[BetRecord, ...]
    summary: PerformanceSummary
    bankroll_curve_jpy: tuple[int, ...]
    final_bankroll_jpy: int


class BacktestSimulator:
    def __init__(self, *, config: BacktestConfig = BacktestConfig()) -> None:
        self.config = config

    def run(
        self,
        *,
        predictions: Sequence[ModelPrediction],
        odds: Sequence[OddsQuote],
        results: Sequence[Result],
    ) -> BacktestResult:
        bankroll_jpy = self.config.initial_bankroll_jpy
        records: list[BetRecord] = []
        bankroll_curve_jpy = [bankroll_jpy]

        sorted_predictions = sorted(
            predictions,
            key=lambda prediction: (
                prediction.as_of,
                str(prediction.race_id),
                str(prediction.runner_id),
            ),
        )

        for prediction in sorted_predictions:
            if prediction.target != PredictionTarget.WIN_PROBABILITY:
                continue

            quote = _latest_quote_for_prediction(prediction, odds)
            race_result = _result_for_prediction(prediction, results)
            decision = calculate_kelly_stake(
                probability=prediction.probability,
                odds=quote.odds,
                bankroll_jpy=bankroll_jpy,
                config=self.config.kelly_config,
            )
            if decision.stake_jpy <= 0:
                continue

            is_win = race_result.did_win
            payout_jpy = int(decision.stake_jpy * quote.odds) if is_win else 0
            profit_jpy = payout_jpy - decision.stake_jpy
            bankroll_jpy += profit_jpy
            bankroll_curve_jpy.append(bankroll_jpy)

            records.append(
                BetRecord(
                    race_id=prediction.race_id,
                    runner_id=prediction.runner_id,
                    bet_type=BetType.WIN,
                    probability=prediction.probability,
                    odds=quote.odds,
                    edge=decision.edge,
                    kelly_fraction=decision.full_kelly_fraction,
                    stake_fraction=decision.stake_fraction,
                    stake_jpy=decision.stake_jpy,
                    payout_jpy=payout_jpy,
                    profit_jpy=profit_jpy,
                    bankroll_after_jpy=bankroll_jpy,
                    is_win=is_win,
                )
            )

        summary = summarize_performance(
            initial_bankroll_jpy=self.config.initial_bankroll_jpy,
            final_bankroll_jpy=bankroll_jpy,
            stakes_jpy=[record.stake_jpy for record in records],
            payouts_jpy=[record.payout_jpy for record in records],
            wins=[record.is_win for record in records],
            bankroll_curve_jpy=bankroll_curve_jpy,
        )

        return BacktestResult(
            records=tuple(records),
            summary=summary,
            bankroll_curve_jpy=tuple(bankroll_curve_jpy),
            final_bankroll_jpy=bankroll_jpy,
        )


def _latest_quote_for_prediction(
    prediction: ModelPrediction,
    odds: Sequence[OddsQuote],
) -> OddsQuote:
    matching_quotes = [
        quote
        for quote in odds
        if quote.race_id == prediction.race_id
        and quote.runner_id == prediction.runner_id
        and quote.bet_type == BetType.WIN
        and quote.captured_at <= prediction.as_of
    ]
    if not matching_quotes:
        raise ValueError(
            "Missing odds for "
            f"race_id={prediction.race_id!r}, runner_id={prediction.runner_id!r}"
        )
    return max(matching_quotes, key=lambda quote: quote.captured_at)


def _result_for_prediction(
    prediction: ModelPrediction,
    results: Sequence[Result],
) -> Result:
    for result in results:
        if result.race_id == prediction.race_id and result.runner_id == prediction.runner_id:
            return result
    raise ValueError(
        "Missing result for "
        f"race_id={prediction.race_id!r}, runner_id={prediction.runner_id!r}"
    )
```

Replace `src/horse_lab/backtesting/__init__.py` with:

```python
"""Historical betting simulation."""

from horse_lab.backtesting.simulator import (
    BacktestConfig,
    BacktestResult,
    BacktestSimulator,
    BetRecord,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestSimulator",
    "BetRecord",
]
```

- [ ] **Step 4: Run simulator tests to verify they pass**

Run:

```bash
PYTHONPATH=src pytest tests/backtesting/test_simulator.py -v
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/horse_lab/backtesting tests/backtesting/test_simulator.py
git commit -m "feat: add kelly backtest simulator"
```

---

### Task 6: Repository And Feature Builder Protocols

**Files:**
- Create: `src/horse_lab/data/__init__.py`
- Create: `src/horse_lab/data/repositories.py`
- Create: `src/horse_lab/features/__init__.py`
- Create: `src/horse_lab/features/builders.py`
- Modify: `tests/test_package_imports.py`

- [ ] **Step 1: Extend smoke test for protocols**

Replace `tests/test_package_imports.py` with:

```python
from horse_lab.schemas import BetType, PredictionTarget


def test_core_package_imports():
    assert BetType.WIN.value == "win"
    assert PredictionTarget.WIN_PROBABILITY.value == "win_probability"


def test_baseline_modules_import():
    from horse_lab.betting import KellyConfig, calculate_kelly_stake
    from horse_lab.backtesting import BacktestConfig, BacktestSimulator
    from horse_lab.evaluation import PerformanceSummary
    from horse_lab.models import MarketImpliedProbabilityModel

    assert KellyConfig.__name__ == "KellyConfig"
    assert callable(calculate_kelly_stake)
    assert BacktestConfig.__name__ == "BacktestConfig"
    assert BacktestSimulator.__name__ == "BacktestSimulator"
    assert PerformanceSummary.__name__ == "PerformanceSummary"
    assert MarketImpliedProbabilityModel.__name__ == "MarketImpliedProbabilityModel"


def test_storage_and_feature_protocols_import():
    from horse_lab.data import RaceRepository, ResultRepository
    from horse_lab.features import FeatureBuilder

    assert RaceRepository.__name__ == "RaceRepository"
    assert ResultRepository.__name__ == "ResultRepository"
    assert FeatureBuilder.__name__ == "FeatureBuilder"
```

- [ ] **Step 2: Run smoke test to verify it fails**

Run:

```bash
PYTHONPATH=src pytest tests/test_package_imports.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'horse_lab.data'`.

- [ ] **Step 3: Implement repository protocols**

Create `src/horse_lab/data/repositories.py`:

```python
"""Repository protocols for historical racing data."""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, Sequence

from horse_lab.schemas import FeatureRow, OddsQuote, Race, RaceId, Result


class RaceRepository(Protocol):
    def list_races(self, *, start_date: date, end_date: date) -> Sequence[Race]:
        """Return races whose race dates are within the inclusive date range."""


class OddsRepository(Protocol):
    def list_odds(
        self,
        *,
        race_ids: Sequence[RaceId],
        captured_at_or_before: datetime,
    ) -> Sequence[OddsQuote]:
        """Return odds snapshots available at or before the requested time."""


class ResultRepository(Protocol):
    def list_results(self, *, race_ids: Sequence[RaceId]) -> Sequence[Result]:
        """Return official race results for the requested races."""


class FeatureRepository(Protocol):
    def list_feature_rows(
        self,
        *,
        race_ids: Sequence[RaceId],
        feature_version: str,
        as_of: datetime,
    ) -> Sequence[FeatureRow]:
        """Return point-in-time runner-level feature rows."""
```

Create `src/horse_lab/data/__init__.py`:

```python
"""Data access protocols."""

from horse_lab.data.repositories import (
    FeatureRepository,
    OddsRepository,
    RaceRepository,
    ResultRepository,
)

__all__ = [
    "FeatureRepository",
    "OddsRepository",
    "RaceRepository",
    "ResultRepository",
]
```

- [ ] **Step 4: Implement feature builder protocol**

Create `src/horse_lab/features/builders.py`:

```python
"""Feature builder protocols."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from horse_lab.schemas import Entry, FeatureRow, OddsQuote, Race


class FeatureBuilder(Protocol):
    feature_version: str

    def build(
        self,
        *,
        races: Sequence[Race],
        entries: Sequence[Entry],
        odds: Sequence[OddsQuote],
        as_of: datetime,
    ) -> Sequence[FeatureRow]:
        """Build runner-level features using only information available at as_of."""
```

Create `src/horse_lab/features/__init__.py`:

```python
"""Feature generation contracts."""

from horse_lab.features.builders import FeatureBuilder

__all__ = ["FeatureBuilder"]
```

- [ ] **Step 5: Run smoke test to verify it passes**

Run:

```bash
PYTHONPATH=src pytest tests/test_package_imports.py -v
```

Expected: PASS with `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/horse_lab/data src/horse_lab/features tests/test_package_imports.py
git commit -m "feat: add data and feature protocols"
```

---

### Task 7: Full Verification And Documentation Update

**Files:**
- Modify: `docs/architecture.md`

- [ ] **Step 1: Update architecture doc with implemented MVP modules**

Append this section to `docs/architecture.md`:

```markdown

## Implemented MVP Baseline

The first executable baseline is dependency-light and supports win bets only.

- `horse_lab.models.market.MarketImpliedProbabilityModel` converts latest pre-race win odds into normalized race-level probabilities.
- `horse_lab.betting.kelly` calculates edge, full Kelly fraction, fractional Kelly stake fraction, and yen stake size.
- `horse_lab.backtesting.simulator.BacktestSimulator` simulates runner-level win bets over historical predictions, odds, and results.
- `horse_lab.evaluation.metrics` summarizes ROI, hit rate, turnover, and max drawdown.
- `horse_lab.data` and `horse_lab.features` define protocols for storage and point-in-time feature generation.
```

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/horse-lab-pycache PYTHONPATH=src pytest -v
```

Expected: PASS with all tests passing.

- [ ] **Step 3: Run Python compile verification**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/horse-lab-pycache python3 -m compileall src tests
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: shows only intentional new or modified project files.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: document implemented mvp baseline"
```

---

## Self-Review Checklist

- Spec coverage:
  - Modular structure: Tasks 1, 6, and 7.
  - Market-implied baseline: Task 3.
  - Kelly stake sizing: Task 2.
  - Backtest engine: Task 5.
  - Evaluation metrics: Task 4.
  - Focused tests: Tasks 1 through 6.
- Type consistency:
  - `RaceId`, `RunnerId`, `OddsQuote`, `ModelPrediction`, and `Result` come from `horse_lab.schemas`.
  - `InferenceContext`, `TrainingContext`, `TrainingDataset`, and `ModelArtifact` come from `horse_lab.models.base`.
  - `KellyConfig` is shared by betting and backtesting.
- Verification:
  - Every implementation task has a failing-test step and a passing-test step.
  - Final verification runs pytest and compileall.
