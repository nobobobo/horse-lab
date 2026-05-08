import datetime as dt
from pathlib import Path

import pytest

from horse_lab.data import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.pipelines import ReplayResult, run_market_replay
from horse_lab.schemas import RunnerId


SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"


def test_run_market_replay_returns_deterministic_sample_metrics():
    result = run_market_replay(
        race_repository=CsvRaceRepository(SAMPLE_DATA / "races.csv"),
        odds_repository=CsvOddsRepository(SAMPLE_DATA / "odds.csv"),
        result_repository=CsvResultRepository(SAMPLE_DATA / "results.csv"),
        feature_repository=CsvFeatureRepository(SAMPLE_DATA / "features.csv"),
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 9, 55),
        feature_version="fixture-v1",
    )

    assert isinstance(result, ReplayResult)
    assert len(result.races) == 2
    assert len(result.feature_rows) == 4
    assert len(result.odds) == 5
    assert len(result.results) == 4
    assert len(result.predictions) == 4

    summary = result.summary
    assert result.backtest_result.final_bankroll_jpy == 108_000
    assert summary.final_bankroll_jpy == 108_000
    assert summary.total_bets == 4
    assert summary.wins == 2
    assert summary.total_staked_jpy == 8_000
    assert summary.total_payout_jpy == 16_000
    assert summary.net_profit_jpy == 8_000
    assert summary.roi == pytest.approx(1.0)
    assert summary.hit_rate == pytest.approx(0.5)
    assert summary.turnover == pytest.approx(0.08)
    assert summary.max_drawdown == pytest.approx(0.0)
    assert result.backtest_result.bankroll_curve_jpy == (100_000, 102_000, 108_000)


def test_run_market_replay_uses_point_in_time_market_odds():
    result = run_market_replay(
        race_repository=CsvRaceRepository(SAMPLE_DATA / "races.csv"),
        odds_repository=CsvOddsRepository(SAMPLE_DATA / "odds.csv"),
        result_repository=CsvResultRepository(SAMPLE_DATA / "results.csv"),
        feature_repository=CsvFeatureRepository(SAMPLE_DATA / "features.csv"),
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 9, 55),
        feature_version="fixture-v1",
    )

    by_runner = {prediction.runner_id: prediction for prediction in result.predictions}

    assert by_runner[RunnerId("202605080101-01")].metadata["market_odds"] == 3.0
