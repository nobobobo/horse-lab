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
from horse_lab.schemas import (
    BetType,
    FeatureName,
    FeatureRow,
    OddsQuote,
    Race,
    RaceId,
    Result,
    RunnerId,
    Surface,
)


SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"


class _RaceRepository:
    def __init__(self, races):
        self.races = tuple(races)

    def list_races(self, *, start_date, end_date):
        return self.races


class _FeatureRepository:
    def __init__(self, feature_rows):
        self.feature_rows = tuple(feature_rows)

    def list_feature_rows(self, *, race_ids, feature_version, as_of):
        return self.feature_rows


class _OddsRepository:
    def __init__(self, odds):
        self.odds = tuple(odds)

    def list_odds(self, *, race_ids, captured_at_or_before):
        return self.odds


class _ResultRepository:
    def __init__(self, results):
        self.results = tuple(results)

    def list_results(self, *, race_ids):
        return self.results


def _race(field_size=2):
    return Race(
        race_id=RaceId("race-1"),
        race_date=dt.date(2026, 5, 8),
        venue="Tokyo",
        race_number=1,
        name="Fixture",
        surface=Surface.TURF,
        distance_m=1200,
        field_size=field_size,
    )


def _feature_row(runner_id):
    return FeatureRow(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        as_of=dt.datetime(2026, 5, 8, 9, 55),
        feature_version="fixture-v1",
        values={FeatureName("recent_speed"): 70},
    )


def _quote(runner_id):
    return OddsQuote(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        bet_type=BetType.WIN,
        captured_at=dt.datetime(2026, 5, 8, 9, 55),
        odds=2.0,
    )


def _result(runner_id, finish_position):
    return Result(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        finish_position=finish_position,
    )


def _run_replay_with_rows(*, races, feature_rows, odds, results):
    return run_market_replay(
        race_repository=_RaceRepository(races),
        odds_repository=_OddsRepository(odds),
        result_repository=_ResultRepository(results),
        feature_repository=_FeatureRepository(feature_rows),
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 9, 55),
        feature_version="fixture-v1",
    )


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
    assert result.probability_summary.observations == 4
    assert result.probability_summary.positives == 2
    assert result.probability_summary.brier_score > 0.0


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


def test_run_market_replay_rejects_feature_count_that_does_not_match_field_size():
    with pytest.raises(ValueError, match="Feature row count does not match field_size"):
        _run_replay_with_rows(
            races=[_race(field_size=2)],
            feature_rows=[_feature_row("runner-1")],
            odds=[_quote("runner-1")],
            results=[_result("runner-1", 1)],
        )


def test_run_market_replay_rejects_result_runner_mismatch():
    with pytest.raises(ValueError, match="Result runners do not match feature runners"):
        _run_replay_with_rows(
            races=[_race(field_size=2)],
            feature_rows=[_feature_row("runner-1"), _feature_row("runner-2")],
            odds=[_quote("runner-1"), _quote("runner-2")],
            results=[_result("runner-1", 1), _result("runner-3", 2)],
        )


def test_run_market_replay_rejects_odds_runner_mismatch():
    with pytest.raises(ValueError, match="Odds runners do not match feature runners"):
        _run_replay_with_rows(
            races=[_race(field_size=2)],
            feature_rows=[_feature_row("runner-1"), _feature_row("runner-2")],
            odds=[_quote("runner-1"), _quote("runner-3")],
            results=[_result("runner-1", 1), _result("runner-2", 2)],
        )
