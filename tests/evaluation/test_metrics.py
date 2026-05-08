import datetime as dt
import math

import pytest

from horse_lab.evaluation import (
    PerformanceSummary,
    ProbabilitySummary,
    compute_max_drawdown,
    summarize_performance,
    summarize_win_probability_predictions,
)
from horse_lab.schemas import (
    ModelName,
    ModelPrediction,
    PredictionTarget,
    RaceId,
    Result,
    RunnerId,
)


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


def _prediction(runner_id: str, probability: float) -> ModelPrediction:
    return ModelPrediction(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        model_name=ModelName("fixture"),
        model_version="v1",
        target=PredictionTarget.WIN_PROBABILITY,
        probability=probability,
        as_of=dt.datetime(2026, 5, 8, 9, 55),
    )


def _result(runner_id: str, finish_position: int) -> Result:
    return Result(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        finish_position=finish_position,
    )


def test_summarize_win_probability_predictions_computes_quality_metrics():
    summary = summarize_win_probability_predictions(
        predictions=[
            _prediction("runner-1", 0.7),
            _prediction("runner-2", 0.2),
            _prediction("runner-3", 0.1),
        ],
        results=[
            _result("runner-1", 1),
            _result("runner-2", 2),
            _result("runner-3", 3),
        ],
        bin_count=5,
    )

    assert isinstance(summary, ProbabilitySummary)
    assert summary.observations == 3
    assert summary.positives == 1
    assert summary.mean_predicted_probability == pytest.approx(1.0 / 3.0)
    assert summary.empirical_rate == pytest.approx(1.0 / 3.0)
    assert summary.log_loss == pytest.approx(
        (-math.log(0.7) - math.log(0.8) - math.log(0.9)) / 3
    )
    assert summary.brier_score == pytest.approx(
        ((0.7 - 1.0) ** 2 + (0.2 - 0.0) ** 2 + (0.1 - 0.0) ** 2) / 3
    )
    assert len(summary.bins) == 5
    assert summary.bins[0].count == 1
    assert summary.bins[0].mean_predicted_probability == pytest.approx(0.1)
    assert summary.bins[1].count == 1
    assert summary.bins[3].count == 1
    assert summary.expected_calibration_error == pytest.approx(
        ((1 / 3) * 0.1) + ((1 / 3) * 0.2) + ((1 / 3) * 0.3)
    )


def test_summarize_win_probability_predictions_requires_matching_result():
    with pytest.raises(ValueError, match="Missing result"):
        summarize_win_probability_predictions(
            predictions=[_prediction("runner-1", 0.5)],
            results=[],
        )


def test_summarize_win_probability_predictions_handles_empty_predictions():
    summary = summarize_win_probability_predictions(
        predictions=[],
        results=[],
        bin_count=3,
    )

    assert summary.observations == 0
    assert summary.log_loss == 0.0
    assert summary.brier_score == 0.0
    assert len(summary.bins) == 3
