"""Performance metric helpers for betting backtests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from horse_lab.schemas import ModelPrediction, PredictionTarget, Result


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


@dataclass(frozen=True)
class ProbabilityCalibrationBin:
    lower_bound: float
    upper_bound: float
    count: int
    positives: int
    mean_predicted_probability: float
    empirical_rate: float
    absolute_error: float


@dataclass(frozen=True)
class ProbabilitySummary:
    observations: int
    positives: int
    mean_predicted_probability: float
    empirical_rate: float
    log_loss: float
    brier_score: float
    expected_calibration_error: float
    bins: tuple[ProbabilityCalibrationBin, ...]


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


def summarize_win_probability_predictions(
    *,
    predictions: Sequence[ModelPrediction],
    results: Sequence[Result],
    bin_count: int = 10,
    epsilon: float = 1e-15,
) -> ProbabilitySummary:
    """Summarize binary win-probability quality against settled results."""

    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0.0 and 0.5")

    result_by_runner = {
        (result.race_id, result.runner_id): result
        for result in results
    }

    observations: list[tuple[float, int]] = []
    for prediction in predictions:
        if prediction.target != PredictionTarget.WIN_PROBABILITY:
            continue
        result = result_by_runner.get((prediction.race_id, prediction.runner_id))
        if result is None:
            raise ValueError(
                "Missing result for "
                f"race_id={prediction.race_id!r}, runner_id={prediction.runner_id!r}"
            )
        observations.append((prediction.probability, 1 if result.did_win else 0))

    if not observations:
        return ProbabilitySummary(
            observations=0,
            positives=0,
            mean_predicted_probability=0.0,
            empirical_rate=0.0,
            log_loss=0.0,
            brier_score=0.0,
            expected_calibration_error=0.0,
            bins=_empty_calibration_bins(bin_count),
        )

    count = len(observations)
    positives = sum(label for _, label in observations)
    mean_probability = sum(probability for probability, _ in observations) / count
    empirical_rate = positives / count
    log_loss = sum(
        _binary_log_loss(probability, label, epsilon=epsilon)
        for probability, label in observations
    ) / count
    brier_score = sum(
        (probability - label) ** 2 for probability, label in observations
    ) / count
    bins = _calibration_bins(observations, bin_count=bin_count)
    expected_calibration_error = sum(
        (calibration_bin.count / count) * calibration_bin.absolute_error
        for calibration_bin in bins
    )

    return ProbabilitySummary(
        observations=count,
        positives=positives,
        mean_predicted_probability=mean_probability,
        empirical_rate=empirical_rate,
        log_loss=log_loss,
        brier_score=brier_score,
        expected_calibration_error=expected_calibration_error,
        bins=bins,
    )


def _binary_log_loss(probability: float, label: int, *, epsilon: float) -> float:
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    if label:
        return -math.log(clipped)
    return -math.log(1.0 - clipped)


def _empty_calibration_bins(bin_count: int) -> tuple[ProbabilityCalibrationBin, ...]:
    return tuple(
        ProbabilityCalibrationBin(
            lower_bound=index / bin_count,
            upper_bound=(index + 1) / bin_count,
            count=0,
            positives=0,
            mean_predicted_probability=0.0,
            empirical_rate=0.0,
            absolute_error=0.0,
        )
        for index in range(bin_count)
    )


def _calibration_bins(
    observations: Sequence[tuple[float, int]],
    *,
    bin_count: int,
) -> tuple[ProbabilityCalibrationBin, ...]:
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
    for probability, label in observations:
        index = min(bin_count - 1, int(probability * bin_count))
        grouped[index].append((probability, label))

    bins: list[ProbabilityCalibrationBin] = []
    for index, values in enumerate(grouped):
        if values:
            count = len(values)
            positives = sum(label for _, label in values)
            mean_probability = sum(probability for probability, _ in values) / count
            empirical_rate = positives / count
            absolute_error = abs(mean_probability - empirical_rate)
        else:
            count = 0
            positives = 0
            mean_probability = 0.0
            empirical_rate = 0.0
            absolute_error = 0.0
        bins.append(
            ProbabilityCalibrationBin(
                lower_bound=index / bin_count,
                upper_bound=(index + 1) / bin_count,
                count=count,
                positives=positives,
                mean_predicted_probability=mean_probability,
                empirical_rate=empirical_rate,
                absolute_error=absolute_error,
            )
        )
    return tuple(bins)
