"""Historical betting simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Sequence

from horse_lab.betting import KellyConfig, calculate_kelly_stake
from horse_lab.evaluation import PerformanceSummary, summarize_performance
from horse_lab.schemas import (
    BetType,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    Race,
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
        races: Sequence[Race] = (),
    ) -> BacktestResult:
        bankroll_jpy = self.config.initial_bankroll_jpy
        records: list[BetRecord] = []
        bankroll_curve_jpy = [bankroll_jpy]

        predictions_by_race: dict[RaceId, list[ModelPrediction]] = {}
        for prediction in predictions:
            if prediction.target != PredictionTarget.WIN_PROBABILITY:
                continue
            predictions_by_race.setdefault(prediction.race_id, []).append(prediction)

        win_quotes_by_runner = _win_quotes_by_runner(odds)
        results_by_runner = {
            (result.race_id, result.runner_id): result for result in results
        }

        race_groups = [
            sorted(
                race_predictions,
                key=lambda prediction: (
                    prediction.as_of,
                    str(prediction.runner_id),
                ),
            )
            for race_predictions in predictions_by_race.values()
        ]
        if races:
            race_by_id = {race.race_id: race for race in races}
            race_groups = sorted(
                race_groups,
                key=lambda race_predictions: _race_chronology_key(
                    race_predictions[0],
                    race_by_id,
                ),
            )
        else:
            race_groups = sorted(
                race_groups,
                key=lambda race_predictions: (
                    race_predictions[0].as_of,
                    str(race_predictions[0].race_id),
                ),
            )

        for prediction_group in race_groups:
            group_records: list[BetRecord] = []
            group_profit_jpy = 0

            for prediction in prediction_group:
                quote = _latest_quote_for_prediction(
                    prediction,
                    win_quotes_by_runner,
                )
                decision = calculate_kelly_stake(
                    probability=prediction.probability,
                    odds=quote.odds,
                    bankroll_jpy=bankroll_jpy,
                    config=self.config.kelly_config,
                )
                if decision.stake_jpy <= 0:
                    continue

                race_result = _result_for_prediction(prediction, results_by_runner)
                is_win = race_result.did_win
                payout_jpy = int(round(decision.stake_jpy * quote.odds)) if is_win else 0
                profit_jpy = payout_jpy - decision.stake_jpy
                group_profit_jpy += profit_jpy

                group_records.append(
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
                        bankroll_after_jpy=0,
                        is_win=is_win,
                    )
                )

            if not group_records:
                continue

            bankroll_jpy += group_profit_jpy
            bankroll_curve_jpy.append(bankroll_jpy)
            records.extend(
                BetRecord(
                    race_id=record.race_id,
                    runner_id=record.runner_id,
                    bet_type=record.bet_type,
                    probability=record.probability,
                    odds=record.odds,
                    edge=record.edge,
                    kelly_fraction=record.kelly_fraction,
                    stake_fraction=record.stake_fraction,
                    stake_jpy=record.stake_jpy,
                    payout_jpy=record.payout_jpy,
                    profit_jpy=record.profit_jpy,
                    bankroll_after_jpy=bankroll_jpy,
                    is_win=record.is_win,
                )
                for record in group_records
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


def _win_quotes_by_runner(
    odds: Sequence[OddsQuote],
) -> dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]]:
    grouped: dict[tuple[RaceId, RunnerId], list[OddsQuote]] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN:
            continue
        grouped.setdefault((quote.race_id, quote.runner_id), []).append(quote)
    return {
        key: tuple(sorted(values, key=lambda quote: quote.captured_at))
        for key, values in grouped.items()
    }


def _latest_quote_for_prediction(
    prediction: ModelPrediction,
    win_quotes_by_runner: dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]],
) -> OddsQuote:
    quotes = win_quotes_by_runner.get((prediction.race_id, prediction.runner_id), ())
    for quote in reversed(quotes):
        if quote.captured_at <= prediction.as_of:
            return quote
    raise ValueError(
        "Missing odds for "
        f"race_id={prediction.race_id!r}, runner_id={prediction.runner_id!r}"
    )


def _race_chronology_key(
    prediction: ModelPrediction,
    race_by_id: dict[RaceId, Race],
) -> tuple[int, datetime, date, int, str]:
    race = race_by_id.get(prediction.race_id)
    if race is None:
        return (
            1,
            prediction.as_of,
            date.max,
            0,
            str(prediction.race_id),
        )

    chronological_time = race.start_time or datetime.combine(race.race_date, time.min)
    return (
        0,
        chronological_time,
        race.race_date,
        race.race_number,
        str(race.race_id),
    )


def _result_for_prediction(
    prediction: ModelPrediction,
    results_by_runner: dict[tuple[RaceId, RunnerId], Result],
) -> Result:
    result = results_by_runner.get((prediction.race_id, prediction.runner_id))
    if result is not None:
        return result
    raise ValueError(
        "Missing result for "
        f"race_id={prediction.race_id!r}, runner_id={prediction.runner_id!r}"
    )
