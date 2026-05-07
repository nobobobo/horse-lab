"""Historical betting simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
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

        for _, prediction_group in groupby(
            sorted_predictions,
            key=lambda prediction: (prediction.as_of, prediction.race_id),
        ):
            group_records: list[BetRecord] = []
            group_profit_jpy = 0

            for prediction in prediction_group:
                if prediction.target != PredictionTarget.WIN_PROBABILITY:
                    continue

                quote = _latest_quote_for_prediction(prediction, odds)
                decision = calculate_kelly_stake(
                    probability=prediction.probability,
                    odds=quote.odds,
                    bankroll_jpy=bankroll_jpy,
                    config=self.config.kelly_config,
                )
                if decision.stake_jpy <= 0:
                    continue

                race_result = _result_for_prediction(prediction, results)
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
