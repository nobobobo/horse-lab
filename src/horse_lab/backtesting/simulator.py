"""Historical betting simulator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from math import floor
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


class OddsTiming(str, Enum):
    """Bet odds selection policy for historical replay."""

    LATEST_AVAILABLE = "latest_available"
    CLOSING = "closing"
    MINUTES_BEFORE_START = "minutes_before_start"


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll_jpy: int = 100_000
    kelly_config: KellyConfig = field(default_factory=KellyConfig)
    min_odds: float | None = None
    max_odds: float | None = None
    max_stake_per_race_jpy: int | None = None
    max_daily_loss_jpy: int | None = None
    odds_timing: OddsTiming | str = OddsTiming.LATEST_AVAILABLE
    odds_minutes_before_start: int | None = None

    def __post_init__(self) -> None:
        if self.initial_bankroll_jpy <= 0:
            raise ValueError("initial_bankroll_jpy must be positive")
        if self.min_odds is not None and self.min_odds <= 1.0:
            raise ValueError("min_odds must be greater than 1.0 when present")
        if self.max_odds is not None and self.max_odds <= 1.0:
            raise ValueError("max_odds must be greater than 1.0 when present")
        if (
            self.min_odds is not None
            and self.max_odds is not None
            and self.min_odds > self.max_odds
        ):
            raise ValueError("min_odds must be less than or equal to max_odds")
        if self.max_stake_per_race_jpy is not None and self.max_stake_per_race_jpy <= 0:
            raise ValueError(
                "max_stake_per_race_jpy must be positive when present"
            )
        if self.max_daily_loss_jpy is not None and self.max_daily_loss_jpy <= 0:
            raise ValueError("max_daily_loss_jpy must be positive when present")

        timing = OddsTiming(self.odds_timing)
        object.__setattr__(self, "odds_timing", timing)
        if timing == OddsTiming.MINUTES_BEFORE_START:
            if self.odds_minutes_before_start is None:
                raise ValueError(
                    "odds_minutes_before_start is required for "
                    "minutes_before_start odds timing"
                )
            if self.odds_minutes_before_start < 0:
                raise ValueError("odds_minutes_before_start must be non-negative")
        elif self.odds_minutes_before_start is not None:
            if self.odds_minutes_before_start < 0:
                raise ValueError("odds_minutes_before_start must be non-negative")


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
    odds_captured_at: datetime | None = None
    fair_odds: float | None = None
    closing_odds: float | None = None
    closing_captured_at: datetime | None = None
    clv_odds_delta: float | None = None
    clv_implied_probability_delta: float | None = None


@dataclass(frozen=True)
class BetDecision:
    race_id: RaceId
    runner_id: RunnerId
    bet_type: BetType
    probability: float
    fair_odds: float | None
    odds: float
    odds_captured_at: datetime
    odds_timing: OddsTiming
    edge: float
    kelly_fraction: float
    stake_fraction: float
    stake_jpy: int
    payout_jpy: int
    profit_jpy: int
    bankroll_before_jpy: int
    bankroll_after_jpy: int
    is_win: bool | None
    did_bet: bool
    skip_reason: str | None
    prediction_as_of: datetime
    closing_odds: float | None = None
    closing_captured_at: datetime | None = None
    clv_odds_delta: float | None = None
    clv_implied_probability_delta: float | None = None


@dataclass(frozen=True)
class BacktestResult:
    records: tuple[BetRecord, ...]
    decisions: tuple[BetDecision, ...]
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
        decisions: list[BetDecision] = []
        bankroll_curve_jpy = [bankroll_jpy]
        daily_profit_jpy: dict[date, int] = {}

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
            race_by_id = {}
            race_groups = sorted(
                race_groups,
                key=lambda race_predictions: (
                    race_predictions[0].as_of,
                    str(race_predictions[0].race_id),
                ),
            )

        for prediction_group in race_groups:
            group_records: list[BetRecord] = []
            group_decisions: list[BetDecision] = []
            group_profit_jpy = 0
            group_staked_jpy = 0
            race_date = _race_date_for_group(prediction_group[0], race_by_id)
            if _daily_stop_loss_triggered(
                race_date=race_date,
                daily_profit_jpy=daily_profit_jpy,
                config=self.config,
            ):
                for prediction in prediction_group:
                    quote = _quote_for_prediction(
                        prediction,
                        win_quotes_by_runner,
                        race_by_id,
                        self.config,
                    )
                    group_decisions.append(
                        _skipped_decision(
                            prediction=prediction,
                            quote=quote,
                            closing_quote=_closing_quote_for_prediction(
                                prediction,
                                win_quotes_by_runner,
                                race_by_id,
                            ),
                            edge=prediction.probability * quote.odds - 1.0,
                            bankroll_jpy=bankroll_jpy,
                            config=self.config,
                            skip_reason="daily_stop_loss",
                        )
                    )
                decisions.extend(group_decisions)
                continue

            for prediction in prediction_group:
                quote = _quote_for_prediction(
                    prediction,
                    win_quotes_by_runner,
                    race_by_id,
                    self.config,
                )
                closing_quote = _closing_quote_for_prediction(
                    prediction,
                    win_quotes_by_runner,
                    race_by_id,
                )
                odds_skip_reason = _odds_skip_reason(quote.odds, self.config)
                edge = prediction.probability * quote.odds - 1.0
                if odds_skip_reason is not None:
                    group_decisions.append(
                        _skipped_decision(
                            prediction=prediction,
                            quote=quote,
                            closing_quote=closing_quote,
                            edge=edge,
                            bankroll_jpy=bankroll_jpy,
                            config=self.config,
                            skip_reason=odds_skip_reason,
                        )
                    )
                    continue

                decision = calculate_kelly_stake(
                    probability=prediction.probability,
                    odds=quote.odds,
                    bankroll_jpy=bankroll_jpy,
                    config=self.config.kelly_config,
                )
                if decision.stake_jpy <= 0:
                    group_decisions.append(
                        _skipped_decision(
                            prediction=prediction,
                            quote=quote,
                            closing_quote=closing_quote,
                            edge=decision.edge,
                            bankroll_jpy=bankroll_jpy,
                            config=self.config,
                            skip_reason=_stake_skip_reason(decision, self.config),
                            kelly_fraction=decision.full_kelly_fraction,
                            stake_fraction=decision.stake_fraction,
                        )
                    )
                    continue

                stake_jpy = _cap_stake_for_race(
                    requested_stake_jpy=decision.stake_jpy,
                    already_staked_jpy=group_staked_jpy,
                    config=self.config,
                )
                if stake_jpy <= 0:
                    group_decisions.append(
                        _skipped_decision(
                            prediction=prediction,
                            quote=quote,
                            closing_quote=closing_quote,
                            edge=decision.edge,
                            bankroll_jpy=bankroll_jpy,
                            config=self.config,
                            skip_reason="race_stake_limit",
                            kelly_fraction=decision.full_kelly_fraction,
                            stake_fraction=0.0,
                        )
                    )
                    continue

                race_result = _result_for_prediction(prediction, results_by_runner)
                is_win = race_result.did_win
                payout_jpy = int(round(stake_jpy * quote.odds)) if is_win else 0
                profit_jpy = payout_jpy - stake_jpy
                group_profit_jpy += profit_jpy
                group_staked_jpy += stake_jpy
                stake_fraction = stake_jpy / bankroll_jpy

                group_records.append(
                    BetRecord(
                        race_id=prediction.race_id,
                        runner_id=prediction.runner_id,
                        bet_type=BetType.WIN,
                        probability=prediction.probability,
                        odds=quote.odds,
                        edge=decision.edge,
                        kelly_fraction=decision.full_kelly_fraction,
                        stake_fraction=stake_fraction,
                        stake_jpy=stake_jpy,
                        payout_jpy=payout_jpy,
                        profit_jpy=profit_jpy,
                        bankroll_after_jpy=0,
                        is_win=is_win,
                        odds_captured_at=quote.captured_at,
                        fair_odds=_fair_odds(prediction.probability),
                        closing_odds=closing_quote.odds if closing_quote else None,
                        closing_captured_at=(
                            closing_quote.captured_at if closing_quote else None
                        ),
                        clv_odds_delta=_clv_odds_delta(quote, closing_quote),
                        clv_implied_probability_delta=(
                            _clv_implied_probability_delta(quote, closing_quote)
                        ),
                    )
                )
                group_decisions.append(
                    BetDecision(
                        race_id=prediction.race_id,
                        runner_id=prediction.runner_id,
                        bet_type=BetType.WIN,
                        probability=prediction.probability,
                        fair_odds=_fair_odds(prediction.probability),
                        odds=quote.odds,
                        odds_captured_at=quote.captured_at,
                        odds_timing=self.config.odds_timing,
                        edge=decision.edge,
                        kelly_fraction=decision.full_kelly_fraction,
                        stake_fraction=stake_fraction,
                        stake_jpy=stake_jpy,
                        payout_jpy=payout_jpy,
                        profit_jpy=profit_jpy,
                        bankroll_before_jpy=bankroll_jpy,
                        bankroll_after_jpy=0,
                        is_win=is_win,
                        did_bet=True,
                        skip_reason=None,
                        prediction_as_of=prediction.as_of,
                        closing_odds=closing_quote.odds if closing_quote else None,
                        closing_captured_at=(
                            closing_quote.captured_at if closing_quote else None
                        ),
                        clv_odds_delta=_clv_odds_delta(quote, closing_quote),
                        clv_implied_probability_delta=(
                            _clv_implied_probability_delta(quote, closing_quote)
                        ),
                    )
                )

            if not group_records:
                decisions.extend(
                    replace(decision, bankroll_after_jpy=bankroll_jpy)
                    for decision in group_decisions
                )
                continue

            bankroll_jpy += group_profit_jpy
            daily_profit_jpy[race_date] = (
                daily_profit_jpy.get(race_date, 0) + group_profit_jpy
            )
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
                    odds_captured_at=record.odds_captured_at,
                    fair_odds=record.fair_odds,
                    closing_odds=record.closing_odds,
                    closing_captured_at=record.closing_captured_at,
                    clv_odds_delta=record.clv_odds_delta,
                    clv_implied_probability_delta=(
                        record.clv_implied_probability_delta
                    ),
                )
                for record in group_records
            )
            decisions.extend(
                replace(decision, bankroll_after_jpy=bankroll_jpy)
                for decision in group_decisions
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
            decisions=tuple(decisions),
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


def _quote_for_prediction(
    prediction: ModelPrediction,
    win_quotes_by_runner: dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]],
    race_by_id: dict[RaceId, Race],
    config: BacktestConfig,
) -> OddsQuote:
    cutoff = _odds_cutoff_for_prediction(prediction, race_by_id, config)
    return _latest_quote_at_or_before(
        prediction=prediction,
        win_quotes_by_runner=win_quotes_by_runner,
        captured_at_or_before=cutoff,
    )


def _latest_quote_at_or_before(
    *,
    prediction: ModelPrediction,
    win_quotes_by_runner: dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]],
    captured_at_or_before: datetime,
) -> OddsQuote:
    quotes = win_quotes_by_runner.get((prediction.race_id, prediction.runner_id), ())
    for quote in reversed(quotes):
        if quote.captured_at <= captured_at_or_before:
            return quote
    raise ValueError(
        "Missing odds for "
        f"race_id={prediction.race_id!r}, runner_id={prediction.runner_id!r}"
    )


def _closing_quote_for_prediction(
    prediction: ModelPrediction,
    win_quotes_by_runner: dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]],
    race_by_id: dict[RaceId, Race],
) -> OddsQuote | None:
    race = race_by_id.get(prediction.race_id)
    cutoff = race.start_time if race is not None and race.start_time else prediction.as_of
    try:
        return _latest_quote_at_or_before(
            prediction=prediction,
            win_quotes_by_runner=win_quotes_by_runner,
            captured_at_or_before=cutoff,
        )
    except ValueError:
        return None


def _odds_cutoff_for_prediction(
    prediction: ModelPrediction,
    race_by_id: dict[RaceId, Race],
    config: BacktestConfig,
) -> datetime:
    if config.odds_timing == OddsTiming.LATEST_AVAILABLE:
        return prediction.as_of

    race = race_by_id.get(prediction.race_id)
    if race is None or race.start_time is None:
        raise ValueError(
            "Race start_time is required for "
            f"{config.odds_timing.value!r} odds timing"
        )

    if config.odds_timing == OddsTiming.CLOSING:
        return race.start_time
    if config.odds_timing == OddsTiming.MINUTES_BEFORE_START:
        minutes = config.odds_minutes_before_start
        if minutes is None:
            raise ValueError("odds_minutes_before_start is required")
        return race.start_time - timedelta(minutes=minutes)
    raise ValueError(f"Unsupported odds timing: {config.odds_timing!r}")


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


def _race_date_for_group(
    prediction: ModelPrediction,
    race_by_id: dict[RaceId, Race],
) -> date:
    race = race_by_id.get(prediction.race_id)
    if race is not None:
        return race.race_date
    return prediction.as_of.date()


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


def _odds_skip_reason(odds: float, config: BacktestConfig) -> str | None:
    if config.min_odds is not None and odds < config.min_odds:
        return "odds_below_minimum"
    if config.max_odds is not None and odds > config.max_odds:
        return "odds_above_maximum"
    return None


def _stake_skip_reason(decision: object, config: BacktestConfig) -> str:
    edge = getattr(decision, "edge")
    stake_fraction = getattr(decision, "stake_fraction")
    if edge < config.kelly_config.minimum_edge:
        return "edge_below_minimum"
    if stake_fraction <= 0.0:
        return "no_stake_fraction"
    return "stake_below_unit"


def _cap_stake_for_race(
    *,
    requested_stake_jpy: int,
    already_staked_jpy: int,
    config: BacktestConfig,
) -> int:
    if config.max_stake_per_race_jpy is None:
        return requested_stake_jpy
    remaining_jpy = config.max_stake_per_race_jpy - already_staked_jpy
    capped_jpy = min(requested_stake_jpy, max(remaining_jpy, 0))
    unit = config.kelly_config.stake_unit_jpy
    return floor(capped_jpy / unit) * unit


def _daily_stop_loss_triggered(
    *,
    race_date: date,
    daily_profit_jpy: dict[date, int],
    config: BacktestConfig,
) -> bool:
    if config.max_daily_loss_jpy is None:
        return False
    return daily_profit_jpy.get(race_date, 0) <= -config.max_daily_loss_jpy


def _skipped_decision(
    *,
    prediction: ModelPrediction,
    quote: OddsQuote,
    closing_quote: OddsQuote | None,
    edge: float,
    bankroll_jpy: int,
    config: BacktestConfig,
    skip_reason: str,
    kelly_fraction: float = 0.0,
    stake_fraction: float = 0.0,
) -> BetDecision:
    return BetDecision(
        race_id=prediction.race_id,
        runner_id=prediction.runner_id,
        bet_type=BetType.WIN,
        probability=prediction.probability,
        fair_odds=_fair_odds(prediction.probability),
        odds=quote.odds,
        odds_captured_at=quote.captured_at,
        odds_timing=config.odds_timing,
        edge=edge,
        kelly_fraction=kelly_fraction,
        stake_fraction=stake_fraction,
        stake_jpy=0,
        payout_jpy=0,
        profit_jpy=0,
        bankroll_before_jpy=bankroll_jpy,
        bankroll_after_jpy=bankroll_jpy,
        is_win=None,
        did_bet=False,
        skip_reason=skip_reason,
        prediction_as_of=prediction.as_of,
        closing_odds=closing_quote.odds if closing_quote else None,
        closing_captured_at=closing_quote.captured_at if closing_quote else None,
        clv_odds_delta=_clv_odds_delta(quote, closing_quote),
        clv_implied_probability_delta=_clv_implied_probability_delta(
            quote,
            closing_quote,
        ),
    )


def _fair_odds(probability: float) -> float | None:
    if probability <= 0.0:
        return None
    return 1.0 / probability


def _clv_odds_delta(
    selected_quote: OddsQuote,
    closing_quote: OddsQuote | None,
) -> float | None:
    if closing_quote is None:
        return None
    return selected_quote.odds - closing_quote.odds


def _clv_implied_probability_delta(
    selected_quote: OddsQuote,
    closing_quote: OddsQuote | None,
) -> float | None:
    if closing_quote is None:
        return None
    return (1.0 / closing_quote.odds) - (1.0 / selected_quote.odds)
