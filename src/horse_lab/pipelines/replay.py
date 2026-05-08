"""Historical replay pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from horse_lab.backtesting import BacktestConfig, BacktestResult, BacktestSimulator
from horse_lab.betting import KellyConfig
from horse_lab.data.repositories import (
    FeatureRepository,
    OddsRepository,
    RaceRepository,
    ResultRepository,
)
from horse_lab.evaluation import PerformanceSummary
from horse_lab.models import InferenceContext, MarketImpliedProbabilityModel
from horse_lab.schemas import (
    BetType,
    FeatureRow,
    ModelPrediction,
    OddsQuote,
    Race,
    RaceId,
    Result,
    RunnerId,
)


@dataclass(frozen=True)
class ReplayResult:
    races: tuple[Race, ...]
    feature_rows: tuple[FeatureRow, ...]
    odds: tuple[OddsQuote, ...]
    results: tuple[Result, ...]
    predictions: tuple[ModelPrediction, ...]
    backtest_result: BacktestResult
    summary: PerformanceSummary


def run_market_replay(
    *,
    race_repository: RaceRepository,
    odds_repository: OddsRepository,
    result_repository: ResultRepository,
    feature_repository: FeatureRepository,
    start_date: date,
    end_date: date,
    as_of: datetime,
    feature_version: str,
    initial_bankroll_jpy: int = 100_000,
    kelly_config: KellyConfig | None = None,
    model_version: str = "market-implied-v1",
) -> ReplayResult:
    """Run a point-in-time market-implied replay over repository data.

    The market baseline uses the latest per-runner win odds at or before
    ``as_of``. This is point-in-time safe, but it is not a same-timestamp
    market snapshot selector.
    """

    races = tuple(
        race_repository.list_races(start_date=start_date, end_date=end_date)
    )
    race_ids = tuple(race.race_id for race in races)
    feature_rows = tuple(
        feature_repository.list_feature_rows(
            race_ids=race_ids,
            feature_version=feature_version,
            as_of=as_of,
        )
    )
    odds = tuple(
        odds_repository.list_odds(
            race_ids=race_ids,
            captured_at_or_before=as_of,
        )
    )
    results = tuple(result_repository.list_results(race_ids=race_ids))

    _validate_replay_inputs(
        races=races,
        feature_rows=feature_rows,
        odds=odds,
        results=results,
    )

    model = MarketImpliedProbabilityModel(model_version=model_version)
    predictions = tuple(
        model.predict(
            feature_rows,
            context=InferenceContext(
                as_of=as_of,
                feature_version=feature_version,
                races=races,
                odds=odds,
            ),
        )
    )

    backtest_result = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=initial_bankroll_jpy,
            kelly_config=kelly_config or KellyConfig(),
        )
    ).run(predictions=predictions, odds=odds, results=results, races=races)

    return ReplayResult(
        races=races,
        feature_rows=feature_rows,
        odds=odds,
        results=results,
        predictions=predictions,
        backtest_result=backtest_result,
        summary=backtest_result.summary,
    )


def _validate_replay_inputs(
    *,
    races: tuple[Race, ...],
    feature_rows: tuple[FeatureRow, ...],
    odds: tuple[OddsQuote, ...],
    results: tuple[Result, ...],
) -> None:
    feature_runners_by_race = _feature_runners_by_race(feature_rows)
    result_runners_by_race = _result_runners_by_race(results)
    odds_runners_by_race = _win_odds_runners_by_race(odds)

    for race in races:
        feature_runners = feature_runners_by_race.get(race.race_id, set())
        if race.field_size is not None and len(feature_runners) != race.field_size:
            raise ValueError(
                "Feature row count does not match field_size "
                f"for race_id={race.race_id!r}"
            )

        result_runners = result_runners_by_race.get(race.race_id, set())
        if result_runners != feature_runners:
            raise ValueError(
                "Result runners do not match feature runners "
                f"for race_id={race.race_id!r}"
            )

        odds_runners = odds_runners_by_race.get(race.race_id, set())
        if odds_runners != feature_runners:
            raise ValueError(
                "Odds runners do not match feature runners "
                f"for race_id={race.race_id!r}"
            )


def _feature_runners_by_race(
    feature_rows: tuple[FeatureRow, ...],
) -> dict[RaceId, set[RunnerId]]:
    runners_by_race: dict[RaceId, set[RunnerId]] = {}
    for row in feature_rows:
        runners_by_race.setdefault(row.race_id, set()).add(row.runner_id)
    return runners_by_race


def _result_runners_by_race(results: tuple[Result, ...]) -> dict[RaceId, set[RunnerId]]:
    runners_by_race: dict[RaceId, set[RunnerId]] = {}
    for result in results:
        runners_by_race.setdefault(result.race_id, set()).add(result.runner_id)
    return runners_by_race


def _win_odds_runners_by_race(
    odds: tuple[OddsQuote, ...],
) -> dict[RaceId, set[RunnerId]]:
    runners_by_race: dict[RaceId, set[RunnerId]] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN:
            continue
        runners_by_race.setdefault(quote.race_id, set()).add(quote.runner_id)
    return runners_by_race
