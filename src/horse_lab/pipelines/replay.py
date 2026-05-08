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
from horse_lab.schemas import FeatureRow, ModelPrediction, OddsQuote, Race, Result


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
    """Run a point-in-time market-implied replay over repository data."""

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
    ).run(predictions=predictions, odds=odds, results=results)

    return ReplayResult(
        races=races,
        feature_rows=feature_rows,
        odds=odds,
        results=results,
        predictions=predictions,
        backtest_result=backtest_result,
        summary=backtest_result.summary,
    )
