"""Live-like daily paper-trading pipeline for Phase 6."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.backtesting import BacktestConfig
from horse_lab.data import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.data.repositories import (
    FeatureRepository,
    OddsRepository,
    RaceRepository,
    ResultRepository,
)
from horse_lab.models import (
    InferenceContext,
    LightGBMWinProbabilityModel,
    MarketImpliedProbabilityModel,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.models.lightgbm import EstimatorFactory
from horse_lab.pipelines.lightgbm_training import MARKET_FEATURE_NAMES
from horse_lab.pipelines.paper_trading import (
    PaperTradingResult,
    paper_trading_result_to_dict,
    run_paper_trading_from_csv,
)
from horse_lab.schemas import (
    FeatureName,
    FeatureRow,
    ModelName,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    Race,
    RaceId,
    Result,
    RunnerId,
    TrainingLabel,
)


DAILY_PREDICTION_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "fold_id",
    "label",
    "method",
    "probability",
)


@dataclass(frozen=True)
class DailyPaperTradingResult:
    report_path: Path
    level0_predictions_path: Path
    candidate_predictions_path: Path
    paper_result: PaperTradingResult
    report: dict[str, Any]


@dataclass(frozen=True)
class Level0Spec:
    method_column: str
    model_key: str
    model_version: str
    excluded_feature_names: tuple[FeatureName, ...] = ()


DEFAULT_LEVEL0_SPECS: tuple[Level0Spec, ...] = (
    Level0Spec(
        method_column="pred__market_implied_probability__market_implied_oof_v1",
        model_key="market",
        model_version="market-implied-oof-v1",
    ),
    Level0Spec(
        method_column="pred__lightgbm_win_probability__lightgbm_full_oof_v1",
        model_key="lightgbm_full",
        model_version="lightgbm-full-oof-v1",
    ),
    Level0Spec(
        method_column="pred__lightgbm_win_probability__lightgbm_no_market_oof_v1",
        model_key="lightgbm_no_market",
        model_version="lightgbm-no-market-oof-v1",
        excluded_feature_names=MARKET_FEATURE_NAMES,
    ),
)


def run_daily_paper_trading_from_csv(
    dataset_dir: Path | str,
    model_registry_path: Path | str,
    artifact_dir: Path | str,
    *,
    start_date: date,
    end_date: date,
    as_of: datetime,
    feature_version: str,
    train_end_date: date | None = None,
    candidate_method: str | None = None,
    backtest_config: BacktestConfig = BacktestConfig(),
    use_odds_timeseries: bool = True,
    require_registry_approval: bool = True,
    estimator_factory: EstimatorFactory | None = None,
) -> DailyPaperTradingResult:
    dataset_path = Path(dataset_dir)
    odds_path = _replay_odds_csv_path(
        dataset_path,
        use_odds_timeseries=use_odds_timeseries,
    )
    return run_daily_paper_trading(
        race_repository=CsvRaceRepository(dataset_path / "races.csv"),
        odds_repository=CsvOddsRepository(odds_path),
        result_repository=CsvResultRepository(dataset_path / "results.csv"),
        feature_repository=CsvFeatureRepository(dataset_path / "features.csv"),
        model_registry_path=model_registry_path,
        artifact_dir=artifact_dir,
        dataset_dir=dataset_path,
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
        feature_version=feature_version,
        train_end_date=train_end_date,
        candidate_method=candidate_method,
        backtest_config=backtest_config,
        use_odds_timeseries=use_odds_timeseries,
        require_registry_approval=require_registry_approval,
        estimator_factory=estimator_factory,
    )


def run_daily_paper_trading(
    *,
    race_repository: RaceRepository,
    odds_repository: OddsRepository,
    result_repository: ResultRepository,
    feature_repository: FeatureRepository,
    model_registry_path: Path | str,
    artifact_dir: Path | str,
    dataset_dir: Path,
    start_date: date,
    end_date: date,
    as_of: datetime,
    feature_version: str,
    train_end_date: date | None = None,
    candidate_method: str | None = None,
    backtest_config: BacktestConfig = BacktestConfig(),
    use_odds_timeseries: bool = True,
    require_registry_approval: bool = True,
    estimator_factory: EstimatorFactory | None = None,
) -> DailyPaperTradingResult:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    registry_path = Path(model_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        require_registry_approval
        and not registry["approval_gate"]["approved_for_paper_trading"]
    ):
        raise ValueError("model registry candidate is not approved for paper trading")

    selected_method = candidate_method or registry["candidate"]["method"]
    selected_train_end = train_end_date or (start_date - timedelta(days=1))

    all_races = tuple(
        race_repository.list_races(
            start_date=date.min,
            end_date=end_date,
        )
    )
    train_races = tuple(
        race for race in all_races if race.race_date <= selected_train_end
    )
    target_races = tuple(
        race for race in all_races if start_date <= race.race_date <= end_date
    )
    _require_non_empty(train_races, "training races")
    _require_non_empty(target_races, "target races")

    train_race_ids = tuple(race.race_id for race in train_races)
    target_race_ids = tuple(race.race_id for race in target_races)
    train_features = tuple(
        feature_repository.list_feature_rows(
            race_ids=train_race_ids,
            feature_version=feature_version,
            as_of=as_of,
        )
    )
    target_features = tuple(
        feature_repository.list_feature_rows(
            race_ids=target_race_ids,
            feature_version=feature_version,
            as_of=as_of,
        )
    )
    train_results = tuple(result_repository.list_results(race_ids=train_race_ids))
    target_results = tuple(result_repository.list_results(race_ids=target_race_ids))
    target_odds = tuple(
        odds_repository.list_odds(
            race_ids=target_race_ids,
            captured_at_or_before=as_of,
        )
    )
    _validate_rows_have_results(
        feature_rows=train_features,
        results=train_results,
        label="training",
    )
    _validate_rows_have_results(
        feature_rows=target_features,
        results=target_results,
        label="target",
    )

    level0_by_column = _generate_level0_predictions(
        specs=DEFAULT_LEVEL0_SPECS,
        train_races=train_races,
        train_features=train_features,
        train_results=train_results,
        target_races=target_races,
        target_features=target_features,
        target_odds=target_odds,
        train_end_date=selected_train_end,
        start_date=start_date,
        as_of=as_of,
        feature_version=feature_version,
        estimator_factory=estimator_factory,
    )
    weights = _registry_weights(registry)
    candidate_predictions = _blend_predictions(
        level0_by_column=level0_by_column,
        weights=weights,
        method=selected_method,
        model_version=registry["candidate"]["model_version"],
        as_of=as_of,
    )

    output_path = Path(artifact_dir)
    level0_predictions_path = output_path / "level0_predictions.csv"
    candidate_predictions_path = output_path / "candidate_predictions.csv"
    report_path = output_path / "daily_paper_trading_report.json"
    _write_prediction_records(
        level0_predictions_path,
        predictions=[
            (method, prediction)
            for method, predictions in level0_by_column.items()
            for prediction in predictions
        ],
        results=target_results,
    )
    _write_prediction_records(
        candidate_predictions_path,
        predictions=[(selected_method, prediction) for prediction in candidate_predictions],
        results=target_results,
    )

    paper_result = run_paper_trading_from_csv(
        candidate_predictions_path,
        dataset_dir,
        output_path / "paper",
        method=selected_method,
        as_of=as_of,
        model_version=registry["candidate"]["model_version"],
        backtest_config=backtest_config,
        use_odds_timeseries=use_odds_timeseries,
    )
    report = {
        "run_type": "daily-paper-trading",
        "dataset_dir": str(dataset_dir),
        "model_registry_path": str(registry_path),
        "candidate_method": selected_method,
        "model_version": registry["candidate"]["model_version"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "train_end_date": selected_train_end.isoformat(),
        "as_of": as_of.isoformat(),
        "feature_version": feature_version,
        "counts": {
            "train_races": len(train_races),
            "target_races": len(target_races),
            "train_features": len(train_features),
            "target_features": len(target_features),
            "target_odds": len(target_odds),
            "candidate_predictions": len(candidate_predictions),
        },
        "serving_policy": registry["serving_policy"],
        "paper_trading": paper_result.report,
        "artifacts": {
            "level0_predictions_path": str(level0_predictions_path),
            "candidate_predictions_path": str(candidate_predictions_path),
            "paper_report_path": str(paper_result.report_path),
            "paper_bet_decisions_path": str(paper_result.bet_decisions_path),
            "paper_clv_report_path": str(paper_result.clv_report_path),
        },
    }
    _write_json(report_path, report)
    return DailyPaperTradingResult(
        report_path=report_path,
        level0_predictions_path=level0_predictions_path,
        candidate_predictions_path=candidate_predictions_path,
        paper_result=paper_result,
        report=report,
    )


def daily_paper_trading_result_to_dict(
    result: DailyPaperTradingResult,
) -> dict[str, Any]:
    return {
        "report_path": str(result.report_path),
        "level0_predictions_path": str(result.level0_predictions_path),
        "candidate_predictions_path": str(result.candidate_predictions_path),
        "paper_trading": paper_trading_result_to_dict(result.paper_result),
        "report": result.report,
    }


def _generate_level0_predictions(
    *,
    specs: Sequence[Level0Spec],
    train_races: tuple[Race, ...],
    train_features: tuple[FeatureRow, ...],
    train_results: tuple[Result, ...],
    target_races: tuple[Race, ...],
    target_features: tuple[FeatureRow, ...],
    target_odds: tuple[OddsQuote, ...],
    train_end_date: date,
    start_date: date,
    as_of: datetime,
    feature_version: str,
    estimator_factory: EstimatorFactory | None,
) -> dict[str, tuple[ModelPrediction, ...]]:
    predictions_by_column: dict[str, tuple[ModelPrediction, ...]] = {}
    for spec in specs:
        if spec.model_key == "market":
            model = MarketImpliedProbabilityModel(model_version=spec.model_version)
            predictions_by_column[spec.method_column] = tuple(
                model.predict(
                    target_features,
                    context=InferenceContext(
                        as_of=as_of,
                        feature_version=feature_version,
                        races=target_races,
                        odds=target_odds,
                    ),
                )
            )
            continue

        filtered_train = _filter_feature_rows(
            train_features,
            spec.excluded_feature_names,
        )
        filtered_target = _filter_feature_rows(
            target_features,
            spec.excluded_feature_names,
        )
        model = LightGBMWinProbabilityModel(
            model_version=spec.model_version,
            estimator_factory=estimator_factory,
        )
        model.fit(
            TrainingDataset(
                feature_rows=filtered_train,
                labels=_training_labels(filtered_train, train_results),
                feature_names=_feature_names(filtered_train),
                target=PredictionTarget.WIN_PROBABILITY,
                metadata={"results": train_results},
            ),
            context=TrainingContext(
                train_start=min(race.race_date for race in train_races),
                train_end=train_end_date,
                validation_start=start_date,
                feature_version=feature_version,
                random_seed=42,
            ),
        )
        predictions_by_column[spec.method_column] = tuple(
            model.predict(
                filtered_target,
                context=InferenceContext(
                    as_of=as_of,
                    feature_version=feature_version,
                    races=target_races,
                    odds=target_odds,
                ),
            )
        )
    return predictions_by_column


def _blend_predictions(
    *,
    level0_by_column: Mapping[str, tuple[ModelPrediction, ...]],
    weights: Mapping[str, float],
    method: str,
    model_version: str,
    as_of: datetime,
) -> tuple[ModelPrediction, ...]:
    available = set(level0_by_column)
    missing = set(weights) - available
    if missing:
        raise ValueError(f"Missing Level 0 prediction columns: {sorted(missing)}")

    reference_column = next(iter(weights))
    reference = level0_by_column[reference_column]
    raw_values: list[tuple[ModelPrediction, float]] = []
    for index, prediction in enumerate(reference):
        key = (prediction.race_id, prediction.runner_id)
        probability = 0.0
        for column, weight in weights.items():
            column_prediction = level0_by_column[column][index]
            if (column_prediction.race_id, column_prediction.runner_id) != key:
                raise ValueError("Level 0 predictions are not aligned")
            probability += weight * column_prediction.probability
        raw_values.append((prediction, probability))

    totals: dict[RaceId, float] = {}
    counts: dict[RaceId, int] = {}
    for prediction, probability in raw_values:
        totals[prediction.race_id] = totals.get(prediction.race_id, 0.0) + probability
        counts[prediction.race_id] = counts.get(prediction.race_id, 0) + 1

    blended = []
    for prediction, probability in raw_values:
        total = totals[prediction.race_id]
        normalized = probability / total if total > 0.0 else 1.0 / counts[prediction.race_id]
        blended.append(
            ModelPrediction(
                race_id=prediction.race_id,
                runner_id=prediction.runner_id,
                model_name=ModelName(method),
                model_version=model_version,
                target=PredictionTarget.WIN_PROBABILITY,
                probability=normalized,
                as_of=as_of,
                metadata={"weights": dict(weights), "raw_probability": probability},
            )
        )
    return tuple(blended)


def _registry_weights(registry: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["feature_column"]): float(row["weight"])
        for row in registry["serving_policy"]["weights"]
    }


def _write_prediction_records(
    path: Path,
    *,
    predictions: Sequence[tuple[str, ModelPrediction]],
    results: Sequence[Result],
) -> None:
    result_by_runner = _result_by_runner(tuple(results))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DAILY_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                "race_id": str(prediction.race_id),
                "runner_id": str(prediction.runner_id),
                "fold_id": "daily",
                "label": (
                    "1"
                    if result_by_runner[
                        (prediction.race_id, str(prediction.runner_id))
                    ].did_win
                    else "0"
                ),
                "method": method,
                "probability": str(prediction.probability),
            }
            for method, prediction in predictions
        )


def _training_labels(
    feature_rows: tuple[FeatureRow, ...],
    results: tuple[Result, ...],
) -> tuple[TrainingLabel, ...]:
    result_by_runner = _result_by_runner(results)
    return tuple(
        TrainingLabel(
            race_id=row.race_id,
            runner_id=row.runner_id,
            target=PredictionTarget.WIN_PROBABILITY,
            value=(
                1.0
                if result_by_runner[(row.race_id, str(row.runner_id))].did_win
                else 0.0
            ),
        )
        for row in feature_rows
    )


def _feature_names(feature_rows: tuple[FeatureRow, ...]) -> tuple[FeatureName, ...]:
    names = {
        feature_name
        for row in feature_rows
        for feature_name in row.values
    }
    if not names:
        raise ValueError("Training feature rows must contain at least one feature")
    return tuple(sorted(names, key=str))


def _filter_feature_rows(
    feature_rows: tuple[FeatureRow, ...],
    excluded_feature_names: Sequence[FeatureName],
) -> tuple[FeatureRow, ...]:
    excluded = set(excluded_feature_names)
    if not excluded:
        return feature_rows
    return tuple(
        FeatureRow(
            race_id=row.race_id,
            runner_id=row.runner_id,
            as_of=row.as_of,
            feature_version=row.feature_version,
            values={
                feature_name: value
                for feature_name, value in row.values.items()
                if feature_name not in excluded
            },
            metadata=row.metadata,
        )
        for row in feature_rows
    )


def _validate_rows_have_results(
    *,
    feature_rows: tuple[FeatureRow, ...],
    results: tuple[Result, ...],
    label: str,
) -> None:
    _require_non_empty(feature_rows, f"{label} feature rows")
    _require_non_empty(results, f"{label} results")
    result_by_runner = _result_by_runner(results)
    missing = [
        (row.race_id, row.runner_id)
        for row in feature_rows
        if (row.race_id, str(row.runner_id)) not in result_by_runner
    ]
    if missing:
        race_id, runner_id = missing[0]
        raise ValueError(
            f"Missing {label} result for race_id={race_id!r}, runner_id={runner_id!r}"
        )


def _result_by_runner(results: tuple[Result, ...]) -> dict[tuple[RaceId, str], Result]:
    return {
        (result.race_id, str(result.runner_id)): result
        for result in results
    }


def _replay_odds_csv_path(
    dataset_path: Path,
    *,
    use_odds_timeseries: bool,
) -> Path:
    odds_timeseries_path = dataset_path / "odds_timeseries.csv"
    odds_path = dataset_path / "odds.csv"
    if use_odds_timeseries and odds_timeseries_path.exists():
        return odds_timeseries_path
    if odds_path.exists():
        return odds_path
    return odds_timeseries_path


def _require_non_empty(rows: Sequence[object], label: str) -> None:
    if not rows:
        raise ValueError(f"No {label} found")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
