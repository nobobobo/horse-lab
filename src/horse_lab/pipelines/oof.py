"""Out-of-fold prediction generation for Phase 4 stacking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

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
from horse_lab.evaluation import ProbabilitySummary, summarize_win_probability_predictions
from horse_lab.models import (
    InferenceContext,
    LightGBMWinProbabilityModel,
    MarketImpliedProbabilityModel,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.models.lightgbm import EstimatorFactory
from horse_lab.pipelines.lightgbm_training import MARKET_FEATURE_NAMES
from horse_lab.schemas import (
    FeatureName,
    FeatureRow,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    Race,
    RaceId,
    Result,
    TrainingLabel,
)
from horse_lab.stacking import (
    PredictionRole,
    StoredPrediction,
    model_prediction_to_stored_prediction,
    write_prediction_store_csv,
)


@dataclass(frozen=True)
class OOFFold:
    fold_id: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date


@dataclass(frozen=True)
class OOFModelSpec:
    model_key: str
    model_version: str
    exclude_feature_names: tuple[FeatureName, ...] = ()


@dataclass(frozen=True)
class OOFRunResult:
    predictions_path: Path
    report_path: Path
    stored_predictions: tuple[StoredPrediction, ...]
    folds: tuple[OOFFold, ...]
    report: dict[str, Any]


DEFAULT_OOF_MODEL_SPECS: tuple[OOFModelSpec, ...] = (
    OOFModelSpec("market", "market-implied-oof-v1"),
    OOFModelSpec("lightgbm_full", "lightgbm-full-oof-v1"),
    OOFModelSpec(
        "lightgbm_no_market",
        "lightgbm-no-market-oof-v1",
        exclude_feature_names=MARKET_FEATURE_NAMES,
    ),
)


def run_level0_oof_from_csv(
    dataset_dir: Path | str,
    artifact_dir: Path | str,
    *,
    validation_start_date: date,
    validation_end_date: date,
    as_of: datetime,
    feature_version: str,
    model_keys: Sequence[str] | None = None,
    random_seed: int = 42,
    estimator_factory: EstimatorFactory | None = None,
    use_odds_timeseries: bool = False,
) -> OOFRunResult:
    dataset_path = Path(dataset_dir)
    return run_level0_oof(
        race_repository=CsvRaceRepository(dataset_path / "races.csv"),
        odds_repository=CsvOddsRepository(
            _replay_odds_csv_path(
                dataset_path,
                use_odds_timeseries=use_odds_timeseries,
            )
        ),
        result_repository=CsvResultRepository(dataset_path / "results.csv"),
        feature_repository=CsvFeatureRepository(dataset_path / "features.csv"),
        artifact_dir=artifact_dir,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
        as_of=as_of,
        feature_version=feature_version,
        model_specs=_select_model_specs(model_keys),
        random_seed=random_seed,
        estimator_factory=estimator_factory,
    )


def run_level0_oof(
    *,
    race_repository: RaceRepository,
    odds_repository: OddsRepository,
    result_repository: ResultRepository,
    feature_repository: FeatureRepository,
    artifact_dir: Path | str,
    validation_start_date: date,
    validation_end_date: date,
    as_of: datetime,
    feature_version: str,
    model_specs: Sequence[OOFModelSpec] = DEFAULT_OOF_MODEL_SPECS,
    random_seed: int = 42,
    estimator_factory: EstimatorFactory | None = None,
) -> OOFRunResult:
    races = tuple(
        race_repository.list_races(
            start_date=date.min,
            end_date=validation_end_date,
        )
    )
    _require_non_empty(races, "races")
    folds = _monthly_folds(
        races,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
    )
    _require_non_empty(folds, "OOF folds")

    stored_predictions: list[StoredPrediction] = []
    summaries: dict[str, dict[str, Any]] = {}
    validation_results_for_summary: dict[str, list[Result]] = {
        spec.model_key: [] for spec in model_specs
    }
    model_predictions_for_summary: dict[str, list[ModelPrediction]] = {
        spec.model_key: [] for spec in model_specs
    }

    for fold in folds:
        fold_train_races = _races_in_window(
            races,
            start_date=fold.train_start,
            end_date=fold.train_end,
        )
        fold_validation_races = _races_in_window(
            races,
            start_date=fold.validation_start,
            end_date=fold.validation_end,
        )
        if not fold_train_races or not fold_validation_races:
            continue

        train_race_ids = tuple(race.race_id for race in fold_train_races)
        validation_race_ids = tuple(race.race_id for race in fold_validation_races)
        train_feature_rows = tuple(
            feature_repository.list_feature_rows(
                race_ids=train_race_ids,
                feature_version=feature_version,
                as_of=as_of,
            )
        )
        validation_feature_rows = tuple(
            feature_repository.list_feature_rows(
                race_ids=validation_race_ids,
                feature_version=feature_version,
                as_of=as_of,
            )
        )
        train_results = tuple(result_repository.list_results(race_ids=train_race_ids))
        validation_results = tuple(
            result_repository.list_results(race_ids=validation_race_ids)
        )
        validation_odds = tuple(
            odds_repository.list_odds(
                race_ids=validation_race_ids,
                captured_at_or_before=as_of,
            )
        )

        _validate_rows_have_results(
            feature_rows=train_feature_rows,
            results=train_results,
            label=f"{fold.fold_id} training",
        )
        _validate_rows_have_results(
            feature_rows=validation_feature_rows,
            results=validation_results,
            label=f"{fold.fold_id} validation",
        )

        for spec in model_specs:
            predictions = _predict_fold(
                spec=spec,
                train_races=fold_train_races,
                train_feature_rows=train_feature_rows,
                train_results=train_results,
                validation_races=fold_validation_races,
                validation_feature_rows=validation_feature_rows,
                validation_odds=validation_odds,
                fold=fold,
                as_of=as_of,
                feature_version=feature_version,
                random_seed=random_seed,
                estimator_factory=estimator_factory,
            )
            stored_predictions.extend(
                model_prediction_to_stored_prediction(
                    prediction,
                    prediction_role=PredictionRole.OOF,
                    feature_version=feature_version,
                    fold_id=fold.fold_id,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    validation_start=fold.validation_start,
                    validation_end=fold.validation_end,
                )
                for prediction in predictions
            )
            model_predictions_for_summary[spec.model_key].extend(predictions)
            validation_results_for_summary[spec.model_key].extend(validation_results)

    artifact_path = Path(artifact_dir)
    predictions_path = artifact_path / "oof_predictions.csv"
    report_path = artifact_path / "oof_report.json"
    stored = tuple(stored_predictions)
    write_prediction_store_csv(predictions_path, stored)

    for spec in model_specs:
        predictions = tuple(model_predictions_for_summary[spec.model_key])
        results = tuple(validation_results_for_summary[spec.model_key])
        summaries[spec.model_key] = _model_summary(
            predictions=predictions,
            results=results,
            spec=spec,
        )

    report = {
        "feature_version": feature_version,
        "as_of": as_of.isoformat(),
        "validation_start_date": validation_start_date.isoformat(),
        "validation_end_date": validation_end_date.isoformat(),
        "folds": [_fold_to_dict(fold) for fold in folds],
        "models": summaries,
        "counts": {
            "stored_predictions": len(stored),
            "folds": len(folds),
            "models": len(model_specs),
        },
        "predictions_path": str(predictions_path),
    }
    _write_json(report_path, report)
    return OOFRunResult(
        predictions_path=predictions_path,
        report_path=report_path,
        stored_predictions=stored,
        folds=folds,
        report=report,
    )


def oof_run_result_to_dict(result: OOFRunResult) -> dict[str, Any]:
    return {
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def _predict_fold(
    *,
    spec: OOFModelSpec,
    train_races: tuple[Race, ...],
    train_feature_rows: tuple[FeatureRow, ...],
    train_results: tuple[Result, ...],
    validation_races: tuple[Race, ...],
    validation_feature_rows: tuple[FeatureRow, ...],
    validation_odds: tuple[OddsQuote, ...],
    fold: OOFFold,
    as_of: datetime,
    feature_version: str,
    random_seed: int,
    estimator_factory: EstimatorFactory | None,
) -> tuple[ModelPrediction, ...]:
    if spec.model_key == "market":
        model = MarketImpliedProbabilityModel(model_version=spec.model_version)
        return tuple(
            model.predict(
                validation_feature_rows,
                context=InferenceContext(
                    as_of=as_of,
                    feature_version=feature_version,
                    races=validation_races,
                    odds=validation_odds,
                ),
            )
        )

    if not spec.model_key.startswith("lightgbm"):
        raise ValueError(f"Unsupported OOF model_key: {spec.model_key!r}")

    filtered_train = _filter_feature_rows(
        train_feature_rows,
        spec.exclude_feature_names,
    )
    filtered_validation = _filter_feature_rows(
        validation_feature_rows,
        spec.exclude_feature_names,
    )
    training_dataset = TrainingDataset(
        feature_rows=filtered_train,
        labels=_training_labels(filtered_train, train_results),
        feature_names=_feature_names(filtered_train),
        target=PredictionTarget.WIN_PROBABILITY,
        metadata={"results": train_results},
    )
    model = LightGBMWinProbabilityModel(
        model_version=spec.model_version,
        estimator_factory=estimator_factory,
    )
    model.fit(
        training_dataset,
        context=TrainingContext(
            train_start=fold.train_start,
            train_end=fold.train_end,
            validation_start=fold.validation_start,
            feature_version=feature_version,
            random_seed=random_seed,
        ),
    )
    return tuple(
        model.predict(
            filtered_validation,
            context=InferenceContext(
                as_of=as_of,
                feature_version=feature_version,
                races=validation_races,
                odds=validation_odds,
            ),
        )
    )


def _select_model_specs(model_keys: Sequence[str] | None) -> tuple[OOFModelSpec, ...]:
    available = {spec.model_key: spec for spec in DEFAULT_OOF_MODEL_SPECS}
    if model_keys is None:
        return DEFAULT_OOF_MODEL_SPECS
    selected: list[OOFModelSpec] = []
    for key in model_keys:
        if key not in available:
            raise ValueError(
                f"Unsupported OOF model key {key!r}; "
                f"available={tuple(available)}"
            )
        selected.append(available[key])
    return tuple(selected)


def _monthly_folds(
    races: tuple[Race, ...],
    *,
    validation_start_date: date,
    validation_end_date: date,
) -> tuple[OOFFold, ...]:
    if validation_start_date > validation_end_date:
        raise ValueError("validation_start_date must be on or before validation_end_date")
    min_race_date = min(race.race_date for race in races)
    folds: list[OOFFold] = []
    cursor = date(validation_start_date.year, validation_start_date.month, 1)
    if cursor < validation_start_date:
        cursor = validation_start_date
    while cursor <= validation_end_date:
        month_end = min(_last_day_of_month(cursor), validation_end_date)
        train_end = cursor - timedelta(days=1)
        if train_end >= min_race_date:
            folds.append(
                OOFFold(
                    fold_id=f"{cursor:%Y%m}",
                    train_start=min_race_date,
                    train_end=train_end,
                    validation_start=cursor,
                    validation_end=month_end,
                )
            )
        cursor = month_end + timedelta(days=1)
    return tuple(folds)


def _last_day_of_month(value: date) -> date:
    if value.month == 12:
        next_month = date(value.year + 1, 1, 1)
    else:
        next_month = date(value.year, value.month + 1, 1)
    return next_month - timedelta(days=1)


def _races_in_window(
    races: tuple[Race, ...],
    *,
    start_date: date,
    end_date: date,
) -> tuple[Race, ...]:
    return tuple(race for race in races if start_date <= race.race_date <= end_date)


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


def _model_summary(
    *,
    predictions: tuple[ModelPrediction, ...],
    results: tuple[Result, ...],
    spec: OOFModelSpec,
) -> dict[str, Any]:
    summary = summarize_win_probability_predictions(
        predictions=predictions,
        results=results,
    )
    return {
        "model_key": spec.model_key,
        "model_version": spec.model_version,
        "excluded_feature_names": [str(name) for name in spec.exclude_feature_names],
        "prediction_count": len(predictions),
        "probability": _probability_summary_to_dict(summary),
    }


def _probability_summary_to_dict(summary: ProbabilitySummary) -> dict[str, Any]:
    return {
        "observations": summary.observations,
        "positives": summary.positives,
        "mean_predicted_probability": summary.mean_predicted_probability,
        "empirical_rate": summary.empirical_rate,
        "log_loss": summary.log_loss,
        "brier_score": summary.brier_score,
        "expected_calibration_error": summary.expected_calibration_error,
    }


def _fold_to_dict(fold: OOFFold) -> dict[str, str]:
    return {
        "fold_id": fold.fold_id,
        "train_start": fold.train_start.isoformat(),
        "train_end": fold.train_end.isoformat(),
        "validation_start": fold.validation_start.isoformat(),
        "validation_end": fold.validation_end.isoformat(),
    }


def _require_non_empty(rows: Sequence[object], label: str) -> None:
    if not rows:
        raise ValueError(f"No {label} found")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _replay_odds_csv_path(
    dataset_path: Path,
    *,
    use_odds_timeseries: bool,
) -> Path:
    odds_path = dataset_path / "odds.csv"
    odds_timeseries_path = dataset_path / "odds_timeseries.csv"
    if use_odds_timeseries and odds_timeseries_path.exists():
        return odds_timeseries_path
    if odds_path.exists():
        return odds_path
    if odds_timeseries_path.exists():
        return odds_timeseries_path
    return odds_path
