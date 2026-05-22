"""LightGBM training pipeline over replay-ready historical CSV data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from horse_lab.evaluation import (
    ProbabilitySummary,
    summarize_win_probability_predictions,
)
from horse_lab.features import PAST_PERFORMANCE_FEATURE_NAMES
from horse_lab.models import (
    InferenceContext,
    LightGBMWinProbabilityModel,
    ModelArtifact,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.models.lightgbm import EstimatorFactory
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


@dataclass(frozen=True)
class LightGBMTrainingResult:
    train_races: tuple[Race, ...]
    train_feature_rows: tuple[FeatureRow, ...]
    train_results: tuple[Result, ...]
    validation_races: tuple[Race, ...]
    validation_feature_rows: tuple[FeatureRow, ...]
    validation_results: tuple[Result, ...]
    validation_odds: tuple[OddsQuote, ...]
    predictions: tuple[ModelPrediction, ...]
    probability_summary: ProbabilitySummary
    model_artifact: ModelArtifact
    model_path: Path
    summary_path: Path
    feature_importance_path: Path
    feature_importances: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LightGBMFeatureSetScenario:
    include_feature_names: tuple[FeatureName, ...] | None = None
    exclude_feature_names: tuple[FeatureName, ...] = ()


MARKET_FEATURE_NAMES: tuple[FeatureName, ...] = (
    FeatureName("entry_win_odds"),
    FeatureName("entry_popularity_rank"),
    FeatureName("odds_open"),
    FeatureName("odds_latest"),
    FeatureName("odds_min"),
    FeatureName("odds_max"),
    FeatureName("odds_snapshot_count"),
    FeatureName("odds_change_open_to_latest"),
    FeatureName("implied_probability_change_open_to_latest"),
    FeatureName("pool_size_latest_jpy"),
    FeatureName("last_odds"),
    FeatureName("avg_odds_last3"),
)

ENTRY_PROFILE_FEATURE_NAMES: tuple[FeatureName, ...] = (
    FeatureName("horse_number"),
    FeatureName("gate_number"),
    FeatureName("carried_weight_kg"),
    FeatureName("age"),
    FeatureName("sex"),
    FeatureName("horse_symbol_code"),
    FeatureName("breed_code"),
    FeatureName("coat_color_code"),
    FeatureName("trainer_affiliation_code"),
    FeatureName("body_weight_kg"),
    FeatureName("body_weight_diff_kg"),
)

RACE_CONDITION_FEATURE_NAMES: tuple[FeatureName, ...] = (
    FeatureName("race_venue"),
    FeatureName("race_surface"),
    FeatureName("race_distance_m"),
    FeatureName("race_direction"),
    FeatureName("race_track_condition"),
    FeatureName("race_weather"),
    FeatureName("race_grade"),
    FeatureName("race_grade_group"),
    FeatureName("race_title_type"),
    FeatureName("race_has_title"),
    FeatureName("race_field_size"),
)

PEDIGREE_RATING_FEATURE_NAMES: tuple[FeatureName, ...] = (
    FeatureName("pedigree_sire_id"),
    FeatureName("pedigree_dam_id"),
    FeatureName("pedigree_damsire_id"),
    FeatureName("horse_birth_year"),
    FeatureName("horse_age_days_from_birth"),
    FeatureName("horse_rating"),
    FeatureName("horse_rating_delta_to_field_mean"),
    FeatureName("horse_rating_rank_in_race"),
    FeatureName("horse_rating_source"),
)

PERSON_ID_FEATURE_NAMES: tuple[FeatureName, ...] = (
    FeatureName("jockey_id"),
    FeatureName("trainer_id"),
)

CORE_FORM_FEATURE_NAMES: tuple[FeatureName, ...] = tuple(
    name for name in PAST_PERFORMANCE_FEATURE_NAMES if name not in MARKET_FEATURE_NAMES
)


def _unique_feature_names(feature_names: Sequence[FeatureName]) -> tuple[FeatureName, ...]:
    return tuple(sorted(set(feature_names), key=str))


ODDS_MOVEMENT_FEATURE_NAMES: tuple[FeatureName, ...] = (
    FeatureName("odds_open"),
    FeatureName("odds_latest"),
    FeatureName("odds_min"),
    FeatureName("odds_max"),
    FeatureName("odds_snapshot_count"),
    FeatureName("odds_change_open_to_latest"),
    FeatureName("implied_probability_change_open_to_latest"),
    FeatureName("pool_size_latest_jpy"),
)

DEFAULT_LIGHTGBM_ABLATION_SCENARIOS: dict[str, tuple[FeatureName, ...]] = {
    "full": (),
    "no_market": MARKET_FEATURE_NAMES,
    "no_movement": ODDS_MOVEMENT_FEATURE_NAMES,
}

DEFAULT_LIGHTGBM_FEATURE_SET_SCENARIOS: dict[str, LightGBMFeatureSetScenario] = {
    "full": LightGBMFeatureSetScenario(),
    "market_only": LightGBMFeatureSetScenario(
        include_feature_names=MARKET_FEATURE_NAMES,
    ),
    "no_market_selected": LightGBMFeatureSetScenario(
        include_feature_names=_unique_feature_names(
            (
                *ENTRY_PROFILE_FEATURE_NAMES,
                *RACE_CONDITION_FEATURE_NAMES,
                *PERSON_ID_FEATURE_NAMES,
                *CORE_FORM_FEATURE_NAMES,
            )
        ),
    ),
    "profile_pedigree_rating": LightGBMFeatureSetScenario(
        include_feature_names=_unique_feature_names(
            (
                *ENTRY_PROFILE_FEATURE_NAMES,
                *RACE_CONDITION_FEATURE_NAMES,
                *PERSON_ID_FEATURE_NAMES,
                *CORE_FORM_FEATURE_NAMES,
                *PEDIGREE_RATING_FEATURE_NAMES,
            )
        ),
    ),
}


def run_lightgbm_training_from_csv(
    dataset_dir: Path | str,
    artifact_dir: Path | str,
    *,
    train_end_date: date,
    valid_start_date: date,
    valid_end_date: date,
    as_of: datetime,
    feature_version: str,
    random_seed: int = 42,
    model_version: str = "lightgbm-win-v1",
    estimator_factory: EstimatorFactory | None = None,
    include_feature_names: Sequence[str | FeatureName] | None = None,
    exclude_feature_names: Sequence[str | FeatureName] = (),
) -> LightGBMTrainingResult:
    """Train and validate the LightGBM baseline from replay-ready CSV files."""

    dataset_path = Path(dataset_dir)
    return run_lightgbm_training(
        race_repository=CsvRaceRepository(dataset_path / "races.csv"),
        odds_repository=CsvOddsRepository(_replay_odds_csv_path(dataset_path)),
        result_repository=CsvResultRepository(dataset_path / "results.csv"),
        feature_repository=CsvFeatureRepository(dataset_path / "features.csv"),
        artifact_dir=artifact_dir,
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
        valid_end_date=valid_end_date,
        as_of=as_of,
        feature_version=feature_version,
        random_seed=random_seed,
        model_version=model_version,
        estimator_factory=estimator_factory,
        include_feature_names=include_feature_names,
        exclude_feature_names=exclude_feature_names,
    )


def run_lightgbm_ablation_from_csv(
    dataset_dir: Path | str,
    artifact_dir: Path | str,
    *,
    train_end_date: date,
    valid_start_date: date,
    valid_end_date: date,
    as_of: datetime,
    feature_version: str,
    random_seed: int = 42,
    model_version: str = "lightgbm-win-v1",
    scenarios: Mapping[str, Sequence[str | FeatureName]] | None = None,
    estimator_factory: EstimatorFactory | None = None,
) -> dict[str, Any]:
    """Run focused LightGBM feature ablations and write an aggregate summary."""

    selected_scenarios = scenarios or DEFAULT_LIGHTGBM_ABLATION_SCENARIOS
    artifact_path = Path(artifact_dir)
    scenario_summaries: dict[str, Any] = {}
    for scenario_name, excluded_features in selected_scenarios.items():
        result = run_lightgbm_training_from_csv(
            dataset_dir,
            artifact_path / scenario_name,
            train_end_date=train_end_date,
            valid_start_date=valid_start_date,
            valid_end_date=valid_end_date,
            as_of=as_of,
            feature_version=feature_version,
            random_seed=random_seed,
            model_version=f"{model_version}-{scenario_name}",
            estimator_factory=estimator_factory,
            exclude_feature_names=excluded_features,
        )
        summary = lightgbm_training_result_to_dict(result)
        summary["excluded_feature_names"] = [
            str(name) for name in _normalize_feature_names(excluded_features)
        ]
        scenario_summaries[scenario_name] = summary

    best_by_log_loss = min(
        scenario_summaries,
        key=lambda name: scenario_summaries[name]["probability"]["log_loss"],
    )
    aggregate = {
        "dataset_dir": str(dataset_dir),
        "artifact_dir": str(artifact_path),
        "split": {
            "train_end_date": train_end_date.isoformat(),
            "valid_start_date": valid_start_date.isoformat(),
            "valid_end_date": valid_end_date.isoformat(),
            "as_of": as_of.isoformat(),
        },
        "feature_version": feature_version,
        "scenarios": scenario_summaries,
        "best_by_log_loss": best_by_log_loss,
    }
    _write_json_file(artifact_path / "ablation_summary.json", aggregate)
    return aggregate


def run_lightgbm_feature_set_study_from_csv(
    dataset_dir: Path | str,
    artifact_dir: Path | str,
    *,
    train_end_date: date,
    valid_start_date: date,
    valid_end_date: date,
    as_of: datetime,
    feature_version: str,
    random_seed: int = 42,
    model_version: str = "lightgbm-win-v1",
    scenarios: Mapping[str, LightGBMFeatureSetScenario] | None = None,
    estimator_factory: EstimatorFactory | None = None,
) -> dict[str, Any]:
    """Run named include/exclude feature-set studies for specialist models."""

    selected_scenarios = scenarios or DEFAULT_LIGHTGBM_FEATURE_SET_SCENARIOS
    artifact_path = Path(artifact_dir)
    scenario_summaries: dict[str, Any] = {}
    for scenario_name, scenario in selected_scenarios.items():
        result = run_lightgbm_training_from_csv(
            dataset_dir,
            artifact_path / scenario_name,
            train_end_date=train_end_date,
            valid_start_date=valid_start_date,
            valid_end_date=valid_end_date,
            as_of=as_of,
            feature_version=feature_version,
            random_seed=random_seed,
            model_version=f"{model_version}-{scenario_name}",
            estimator_factory=estimator_factory,
            include_feature_names=scenario.include_feature_names,
            exclude_feature_names=scenario.exclude_feature_names,
        )
        summary = lightgbm_training_result_to_dict(result)
        summary["included_feature_names"] = (
            [
                str(name)
                for name in _normalize_optional_feature_names(
                    scenario.include_feature_names
                )
            ]
            if scenario.include_feature_names is not None
            else None
        )
        summary["excluded_feature_names"] = [
            str(name) for name in _normalize_feature_names(scenario.exclude_feature_names)
        ]
        scenario_summaries[scenario_name] = summary

    best_by_log_loss = min(
        scenario_summaries,
        key=lambda name: scenario_summaries[name]["probability"]["log_loss"],
    )
    aggregate = {
        "dataset_dir": str(dataset_dir),
        "artifact_dir": str(artifact_path),
        "split": {
            "train_end_date": train_end_date.isoformat(),
            "valid_start_date": valid_start_date.isoformat(),
            "valid_end_date": valid_end_date.isoformat(),
            "as_of": as_of.isoformat(),
        },
        "feature_version": feature_version,
        "scenarios": scenario_summaries,
        "best_by_log_loss": best_by_log_loss,
    }
    _write_json_file(artifact_path / "feature_set_study_summary.json", aggregate)
    return aggregate


def _replay_odds_csv_path(dataset_path: Path) -> Path:
    odds_timeseries_path = dataset_path / "odds_timeseries.csv"
    if odds_timeseries_path.exists():
        return odds_timeseries_path
    return dataset_path / "odds.csv"


def run_lightgbm_training(
    *,
    race_repository: RaceRepository,
    odds_repository: OddsRepository,
    result_repository: ResultRepository,
    feature_repository: FeatureRepository,
    artifact_dir: Path | str,
    train_end_date: date,
    valid_start_date: date,
    valid_end_date: date,
    as_of: datetime,
    feature_version: str,
    random_seed: int = 42,
    model_version: str = "lightgbm-win-v1",
    estimator_factory: EstimatorFactory | None = None,
    include_feature_names: Sequence[str | FeatureName] | None = None,
    exclude_feature_names: Sequence[str | FeatureName] = (),
) -> LightGBMTrainingResult:
    """Train on races up to ``train_end_date`` and validate on a later window."""

    _validate_time_split(
        train_end_date=train_end_date,
        valid_start_date=valid_start_date,
        valid_end_date=valid_end_date,
    )

    train_races = tuple(
        race_repository.list_races(
            start_date=date.min,
            end_date=train_end_date,
        )
    )
    validation_races = tuple(
        race_repository.list_races(
            start_date=valid_start_date,
            end_date=valid_end_date,
        )
    )
    _require_non_empty(train_races, "training races")
    _require_non_empty(validation_races, "validation races")

    train_race_ids = tuple(race.race_id for race in train_races)
    validation_race_ids = tuple(race.race_id for race in validation_races)
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
        label="training",
    )
    _validate_rows_have_results(
        feature_rows=validation_feature_rows,
        results=validation_results,
        label="validation",
    )

    included = _normalize_optional_feature_names(include_feature_names)
    excluded = _normalize_feature_names(exclude_feature_names)
    if included is not None:
        train_feature_rows = _include_feature_rows(train_feature_rows, included)
        validation_feature_rows = _include_feature_rows(
            validation_feature_rows,
            included,
        )
    if excluded:
        train_feature_rows = _exclude_feature_rows(train_feature_rows, excluded)
        validation_feature_rows = _exclude_feature_rows(validation_feature_rows, excluded)

    training_dataset = TrainingDataset(
        feature_rows=train_feature_rows,
        labels=_training_labels(train_feature_rows, train_results),
        feature_names=_feature_names(train_feature_rows),
        target=PredictionTarget.WIN_PROBABILITY,
        metadata={"results": train_results},
    )
    train_start = min(race.race_date for race in train_races)
    train_context = TrainingContext(
        train_start=train_start,
        train_end=train_end_date,
        validation_start=valid_start_date,
        feature_version=feature_version,
        random_seed=random_seed,
    )

    model = LightGBMWinProbabilityModel(
        model_version=model_version,
        estimator_factory=estimator_factory,
    )
    model_artifact = model.fit(training_dataset, context=train_context)
    predictions = tuple(
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
    probability_summary = summarize_win_probability_predictions(
        predictions=predictions,
        results=validation_results,
    )

    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "model"
    summary_path = artifact_path / "evaluation_summary.json"
    feature_importance_path = artifact_path / "feature_importance.csv"
    feature_importances = model.feature_importances()
    model.save(model_path)

    result = LightGBMTrainingResult(
        train_races=train_races,
        train_feature_rows=train_feature_rows,
        train_results=train_results,
        validation_races=validation_races,
        validation_feature_rows=validation_feature_rows,
        validation_results=validation_results,
        validation_odds=validation_odds,
        predictions=predictions,
        probability_summary=probability_summary,
        model_artifact=model_artifact,
        model_path=model_path,
        summary_path=summary_path,
        feature_importance_path=feature_importance_path,
        feature_importances=feature_importances,
    )
    _write_feature_importance_csv(feature_importance_path, feature_importances)
    _write_json_file(summary_path, lightgbm_training_result_to_dict(result))
    return result


def lightgbm_training_result_to_dict(
    result: LightGBMTrainingResult,
) -> dict[str, Any]:
    """Serialize a training result for CLI output and artifact reports."""

    metadata = _jsonable_mapping(result.model_artifact.metadata)
    numeric_feature_count = len(metadata.get("numeric_features", ()))
    categorical_feature_count = len(metadata.get("categorical_features", ()))

    return {
        "artifact": {
            "model_name": str(result.model_artifact.model_name),
            "model_version": result.model_artifact.model_version,
            "trained_at": result.model_artifact.trained_at.isoformat(),
            "feature_version": result.model_artifact.feature_version,
            "target": result.model_artifact.target.value,
            "model_path": str(result.model_path),
            "summary_path": str(result.summary_path),
            "feature_importance_path": str(result.feature_importance_path),
            "metadata": metadata,
        },
        "counts": {
            "train_races": len(result.train_races),
            "train_feature_rows": len(result.train_feature_rows),
            "train_results": len(result.train_results),
            "validation_races": len(result.validation_races),
            "validation_feature_rows": len(result.validation_feature_rows),
            "validation_results": len(result.validation_results),
            "validation_odds": len(result.validation_odds),
            "predictions": len(result.predictions),
            "model_features": numeric_feature_count + categorical_feature_count,
            "numeric_features": numeric_feature_count,
            "categorical_features": categorical_feature_count,
        },
        "probability": _probability_summary_to_dict(result.probability_summary),
        "feature_importances": [
            _jsonable_mapping(row) for row in result.feature_importances
        ],
    }


def _validate_time_split(
    *,
    train_end_date: date,
    valid_start_date: date,
    valid_end_date: date,
) -> None:
    if valid_start_date > valid_end_date:
        raise ValueError("valid_start_date must be on or before valid_end_date")
    if train_end_date >= valid_start_date:
        raise ValueError("train_end_date must be before valid_start_date")


def _require_non_empty(rows: tuple[object, ...], label: str) -> None:
    if not rows:
        raise ValueError(f"No {label} found for the requested date range")


def _training_labels(
    feature_rows: tuple[FeatureRow, ...],
    results: tuple[Result, ...],
) -> tuple[TrainingLabel, ...]:
    result_by_runner = _result_by_runner(results)
    labels: list[TrainingLabel] = []
    for row in feature_rows:
        result = result_by_runner[(row.race_id, row.runner_id)]
        labels.append(
            TrainingLabel(
                race_id=row.race_id,
                runner_id=row.runner_id,
                target=PredictionTarget.WIN_PROBABILITY,
                value=1.0 if result.did_win else 0.0,
            )
        )
    return tuple(labels)


def _feature_names(feature_rows: tuple[FeatureRow, ...]) -> tuple[FeatureName, ...]:
    names = {
        feature_name
        for row in feature_rows
        for feature_name in row.values.keys()
    }
    if not names:
        raise ValueError("Training feature rows must contain at least one feature")
    return tuple(sorted(names, key=str))


def _normalize_feature_names(
    feature_names: Sequence[str | FeatureName],
) -> tuple[FeatureName, ...]:
    return tuple(sorted({FeatureName(str(name)) for name in feature_names}, key=str))


def _normalize_optional_feature_names(
    feature_names: Sequence[str | FeatureName] | None,
) -> tuple[FeatureName, ...] | None:
    if feature_names is None:
        return None
    return _normalize_feature_names(feature_names)


def _include_feature_rows(
    feature_rows: tuple[FeatureRow, ...],
    included_feature_names: Sequence[FeatureName],
) -> tuple[FeatureRow, ...]:
    included = set(included_feature_names)
    return tuple(
        FeatureRow(
            race_id=row.race_id,
            runner_id=row.runner_id,
            as_of=row.as_of,
            feature_version=row.feature_version,
            values={
                feature_name: value
                for feature_name, value in row.values.items()
                if feature_name in included
            },
            metadata=row.metadata,
        )
        for row in feature_rows
    )


def _exclude_feature_rows(
    feature_rows: tuple[FeatureRow, ...],
    excluded_feature_names: Sequence[FeatureName],
) -> tuple[FeatureRow, ...]:
    excluded = set(excluded_feature_names)
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
        if (row.race_id, row.runner_id) not in result_by_runner
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


def _probability_summary_to_dict(summary: ProbabilitySummary) -> dict[str, Any]:
    return {
        "observations": summary.observations,
        "positives": summary.positives,
        "mean_predicted_probability": summary.mean_predicted_probability,
        "empirical_rate": summary.empirical_rate,
        "log_loss": summary.log_loss,
        "brier_score": summary.brier_score,
        "expected_calibration_error": summary.expected_calibration_error,
        "bins": [
            {
                "lower_bound": calibration_bin.lower_bound,
                "upper_bound": calibration_bin.upper_bound,
                "count": calibration_bin.count,
                "positives": calibration_bin.positives,
                "mean_predicted_probability": calibration_bin.mean_predicted_probability,
                "empirical_rate": calibration_bin.empirical_rate,
                "absolute_error": calibration_bin.absolute_error,
            }
            for calibration_bin in summary.bins
        ],
    }


def _jsonable_mapping(mapping: Any) -> dict[str, Any]:
    return {
        str(key): _jsonable_value(value)
        for key, value in dict(mapping).items()
    }


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return value


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_feature_importance_csv(
    path: Path,
    feature_importances: tuple[dict[str, Any], ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "feature_name",
        "feature_type",
        "split_importance",
        "gain_importance",
        "split_fraction",
        "gain_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_importances:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
