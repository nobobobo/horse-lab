"""Segment-specific calibration studies for market probabilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.data.csv_parsing import parse_race_row, read_csv_rows
from horse_lab.schemas import Race
from horse_lab.stacking.calibration import (
    MarketCalibrationModel,
    _fit_model,
    _infer_market_column,
    _normalize_by_race,
    _predict_raw,
    _probability_summary,
)
from horse_lab.stacking.meta_learner import MetaFeatureRow, read_meta_feature_rows


SEGMENT_CALIBRATION_PREDICTION_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "fold_id",
    "label",
    "method",
    "segment_name",
    "segment_value",
    "model_scope",
    "market_probability",
    "raw_probability",
    "probability",
)


@dataclass(frozen=True)
class SegmentCalibrationStudyResult:
    predictions_path: Path
    report_path: Path
    report: dict[str, Any]


def run_segment_calibration_study_from_csv(
    meta_features_csv: Path | str,
    artifact_dir: Path | str,
    *,
    races_csv: Path | str | None = None,
    market_column: str | None = None,
    segment_name: str = "market_probability_band",
    min_train_folds: int = 3,
    min_segment_train_rows: int = 500,
    model_version: str = "segment-calibration-v1",
    learning_rate: float = 0.05,
    max_iterations: int = 2000,
    l2: float = 1e-3,
) -> SegmentCalibrationStudyResult:
    rows = read_meta_feature_rows(meta_features_csv)
    races = (
        tuple(parse_race_row(row) for row in read_csv_rows(races_csv))
        if races_csv is not None
        else ()
    )
    return run_segment_calibration_study(
        rows=rows,
        races=races,
        artifact_dir=artifact_dir,
        market_column=market_column,
        segment_name=segment_name,
        min_train_folds=min_train_folds,
        min_segment_train_rows=min_segment_train_rows,
        model_version=model_version,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        l2=l2,
    )


def run_segment_calibration_study(
    *,
    rows: Sequence[MetaFeatureRow],
    races: Sequence[Race] = (),
    artifact_dir: Path | str,
    market_column: str | None = None,
    segment_name: str = "market_probability_band",
    min_train_folds: int = 3,
    min_segment_train_rows: int = 500,
    model_version: str = "segment-calibration-v1",
    learning_rate: float = 0.05,
    max_iterations: int = 2000,
    l2: float = 1e-3,
) -> SegmentCalibrationStudyResult:
    if min_train_folds <= 0:
        raise ValueError("min_train_folds must be positive")
    if min_segment_train_rows <= 0:
        raise ValueError("min_segment_train_rows must be positive")
    if segment_name not in _SUPPORTED_SEGMENTS:
        raise ValueError(f"unsupported segment_name={segment_name!r}")
    if segment_name != "market_probability_band" and not races:
        raise ValueError(f"races are required for segment_name={segment_name!r}")
    if not rows:
        raise ValueError("meta feature rows must not be empty")

    selected_market_column = market_column or _infer_market_column(rows)
    folds = tuple(sorted({row.fold_id for row in rows if row.fold_id}))
    if len(folds) <= min_train_folds:
        raise ValueError("not enough folds for segment calibration study")

    race_by_id = {str(race.race_id): race for race in races}
    prediction_rows: list[dict[str, str]] = []
    fold_reports: list[dict[str, Any]] = []
    all_calibrated: list[float] = []
    all_market: list[float] = []
    all_labels: list[int] = []

    for holdout_index in range(min_train_folds, len(folds)):
        holdout_fold_id = folds[holdout_index]
        train_fold_ids = folds[:holdout_index]
        train_rows = tuple(row for row in rows if row.fold_id in train_fold_ids)
        holdout_rows = tuple(row for row in rows if row.fold_id == holdout_fold_id)
        if not train_rows or not holdout_rows:
            continue

        global_model, global_train_loss = _fit_model(
            train_rows,
            market_column=selected_market_column,
            model_version=model_version,
            train_fold_ids=train_fold_ids,
            holdout_fold_id=holdout_fold_id,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            l2=l2,
        )
        segment_models = _fit_segment_models(
            train_rows,
            race_by_id=race_by_id,
            segment_name=segment_name,
            market_column=selected_market_column,
            model_version=model_version,
            train_fold_ids=train_fold_ids,
            holdout_fold_id=holdout_fold_id,
            min_segment_train_rows=min_segment_train_rows,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            l2=l2,
        )
        raw_probabilities: list[float] = []
        model_scopes: list[str] = []
        segment_values: list[str] = []
        for row in holdout_rows:
            segment_value = _segment_value(
                row,
                race_by_id=race_by_id,
                segment_name=segment_name,
                market_column=selected_market_column,
            )
            model = segment_models.get(segment_value)
            if model is None:
                model = global_model
                model_scope = "global_fallback"
            else:
                model_scope = "segment"
            raw_probabilities.append(
                _predict_raw(model, row.feature_values[selected_market_column])
            )
            model_scopes.append(model_scope)
            segment_values.append(segment_value)

        calibrated = _normalize_by_race(holdout_rows, tuple(raw_probabilities))
        market = _normalize_by_race(
            holdout_rows,
            tuple(row.feature_values[selected_market_column] for row in holdout_rows),
        )
        labels = tuple(row.label for row in holdout_rows)

        all_calibrated.extend(calibrated)
        all_market.extend(market)
        all_labels.extend(labels)
        prediction_rows.extend(
            _prediction_rows(
                holdout_rows,
                market_column=selected_market_column,
                raw_probabilities=raw_probabilities,
                calibrated_probabilities=calibrated,
                segment_name=segment_name,
                segment_values=segment_values,
                model_scopes=model_scopes,
            )
        )
        fold_reports.append(
            {
                "holdout_fold_id": holdout_fold_id,
                "train_fold_ids": list(train_fold_ids),
                "counts": {
                    "train_rows": len(train_rows),
                    "holdout_rows": len(holdout_rows),
                    "segment_models": len(segment_models),
                    "global_fallback_holdout_rows": sum(
                        1 for scope in model_scopes if scope == "global_fallback"
                    ),
                },
                "global_model": {
                    "intercept": global_model.intercept,
                    "slope": global_model.slope,
                    "train_raw_log_loss": global_train_loss,
                },
                "segment_models": {
                    value: {
                        "intercept": model.intercept,
                        "slope": model.slope,
                    }
                    for value, model in sorted(segment_models.items())
                },
                "segment_calibrated_probability": _probability_summary(
                    calibrated,
                    labels,
                ),
                "market_baseline": _probability_summary(market, labels),
            }
        )

    if not fold_reports:
        raise ValueError("no segment calibration folds were evaluated")

    calibrated_summary = _probability_summary(all_calibrated, all_labels)
    market_summary = _probability_summary(all_market, all_labels)
    improvement = market_summary["log_loss"] - calibrated_summary["log_loss"]

    output_path = Path(artifact_dir)
    predictions_path = output_path / "segment_calibrated_predictions.csv"
    report_path = output_path / "segment_calibration_report.json"
    _write_predictions(predictions_path, prediction_rows)
    report = {
        "model_version": model_version,
        "market_column": selected_market_column,
        "segment_name": segment_name,
        "fold_ids": list(folds),
        "holdout_fold_ids": [fold["holdout_fold_id"] for fold in fold_reports],
        "counts": {
            "rows": len(rows),
            "holdout_rows": len(all_labels),
            "evaluated_folds": len(fold_reports),
        },
        "config": {
            "min_train_folds": min_train_folds,
            "min_segment_train_rows": min_segment_train_rows,
            "learning_rate": learning_rate,
            "max_iterations": max_iterations,
            "l2": l2,
        },
        "overall": {
            "segment_calibrated": calibrated_summary,
            "market_baseline": market_summary,
        },
        "recommendation": {
            "action": (
                "promote_segment_calibration_candidate"
                if improvement > 0.0
                else "keep_market_baseline"
            ),
            "log_loss_improvement_vs_market": improvement,
        },
        "folds_detail": fold_reports,
        "predictions_path": str(predictions_path),
    }
    _write_json(report_path, report)
    return SegmentCalibrationStudyResult(
        predictions_path=predictions_path,
        report_path=report_path,
        report=report,
    )


def segment_calibration_study_result_to_dict(
    result: SegmentCalibrationStudyResult,
) -> dict[str, Any]:
    return {
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def _fit_segment_models(
    rows: Sequence[MetaFeatureRow],
    *,
    race_by_id: Mapping[str, Race],
    segment_name: str,
    market_column: str,
    model_version: str,
    train_fold_ids: tuple[str, ...],
    holdout_fold_id: str,
    min_segment_train_rows: int,
    learning_rate: float,
    max_iterations: int,
    l2: float,
) -> dict[str, MarketCalibrationModel]:
    grouped: dict[str, list[MetaFeatureRow]] = {}
    for row in rows:
        segment_value = _segment_value(
            row,
            race_by_id=race_by_id,
            segment_name=segment_name,
            market_column=market_column,
        )
        grouped.setdefault(segment_value, []).append(row)

    models: dict[str, MarketCalibrationModel] = {}
    for segment_value, segment_rows in grouped.items():
        labels = {row.label for row in segment_rows}
        if len(segment_rows) < min_segment_train_rows or labels != {0, 1}:
            continue
        model, _ = _fit_model(
            tuple(segment_rows),
            market_column=market_column,
            model_version=f"{model_version}-{segment_value}",
            train_fold_ids=train_fold_ids,
            holdout_fold_id=holdout_fold_id,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            l2=l2,
        )
        models[segment_value] = model
    return models


def _prediction_rows(
    rows: Sequence[MetaFeatureRow],
    *,
    market_column: str,
    raw_probabilities: Sequence[float],
    calibrated_probabilities: Sequence[float],
    segment_name: str,
    segment_values: Sequence[str],
    model_scopes: Sequence[str],
) -> list[dict[str, str]]:
    return [
        {
            "race_id": row.race_id,
            "runner_id": row.runner_id,
            "fold_id": row.fold_id,
            "label": str(row.label),
            "method": "segment_calibrated",
            "segment_name": segment_name,
            "segment_value": segment_value,
            "model_scope": model_scope,
            "market_probability": str(row.feature_values[market_column]),
            "raw_probability": str(raw_probability),
            "probability": str(probability),
        }
        for row, raw_probability, probability, segment_value, model_scope in zip(
            rows,
            raw_probabilities,
            calibrated_probabilities,
            segment_values,
            model_scopes,
        )
    ]


def _segment_value(
    row: MetaFeatureRow,
    *,
    race_by_id: Mapping[str, Race],
    segment_name: str,
    market_column: str,
) -> str:
    market_probability = row.feature_values[market_column]
    if segment_name == "market_probability_band":
        return _market_probability_band(market_probability)

    race = race_by_id.get(row.race_id)
    if race is None:
        return "unknown"
    if segment_name == "venue":
        return race.venue
    if segment_name == "surface":
        return race.surface.value
    if segment_name == "distance_bucket":
        return _distance_bucket(race.distance_m)
    if segment_name == "field_size_bucket":
        return _field_size_bucket(race.field_size)
    raise ValueError(f"unsupported segment_name={segment_name!r}")


def _market_probability_band(probability: float) -> str:
    if probability < 0.03:
        return "lt_003"
    if probability < 0.05:
        return "003_005"
    if probability < 0.10:
        return "005_010"
    if probability < 0.20:
        return "010_020"
    return "gte_020"


def _distance_bucket(distance_m: int) -> str:
    if distance_m <= 1200:
        return "lte_1200"
    if distance_m <= 1600:
        return "1201_1600"
    if distance_m <= 2000:
        return "1601_2000"
    return "gt_2000"


def _field_size_bucket(field_size: int | None) -> str:
    if field_size is None:
        return "unknown"
    if field_size <= 10:
        return "lte_10"
    if field_size <= 14:
        return "11_14"
    return "gte_15"


def _write_predictions(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SEGMENT_CALIBRATION_PREDICTION_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


_SUPPORTED_SEGMENTS = {
    "market_probability_band",
    "venue",
    "surface",
    "distance_bucket",
    "field_size_bucket",
}
