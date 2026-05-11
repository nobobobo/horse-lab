"""Build meta-learning datasets from stored Level 0 predictions."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.data.csv_parsing import parse_result_row, read_csv_rows
from horse_lab.schemas import PredictionTarget, RaceId, Result, RunnerId
from horse_lab.stacking.prediction_store import (
    PredictionRole,
    StoredPrediction,
    read_prediction_store_csv,
)


META_DATASET_BASE_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "target",
    "label",
    "fold_id",
    "validation_start",
    "validation_end",
    "as_of",
    "feature_version",
)


@dataclass(frozen=True)
class MetaDatasetBuildResult:
    meta_dataset_path: Path
    report_path: Path
    report: dict[str, Any]


def build_meta_dataset_from_csv(
    predictions_csv: Path | str,
    results_csv: Path | str,
    output_dir: Path | str,
    *,
    prediction_role: PredictionRole | str = PredictionRole.OOF,
    target: PredictionTarget | str = PredictionTarget.WIN_PROBABILITY,
    drop_incomplete_rows: bool = True,
) -> MetaDatasetBuildResult:
    predictions = read_prediction_store_csv(predictions_csv)
    results = tuple(parse_result_row(row) for row in read_csv_rows(results_csv))
    return build_meta_dataset(
        predictions=predictions,
        results=results,
        output_dir=output_dir,
        prediction_role=prediction_role,
        target=target,
        drop_incomplete_rows=drop_incomplete_rows,
    )


def build_meta_dataset(
    *,
    predictions: Sequence[StoredPrediction],
    results: Sequence[Result],
    output_dir: Path | str,
    prediction_role: PredictionRole | str = PredictionRole.OOF,
    target: PredictionTarget | str = PredictionTarget.WIN_PROBABILITY,
    drop_incomplete_rows: bool = True,
) -> MetaDatasetBuildResult:
    role = PredictionRole(prediction_role)
    prediction_target = PredictionTarget(target)
    selected_predictions = tuple(
        prediction
        for prediction in predictions
        if prediction.prediction_role == role and prediction.target == prediction_target
    )
    result_by_runner = _result_by_runner(results)
    grouped = _group_predictions(selected_predictions)
    model_columns = _model_columns(selected_predictions)

    rows: list[dict[str, str]] = []
    dropped_missing_results = 0
    dropped_incomplete_rows = 0
    duplicate_prediction_keys = 0

    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        race_id, runner_id = key
        result = result_by_runner.get(key)
        if result is None:
            dropped_missing_results += 1
            continue

        predictions_by_model: dict[str, StoredPrediction] = {}
        for prediction in grouped[key]:
            column = model_prediction_column(prediction)
            if column in predictions_by_model:
                duplicate_prediction_keys += 1
                continue
            predictions_by_model[column] = prediction

        missing_columns = [
            column for column in model_columns if column not in predictions_by_model
        ]
        if missing_columns and drop_incomplete_rows:
            dropped_incomplete_rows += 1
            continue

        row = _base_row(
            race_id=race_id,
            runner_id=runner_id,
            result=result,
            predictions=tuple(predictions_by_model.values()),
        )
        for column in model_columns:
            prediction = predictions_by_model.get(column)
            row[column] = "" if prediction is None else str(prediction.probability)
        rows.append(row)

    output_path = Path(output_dir)
    meta_dataset_path = output_path / "meta_features.csv"
    report_path = output_path / "meta_dataset_report.json"
    _write_meta_dataset(meta_dataset_path, rows, model_columns)
    report = {
        "prediction_role": role.value,
        "target": prediction_target.value,
        "model_columns": list(model_columns),
        "counts": {
            "input_predictions": len(predictions),
            "selected_predictions": len(selected_predictions),
            "results": len(results),
            "meta_rows": len(rows),
            "dropped_missing_results": dropped_missing_results,
            "dropped_incomplete_rows": dropped_incomplete_rows,
            "duplicate_prediction_keys": duplicate_prediction_keys,
        },
        "labels": {
            "positives": sum(int(row["label"]) for row in rows),
            "empirical_rate": (
                sum(int(row["label"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        },
        "folds": sorted({row["fold_id"] for row in rows if row["fold_id"]}),
        "meta_dataset_path": str(meta_dataset_path),
    }
    _write_json(report_path, report)
    return MetaDatasetBuildResult(
        meta_dataset_path=meta_dataset_path,
        report_path=report_path,
        report=report,
    )


def meta_dataset_build_result_to_dict(
    result: MetaDatasetBuildResult,
) -> dict[str, Any]:
    return {
        "meta_dataset_path": str(result.meta_dataset_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def model_prediction_column(prediction: StoredPrediction) -> str:
    return (
        "pred__"
        + _slug(str(prediction.model_name))
        + "__"
        + _slug(prediction.model_version)
    )


def _group_predictions(
    predictions: Sequence[StoredPrediction],
) -> dict[tuple[RaceId, RunnerId], list[StoredPrediction]]:
    grouped: dict[tuple[RaceId, RunnerId], list[StoredPrediction]] = {}
    for prediction in predictions:
        grouped.setdefault((prediction.race_id, prediction.runner_id), []).append(
            prediction
        )
    return grouped


def _model_columns(predictions: Sequence[StoredPrediction]) -> tuple[str, ...]:
    return tuple(
        sorted({model_prediction_column(prediction) for prediction in predictions})
    )


def _result_by_runner(
    results: Sequence[Result],
) -> Mapping[tuple[RaceId, RunnerId], Result]:
    return {(result.race_id, result.runner_id): result for result in results}


def _base_row(
    *,
    race_id: RaceId,
    runner_id: RunnerId,
    result: Result,
    predictions: tuple[StoredPrediction, ...],
) -> dict[str, str]:
    prediction = sorted(
        predictions,
        key=lambda item: (
            item.validation_start is None,
            item.validation_start,
            str(item.model_name),
            item.model_version,
        ),
    )[0]
    return {
        "race_id": str(race_id),
        "runner_id": str(runner_id),
        "target": prediction.target.value,
        "label": "1" if result.did_win else "0",
        "fold_id": prediction.fold_id or "",
        "validation_start": (
            prediction.validation_start.isoformat()
            if prediction.validation_start is not None
            else ""
        ),
        "validation_end": (
            prediction.validation_end.isoformat()
            if prediction.validation_end is not None
            else ""
        ),
        "as_of": prediction.as_of.isoformat(),
        "feature_version": prediction.feature_version,
    }


def _write_meta_dataset(
    path: Path,
    rows: Sequence[dict[str, str]],
    model_columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(META_DATASET_BASE_FIELDS) + tuple(model_columns),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    return normalized.strip("_") or "unknown"
