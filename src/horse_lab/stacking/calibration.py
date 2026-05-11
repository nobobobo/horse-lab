"""Market-probability calibration studies for Level 0 ensembles."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from horse_lab.stacking.meta_learner import MetaFeatureRow, read_meta_feature_rows


CALIBRATED_PREDICTION_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "fold_id",
    "label",
    "method",
    "market_probability",
    "raw_probability",
    "probability",
)


@dataclass(frozen=True)
class MarketCalibrationModel:
    model_version: str
    market_column: str
    intercept: float
    slope: float
    train_fold_ids: tuple[str, ...]
    holdout_fold_id: str
    learning_rate: float
    max_iterations: int
    l2: float


@dataclass(frozen=True)
class MarketCalibrationStudyResult:
    predictions_path: Path
    report_path: Path
    report: dict[str, Any]


def run_market_calibration_study_from_csv(
    meta_features_csv: Path | str,
    artifact_dir: Path | str,
    *,
    market_column: str | None = None,
    min_train_folds: int = 3,
    model_version: str = "market-calibration-v1",
    learning_rate: float = 0.05,
    max_iterations: int = 2000,
    l2: float = 1e-3,
) -> MarketCalibrationStudyResult:
    rows = read_meta_feature_rows(meta_features_csv)
    return run_market_calibration_study(
        rows=rows,
        artifact_dir=artifact_dir,
        market_column=market_column,
        min_train_folds=min_train_folds,
        model_version=model_version,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        l2=l2,
    )


def run_market_calibration_study(
    *,
    rows: Sequence[MetaFeatureRow],
    artifact_dir: Path | str,
    market_column: str | None = None,
    min_train_folds: int = 3,
    model_version: str = "market-calibration-v1",
    learning_rate: float = 0.05,
    max_iterations: int = 2000,
    l2: float = 1e-3,
) -> MarketCalibrationStudyResult:
    _validate_config(
        min_train_folds=min_train_folds,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        l2=l2,
    )
    if not rows:
        raise ValueError("meta feature rows must not be empty")

    selected_market_column = market_column or _infer_market_column(rows)
    folds = tuple(sorted({row.fold_id for row in rows if row.fold_id}))
    if len(folds) <= min_train_folds:
        raise ValueError("not enough folds for market calibration study")

    fold_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, str]] = []
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

        model, train_raw_loss = _fit_model(
            train_rows,
            market_column=selected_market_column,
            model_version=model_version,
            train_fold_ids=train_fold_ids,
            holdout_fold_id=holdout_fold_id,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            l2=l2,
        )
        holdout_raw = tuple(
            _predict_raw(model, row.feature_values[selected_market_column])
            for row in holdout_rows
        )
        holdout_calibrated = _normalize_by_race(holdout_rows, holdout_raw)
        holdout_market = _normalize_by_race(
            holdout_rows,
            tuple(row.feature_values[selected_market_column] for row in holdout_rows),
        )
        labels = tuple(row.label for row in holdout_rows)

        all_calibrated.extend(holdout_calibrated)
        all_market.extend(holdout_market)
        all_labels.extend(labels)
        prediction_rows.extend(
            _prediction_rows(
                holdout_rows,
                market_column=selected_market_column,
                raw_probabilities=holdout_raw,
                calibrated_probabilities=holdout_calibrated,
            )
        )
        fold_reports.append(
            {
                "holdout_fold_id": holdout_fold_id,
                "train_fold_ids": list(train_fold_ids),
                "model": {
                    "intercept": model.intercept,
                    "slope": model.slope,
                    "train_raw_log_loss": train_raw_loss,
                },
                "counts": {
                    "train_rows": len(train_rows),
                    "holdout_rows": len(holdout_rows),
                },
                "calibrated_probability": _probability_summary(
                    holdout_calibrated,
                    labels,
                ),
                "market_baseline": _probability_summary(holdout_market, labels),
            }
        )

    if not fold_reports:
        raise ValueError("no calibration folds were evaluated")

    calibrated_summary = _probability_summary(all_calibrated, all_labels)
    market_summary = _probability_summary(all_market, all_labels)
    improvement = market_summary["log_loss"] - calibrated_summary["log_loss"]

    output_path = Path(artifact_dir)
    predictions_path = output_path / "market_calibrated_predictions.csv"
    report_path = output_path / "market_calibration_report.json"
    _write_predictions(predictions_path, prediction_rows)
    report = {
        "model_version": model_version,
        "market_column": selected_market_column,
        "fold_ids": list(folds),
        "holdout_fold_ids": [fold["holdout_fold_id"] for fold in fold_reports],
        "counts": {
            "rows": len(rows),
            "holdout_rows": len(all_labels),
            "evaluated_folds": len(fold_reports),
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "max_iterations": max_iterations,
            "l2": l2,
        },
        "overall": {
            "market_calibrated": calibrated_summary,
            "market_baseline": market_summary,
        },
        "recommendation": {
            "action": (
                "promote_calibration_candidate"
                if improvement > 0.0
                else "keep_market_baseline"
            ),
            "log_loss_improvement": improvement,
        },
        "folds_detail": fold_reports,
        "predictions_path": str(predictions_path),
    }
    _write_json(report_path, report)
    return MarketCalibrationStudyResult(
        predictions_path=predictions_path,
        report_path=report_path,
        report=report,
    )


def market_calibration_study_result_to_dict(
    result: MarketCalibrationStudyResult,
) -> dict[str, Any]:
    return {
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def _fit_model(
    rows: Sequence[MetaFeatureRow],
    *,
    market_column: str,
    model_version: str,
    train_fold_ids: tuple[str, ...],
    holdout_fold_id: str,
    learning_rate: float,
    max_iterations: int,
    l2: float,
) -> tuple[MarketCalibrationModel, float]:
    intercept = 0.0
    slope = 1.0
    labels = tuple(row.label for row in rows)
    logits = tuple(_logit(_clip_probability(row.feature_values[market_column])) for row in rows)
    n_rows = len(rows)

    for _ in range(max_iterations):
        intercept_gradient = 0.0
        slope_gradient = 0.0
        for value, label in zip(logits, labels):
            probability = _sigmoid(intercept + slope * value)
            error = probability - label
            intercept_gradient += error
            slope_gradient += error * value
        intercept -= learning_rate * (intercept_gradient / n_rows)
        slope -= learning_rate * ((slope_gradient / n_rows) + l2 * (slope - 1.0))

    train_raw = tuple(_sigmoid(intercept + slope * value) for value in logits)
    return (
        MarketCalibrationModel(
            model_version=model_version,
            market_column=market_column,
            intercept=intercept,
            slope=slope,
            train_fold_ids=train_fold_ids,
            holdout_fold_id=holdout_fold_id,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            l2=l2,
        ),
        _binary_log_loss(train_raw, labels),
    )


def _predict_raw(model: MarketCalibrationModel, market_probability: float) -> float:
    return _sigmoid(
        model.intercept + model.slope * _logit(_clip_probability(market_probability))
    )


def _prediction_rows(
    rows: Sequence[MetaFeatureRow],
    *,
    market_column: str,
    raw_probabilities: Sequence[float],
    calibrated_probabilities: Sequence[float],
) -> list[dict[str, str]]:
    return [
        {
            "race_id": row.race_id,
            "runner_id": row.runner_id,
            "fold_id": row.fold_id,
            "label": str(row.label),
            "method": "market_calibrated",
            "market_probability": str(row.feature_values[market_column]),
            "raw_probability": str(raw_probability),
            "probability": str(probability),
        }
        for row, raw_probability, probability in zip(
            rows,
            raw_probabilities,
            calibrated_probabilities,
        )
    ]


def _normalize_by_race(
    rows: Sequence[MetaFeatureRow],
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row, probability in zip(rows, probabilities):
        totals[row.race_id] = totals.get(row.race_id, 0.0) + max(probability, 0.0)
        counts[row.race_id] = counts.get(row.race_id, 0) + 1

    normalized: list[float] = []
    for row, probability in zip(rows, probabilities):
        total = totals[row.race_id]
        if total > 0.0:
            normalized.append(max(probability, 0.0) / total)
        else:
            normalized.append(1.0 / counts[row.race_id])
    return tuple(normalized)


def _probability_summary(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> dict[str, float | int]:
    if not probabilities:
        return {
            "observations": 0,
            "positives": 0,
            "mean_predicted_probability": 0.0,
            "empirical_rate": 0.0,
            "log_loss": 0.0,
            "brier_score": 0.0,
            "expected_calibration_error": 0.0,
        }
    count = len(probabilities)
    positives = sum(labels)
    return {
        "observations": count,
        "positives": positives,
        "mean_predicted_probability": sum(probabilities) / count,
        "empirical_rate": positives / count,
        "log_loss": _binary_log_loss(probabilities, labels),
        "brier_score": sum(
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels)
        )
        / count,
        "expected_calibration_error": _expected_calibration_error(
            probabilities,
            labels,
        ),
    }


def _expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> float:
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not in_bin:
            continue
        confidence = sum(probability for probability, _ in in_bin) / len(in_bin)
        accuracy = sum(label for _, label in in_bin) / len(in_bin)
        error += (len(in_bin) / total) * abs(confidence - accuracy)
    return error


def _infer_market_column(rows: Sequence[MetaFeatureRow]) -> str:
    columns = sorted(rows[0].feature_values)
    market_columns = [
        column for column in columns if "market_implied_probability" in column
    ]
    if not market_columns:
        market_columns = [column for column in columns if "market" in column]
    if len(market_columns) != 1:
        raise ValueError(
            "market_column must be provided when exactly one market column "
            "cannot be inferred"
        )
    return market_columns[0]


def _binary_log_loss(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    epsilon: float = 1e-15,
) -> float:
    return sum(
        -math.log(_clip_probability(probability, epsilon=epsilon))
        if label
        else -math.log(1.0 - _clip_probability(probability, epsilon=epsilon))
        for probability, label in zip(probabilities, labels)
    ) / len(probabilities)


def _clip_probability(probability: float, *, epsilon: float = 1e-9) -> float:
    return min(max(probability, epsilon), 1.0 - epsilon)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _validate_config(
    *,
    min_train_folds: int,
    learning_rate: float,
    max_iterations: int,
    l2: float,
) -> None:
    if min_train_folds <= 0:
        raise ValueError("min_train_folds must be positive")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")


def _write_predictions(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CALIBRATED_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
