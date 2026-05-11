"""Dependency-light Level 1 logistic meta learner for stacking."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


META_PREDICTION_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "fold_id",
    "label",
    "raw_probability",
    "probability",
)


@dataclass(frozen=True)
class MetaFeatureRow:
    race_id: str
    runner_id: str
    fold_id: str
    label: int
    feature_values: Mapping[str, float]


@dataclass(frozen=True)
class LogisticMetaModel:
    model_version: str
    feature_columns: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    train_fold_ids: tuple[str, ...]
    holdout_fold_id: str
    l2: float
    max_iterations: int
    learning_rate: float


@dataclass(frozen=True)
class MetaLearnerTrainingResult:
    model_path: Path
    predictions_path: Path
    report_path: Path
    report: dict[str, Any]


def train_logistic_meta_learner_from_csv(
    meta_features_csv: Path | str,
    artifact_dir: Path | str,
    *,
    holdout_fold_id: str | None = None,
    feature_columns: Sequence[str] | None = None,
    model_version: str = "logistic-meta-v1",
    learning_rate: float = 0.05,
    max_iterations: int = 2000,
    l2: float = 1e-3,
) -> MetaLearnerTrainingResult:
    rows = read_meta_feature_rows(meta_features_csv, feature_columns=feature_columns)
    return train_logistic_meta_learner(
        rows=rows,
        artifact_dir=artifact_dir,
        holdout_fold_id=holdout_fold_id,
        feature_columns=feature_columns,
        model_version=model_version,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        l2=l2,
    )


def train_logistic_meta_learner(
    *,
    rows: Sequence[MetaFeatureRow],
    artifact_dir: Path | str,
    holdout_fold_id: str | None = None,
    feature_columns: Sequence[str] | None = None,
    model_version: str = "logistic-meta-v1",
    learning_rate: float = 0.05,
    max_iterations: int = 2000,
    l2: float = 1e-3,
) -> MetaLearnerTrainingResult:
    _validate_optimizer_config(
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        l2=l2,
    )
    if not rows:
        raise ValueError("meta feature rows must not be empty")

    columns = tuple(feature_columns or _feature_columns(rows))
    if not columns:
        raise ValueError("meta feature rows must contain at least one feature column")

    selected_holdout = holdout_fold_id or _latest_fold_id(rows)
    train_rows = tuple(row for row in rows if row.fold_id != selected_holdout)
    holdout_rows = tuple(row for row in rows if row.fold_id == selected_holdout)
    if not train_rows:
        raise ValueError("training rows must not be empty")
    if not holdout_rows:
        raise ValueError(f"no rows found for holdout_fold_id={selected_holdout!r}")

    train_matrix = _design_matrix(train_rows, columns)
    holdout_matrix = _design_matrix(holdout_rows, columns)
    means, scales = _fit_standardizer(train_matrix)
    standardized_train = _standardize(train_matrix, means=means, scales=scales)
    standardized_holdout = _standardize(holdout_matrix, means=means, scales=scales)

    labels = tuple(row.label for row in train_rows)
    intercept, coefficients, final_loss = _fit_logistic_regression(
        standardized_train,
        labels,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        l2=l2,
    )
    model = LogisticMetaModel(
        model_version=model_version,
        feature_columns=columns,
        intercept=intercept,
        coefficients=coefficients,
        means=means,
        scales=scales,
        train_fold_ids=tuple(sorted({row.fold_id for row in train_rows})),
        holdout_fold_id=selected_holdout,
        l2=l2,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
    )

    train_raw = _predict_matrix(standardized_train, model)
    holdout_raw = _predict_matrix(standardized_holdout, model)
    train_predictions = _normalize_by_race(train_rows, train_raw)
    holdout_predictions = _normalize_by_race(holdout_rows, holdout_raw)

    output_path = Path(artifact_dir)
    model_path = output_path / "meta_model.json"
    predictions_path = output_path / "meta_predictions.csv"
    report_path = output_path / "meta_evaluation.json"
    _write_model(model_path, model)
    _write_predictions(predictions_path, holdout_rows, holdout_raw, holdout_predictions)

    report = {
        "model_version": model_version,
        "feature_columns": list(columns),
        "train_fold_ids": list(model.train_fold_ids),
        "holdout_fold_id": selected_holdout,
        "counts": {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "features": len(columns),
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "max_iterations": max_iterations,
            "l2": l2,
            "final_training_log_loss": final_loss,
        },
        "train_probability": _probability_summary(
            train_predictions,
            tuple(row.label for row in train_rows),
        ),
        "holdout_probability": _probability_summary(
            holdout_predictions,
            tuple(row.label for row in holdout_rows),
        ),
        "holdout_baselines": {
            column: _probability_summary(
                _normalize_by_race(
                    holdout_rows,
                    tuple(row.feature_values[column] for row in holdout_rows),
                ),
                tuple(row.label for row in holdout_rows),
            )
            for column in columns
        },
        "coefficients": [
            {
                "feature_column": column,
                "coefficient": coefficient,
                "mean": mean,
                "scale": scale,
            }
            for column, coefficient, mean, scale in zip(
                columns,
                coefficients,
                means,
                scales,
            )
        ],
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }
    _write_json(report_path, report)
    return MetaLearnerTrainingResult(
        model_path=model_path,
        predictions_path=predictions_path,
        report_path=report_path,
        report=report,
    )


def meta_learner_training_result_to_dict(
    result: MetaLearnerTrainingResult,
) -> dict[str, Any]:
    return {
        "model_path": str(result.model_path),
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def read_meta_feature_rows(
    path: Path | str,
    *,
    feature_columns: Sequence[str] | None = None,
) -> tuple[MetaFeatureRow, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("meta feature CSV must contain a header")
        columns = tuple(
            feature_columns
            or [field for field in reader.fieldnames if field.startswith("pred__")]
        )
        rows = []
        for csv_row in reader:
            rows.append(
                MetaFeatureRow(
                    race_id=csv_row["race_id"],
                    runner_id=csv_row["runner_id"],
                    fold_id=csv_row["fold_id"],
                    label=int(csv_row["label"]),
                    feature_values={
                        column: _parse_probability(csv_row[column], column=column)
                        for column in columns
                    },
                )
            )
    return tuple(rows)


def _fit_logistic_regression(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    learning_rate: float,
    max_iterations: int,
    l2: float,
) -> tuple[float, tuple[float, ...], float]:
    n_rows = len(matrix)
    n_features = len(matrix[0])
    positive_rate = sum(labels) / n_rows
    intercept = _logit(_clip_probability(positive_rate))
    coefficients = [0.0 for _ in range(n_features)]

    for _ in range(max_iterations):
        intercept_gradient = 0.0
        coefficient_gradients = [0.0 for _ in range(n_features)]
        for values, label in zip(matrix, labels):
            probability = _sigmoid(
                intercept
                + sum(weight * value for weight, value in zip(coefficients, values))
            )
            error = probability - label
            intercept_gradient += error
            for index, value in enumerate(values):
                coefficient_gradients[index] += error * value

        intercept -= learning_rate * (intercept_gradient / n_rows)
        for index, gradient in enumerate(coefficient_gradients):
            regularized_gradient = gradient / n_rows + l2 * coefficients[index]
            coefficients[index] -= learning_rate * regularized_gradient

    predictions = tuple(_sigmoid(_linear_score(row, intercept, coefficients)) for row in matrix)
    return (
        intercept,
        tuple(coefficients),
        _binary_log_loss(predictions, labels),
    )


def _predict_matrix(
    matrix: Sequence[Sequence[float]],
    model: LogisticMetaModel,
) -> tuple[float, ...]:
    return tuple(
        _sigmoid(_linear_score(row, model.intercept, model.coefficients))
        for row in matrix
    )


def _linear_score(
    values: Sequence[float],
    intercept: float,
    coefficients: Sequence[float],
) -> float:
    return intercept + sum(weight * value for weight, value in zip(coefficients, values))


def _design_matrix(
    rows: Sequence[MetaFeatureRow],
    columns: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(_logit(_clip_probability(row.feature_values[column])) for column in columns)
        for row in rows
    )


def _fit_standardizer(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n_rows = len(matrix)
    n_features = len(matrix[0])
    means = tuple(
        sum(row[index] for row in matrix) / n_rows
        for index in range(n_features)
    )
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in matrix) / n_rows
        scale = math.sqrt(variance)
        scales.append(scale if scale > 1e-12 else 1.0)
    return means, tuple(scales)


def _standardize(
    matrix: Sequence[Sequence[float]],
    *,
    means: Sequence[float],
    scales: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple((value - mean) / scale for value, mean, scale in zip(row, means, scales))
        for row in matrix
    )


def _normalize_by_race(
    rows: Sequence[MetaFeatureRow],
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row, probability in zip(rows, probabilities):
        totals[row.race_id] = totals.get(row.race_id, 0.0) + probability
        counts[row.race_id] = counts.get(row.race_id, 0) + 1

    normalized = []
    for row, probability in zip(rows, probabilities):
        total = totals[row.race_id]
        if total > 0.0:
            normalized.append(probability / total)
        else:
            normalized.append(1.0 / counts[row.race_id])
    return tuple(normalized)


def _write_model(path: Path, model: LogisticMetaModel) -> None:
    payload = {
        "model_version": model.model_version,
        "feature_columns": list(model.feature_columns),
        "intercept": model.intercept,
        "coefficients": list(model.coefficients),
        "means": list(model.means),
        "scales": list(model.scales),
        "train_fold_ids": list(model.train_fold_ids),
        "holdout_fold_id": model.holdout_fold_id,
        "l2": model.l2,
        "max_iterations": model.max_iterations,
        "learning_rate": model.learning_rate,
    }
    _write_json(path, payload)


def _write_predictions(
    path: Path,
    rows: Sequence[MetaFeatureRow],
    raw_probabilities: Sequence[float],
    probabilities: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=META_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                "race_id": row.race_id,
                "runner_id": row.runner_id,
                "fold_id": row.fold_id,
                "label": str(row.label),
                "raw_probability": str(raw_probability),
                "probability": str(probability),
            }
            for row, raw_probability, probability in zip(
                rows,
                raw_probabilities,
                probabilities,
            )
        )


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
    }


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


def _feature_columns(rows: Sequence[MetaFeatureRow]) -> tuple[str, ...]:
    return tuple(sorted(rows[0].feature_values))


def _latest_fold_id(rows: Sequence[MetaFeatureRow]) -> str:
    fold_ids = sorted({row.fold_id for row in rows if row.fold_id})
    if not fold_ids:
        raise ValueError("meta feature rows must contain fold_id values")
    return fold_ids[-1]


def _parse_probability(value: str, *, column: str) -> float:
    if value == "":
        raise ValueError(f"Missing probability for column={column!r}")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability in column={column!r} must be between 0 and 1")
    return probability


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


def _validate_optimizer_config(
    *,
    learning_rate: float,
    max_iterations: int,
    l2: float,
) -> None:
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
