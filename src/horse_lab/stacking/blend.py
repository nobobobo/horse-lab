"""Convex blending for Level 0 OOF prediction columns."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from horse_lab.stacking.meta_learner import (
    META_PREDICTION_FIELDS,
    MetaFeatureRow,
    read_meta_feature_rows,
)


@dataclass(frozen=True)
class ConvexBlendModel:
    model_version: str
    feature_columns: tuple[str, ...]
    weights: tuple[float, ...]
    train_fold_ids: tuple[str, ...]
    holdout_fold_id: str
    grid_step: float


@dataclass(frozen=True)
class BlendSearchResult:
    model_path: Path
    predictions_path: Path
    report_path: Path
    report: dict[str, Any]


def search_convex_blend_from_csv(
    meta_features_csv: Path | str,
    artifact_dir: Path | str,
    *,
    holdout_fold_id: str | None = None,
    feature_columns: Sequence[str] | None = None,
    model_version: str = "convex-blend-v1",
    grid_step: float = 0.05,
) -> BlendSearchResult:
    rows = read_meta_feature_rows(meta_features_csv, feature_columns=feature_columns)
    return search_convex_blend(
        rows=rows,
        artifact_dir=artifact_dir,
        holdout_fold_id=holdout_fold_id,
        feature_columns=feature_columns,
        model_version=model_version,
        grid_step=grid_step,
    )


def search_convex_blend(
    *,
    rows: Sequence[MetaFeatureRow],
    artifact_dir: Path | str,
    holdout_fold_id: str | None = None,
    feature_columns: Sequence[str] | None = None,
    model_version: str = "convex-blend-v1",
    grid_step: float = 0.05,
) -> BlendSearchResult:
    if not rows:
        raise ValueError("meta feature rows must not be empty")
    columns = tuple(feature_columns or _feature_columns(rows))
    if not columns:
        raise ValueError("meta feature rows must contain at least one feature column")
    _validate_grid_step(grid_step)

    selected_holdout = holdout_fold_id or _latest_fold_id(rows)
    train_rows = tuple(row for row in rows if row.fold_id != selected_holdout)
    holdout_rows = tuple(row for row in rows if row.fold_id == selected_holdout)
    if not train_rows:
        raise ValueError("training rows must not be empty")
    if not holdout_rows:
        raise ValueError(f"no rows found for holdout_fold_id={selected_holdout!r}")

    candidates = _candidate_weights(len(columns), grid_step=grid_step)
    scored_candidates = [
        {
            "weights": weights,
            "train_log_loss": _probability_summary(
                _blend_probabilities(train_rows, columns, weights),
                tuple(row.label for row in train_rows),
            )["log_loss"],
        }
        for weights in candidates
    ]
    best_candidate = min(
        scored_candidates,
        key=lambda row: (
            row["train_log_loss"],
            _weight_tiebreaker(row["weights"]),
        ),
    )
    best_weights = tuple(float(value) for value in best_candidate["weights"])

    train_predictions = _blend_probabilities(train_rows, columns, best_weights)
    holdout_predictions = _blend_probabilities(holdout_rows, columns, best_weights)
    model = ConvexBlendModel(
        model_version=model_version,
        feature_columns=columns,
        weights=best_weights,
        train_fold_ids=tuple(sorted({row.fold_id for row in train_rows})),
        holdout_fold_id=selected_holdout,
        grid_step=grid_step,
    )

    output_path = Path(artifact_dir)
    model_path = output_path / "blend_model.json"
    predictions_path = output_path / "blend_predictions.csv"
    report_path = output_path / "blend_evaluation.json"
    _write_model(model_path, model)
    _write_predictions(predictions_path, holdout_rows, holdout_predictions)

    report = {
        "model_version": model_version,
        "feature_columns": list(columns),
        "weights": [
            {"feature_column": column, "weight": weight}
            for column, weight in zip(columns, best_weights)
        ],
        "train_fold_ids": list(model.train_fold_ids),
        "holdout_fold_id": selected_holdout,
        "counts": {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "features": len(columns),
            "candidates": len(candidates),
        },
        "search": {
            "grid_step": grid_step,
            "best_train_log_loss": best_candidate["train_log_loss"],
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
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }
    _write_json(report_path, report)
    return BlendSearchResult(
        model_path=model_path,
        predictions_path=predictions_path,
        report_path=report_path,
        report=report,
    )


def blend_search_result_to_dict(result: BlendSearchResult) -> dict[str, Any]:
    return {
        "model_path": str(result.model_path),
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def _blend_probabilities(
    rows: Sequence[MetaFeatureRow],
    columns: Sequence[str],
    weights: Sequence[float],
) -> tuple[float, ...]:
    raw = tuple(
        sum(
            weight * row.feature_values[column]
            for column, weight in zip(columns, weights)
        )
        for row in rows
    )
    return _normalize_by_race(rows, raw)


def _candidate_weights(
    n_features: int,
    *,
    grid_step: float,
) -> tuple[tuple[float, ...], ...]:
    units = round(1.0 / grid_step)

    def build(
        prefix: tuple[int, ...],
        remaining: int,
        slots: int,
    ) -> list[tuple[int, ...]]:
        if slots == 1:
            return [prefix + (remaining,)]
        values: list[tuple[int, ...]] = []
        for value in range(remaining + 1):
            values.extend(build(prefix + (value,), remaining - value, slots - 1))
        return values

    return tuple(
        tuple(value / units for value in candidate)
        for candidate in build((), units, n_features)
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


def _write_model(path: Path, model: ConvexBlendModel) -> None:
    payload = {
        "model_version": model.model_version,
        "feature_columns": list(model.feature_columns),
        "weights": list(model.weights),
        "train_fold_ids": list(model.train_fold_ids),
        "holdout_fold_id": model.holdout_fold_id,
        "grid_step": model.grid_step,
    }
    _write_json(path, payload)


def _write_predictions(
    path: Path,
    rows: Sequence[MetaFeatureRow],
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
                "raw_probability": str(probability),
                "probability": str(probability),
            }
            for row, probability in zip(rows, probabilities)
        )


def _feature_columns(rows: Sequence[MetaFeatureRow]) -> tuple[str, ...]:
    return tuple(sorted(rows[0].feature_values))


def _latest_fold_id(rows: Sequence[MetaFeatureRow]) -> str:
    fold_ids = sorted({row.fold_id for row in rows if row.fold_id})
    if not fold_ids:
        raise ValueError("meta feature rows must contain fold_id values")
    return fold_ids[-1]


def _weight_tiebreaker(weights: Sequence[float]) -> tuple[float, ...]:
    return tuple(-weight for weight in weights)


def _clip_probability(probability: float, *, epsilon: float) -> float:
    return min(max(probability, epsilon), 1.0 - epsilon)


def _validate_grid_step(grid_step: float) -> None:
    if not 0.0 < grid_step <= 1.0:
        raise ValueError("grid_step must be in (0, 1]")
    units = round(1.0 / grid_step)
    if not math.isclose(units * grid_step, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("grid_step must divide 1.0 exactly")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
