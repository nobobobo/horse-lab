"""Market-anchored residual overlay studies."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.stacking.meta_learner import MetaFeatureRow, read_meta_feature_rows


RESIDUAL_OVERLAY_PREDICTION_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "fold_id",
    "label",
    "method",
    "market_probability",
    "overlay_probability",
    "alpha",
    "probability",
)


@dataclass(frozen=True)
class ResidualOverlayStudyResult:
    predictions_path: Path
    report_path: Path
    report: dict[str, Any]


def run_residual_overlay_study_from_csv(
    meta_features_csv: Path | str,
    artifact_dir: Path | str,
    *,
    market_column: str | None = None,
    overlay_column: str | None = None,
    min_train_folds: int = 3,
    alpha_min: float = -0.5,
    alpha_max: float = 0.5,
    alpha_step: float = 0.05,
    model_version: str = "residual-overlay-v1",
) -> ResidualOverlayStudyResult:
    rows = read_meta_feature_rows(meta_features_csv)
    return run_residual_overlay_study(
        rows=rows,
        artifact_dir=artifact_dir,
        market_column=market_column,
        overlay_column=overlay_column,
        min_train_folds=min_train_folds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        model_version=model_version,
    )


def run_residual_overlay_study(
    *,
    rows: Sequence[MetaFeatureRow],
    artifact_dir: Path | str,
    market_column: str | None = None,
    overlay_column: str | None = None,
    min_train_folds: int = 3,
    alpha_min: float = -0.5,
    alpha_max: float = 0.5,
    alpha_step: float = 0.05,
    model_version: str = "residual-overlay-v1",
) -> ResidualOverlayStudyResult:
    _validate_config(
        min_train_folds=min_train_folds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
    )
    if not rows:
        raise ValueError("meta feature rows must not be empty")

    selected_market_column = market_column or _infer_market_column(rows)
    selected_overlay_column = overlay_column or _infer_overlay_column(
        rows,
        market_column=selected_market_column,
    )
    folds = tuple(sorted({row.fold_id for row in rows if row.fold_id}))
    if len(folds) <= min_train_folds:
        raise ValueError("not enough folds for residual overlay study")

    alphas = _alpha_grid(alpha_min=alpha_min, alpha_max=alpha_max, alpha_step=alpha_step)
    prediction_rows: list[dict[str, str]] = []
    fold_reports: list[dict[str, Any]] = []
    all_overlay: list[float] = []
    all_market: list[float] = []
    all_base_overlay: list[float] = []
    all_labels: list[int] = []

    for holdout_index in range(min_train_folds, len(folds)):
        holdout_fold_id = folds[holdout_index]
        train_fold_ids = folds[:holdout_index]
        train_rows = tuple(row for row in rows if row.fold_id in train_fold_ids)
        holdout_rows = tuple(row for row in rows if row.fold_id == holdout_fold_id)
        if not train_rows or not holdout_rows:
            continue

        best_alpha, train_summary = _select_alpha(
            train_rows,
            market_column=selected_market_column,
            overlay_column=selected_overlay_column,
            alphas=alphas,
        )
        holdout_overlay = _overlay_probabilities(
            holdout_rows,
            market_column=selected_market_column,
            overlay_column=selected_overlay_column,
            alpha=best_alpha,
        )
        holdout_market = _normalized_column(
            holdout_rows,
            selected_market_column,
        )
        holdout_base_overlay = _normalized_column(
            holdout_rows,
            selected_overlay_column,
        )
        labels = tuple(row.label for row in holdout_rows)

        all_overlay.extend(holdout_overlay)
        all_market.extend(holdout_market)
        all_base_overlay.extend(holdout_base_overlay)
        all_labels.extend(labels)
        prediction_rows.extend(
            _prediction_rows(
                holdout_rows,
                market_column=selected_market_column,
                overlay_column=selected_overlay_column,
                probabilities=holdout_overlay,
                alpha=best_alpha,
            )
        )
        fold_reports.append(
            {
                "holdout_fold_id": holdout_fold_id,
                "train_fold_ids": list(train_fold_ids),
                "selected_alpha": best_alpha,
                "counts": {
                    "train_rows": len(train_rows),
                    "holdout_rows": len(holdout_rows),
                },
                "train_selected_alpha_probability": train_summary,
                "residual_overlay": _probability_summary(holdout_overlay, labels),
                "market_baseline": _probability_summary(holdout_market, labels),
                "overlay_baseline": _probability_summary(
                    holdout_base_overlay,
                    labels,
                ),
            }
        )

    if not fold_reports:
        raise ValueError("no residual overlay folds were evaluated")

    overlay_summary = _probability_summary(all_overlay, all_labels)
    market_summary = _probability_summary(all_market, all_labels)
    base_overlay_summary = _probability_summary(all_base_overlay, all_labels)
    improvement = market_summary["log_loss"] - overlay_summary["log_loss"]

    output_path = Path(artifact_dir)
    predictions_path = output_path / "residual_overlay_predictions.csv"
    report_path = output_path / "residual_overlay_report.json"
    _write_predictions(predictions_path, prediction_rows)
    report = {
        "model_version": model_version,
        "market_column": selected_market_column,
        "overlay_column": selected_overlay_column,
        "fold_ids": list(folds),
        "holdout_fold_ids": [fold["holdout_fold_id"] for fold in fold_reports],
        "counts": {
            "rows": len(rows),
            "holdout_rows": len(all_labels),
            "evaluated_folds": len(fold_reports),
        },
        "alpha_grid": {
            "min": alpha_min,
            "max": alpha_max,
            "step": alpha_step,
            "values": list(alphas),
        },
        "overall": {
            "residual_overlay": overlay_summary,
            "market_baseline": market_summary,
            "overlay_baseline": base_overlay_summary,
        },
        "recommendation": {
            "action": (
                "promote_residual_overlay_candidate"
                if improvement > 0.0
                else "keep_market_baseline"
            ),
            "log_loss_improvement_vs_market": improvement,
        },
        "folds_detail": fold_reports,
        "predictions_path": str(predictions_path),
    }
    _write_json(report_path, report)
    return ResidualOverlayStudyResult(
        predictions_path=predictions_path,
        report_path=report_path,
        report=report,
    )


def residual_overlay_study_result_to_dict(
    result: ResidualOverlayStudyResult,
) -> dict[str, Any]:
    return {
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "report": result.report,
    }


def _select_alpha(
    rows: Sequence[MetaFeatureRow],
    *,
    market_column: str,
    overlay_column: str,
    alphas: Sequence[float],
) -> tuple[float, dict[str, float | int]]:
    labels = tuple(row.label for row in rows)
    summaries: list[tuple[float, dict[str, float | int]]] = []
    for alpha in alphas:
        probabilities = _overlay_probabilities(
            rows,
            market_column=market_column,
            overlay_column=overlay_column,
            alpha=alpha,
        )
        summaries.append((alpha, _probability_summary(probabilities, labels)))
    return min(
        summaries,
        key=lambda item: (
            float(item[1]["log_loss"]),
            float(item[1]["brier_score"]),
            abs(item[0]),
        ),
    )


def _overlay_probabilities(
    rows: Sequence[MetaFeatureRow],
    *,
    market_column: str,
    overlay_column: str,
    alpha: float,
) -> tuple[float, ...]:
    raw = tuple(
        _sigmoid(
            _logit(_clip_probability(row.feature_values[market_column]))
            + alpha
            * (
                _logit(_clip_probability(row.feature_values[overlay_column]))
                - _logit(_clip_probability(row.feature_values[market_column]))
            )
        )
        for row in rows
    )
    return _normalize_by_race(rows, raw)


def _normalized_column(
    rows: Sequence[MetaFeatureRow],
    column: str,
) -> tuple[float, ...]:
    return _normalize_by_race(rows, tuple(row.feature_values[column] for row in rows))


def _prediction_rows(
    rows: Sequence[MetaFeatureRow],
    *,
    market_column: str,
    overlay_column: str,
    probabilities: Sequence[float],
    alpha: float,
) -> list[dict[str, str]]:
    return [
        {
            "race_id": row.race_id,
            "runner_id": row.runner_id,
            "fold_id": row.fold_id,
            "label": str(row.label),
            "method": "residual_overlay",
            "market_probability": str(row.feature_values[market_column]),
            "overlay_probability": str(row.feature_values[overlay_column]),
            "alpha": str(alpha),
            "probability": str(probability),
        }
        for row, probability in zip(rows, probabilities)
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


def _infer_overlay_column(
    rows: Sequence[MetaFeatureRow],
    *,
    market_column: str,
) -> str:
    columns = sorted(rows[0].feature_values)
    non_market_columns = [column for column in columns if column != market_column]
    preferred = [column for column in non_market_columns if "no_market" in column]
    if preferred:
        return preferred[0]
    lightgbm = [column for column in non_market_columns if "lightgbm" in column]
    if lightgbm:
        return lightgbm[0]
    if len(non_market_columns) == 1:
        return non_market_columns[0]
    raise ValueError(
        "overlay_column must be provided when it cannot be inferred from "
        "non-market prediction columns"
    )


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


def _alpha_grid(
    *,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
) -> tuple[float, ...]:
    values: list[float] = []
    current = alpha_min
    while current <= alpha_max + (alpha_step / 2.0):
        values.append(round(current, 10))
        current += alpha_step
    return tuple(values)


def _validate_config(
    *,
    min_train_folds: int,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
) -> None:
    if min_train_folds <= 0:
        raise ValueError("min_train_folds must be positive")
    if alpha_step <= 0.0:
        raise ValueError("alpha_step must be positive")
    if alpha_min > alpha_max:
        raise ValueError("alpha_min must be less than or equal to alpha_max")


def _write_predictions(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESIDUAL_OVERLAY_PREDICTION_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
