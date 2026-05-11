"""Phase 4 walk-forward stacking studies."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.data.csv_parsing import parse_race_row, read_csv_rows
from horse_lab.schemas import Race
from horse_lab.stacking.blend import search_convex_blend
from horse_lab.stacking.meta_learner import (
    MetaFeatureRow,
    read_meta_feature_rows,
    train_logistic_meta_learner,
)


WALKFORWARD_PREDICTION_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "fold_id",
    "label",
    "method",
    "probability",
)

SEGMENT_METRIC_FIELDS: tuple[str, ...] = (
    "segment_name",
    "segment_value",
    "method",
    "observations",
    "positives",
    "mean_predicted_probability",
    "empirical_rate",
    "log_loss",
    "brier_score",
    "delta_log_loss_vs_market",
    "delta_brier_vs_market",
)


@dataclass(frozen=True)
class Phase4StudyResult:
    report_path: Path
    predictions_path: Path
    segment_metrics_path: Path
    report: dict[str, Any]


def run_phase4_study_from_csv(
    meta_features_csv: Path | str,
    races_csv: Path | str,
    artifact_dir: Path | str,
    *,
    min_train_folds: int = 3,
    blend_grid_step: float = 0.05,
    logistic_learning_rate: float = 0.05,
    logistic_max_iterations: int = 2000,
    logistic_l2: float = 1e-3,
) -> Phase4StudyResult:
    rows = read_meta_feature_rows(meta_features_csv)
    races = tuple(parse_race_row(row) for row in read_csv_rows(races_csv))
    return run_phase4_study(
        rows=rows,
        races=races,
        artifact_dir=artifact_dir,
        min_train_folds=min_train_folds,
        blend_grid_step=blend_grid_step,
        logistic_learning_rate=logistic_learning_rate,
        logistic_max_iterations=logistic_max_iterations,
        logistic_l2=logistic_l2,
    )


def run_phase4_study(
    *,
    rows: Sequence[MetaFeatureRow],
    races: Sequence[Race],
    artifact_dir: Path | str,
    min_train_folds: int = 3,
    blend_grid_step: float = 0.05,
    logistic_learning_rate: float = 0.05,
    logistic_max_iterations: int = 2000,
    logistic_l2: float = 1e-3,
) -> Phase4StudyResult:
    if min_train_folds <= 0:
        raise ValueError("min_train_folds must be positive")
    if not rows:
        raise ValueError("meta feature rows must not be empty")

    columns = _feature_columns(rows)
    market_column = _market_column(columns)
    folds = _fold_ids(rows)
    if len(folds) <= min_train_folds:
        raise ValueError(
            "not enough folds for walk-forward study: "
            f"folds={len(folds)}, min_train_folds={min_train_folds}"
        )

    output_path = Path(artifact_dir)
    prediction_records: list[dict[str, str]] = []
    fold_reports: list[dict[str, Any]] = []
    for holdout_fold_id in folds[min_train_folds:]:
        candidate_folds = [
            fold_id for fold_id in folds if fold_id <= holdout_fold_id
        ]
        candidate_rows = tuple(row for row in rows if row.fold_id in candidate_folds)
        train_fold_ids = tuple(
            fold_id for fold_id in candidate_folds if fold_id != holdout_fold_id
        )
        holdout_rows = tuple(
            row for row in candidate_rows if row.fold_id == holdout_fold_id
        )

        blend_result = search_convex_blend(
            rows=candidate_rows,
            artifact_dir=output_path / "folds" / holdout_fold_id / "blend",
            holdout_fold_id=holdout_fold_id,
            feature_columns=columns,
            model_version=f"convex-blend-{holdout_fold_id}",
            grid_step=blend_grid_step,
        )
        logistic_result = train_logistic_meta_learner(
            rows=candidate_rows,
            artifact_dir=output_path / "folds" / holdout_fold_id / "logistic_meta",
            holdout_fold_id=holdout_fold_id,
            feature_columns=columns,
            model_version=f"logistic-meta-{holdout_fold_id}",
            learning_rate=logistic_learning_rate,
            max_iterations=logistic_max_iterations,
            l2=logistic_l2,
        )

        method_probabilities = _baseline_probabilities(holdout_rows, columns)
        method_probabilities["convex_blend"] = _read_prediction_probabilities(
            blend_result.predictions_path
        )
        method_probabilities["logistic_meta"] = _read_prediction_probabilities(
            logistic_result.predictions_path
        )
        labels = tuple(row.label for row in holdout_rows)
        fold_summary = {
            "holdout_fold_id": holdout_fold_id,
            "train_fold_ids": list(train_fold_ids),
            "rows": len(holdout_rows),
            "positives": sum(labels),
            "methods": {
                method: _probability_summary(probabilities, labels)
                for method, probabilities in method_probabilities.items()
            },
            "blend_weights": blend_result.report["weights"],
            "logistic_coefficients": logistic_result.report["coefficients"],
        }
        fold_reports.append(fold_summary)

        for method, probabilities in method_probabilities.items():
            prediction_records.extend(
                _prediction_records(
                    rows=holdout_rows,
                    method=method,
                    probabilities=probabilities,
                )
            )

    predictions_path = output_path / "walkforward_predictions.csv"
    segment_metrics_path = output_path / "segment_metrics.csv"
    report_path = output_path / "phase4_study_report.json"
    _write_prediction_records(predictions_path, prediction_records)
    segment_rows = _segment_metrics(
        prediction_records=prediction_records,
        meta_rows=rows,
        races=races,
        market_column=market_column,
    )
    _write_segment_metrics(segment_metrics_path, segment_rows)

    overall = _overall_metrics(prediction_records)
    best_level0 = _best_method(
        {
            method: metrics
            for method, metrics in overall.items()
            if method.startswith("pred__")
        }
    )
    best_ensemble = _best_method(
        {
            method: metrics
            for method, metrics in overall.items()
            if method in {"convex_blend", "logistic_meta"}
        }
    )
    best_overall = _best_method(overall)
    recommendation = _recommendation(
        best_level0=best_level0,
        best_ensemble=best_ensemble,
        best_overall=best_overall,
    )
    report = {
        "feature_columns": list(columns),
        "market_column": market_column,
        "folds": folds,
        "holdout_fold_ids": [
            fold["holdout_fold_id"] for fold in fold_reports
        ],
        "counts": {
            "rows": len(rows),
            "folds": len(folds),
            "holdout_folds": len(fold_reports),
            "walkforward_predictions": len(prediction_records),
        },
        "config": {
            "min_train_folds": min_train_folds,
            "blend_grid_step": blend_grid_step,
            "logistic_learning_rate": logistic_learning_rate,
            "logistic_max_iterations": logistic_max_iterations,
            "logistic_l2": logistic_l2,
        },
        "overall": overall,
        "folds_detail": fold_reports,
        "best_level0": best_level0,
        "best_ensemble": best_ensemble,
        "best_overall": best_overall,
        "recommendation": recommendation,
        "predictions_path": str(predictions_path),
        "segment_metrics_path": str(segment_metrics_path),
    }
    _write_json(report_path, report)
    return Phase4StudyResult(
        report_path=report_path,
        predictions_path=predictions_path,
        segment_metrics_path=segment_metrics_path,
        report=report,
    )


def phase4_study_result_to_dict(result: Phase4StudyResult) -> dict[str, Any]:
    return {
        "report_path": str(result.report_path),
        "predictions_path": str(result.predictions_path),
        "segment_metrics_path": str(result.segment_metrics_path),
        "report": result.report,
    }


def _feature_columns(rows: Sequence[MetaFeatureRow]) -> tuple[str, ...]:
    return tuple(sorted(rows[0].feature_values))


def _market_column(columns: Sequence[str]) -> str:
    market_columns = [
        column for column in columns if "market_implied_probability" in column
    ]
    if not market_columns:
        raise ValueError("meta feature rows must include a market prediction column")
    return market_columns[0]


def _fold_ids(rows: Sequence[MetaFeatureRow]) -> tuple[str, ...]:
    folds = tuple(sorted({row.fold_id for row in rows if row.fold_id}))
    if not folds:
        raise ValueError("meta feature rows must contain fold_id values")
    return folds


def _baseline_probabilities(
    rows: Sequence[MetaFeatureRow],
    columns: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    return {
        column: _normalize_by_race(
            rows,
            tuple(row.feature_values[column] for row in rows),
        )
        for column in columns
    }


def _read_prediction_probabilities(path: Path) -> tuple[float, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(float(row["probability"]) for row in csv.DictReader(handle))


def _prediction_records(
    *,
    rows: Sequence[MetaFeatureRow],
    method: str,
    probabilities: Sequence[float],
) -> list[dict[str, str]]:
    return [
        {
            "race_id": row.race_id,
            "runner_id": row.runner_id,
            "fold_id": row.fold_id,
            "label": str(row.label),
            "method": method,
            "probability": str(probability),
        }
        for row, probability in zip(rows, probabilities)
    ]


def _overall_metrics(
    prediction_records: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for record in prediction_records:
        grouped.setdefault(record["method"], []).append(record)
    return {
        method: _records_probability_summary(records)
        for method, records in sorted(grouped.items())
    }


def _segment_metrics(
    *,
    prediction_records: Sequence[Mapping[str, str]],
    meta_rows: Sequence[MetaFeatureRow],
    races: Sequence[Race],
    market_column: str,
) -> list[dict[str, str]]:
    meta_by_runner = {(row.race_id, row.runner_id): row for row in meta_rows}
    race_by_id = {str(race.race_id): race for race in races}
    grouped: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    for record in prediction_records:
        meta_row = meta_by_runner[(record["race_id"], record["runner_id"])]
        race = race_by_id.get(record["race_id"])
        for segment_name, segment_value in _segments(
            record=record,
            meta_row=meta_row,
            race=race,
            market_column=market_column,
        ):
            grouped.setdefault(
                (segment_name, segment_value, record["method"]),
                [],
            ).append(record)

    market_metrics: dict[tuple[str, str], dict[str, float | int]] = {}
    rows: list[dict[str, str]] = []
    for (segment_name, segment_value, method), records in sorted(grouped.items()):
        metrics = _records_probability_summary(records)
        if method == market_column:
            market_metrics[(segment_name, segment_value)] = metrics
        rows.append(
            _segment_metric_row(
                segment_name=segment_name,
                segment_value=segment_value,
                method=method,
                metrics=metrics,
            )
        )

    for row in rows:
        market = market_metrics.get((row["segment_name"], row["segment_value"]))
        if market is None:
            row["delta_log_loss_vs_market"] = ""
            row["delta_brier_vs_market"] = ""
            continue
        row["delta_log_loss_vs_market"] = str(
            float(row["log_loss"]) - float(market["log_loss"])
        )
        row["delta_brier_vs_market"] = str(
            float(row["brier_score"]) - float(market["brier_score"])
        )
    return rows


def _segments(
    *,
    record: Mapping[str, str],
    meta_row: MetaFeatureRow,
    race: Race | None,
    market_column: str,
) -> tuple[tuple[str, str], ...]:
    market_probability = meta_row.feature_values[market_column]
    values = [
        ("all", "all"),
        ("fold_id", record["fold_id"]),
        ("market_probability_band", _market_probability_band(market_probability)),
    ]
    if race is not None:
        values.extend(
            [
                ("venue", race.venue),
                ("surface", race.surface.value),
                ("distance_bucket", _distance_bucket(race.distance_m)),
                ("field_size_bucket", _field_size_bucket(race.field_size)),
            ]
        )
    return tuple(values)


def _records_probability_summary(
    records: Sequence[Mapping[str, str]],
) -> dict[str, float | int]:
    probabilities = tuple(float(record["probability"]) for record in records)
    labels = tuple(int(record["label"]) for record in records)
    return _probability_summary(probabilities, labels)


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


def _segment_metric_row(
    *,
    segment_name: str,
    segment_value: str,
    method: str,
    metrics: Mapping[str, float | int],
) -> dict[str, str]:
    return {
        "segment_name": segment_name,
        "segment_value": segment_value,
        "method": method,
        "observations": str(metrics["observations"]),
        "positives": str(metrics["positives"]),
        "mean_predicted_probability": str(metrics["mean_predicted_probability"]),
        "empirical_rate": str(metrics["empirical_rate"]),
        "log_loss": str(metrics["log_loss"]),
        "brier_score": str(metrics["brier_score"]),
        "delta_log_loss_vs_market": "",
        "delta_brier_vs_market": "",
    }


def _best_method(
    metrics_by_method: Mapping[str, Mapping[str, float | int]],
) -> dict[str, float | int | str]:
    if not metrics_by_method:
        return {"method": "", "log_loss": 0.0, "brier_score": 0.0}
    method = min(
        metrics_by_method,
        key=lambda name: (
            float(metrics_by_method[name]["log_loss"]),
            float(metrics_by_method[name]["brier_score"]),
            name,
        ),
    )
    return {
        "method": method,
        "log_loss": metrics_by_method[method]["log_loss"],
        "brier_score": metrics_by_method[method]["brier_score"],
    }


def _recommendation(
    *,
    best_level0: Mapping[str, float | int | str],
    best_ensemble: Mapping[str, float | int | str],
    best_overall: Mapping[str, float | int | str],
) -> dict[str, Any]:
    if best_overall["method"] == best_ensemble["method"]:
        action = "promote_ensemble_candidate"
    else:
        action = "keep_best_level0"
    improvement_vs_level0 = (
        float(best_level0["log_loss"]) - float(best_ensemble["log_loss"])
        if best_level0["method"] and best_ensemble["method"]
        else 0.0
    )
    return {
        "action": action,
        "best_level0_method": best_level0["method"],
        "best_ensemble_method": best_ensemble["method"],
        "best_overall_method": best_overall["method"],
        "ensemble_log_loss_improvement_vs_best_level0": improvement_vs_level0,
    }


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


def _clip_probability(probability: float, *, epsilon: float) -> float:
    return min(max(probability, epsilon), 1.0 - epsilon)


def _write_prediction_records(
    path: Path,
    records: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WALKFORWARD_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _write_segment_metrics(
    path: Path,
    records: Sequence[Mapping[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
