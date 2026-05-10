"""CSV contract for out-of-fold and live model predictions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.data.csv_parsing import (
    parse_date,
    parse_datetime,
    parse_float_or_none,
    parse_str_or_none,
)
from horse_lab.schemas import (
    ModelName,
    ModelPrediction,
    PredictionTarget,
    RaceId,
    RunnerId,
)


class PredictionRole(str, Enum):
    OOF = "oof"
    HOLDOUT = "holdout"
    LIVE = "live"
    BACKFILL = "backfill"


OOF_PREDICTION_CSV_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "target",
    "model_name",
    "model_version",
    "prediction_role",
    "fold_id",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "as_of",
    "feature_version",
    "probability",
    "lower_probability",
    "upper_probability",
    "metadata_json",
)


@dataclass(frozen=True)
class StoredPrediction:
    race_id: RaceId
    runner_id: RunnerId
    target: PredictionTarget
    model_name: ModelName
    model_version: str
    prediction_role: PredictionRole | str
    as_of: datetime
    feature_version: str
    probability: float
    fold_id: str | None = None
    train_start: date | None = None
    train_end: date | None = None
    validation_start: date | None = None
    validation_end: date | None = None
    lower_probability: float | None = None
    upper_probability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        object.__setattr__(
            self,
            "prediction_role",
            PredictionRole(self.prediction_role),
        )

    def to_model_prediction(self) -> ModelPrediction:
        return ModelPrediction(
            race_id=self.race_id,
            runner_id=self.runner_id,
            model_name=self.model_name,
            model_version=self.model_version,
            target=self.target,
            probability=self.probability,
            as_of=self.as_of,
            lower_probability=self.lower_probability,
            upper_probability=self.upper_probability,
            metadata={
                **dict(self.metadata),
                "feature_version": self.feature_version,
                "prediction_role": self.prediction_role.value,
                "fold_id": self.fold_id,
            },
        )


def model_prediction_to_stored_prediction(
    prediction: ModelPrediction,
    *,
    prediction_role: PredictionRole | str,
    feature_version: str,
    fold_id: str | None = None,
    train_start: date | None = None,
    train_end: date | None = None,
    validation_start: date | None = None,
    validation_end: date | None = None,
) -> StoredPrediction:
    return StoredPrediction(
        race_id=prediction.race_id,
        runner_id=prediction.runner_id,
        target=prediction.target,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        prediction_role=prediction_role,
        as_of=prediction.as_of,
        feature_version=feature_version,
        probability=prediction.probability,
        fold_id=fold_id,
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        lower_probability=prediction.lower_probability,
        upper_probability=prediction.upper_probability,
        metadata=prediction.metadata,
    )


def write_prediction_store_csv(
    path: Path | str,
    predictions: Sequence[StoredPrediction],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        predictions,
        key=lambda row: (
            str(row.race_id),
            str(row.runner_id),
            row.target.value,
            str(row.model_name),
            row.model_version,
            row.as_of,
        ),
    )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OOF_PREDICTION_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(stored_prediction_to_csv_row(row) for row in rows)


def read_prediction_store_csv(path: Path | str) -> tuple[StoredPrediction, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(_parse_stored_prediction_row(row) for row in csv.DictReader(handle))


def stored_prediction_to_csv_row(prediction: StoredPrediction) -> dict[str, str]:
    return {
        "race_id": str(prediction.race_id),
        "runner_id": str(prediction.runner_id),
        "target": prediction.target.value,
        "model_name": str(prediction.model_name),
        "model_version": prediction.model_version,
        "prediction_role": prediction.prediction_role.value,
        "fold_id": _render_optional(prediction.fold_id),
        "train_start": _render_optional_date(prediction.train_start),
        "train_end": _render_optional_date(prediction.train_end),
        "validation_start": _render_optional_date(prediction.validation_start),
        "validation_end": _render_optional_date(prediction.validation_end),
        "as_of": prediction.as_of.isoformat(),
        "feature_version": prediction.feature_version,
        "probability": str(prediction.probability),
        "lower_probability": _render_optional(prediction.lower_probability),
        "upper_probability": _render_optional(prediction.upper_probability),
        "metadata_json": json.dumps(
            prediction.metadata,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _parse_stored_prediction_row(row: Mapping[str, str]) -> StoredPrediction:
    metadata = json.loads(row.get("metadata_json") or "{}")
    return StoredPrediction(
        race_id=RaceId(row["race_id"]),
        runner_id=RunnerId(row["runner_id"]),
        target=PredictionTarget(row["target"]),
        model_name=ModelName(row["model_name"]),
        model_version=row["model_version"],
        prediction_role=PredictionRole(row["prediction_role"]),
        fold_id=parse_str_or_none(row.get("fold_id")),
        train_start=_parse_date_or_none(row.get("train_start")),
        train_end=_parse_date_or_none(row.get("train_end")),
        validation_start=_parse_date_or_none(row.get("validation_start")),
        validation_end=_parse_date_or_none(row.get("validation_end")),
        as_of=parse_datetime(row["as_of"]),
        feature_version=row["feature_version"],
        probability=float(row["probability"]),
        lower_probability=parse_float_or_none(row.get("lower_probability")),
        upper_probability=parse_float_or_none(row.get("upper_probability")),
        metadata=metadata,
    )


def _parse_date_or_none(value: str | None) -> date | None:
    normalized = (value or "").strip()
    return parse_date(normalized) if normalized else None


def _render_optional(value: object | None) -> str:
    return "" if value is None else str(value)


def _render_optional_date(value: date | None) -> str:
    return value.isoformat() if value is not None else ""
