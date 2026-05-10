import datetime as dt
import json
from pathlib import Path

import pytest

from horse_lab.schemas import (
    ModelName,
    ModelPrediction,
    PredictionTarget,
    RaceId,
    RunnerId,
)
from horse_lab.stacking import (
    PredictionRole,
    StoredPrediction,
    model_prediction_to_stored_prediction,
    read_prediction_store_csv,
    stored_prediction_to_csv_row,
    write_prediction_store_csv,
)


def _stored_prediction(**overrides):
    values = {
        "race_id": RaceId("2026050805010101"),
        "runner_id": RunnerId("2026050805010101-01"),
        "target": PredictionTarget.WIN_PROBABILITY,
        "model_name": ModelName("lightgbm_form"),
        "model_version": "v1",
        "prediction_role": PredictionRole.OOF,
        "fold_id": "fold-1",
        "train_start": dt.date(2025, 5, 10),
        "train_end": dt.date(2026, 2, 28),
        "validation_start": dt.date(2026, 3, 1),
        "validation_end": dt.date(2026, 3, 31),
        "as_of": dt.datetime(2026, 5, 8, 9, 55),
        "feature_version": "jravan-replay-v2",
        "probability": 0.12,
        "lower_probability": 0.08,
        "upper_probability": 0.18,
        "metadata": {"scenario": "full"},
    }
    values.update(overrides)
    return StoredPrediction(**values)


def test_stored_prediction_renders_csv_row_with_fold_metadata():
    row = stored_prediction_to_csv_row(_stored_prediction())

    assert row["prediction_role"] == "oof"
    assert row["fold_id"] == "fold-1"
    assert row["train_end"] == "2026-02-28"
    assert row["feature_version"] == "jravan-replay-v2"
    assert json.loads(row["metadata_json"]) == {"scenario": "full"}


def test_prediction_store_round_trips_csv(tmp_path: Path):
    path = tmp_path / "predictions" / "oof_predictions.csv"
    prediction = _stored_prediction()

    write_prediction_store_csv(path, [prediction])

    loaded = read_prediction_store_csv(path)
    assert loaded == (prediction,)
    assert loaded[0].to_model_prediction().probability == 0.12


def test_model_prediction_can_be_wrapped_for_oof_store():
    prediction = ModelPrediction(
        race_id=RaceId("2026050805010101"),
        runner_id=RunnerId("2026050805010101-01"),
        model_name=ModelName("market"),
        model_version="market-v1",
        target=PredictionTarget.WIN_PROBABILITY,
        probability=0.2,
        as_of=dt.datetime(2026, 5, 8, 9, 55),
        metadata={"raw": 0.19},
    )

    stored = model_prediction_to_stored_prediction(
        prediction,
        prediction_role=PredictionRole.HOLDOUT,
        feature_version="fixture-v1",
        fold_id="holdout-202603",
    )

    assert stored.prediction_role == PredictionRole.HOLDOUT
    assert stored.feature_version == "fixture-v1"
    assert stored.metadata == {"raw": 0.19}


def test_stored_prediction_rejects_invalid_probability():
    with pytest.raises(ValueError, match="probability"):
        _stored_prediction(probability=1.5)
