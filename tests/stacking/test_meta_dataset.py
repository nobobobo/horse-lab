import csv
import datetime as dt
import json
from pathlib import Path

from horse_lab.schemas import (
    ModelName,
    PredictionTarget,
    RaceId,
    RunnerId,
)
from horse_lab.stacking import (
    PredictionRole,
    StoredPrediction,
    build_meta_dataset_from_csv,
    model_prediction_column,
    write_prediction_store_csv,
)


def _stored_prediction(**overrides):
    values = {
        "race_id": RaceId("race-1"),
        "runner_id": RunnerId("runner-1"),
        "target": PredictionTarget.WIN_PROBABILITY,
        "model_name": ModelName("market_implied_probability"),
        "model_version": "market-implied-oof-v1",
        "prediction_role": PredictionRole.OOF,
        "fold_id": "202603",
        "train_start": dt.date(2025, 5, 10),
        "train_end": dt.date(2026, 2, 28),
        "validation_start": dt.date(2026, 3, 1),
        "validation_end": dt.date(2026, 3, 31),
        "as_of": dt.datetime(2026, 5, 10, 23, 59),
        "feature_version": "jravan-replay-v2",
        "probability": 0.21,
    }
    values.update(overrides)
    return StoredPrediction(**values)


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_build_meta_dataset_from_prediction_store(tmp_path):
    predictions_path = tmp_path / "oof_predictions.csv"
    results_path = tmp_path / "results.csv"
    output_dir = tmp_path / "meta"
    lightgbm_prediction = _stored_prediction(
        model_name=ModelName("lightgbm_win_probability"),
        model_version="lightgbm-no-market-oof-v1",
        probability=0.18,
    )
    write_prediction_store_csv(
        predictions_path,
        [
            _stored_prediction(),
            lightgbm_prediction,
            _stored_prediction(
                race_id=RaceId("race-2"),
                runner_id=RunnerId("runner-2"),
                probability=0.11,
            ),
        ],
    )
    _write(
        results_path,
        """
race_id,runner_id,finish_position,is_disqualified,is_dead_heat,final_time_seconds,prize_jpy
race-1,runner-1,1,false,false,70.0,100
race-2,runner-2,2,false,false,71.0,0
""",
    )

    result = build_meta_dataset_from_csv(
        predictions_path,
        results_path,
        output_dir,
    )

    assert result.meta_dataset_path == output_dir / "meta_features.csv"
    assert result.report["counts"]["meta_rows"] == 1
    assert result.report["counts"]["dropped_incomplete_rows"] == 1
    assert result.report["labels"] == {"positives": 1, "empirical_rate": 1.0}

    with result.meta_dataset_path.open(encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["race_id"] == "race-1"
    assert row["label"] == "1"
    assert row["fold_id"] == "202603"
    assert row[model_prediction_column(lightgbm_prediction)] == "0.18"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["model_columns"] == result.report["model_columns"]
