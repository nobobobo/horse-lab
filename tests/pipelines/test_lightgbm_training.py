import datetime as dt
import json
from pathlib import Path

import pytest

from horse_lab.pipelines import (
    run_lightgbm_ablation_from_csv,
    run_lightgbm_training_from_csv,
)


class FakeEstimator:
    def __init__(self, probabilities):
        self.probabilities = tuple(probabilities)
        self.fit_x = None
        self.fit_y = None
        self.feature_importances_ = [2.0, 10.0, 1.0]

    def fit(self, x_train, y_train):
        self.fit_x = x_train
        self.fit_y = y_train

    def predict_proba(self, x_values):
        assert len(x_values) == len(self.probabilities)
        return [[1.0 - probability, probability] for probability in self.probabilities]


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_replay_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    _write(
        dataset_dir / "races.csv",
        """
race_id,race_date,venue,race_number,name,surface,distance_m,direction,track_condition,weather,grade,start_time,field_size
race-train,2026-05-07,Tokyo,1,Train Race,turf,1200,left,firm,Sunny,,2026-05-07T10:00:00,2
race-valid,2026-05-08,Tokyo,1,Valid Race,turf,1200,left,firm,Sunny,,2026-05-08T10:00:00,2
""",
    )
    _write(
        dataset_dir / "features.csv",
        """
race_id,runner_id,as_of,feature_version,feature__speed,feature__gate,feature__venue
race-train,train-1,2026-05-07T09:55:00,fixture-v1,70,1,Tokyo
race-train,train-2,2026-05-07T09:55:00,fixture-v1,65,2,Tokyo
race-valid,valid-1,2026-05-08T09:55:00,fixture-v1,72,1,Tokyo
race-valid,valid-2,2026-05-08T09:55:00,fixture-v1,64,2,Tokyo
""",
    )
    _write(
        dataset_dir / "results.csv",
        """
race_id,runner_id,finish_position,is_disqualified,is_dead_heat,final_time_seconds,prize_jpy
race-train,train-1,1,false,false,70.0,100
race-train,train-2,2,false,false,71.0,0
race-valid,valid-1,1,false,false,70.0,100
race-valid,valid-2,2,false,false,71.0,0
""",
    )
    _write(
        dataset_dir / "odds.csv",
        """
race_id,runner_id,bet_type,captured_at,odds,popularity_rank,pool_size_jpy,source
race-valid,valid-1,win,2026-05-08T09:55:00,2.0,1,1000,fixture
race-valid,valid-2,win,2026-05-08T09:55:00,3.0,2,1000,fixture
""",
    )


def test_lightgbm_training_pipeline_fits_evaluates_and_saves_artifacts(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "artifacts"
    _write_replay_dataset(dataset_dir)
    _write(
        dataset_dir / "odds_timeseries.csv",
        """
race_id,runner_id,bet_type,captured_at,odds,popularity_rank,pool_size_jpy,source
race-valid,valid-1,win,2026-05-08T09:45:00,2.4,1,900,fixture
race-valid,valid-1,win,2026-05-08T09:55:00,2.0,1,1000,fixture
race-valid,valid-2,win,2026-05-08T09:55:00,3.0,2,1000,fixture
""",
    )
    fake = FakeEstimator(probabilities=[0.8, 0.2])

    result = run_lightgbm_training_from_csv(
        dataset_dir,
        artifact_dir,
        train_end_date=dt.date(2026, 5, 7),
        valid_start_date=dt.date(2026, 5, 8),
        valid_end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 23, 59),
        feature_version="fixture-v1",
        estimator_factory=lambda random_seed: fake,
    )

    assert fake.fit_y == [1, 0]
    assert len(fake.fit_x) == 2
    assert len(result.validation_odds) == 3
    assert len(result.predictions) == 2
    assert result.probability_summary.observations == 2
    assert result.probability_summary.positives == 1
    assert result.probability_summary.log_loss > 0.0

    assert (artifact_dir / "model" / "lightgbm_model.json").exists()
    assert (artifact_dir / "model" / "lightgbm_estimator.pkl").exists()
    assert (artifact_dir / "feature_importance.csv").exists()
    summary = json.loads((artifact_dir / "evaluation_summary.json").read_text())
    assert summary["counts"]["train_feature_rows"] == 2
    assert summary["counts"]["validation_feature_rows"] == 2
    assert summary["probability"]["brier_score"] > 0.0
    assert summary["artifact"]["feature_importance_path"] == str(
        artifact_dir / "feature_importance.csv"
    )
    assert summary["feature_importances"][0]["feature_name"] == "speed"
    assert summary["feature_importances"][0]["split_importance"] == 10.0


def test_lightgbm_training_pipeline_can_exclude_features(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "artifacts"
    _write_replay_dataset(dataset_dir)
    fake = FakeEstimator(probabilities=[0.8, 0.2])

    result = run_lightgbm_training_from_csv(
        dataset_dir,
        artifact_dir,
        train_end_date=dt.date(2026, 5, 7),
        valid_start_date=dt.date(2026, 5, 8),
        valid_end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 23, 59),
        feature_version="fixture-v1",
        estimator_factory=lambda random_seed: fake,
        exclude_feature_names=("venue",),
    )

    assert fake.fit_x == [[1.0, 70.0], [2.0, 65.0]]
    assert {row["feature_name"] for row in result.feature_importances} == {
        "gate",
        "speed",
    }


def test_lightgbm_ablation_pipeline_writes_aggregate_summary(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "ablation"
    _write_replay_dataset(dataset_dir)

    summary = run_lightgbm_ablation_from_csv(
        dataset_dir,
        artifact_dir,
        train_end_date=dt.date(2026, 5, 7),
        valid_start_date=dt.date(2026, 5, 8),
        valid_end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 23, 59),
        feature_version="fixture-v1",
        scenarios={"full": (), "no_gate": ("gate",)},
        estimator_factory=lambda random_seed: FakeEstimator(probabilities=[0.8, 0.2]),
    )

    assert set(summary["scenarios"]) == {"full", "no_gate"}
    assert summary["scenarios"]["no_gate"]["excluded_feature_names"] == ["gate"]
    assert (artifact_dir / "ablation_summary.json").exists()


def test_lightgbm_training_pipeline_rejects_overlapping_split(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _write_replay_dataset(dataset_dir)

    with pytest.raises(ValueError, match="train_end_date must be before"):
        run_lightgbm_training_from_csv(
            dataset_dir,
            tmp_path / "artifacts",
            train_end_date=dt.date(2026, 5, 8),
            valid_start_date=dt.date(2026, 5, 8),
            valid_end_date=dt.date(2026, 5, 8),
            as_of=dt.datetime(2026, 5, 8, 23, 59),
            feature_version="fixture-v1",
            estimator_factory=lambda random_seed: FakeEstimator(probabilities=[]),
        )
