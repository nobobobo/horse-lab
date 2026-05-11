import csv
import datetime as dt
import json
from pathlib import Path

from horse_lab.pipelines import run_level0_oof_from_csv
from horse_lab.stacking import PredictionRole, read_prediction_store_csv


class FakeEstimator:
    def __init__(self, probabilities):
        self.probabilities = tuple(probabilities)
        self.fit_x = None
        self.fit_y = None

    def fit(self, x_train, y_train):
        self.fit_x = x_train
        self.fit_y = y_train

    def predict_proba(self, x_values):
        assert len(x_values) == len(self.probabilities)
        return [
            [1.0 - probability, probability]
            for probability in self.probabilities
        ]


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_oof_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    _write(
        dataset_dir / "races.csv",
        """
race_id,race_date,venue,race_number,name,surface,distance_m,direction,track_condition,weather,grade,start_time,field_size
race-jan,2026-01-05,Tokyo,1,Train Race,turf,1200,left,firm,Sunny,,2026-01-05T10:00:00,2
race-feb,2026-02-07,Tokyo,1,Validation Feb,turf,1200,left,firm,Sunny,,2026-02-07T10:00:00,2
race-mar,2026-03-08,Tokyo,1,Validation Mar,turf,1200,left,firm,Sunny,,2026-03-08T10:00:00,2
""",
    )
    _write(
        dataset_dir / "features.csv",
        """
race_id,runner_id,as_of,feature_version,feature__speed,feature__gate,feature__venue,feature__market_win_probability,feature__odds_log
race-jan,jan-1,2026-01-05T09:55:00,fixture-v1,70,1,Tokyo,0.60,0.51
race-jan,jan-2,2026-01-05T09:55:00,fixture-v1,65,2,Tokyo,0.40,0.92
race-feb,feb-1,2026-02-07T09:55:00,fixture-v1,72,1,Tokyo,0.67,0.69
race-feb,feb-2,2026-02-07T09:55:00,fixture-v1,64,2,Tokyo,0.33,1.38
race-mar,mar-1,2026-03-08T09:55:00,fixture-v1,68,1,Tokyo,0.50,1.10
race-mar,mar-2,2026-03-08T09:55:00,fixture-v1,69,2,Tokyo,0.50,1.10
""",
    )
    _write(
        dataset_dir / "results.csv",
        """
race_id,runner_id,finish_position,is_disqualified,is_dead_heat,final_time_seconds,prize_jpy
race-jan,jan-1,1,false,false,70.0,100
race-jan,jan-2,2,false,false,71.0,0
race-feb,feb-1,1,false,false,70.0,100
race-feb,feb-2,2,false,false,71.0,0
race-mar,mar-1,2,false,false,71.0,0
race-mar,mar-2,1,false,false,70.0,100
""",
    )
    _write(
        dataset_dir / "odds_timeseries.csv",
        """
race_id,runner_id,bet_type,captured_at,odds,popularity_rank,pool_size_jpy,source
race-feb,feb-1,win,2026-02-07T09:55:00,2.0,1,1000,fixture
race-feb,feb-2,win,2026-02-07T09:55:00,4.0,2,1000,fixture
race-mar,mar-1,win,2026-03-08T09:55:00,3.0,1,1000,fixture
race-mar,mar-2,win,2026-03-08T09:55:00,3.0,2,1000,fixture
""",
    )


def test_level0_oof_pipeline_generates_monthly_prediction_store(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "oof"
    _write_oof_dataset(dataset_dir)

    result = run_level0_oof_from_csv(
        dataset_dir,
        artifact_dir,
        validation_start_date=dt.date(2026, 2, 1),
        validation_end_date=dt.date(2026, 3, 31),
        as_of=dt.datetime(2026, 3, 31, 23, 59),
        feature_version="fixture-v1",
        model_keys=("market", "lightgbm_no_market"),
        estimator_factory=lambda random_seed: FakeEstimator(probabilities=[0.8, 0.2]),
    )

    assert [fold.fold_id for fold in result.folds] == ["202602", "202603"]
    assert result.predictions_path == artifact_dir / "oof_predictions.csv"
    assert result.report_path == artifact_dir / "oof_report.json"
    assert result.report["counts"] == {
        "stored_predictions": 8,
        "folds": 2,
        "models": 2,
    }

    stored = read_prediction_store_csv(result.predictions_path)
    assert len(stored) == 8
    assert {prediction.prediction_role for prediction in stored} == {PredictionRole.OOF}
    assert {prediction.fold_id for prediction in stored} == {"202602", "202603"}
    assert {
        prediction.model_version for prediction in stored
    } == {"market-implied-oof-v1", "lightgbm-no-market-oof-v1"}

    market_feb = [
        prediction
        for prediction in stored
        if prediction.race_id == "race-feb"
        and prediction.model_version == "market-implied-oof-v1"
    ]
    assert sum(prediction.probability for prediction in market_feb) == 1.0
    assert market_feb[0].validation_start == dt.date(2026, 2, 1)
    assert market_feb[0].train_end == dt.date(2026, 1, 31)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["models"]["market"]["prediction_count"] == 4
    assert report["models"]["lightgbm_no_market"]["prediction_count"] == 4


def test_level0_oof_cli_writes_json_summary(tmp_path, capsys):
    from horse_lab.cli import main

    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "oof"
    _write_oof_dataset(dataset_dir)

    exit_code = main(
        [
            "level0-oof",
            str(dataset_dir),
            str(artifact_dir),
            "--validation-start-date",
            "2026-02-01",
            "--validation-end-date",
            "2026-03-31",
            "--as-of",
            "2026-03-31T23:59:00",
            "--feature-version",
            "fixture-v1",
            "--model-key",
            "market",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_dir"] == str(dataset_dir)
    assert summary["report"]["counts"]["stored_predictions"] == 4
    with (artifact_dir / "oof_predictions.csv").open(encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    assert len(rows) == 4
