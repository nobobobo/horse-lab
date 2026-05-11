import csv
import datetime as dt
import json
from pathlib import Path

from horse_lab.backtesting import BacktestConfig
from horse_lab.betting import KellyConfig
from horse_lab.pipelines import run_daily_paper_trading_from_csv


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
        return [[1.0 - probability, probability] for probability in self.probabilities]


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    _write(
        dataset_dir / "races.csv",
        """
race_id,race_date,venue,race_number,name,surface,distance_m,direction,track_condition,weather,grade,start_time,field_size
race-train,2026-04-30,Tokyo,1,Train Race,turf,1200,left,firm,Sunny,,2026-04-30T10:00:00,2
race-target-1,2026-05-01,Tokyo,1,Target One,turf,1200,left,firm,Sunny,,2026-05-01T10:00:00,2
race-target-2,2026-05-02,Tokyo,1,Target Two,turf,1600,left,firm,Sunny,,2026-05-02T10:00:00,2
""",
    )
    _write(
        dataset_dir / "features.csv",
        """
race_id,runner_id,as_of,feature_version,feature__speed,feature__gate,feature__venue,feature__market_win_probability,feature__odds_log
race-train,train-1,2026-04-30T09:55:00,fixture-v1,70,1,Tokyo,0.60,0.51
race-train,train-2,2026-04-30T09:55:00,fixture-v1,65,2,Tokyo,0.40,0.92
race-target-1,target-1,2026-05-01T09:55:00,fixture-v1,72,1,Tokyo,0.67,0.69
race-target-1,target-2,2026-05-01T09:55:00,fixture-v1,64,2,Tokyo,0.33,1.38
race-target-2,target-3,2026-05-02T09:55:00,fixture-v1,68,1,Tokyo,0.50,1.10
race-target-2,target-4,2026-05-02T09:55:00,fixture-v1,69,2,Tokyo,0.50,1.10
""",
    )
    _write(
        dataset_dir / "results.csv",
        """
race_id,runner_id,finish_position,is_disqualified,is_dead_heat,final_time_seconds,prize_jpy
race-train,train-1,1,false,false,70.0,100
race-train,train-2,2,false,false,71.0,0
race-target-1,target-1,1,false,false,70.0,100
race-target-1,target-2,2,false,false,71.0,0
race-target-2,target-3,2,false,false,71.0,0
race-target-2,target-4,1,false,false,70.0,100
""",
    )
    _write(
        dataset_dir / "odds_timeseries.csv",
        """
race_id,runner_id,bet_type,captured_at,odds,popularity_rank,pool_size_jpy,source
race-target-1,target-1,win,2026-05-01T09:55:00,2.0,1,1000,fixture
race-target-1,target-2,win,2026-05-01T09:55:00,4.0,2,1000,fixture
race-target-2,target-3,win,2026-05-02T09:55:00,3.0,1,1000,fixture
race-target-2,target-4,win,2026-05-02T09:55:00,3.0,2,1000,fixture
""",
    )


def _write_registry(path: Path) -> None:
    payload = {
        "candidate": {
            "method": "convex_blend",
            "model_version": "convex-blend-fixture-v1",
        },
        "approval_gate": {"approved_for_paper_trading": True},
        "serving_policy": {
            "type": "restrained_convex_blend",
            "weights": [
                {
                    "feature_column": (
                        "pred__market_implied_probability__market_implied_oof_v1"
                    ),
                    "weight": 0.50,
                },
                {
                    "feature_column": (
                        "pred__lightgbm_win_probability__lightgbm_full_oof_v1"
                    ),
                    "weight": 0.50,
                },
                {
                    "feature_column": (
                        "pred__lightgbm_win_probability__lightgbm_no_market_oof_v1"
                    ),
                    "weight": 0.0,
                },
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_daily_paper_trading_builds_level0_blend_and_paper_artifacts(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "daily"
    registry_path = tmp_path / "registry" / "model_registry.json"
    _write_dataset(dataset_dir)
    _write_registry(registry_path)

    result = run_daily_paper_trading_from_csv(
        dataset_dir,
        registry_path,
        artifact_dir,
        start_date=dt.date(2026, 5, 1),
        end_date=dt.date(2026, 5, 2),
        as_of=dt.datetime(2026, 5, 3, 23, 59),
        feature_version="fixture-v1",
        backtest_config=BacktestConfig(
            initial_bankroll_jpy=100_000,
            kelly_config=KellyConfig(minimum_edge=0.0),
        ),
        estimator_factory=lambda random_seed: FakeEstimator(
            probabilities=[0.8, 0.2, 0.4, 0.6]
        ),
    )

    assert result.report["counts"]["train_races"] == 1
    assert result.report["counts"]["target_races"] == 2
    assert result.report["counts"]["candidate_predictions"] == 4
    assert result.report["paper_trading"]["counts"]["predictions"] == 4
    assert result.level0_predictions_path.exists()
    assert result.candidate_predictions_path.exists()
    assert result.paper_result.report_path.exists()

    with result.level0_predictions_path.open(encoding="utf-8") as handle:
        level0_rows = tuple(csv.DictReader(handle))
    assert len(level0_rows) == 12
    assert {
        row["method"]
        for row in level0_rows
    } == {
        "pred__market_implied_probability__market_implied_oof_v1",
        "pred__lightgbm_win_probability__lightgbm_full_oof_v1",
        "pred__lightgbm_win_probability__lightgbm_no_market_oof_v1",
    }

    with result.candidate_predictions_path.open(encoding="utf-8") as handle:
        candidate_rows = tuple(csv.DictReader(handle))
    assert {row["method"] for row in candidate_rows} == {"convex_blend"}
    for race_id in {"race-target-1", "race-target-2"}:
        total = sum(
            float(row["probability"])
            for row in candidate_rows
            if row["race_id"] == race_id
        )
        assert total == 1.0


def test_daily_paper_trading_cli(tmp_path, capsys, monkeypatch):
    from horse_lab import cli

    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "daily"
    registry_path = tmp_path / "registry" / "model_registry.json"
    _write_dataset(dataset_dir)
    _write_registry(registry_path)

    monkeypatch.setattr(
        cli,
        "run_daily_paper_trading_from_csv",
        lambda *args, **kwargs: run_daily_paper_trading_from_csv(
            *args,
            **kwargs,
            estimator_factory=lambda random_seed: FakeEstimator(
                probabilities=[0.8, 0.2, 0.4, 0.6]
            ),
        ),
    )

    exit_code = cli.main(
        [
            "daily-paper-trading-run",
            str(dataset_dir),
            str(registry_path),
            str(artifact_dir),
            "--start-date",
            "2026-05-01",
            "--end-date",
            "2026-05-02",
            "--as-of",
            "2026-05-03T23:59:00",
            "--feature-version",
            "fixture-v1",
            "--minimum-edge",
            "0.0",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["counts"]["candidate_predictions"] == 4
    assert Path(summary["report_path"]).exists()
