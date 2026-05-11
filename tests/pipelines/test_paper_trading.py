import csv
import datetime as dt
import json
from pathlib import Path

from horse_lab.backtesting import BacktestConfig
from horse_lab.betting import KellyConfig
from horse_lab.pipelines import run_paper_trading_from_csv


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    _write(
        dataset_dir / "races.csv",
        """
race_id,race_date,venue,race_number,name,surface,distance_m,direction,track_condition,weather,grade,start_time,field_size
race-1,2026-05-01,Tokyo,1,Race One,turf,1200,left,firm,Sunny,,2026-05-01T10:00:00,2
race-2,2026-05-02,Tokyo,1,Race Two,turf,1600,left,firm,Sunny,,2026-05-02T10:00:00,2
""",
    )
    _write(
        dataset_dir / "results.csv",
        """
race_id,runner_id,finish_position,is_disqualified,is_dead_heat,final_time_seconds,prize_jpy
race-1,r1,1,false,false,70.0,100
race-1,r2,2,false,false,71.0,0
race-2,r3,2,false,false,71.0,0
race-2,r4,1,false,false,70.0,100
""",
    )
    _write(
        dataset_dir / "odds_timeseries.csv",
        """
race_id,runner_id,bet_type,captured_at,odds,popularity_rank,pool_size_jpy,source
race-1,r1,win,2026-05-01T09:50:00,2.2,1,900,fixture
race-1,r1,win,2026-05-01T10:00:00,2.0,1,1000,fixture
race-1,r2,win,2026-05-01T10:00:00,4.0,2,1000,fixture
race-2,r3,win,2026-05-02T10:00:00,2.5,1,1000,fixture
race-2,r4,win,2026-05-02T10:00:00,3.5,2,1000,fixture
""",
    )


def _write_predictions(path: Path) -> None:
    _write(
        path,
        """
race_id,runner_id,fold_id,label,method,probability
race-1,r1,202605,1,convex_blend,0.75
race-1,r2,202605,0,convex_blend,0.25
race-2,r3,202605,0,convex_blend,0.40
race-2,r4,202605,1,convex_blend,0.60
race-1,r1,202605,1,market,0.60
race-1,r2,202605,0,market,0.40
""",
    )


def test_paper_trading_replay_writes_monitoring_artifacts(tmp_path):
    dataset_dir = tmp_path / "dataset"
    predictions_path = tmp_path / "walkforward_predictions.csv"
    artifact_dir = tmp_path / "paper"
    _write_dataset(dataset_dir)
    _write_predictions(predictions_path)

    result = run_paper_trading_from_csv(
        predictions_path,
        dataset_dir,
        artifact_dir,
        method="convex_blend",
        as_of=dt.datetime(2026, 5, 3, 23, 59),
        model_version="fixture-paper-v1",
        backtest_config=BacktestConfig(
            initial_bankroll_jpy=100_000,
            kelly_config=KellyConfig(minimum_edge=0.0),
        ),
    )

    assert result.report["counts"]["predictions"] == 4
    assert result.report["counts"]["bet_decisions"] == 4
    assert result.report["decision_summary"]["did_bet"] > 0
    assert result.report["probability"]["log_loss"] > 0.0
    assert result.predictions_path.exists()
    assert result.clv_report_path.exists()
    assert result.bet_decisions_path.exists()

    with result.clv_report_path.open(encoding="utf-8") as handle:
        clv_rows = tuple(csv.DictReader(handle))
    assert len(clv_rows) == 4

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["method"] == "convex_blend"


def test_paper_trading_cli(tmp_path, capsys):
    from horse_lab.cli import main

    dataset_dir = tmp_path / "dataset"
    predictions_path = tmp_path / "walkforward_predictions.csv"
    artifact_dir = tmp_path / "paper"
    _write_dataset(dataset_dir)
    _write_predictions(predictions_path)

    exit_code = main(
        [
            "paper-trading-run",
            str(predictions_path),
            str(dataset_dir),
            str(artifact_dir),
            "--method",
            "convex_blend",
            "--as-of",
            "2026-05-03T23:59:00",
            "--minimum-edge",
            "0.0",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["counts"]["predictions"] == 4
    assert Path(summary["report_path"]).exists()
