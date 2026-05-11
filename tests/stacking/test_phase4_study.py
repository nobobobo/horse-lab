import csv
import json
from pathlib import Path

from horse_lab.stacking import run_phase4_study_from_csv


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_meta_features(path: Path) -> None:
    _write(
        path,
        """
race_id,runner_id,target,label,fold_id,validation_start,validation_end,as_of,feature_version,pred__market_implied_probability__v1,pred__form__v1
race-1,runner-1,win_probability,1,202601,2026-01-01,2026-01-31,2026-04-30T23:59:00,fixture-v1,0.70,0.60
race-1,runner-2,win_probability,0,202601,2026-01-01,2026-01-31,2026-04-30T23:59:00,fixture-v1,0.30,0.40
race-2,runner-1,win_probability,0,202602,2026-02-01,2026-02-28,2026-04-30T23:59:00,fixture-v1,0.45,0.30
race-2,runner-2,win_probability,1,202602,2026-02-01,2026-02-28,2026-04-30T23:59:00,fixture-v1,0.55,0.70
race-3,runner-1,win_probability,1,202603,2026-03-01,2026-03-31,2026-04-30T23:59:00,fixture-v1,0.60,0.80
race-3,runner-2,win_probability,0,202603,2026-03-01,2026-03-31,2026-04-30T23:59:00,fixture-v1,0.40,0.20
race-4,runner-1,win_probability,0,202604,2026-04-01,2026-04-30,2026-04-30T23:59:00,fixture-v1,0.55,0.40
race-4,runner-2,win_probability,1,202604,2026-04-01,2026-04-30,2026-04-30T23:59:00,fixture-v1,0.45,0.60
""",
    )


def _write_races(path: Path) -> None:
    _write(
        path,
        """
race_id,race_date,venue,race_number,name,surface,distance_m,direction,track_condition,weather,grade,start_time,field_size
race-1,2026-01-05,Tokyo,1,Race One,turf,1200,left,firm,Sunny,,2026-01-05T10:00:00,2
race-2,2026-02-07,Kyoto,1,Race Two,dirt,1800,right,firm,Sunny,,2026-02-07T10:00:00,2
race-3,2026-03-08,Tokyo,1,Race Three,turf,1600,left,firm,Sunny,,2026-03-08T10:00:00,2
race-4,2026-04-12,Hanshin,1,Race Four,turf,2000,right,firm,Sunny,,2026-04-12T10:00:00,2
""",
    )


def test_phase4_study_writes_walkforward_and_segment_artifacts(tmp_path):
    meta_features_path = tmp_path / "meta_features.csv"
    races_path = tmp_path / "races.csv"
    artifact_dir = tmp_path / "phase4"
    _write_meta_features(meta_features_path)
    _write_races(races_path)

    result = run_phase4_study_from_csv(
        meta_features_path,
        races_path,
        artifact_dir,
        min_train_folds=2,
        blend_grid_step=0.5,
        logistic_max_iterations=50,
    )

    assert result.report_path == artifact_dir / "phase4_study_report.json"
    assert result.predictions_path == artifact_dir / "walkforward_predictions.csv"
    assert result.segment_metrics_path == artifact_dir / "segment_metrics.csv"
    assert result.report["holdout_fold_ids"] == ["202603", "202604"]
    assert result.report["counts"]["holdout_folds"] == 2
    assert result.report["recommendation"]["action"] in {
        "keep_best_level0",
        "promote_ensemble_candidate",
    }

    with result.predictions_path.open(encoding="utf-8") as handle:
        prediction_rows = tuple(csv.DictReader(handle))
    assert {row["method"] for row in prediction_rows} == {
        "convex_blend",
        "logistic_meta",
        "pred__form__v1",
        "pred__market_implied_probability__v1",
    }

    with result.segment_metrics_path.open(encoding="utf-8") as handle:
        segment_rows = tuple(csv.DictReader(handle))
    assert ("all", "all") in {
        (row["segment_name"], row["segment_value"]) for row in segment_rows
    }
    assert ("venue", "Tokyo") in {
        (row["segment_name"], row["segment_value"]) for row in segment_rows
    }

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["best_overall"]["method"]


def test_phase4_study_cli(tmp_path, capsys):
    from horse_lab.cli import main

    meta_features_path = tmp_path / "meta_features.csv"
    races_path = tmp_path / "races.csv"
    artifact_dir = tmp_path / "phase4"
    _write_meta_features(meta_features_path)
    _write_races(races_path)

    exit_code = main(
        [
            "stacking-phase4-study",
            str(meta_features_path),
            str(races_path),
            str(artifact_dir),
            "--min-train-folds",
            "2",
            "--blend-grid-step",
            "0.5",
            "--logistic-max-iterations",
            "20",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["counts"]["holdout_folds"] == 2
