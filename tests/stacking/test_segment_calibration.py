import csv
import json
from pathlib import Path

from horse_lab.stacking import run_segment_calibration_study_from_csv


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_meta_features(path: Path) -> None:
    _write(
        path,
        """
race_id,runner_id,target,label,fold_id,validation_start,validation_end,as_of,feature_version,pred__market_implied_probability__v1
race-1,runner-1,win_probability,1,202601,2026-01-01,2026-01-31,2026-04-30T23:59:00,fixture-v1,0.70
race-1,runner-2,win_probability,0,202601,2026-01-01,2026-01-31,2026-04-30T23:59:00,fixture-v1,0.30
race-2,runner-1,win_probability,0,202602,2026-02-01,2026-02-28,2026-04-30T23:59:00,fixture-v1,0.55
race-2,runner-2,win_probability,1,202602,2026-02-01,2026-02-28,2026-04-30T23:59:00,fixture-v1,0.45
race-3,runner-1,win_probability,1,202603,2026-03-01,2026-03-31,2026-04-30T23:59:00,fixture-v1,0.60
race-3,runner-2,win_probability,0,202603,2026-03-01,2026-03-31,2026-04-30T23:59:00,fixture-v1,0.40
race-4,runner-1,win_probability,0,202604,2026-04-01,2026-04-30,2026-04-30T23:59:00,fixture-v1,0.40
race-4,runner-2,win_probability,1,202604,2026-04-01,2026-04-30,2026-04-30T23:59:00,fixture-v1,0.60
""",
    )


def test_segment_calibration_study_writes_walk_forward_artifacts(tmp_path):
    meta_features_path = tmp_path / "meta_features.csv"
    artifact_dir = tmp_path / "segment_calibration"
    _write_meta_features(meta_features_path)

    result = run_segment_calibration_study_from_csv(
        meta_features_path,
        artifact_dir,
        min_train_folds=2,
        min_segment_train_rows=2,
        max_iterations=100,
        learning_rate=0.05,
    )

    assert result.report_path == artifact_dir / "segment_calibration_report.json"
    assert result.predictions_path == artifact_dir / "segment_calibrated_predictions.csv"
    assert result.report["segment_name"] == "market_probability_band"
    assert result.report["counts"]["evaluated_folds"] == 2
    assert result.report["overall"]["segment_calibrated"]["observations"] == 4

    with result.predictions_path.open(encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"segment_calibrated"}
    for race_id in {"race-3", "race-4"}:
        total = sum(float(row["probability"]) for row in rows if row["race_id"] == race_id)
        assert total == 1.0

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["holdout_fold_ids"] == ["202603", "202604"]


def test_segment_calibration_study_cli(tmp_path, capsys):
    from horse_lab.cli import main

    meta_features_path = tmp_path / "meta_features.csv"
    artifact_dir = tmp_path / "segment_calibration"
    _write_meta_features(meta_features_path)

    exit_code = main(
        [
            "stacking-segment-calibration-study",
            str(meta_features_path),
            str(artifact_dir),
            "--min-train-folds",
            "2",
            "--min-segment-train-rows",
            "2",
            "--max-iterations",
            "50",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["counts"]["evaluated_folds"] == 2
    assert Path(summary["predictions_path"]).exists()
