import csv
import json
from pathlib import Path

import pytest

from horse_lab.stacking import train_logistic_meta_learner_from_csv


def _write(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_meta_features(path: Path) -> None:
    _write(
        path,
        """
race_id,runner_id,target,label,fold_id,validation_start,validation_end,as_of,feature_version,pred__market__v1,pred__form__v1
race-1,runner-1,win_probability,1,202601,2026-01-01,2026-01-31,2026-03-31T23:59:00,fixture-v1,0.70,0.60
race-1,runner-2,win_probability,0,202601,2026-01-01,2026-01-31,2026-03-31T23:59:00,fixture-v1,0.30,0.40
race-2,runner-1,win_probability,0,202602,2026-02-01,2026-02-28,2026-03-31T23:59:00,fixture-v1,0.45,0.30
race-2,runner-2,win_probability,1,202602,2026-02-01,2026-02-28,2026-03-31T23:59:00,fixture-v1,0.55,0.70
race-3,runner-1,win_probability,1,202603,2026-03-01,2026-03-31,2026-03-31T23:59:00,fixture-v1,0.60,0.80
race-3,runner-2,win_probability,0,202603,2026-03-01,2026-03-31,2026-03-31T23:59:00,fixture-v1,0.40,0.20
""",
    )


def test_train_logistic_meta_learner_uses_latest_fold_as_holdout(tmp_path):
    meta_features_path = tmp_path / "meta_features.csv"
    artifact_dir = tmp_path / "meta_model"
    _write_meta_features(meta_features_path)

    result = train_logistic_meta_learner_from_csv(
        meta_features_path,
        artifact_dir,
        learning_rate=0.1,
        max_iterations=200,
        l2=0.0,
    )

    assert result.model_path == artifact_dir / "meta_model.json"
    assert result.predictions_path == artifact_dir / "meta_predictions.csv"
    assert result.report["holdout_fold_id"] == "202603"
    assert result.report["counts"] == {
        "rows": 6,
        "train_rows": 4,
        "holdout_rows": 2,
        "features": 2,
    }
    assert result.report["holdout_probability"]["observations"] == 2
    assert len(result.report["coefficients"]) == 2

    with result.predictions_path.open(encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    assert len(rows) == 2
    assert sum(float(row["probability"]) for row in rows) == pytest.approx(1.0)
    assert {row["fold_id"] for row in rows} == {"202603"}

    model_payload = json.loads(result.model_path.read_text(encoding="utf-8"))
    assert model_payload["feature_columns"] == ["pred__form__v1", "pred__market__v1"]


def test_stacking_train_meta_cli_writes_artifacts(tmp_path, capsys):
    from horse_lab.cli import main

    meta_features_path = tmp_path / "meta_features.csv"
    artifact_dir = tmp_path / "meta_model"
    _write_meta_features(meta_features_path)

    exit_code = main(
        [
            "stacking-train-meta",
            str(meta_features_path),
            str(artifact_dir),
            "--holdout-fold-id",
            "202602",
            "--max-iterations",
            "50",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["holdout_fold_id"] == "202602"
    assert Path(summary["model_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
