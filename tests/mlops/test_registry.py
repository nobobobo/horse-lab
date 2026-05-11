import json
from pathlib import Path

from horse_lab.mlops import build_phase4_model_registry_from_report


def _write_phase4_report(path: Path) -> None:
    payload = {
        "feature_columns": ["pred__market", "pred__form"],
        "market_column": "pred__market",
        "holdout_fold_ids": ["202604", "202605"],
        "folds_detail": [
            {
                "holdout_fold_id": "202604",
                "blend_weights": [
                    {"feature_column": "pred__market", "weight": 1.0},
                    {"feature_column": "pred__form", "weight": 0.0},
                ],
            },
            {
                "holdout_fold_id": "202605",
                "blend_weights": [
                    {"feature_column": "pred__market", "weight": 0.95},
                    {"feature_column": "pred__form", "weight": 0.05},
                ],
            },
        ],
        "overall": {
            "convex_blend": {"log_loss": 0.20, "brier_score": 0.05},
            "pred__market": {"log_loss": 0.21, "brier_score": 0.051},
        },
        "best_level0": {
            "method": "pred__market",
            "log_loss": 0.21,
            "brier_score": 0.051,
        },
        "best_overall": {
            "method": "convex_blend",
            "log_loss": 0.20,
            "brier_score": 0.05,
        },
        "recommendation": {
            "action": "promote_ensemble_candidate",
            "best_ensemble_method": "convex_blend",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_phase4_model_registry_marks_candidate_for_paper_trading(tmp_path):
    report_path = tmp_path / "phase4_report.json"
    registry_path = tmp_path / "registry" / "model_registry.json"
    _write_phase4_report(report_path)

    result = build_phase4_model_registry_from_report(
        report_path,
        registry_path,
        model_version="fixture-blend-v1",
    )

    assert result.registry_path == registry_path
    assert result.payload["candidate"]["stage"] == "paper_trading_candidate"
    assert result.payload["candidate"]["model_version"] == "fixture-blend-v1"
    assert result.payload["approval_gate"]["approved_for_paper_trading"] is True
    assert result.payload["serving_policy"]["weights"] == [
        {"feature_column": "pred__market", "weight": 0.95},
        {"feature_column": "pred__form", "weight": 0.05},
    ]
    loaded = json.loads(registry_path.read_text(encoding="utf-8"))
    assert loaded["registry_version"] == "horse-lab-model-registry-v1"


def test_model_registry_cli(tmp_path, capsys):
    from horse_lab.cli import main

    report_path = tmp_path / "phase4_report.json"
    registry_path = tmp_path / "registry" / "model_registry.json"
    _write_phase4_report(report_path)

    exit_code = main(
        [
            "model-registry-register-phase4",
            str(report_path),
            str(registry_path),
            "--model-version",
            "fixture-blend-v1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["registry"]["candidate"]["stage"] == "paper_trading_candidate"
