"""Small JSON model registry used by Phase 5 paper-trading gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelRegistryWriteResult:
    registry_path: Path
    payload: dict[str, Any]


def build_phase4_model_registry_from_report(
    phase4_report_path: Path | str,
    output_path: Path | str,
    *,
    candidate_method: str = "convex_blend",
    model_version: str = "convex-blend-phase5-v1",
    minimum_log_loss_improvement: float = 0.0,
) -> ModelRegistryWriteResult:
    phase4_path = Path(phase4_report_path)
    report = json.loads(phase4_path.read_text(encoding="utf-8"))
    recommendation = report["recommendation"]
    overall = report["overall"]
    best_level0 = report["best_level0"]
    candidate_metrics = overall[candidate_method]
    improvement = (
        float(best_level0["log_loss"]) - float(candidate_metrics["log_loss"])
    )
    approved = (
        recommendation["best_ensemble_method"] == candidate_method
        and recommendation["action"] == "promote_ensemble_candidate"
        and improvement >= minimum_log_loss_improvement
    )
    payload = {
        "registry_version": "horse-lab-model-registry-v1",
        "created_at": datetime.utcnow().isoformat(),
        "candidate": {
            "method": candidate_method,
            "model_version": model_version,
            "stage": "paper_trading_candidate" if approved else "rejected",
            "target": "win_probability",
        },
        "approval_gate": {
            "approved_for_paper_trading": approved,
            "minimum_log_loss_improvement": minimum_log_loss_improvement,
            "actual_log_loss_improvement": improvement,
            "reason": (
                "ensemble beats best Level 0 on walk-forward log loss"
                if approved
                else "ensemble did not clear walk-forward promotion gate"
            ),
        },
        "source": {
            "phase4_report_path": str(phase4_path),
            "holdout_fold_ids": report["holdout_fold_ids"],
            "feature_columns": report["feature_columns"],
            "market_column": report["market_column"],
        },
        "serving_policy": {
            "type": "restrained_convex_blend",
            "weights_source": "latest_walk_forward_fold",
            "weights": _latest_blend_weights(report),
            "requires_level0_methods": report["feature_columns"],
        },
        "metrics": {
            "candidate": candidate_metrics,
            "best_level0": best_level0,
            "best_overall": report["best_overall"],
            "overall": overall,
        },
        "monitoring": {
            "primary_metrics": [
                "log_loss",
                "brier_score",
                "expected_calibration_error",
                "clv_implied_probability_delta",
                "roi",
                "max_drawdown",
            ],
            "segment_dimensions": [
                "venue",
                "surface",
                "distance_bucket",
                "field_size_bucket",
                "market_probability_band",
            ],
            "paper_trading_required_before_live": True,
        },
    }
    registry_path = Path(output_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ModelRegistryWriteResult(
        registry_path=registry_path,
        payload=payload,
    )


def model_registry_to_dict(result: ModelRegistryWriteResult) -> dict[str, Any]:
    return {
        "registry_path": str(result.registry_path),
        "registry": result.payload,
    }


def _latest_blend_weights(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds = report["folds_detail"]
    if not folds:
        return []
    latest = sorted(folds, key=lambda row: row["holdout_fold_id"])[-1]
    return latest["blend_weights"]
