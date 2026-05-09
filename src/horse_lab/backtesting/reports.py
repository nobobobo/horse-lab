"""Backtest artifact writers for audit-friendly replay reports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from horse_lab.backtesting.simulator import (
    BacktestConfig,
    BacktestResult,
    BetDecision,
)


@dataclass(frozen=True)
class BacktestArtifactPaths:
    config_path: Path
    decision_report_path: Path


def write_backtest_artifacts(
    *,
    result: BacktestResult,
    config: BacktestConfig,
    output_dir: Path | str,
) -> BacktestArtifactPaths:
    """Write config and per-runner bet decision artifacts."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "backtest_config.json"
    decision_report_path = path / "bet_decisions.csv"

    config_path.write_text(
        json.dumps(backtest_config_to_dict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_decision_report(decision_report_path, result.decisions)

    return BacktestArtifactPaths(
        config_path=config_path,
        decision_report_path=decision_report_path,
    )


def backtest_config_to_dict(config: BacktestConfig) -> dict[str, Any]:
    return {
        "initial_bankroll_jpy": config.initial_bankroll_jpy,
        "kelly_config": {
            "fractional_kelly": config.kelly_config.fractional_kelly,
            "max_stake_fraction": config.kelly_config.max_stake_fraction,
            "minimum_edge": config.kelly_config.minimum_edge,
            "stake_unit_jpy": config.kelly_config.stake_unit_jpy,
        },
        "min_odds": config.min_odds,
        "max_odds": config.max_odds,
        "max_stake_per_race_jpy": config.max_stake_per_race_jpy,
        "max_daily_loss_jpy": config.max_daily_loss_jpy,
        "odds_timing": config.odds_timing.value,
        "odds_minutes_before_start": config.odds_minutes_before_start,
    }


def bet_decision_to_dict(decision: BetDecision) -> dict[str, Any]:
    return {
        "race_id": str(decision.race_id),
        "runner_id": str(decision.runner_id),
        "bet_type": decision.bet_type.value,
        "probability": decision.probability,
        "fair_odds": decision.fair_odds,
        "odds": decision.odds,
        "odds_captured_at": _datetime_or_none(decision.odds_captured_at),
        "odds_timing": decision.odds_timing.value,
        "edge": decision.edge,
        "kelly_fraction": decision.kelly_fraction,
        "stake_fraction": decision.stake_fraction,
        "stake_jpy": decision.stake_jpy,
        "payout_jpy": decision.payout_jpy,
        "profit_jpy": decision.profit_jpy,
        "bankroll_before_jpy": decision.bankroll_before_jpy,
        "bankroll_after_jpy": decision.bankroll_after_jpy,
        "is_win": decision.is_win,
        "did_bet": decision.did_bet,
        "skip_reason": decision.skip_reason,
        "prediction_as_of": _datetime_or_none(decision.prediction_as_of),
        "closing_odds": decision.closing_odds,
        "closing_captured_at": _datetime_or_none(decision.closing_captured_at),
        "clv_odds_delta": decision.clv_odds_delta,
        "clv_implied_probability_delta": decision.clv_implied_probability_delta,
    }


def _write_decision_report(
    path: Path,
    decisions: tuple[BetDecision, ...],
) -> None:
    fieldnames = [
        "race_id",
        "runner_id",
        "bet_type",
        "probability",
        "fair_odds",
        "odds",
        "odds_captured_at",
        "odds_timing",
        "edge",
        "kelly_fraction",
        "stake_fraction",
        "stake_jpy",
        "payout_jpy",
        "profit_jpy",
        "bankroll_before_jpy",
        "bankroll_after_jpy",
        "is_win",
        "did_bet",
        "skip_reason",
        "prediction_as_of",
        "closing_odds",
        "closing_captured_at",
        "clv_odds_delta",
        "clv_implied_probability_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(bet_decision_to_dict(decision))


def _datetime_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
