"""Monitoring summaries for paper-trading runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PaperTradingMonitoringSummaryResult:
    summary_path: Path
    summary: dict[str, Any]


def summarize_paper_trading_reports(
    report_paths: Sequence[Path | str],
    output_path: Path | str,
) -> PaperTradingMonitoringSummaryResult:
    """Aggregate Phase 5/6 paper-trading JSON reports into one monitor payload."""

    materialized_paths = tuple(Path(path) for path in report_paths)
    if not materialized_paths:
        raise ValueError("At least one paper-trading report path is required")

    reports = tuple(_load_paper_report(path) for path in materialized_paths)
    probability_rows = [report.get("probability", {}) for report in reports]
    backtest_rows = [report.get("backtest", {}) for report in reports]
    decision_rows = [report.get("decision_summary", {}) for report in reports]
    clv_rows = [report.get("clv", {}) for report in reports]
    count_rows = [report.get("counts", {}) for report in reports]

    observations = _sum_int(probability_rows, "observations")
    total_bets = _sum_int(backtest_rows, "total_bets")
    total_staked = _sum_float(backtest_rows, "total_staked_jpy")
    net_profit = _sum_float(backtest_rows, "net_profit_jpy")
    final_bankroll_values = [
        float(row["final_bankroll_jpy"])
        for row in backtest_rows
        if row.get("final_bankroll_jpy") is not None
    ]

    summary: dict[str, Any] = {
        "report_count": len(reports),
        "source_report_paths": [str(path) for path in materialized_paths],
        "counts": {
            "races": _sum_int(count_rows, "races"),
            "predictions": _sum_int(count_rows, "predictions"),
            "odds": _sum_int(count_rows, "odds"),
            "results": _sum_int(count_rows, "results"),
            "bet_decisions": _sum_int(count_rows, "bet_decisions"),
            "bet_records": _sum_int(count_rows, "bet_records"),
        },
        "probability": {
            "observations": observations,
            "positives": _sum_int(probability_rows, "positives"),
            "mean_predicted_probability": _weighted_mean(
                probability_rows,
                value_key="mean_predicted_probability",
                weight_key="observations",
            ),
            "empirical_rate": _weighted_mean(
                probability_rows,
                value_key="empirical_rate",
                weight_key="observations",
            ),
            "log_loss": _weighted_mean(
                probability_rows,
                value_key="log_loss",
                weight_key="observations",
            ),
            "brier_score": _weighted_mean(
                probability_rows,
                value_key="brier_score",
                weight_key="observations",
            ),
            "expected_calibration_error": _weighted_mean(
                probability_rows,
                value_key="expected_calibration_error",
                weight_key="observations",
            ),
        },
        "backtest": {
            "total_bets": total_bets,
            "wins": _sum_int(backtest_rows, "wins"),
            "total_staked_jpy": total_staked,
            "total_payout_jpy": _sum_float(backtest_rows, "total_payout_jpy"),
            "net_profit_jpy": net_profit,
            "roi": net_profit / total_staked if total_staked > 0.0 else 0.0,
            "hit_rate": _ratio(
                numerator=_sum_int(backtest_rows, "wins"),
                denominator=total_bets,
            ),
            "turnover": _sum_float(backtest_rows, "turnover"),
            "max_drawdown": max(
                (float(row.get("max_drawdown") or 0.0) for row in backtest_rows),
                default=0.0,
            ),
            "latest_final_bankroll_jpy": (
                final_bankroll_values[-1] if final_bankroll_values else 0.0
            ),
        },
        "decision_summary": {
            "decisions": _sum_int(decision_rows, "decisions"),
            "did_bet": _sum_int(decision_rows, "did_bet"),
            "positive_edge_decisions": _sum_int(
                decision_rows,
                "positive_edge_decisions",
            ),
            "mean_edge": _weighted_mean(
                decision_rows,
                value_key="mean_edge",
                weight_key="decisions",
            ),
            "skip_reason_counts": _sum_nested_ints(
                decision_rows,
                "skip_reason_counts",
            ),
        },
        "clv": {
            "bets_with_clv": _sum_int(clv_rows, "bets_with_clv"),
            "mean_clv_odds_delta": _weighted_mean(
                clv_rows,
                value_key="mean_clv_odds_delta",
                weight_key="bets_with_clv",
            ),
            "mean_clv_implied_probability_delta": _weighted_mean(
                clv_rows,
                value_key="mean_clv_implied_probability_delta",
                weight_key="bets_with_clv",
            ),
            "positive_clv_rate": _weighted_mean(
                clv_rows,
                value_key="positive_clv_rate",
                weight_key="bets_with_clv",
            ),
        },
    }

    summary_path = Path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PaperTradingMonitoringSummaryResult(
        summary_path=summary_path,
        summary=summary,
    )


def paper_trading_monitoring_summary_result_to_dict(
    result: PaperTradingMonitoringSummaryResult,
) -> dict[str, Any]:
    return {
        "summary_path": str(result.summary_path),
        "summary": result.summary,
    }


def _load_paper_report(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "paper_trading" in payload:
        return payload["paper_trading"]
    return payload


def _sum_int(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(float(row.get(key) or 0.0) for row in rows))


def _sum_float(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key) or 0.0) for row in rows))


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    weight_key: str,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(value_key)
        weight = float(row.get(weight_key) or 0.0)
        if value is None or weight <= 0.0:
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _sum_nested_ints(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        nested = row.get(key) or {}
        for nested_key, value in nested.items():
            totals[str(nested_key)] = totals.get(str(nested_key), 0) + int(value)
    return dict(sorted(totals.items()))


def _ratio(*, numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
