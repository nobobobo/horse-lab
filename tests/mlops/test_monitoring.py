import json
from pathlib import Path

from horse_lab.mlops import summarize_paper_trading_reports


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _paper_report(
    *,
    observations: int,
    log_loss: float,
    bets: int,
    profit: float,
    bets_with_clv: int,
) -> dict:
    return {
        "counts": {
            "races": 2,
            "predictions": observations,
            "odds": observations,
            "results": observations,
            "bet_decisions": observations,
            "bet_records": bets,
        },
        "probability": {
            "observations": observations,
            "positives": observations // 2,
            "mean_predicted_probability": 0.5,
            "empirical_rate": 0.5,
            "log_loss": log_loss,
            "brier_score": 0.2,
            "expected_calibration_error": 0.1,
        },
        "backtest": {
            "total_bets": bets,
            "wins": bets // 2,
            "total_staked_jpy": 100.0 * bets,
            "total_payout_jpy": 100.0 * bets + profit,
            "net_profit_jpy": profit,
            "roi": 0.0,
            "hit_rate": 0.0,
            "turnover": float(bets),
            "max_drawdown": 100.0,
            "final_bankroll_jpy": 100_000.0 + profit,
        },
        "decision_summary": {
            "decisions": observations,
            "did_bet": bets,
            "positive_edge_decisions": bets + 1,
            "skip_reason_counts": {"edge_below_threshold": observations - bets},
            "mean_edge": 0.01,
        },
        "clv": {
            "bets_with_clv": bets_with_clv,
            "mean_clv_odds_delta": 0.1,
            "mean_clv_implied_probability_delta": 0.02,
            "positive_clv_rate": 0.5,
        },
    }


def test_summarize_paper_trading_reports_supports_nested_daily_reports(tmp_path):
    paper_path = tmp_path / "paper_report.json"
    daily_path = tmp_path / "daily_report.json"
    output_path = tmp_path / "monitoring" / "summary.json"
    _write_json(
        paper_path,
        _paper_report(
            observations=10,
            log_loss=0.20,
            bets=2,
            profit=100.0,
            bets_with_clv=2,
        ),
    )
    _write_json(
        daily_path,
        {
            "run_type": "daily-paper-trading",
            "paper_trading": _paper_report(
                observations=30,
                log_loss=0.40,
                bets=3,
                profit=-50.0,
                bets_with_clv=3,
            ),
        },
    )

    result = summarize_paper_trading_reports(
        [paper_path, daily_path],
        output_path,
    )

    assert result.summary_path == output_path
    assert result.summary["report_count"] == 2
    assert result.summary["counts"]["predictions"] == 40
    assert result.summary["probability"]["log_loss"] == 0.35
    assert result.summary["backtest"]["total_bets"] == 5
    assert result.summary["backtest"]["net_profit_jpy"] == 50.0
    assert result.summary["decision_summary"]["skip_reason_counts"] == {
        "edge_below_threshold": 35
    }
    assert output_path.exists()


def test_monitoring_summary_cli(tmp_path, capsys):
    from horse_lab.cli import main

    paper_path = tmp_path / "paper_report.json"
    output_path = tmp_path / "monitoring" / "summary.json"
    _write_json(
        paper_path,
        _paper_report(
            observations=10,
            log_loss=0.20,
            bets=2,
            profit=100.0,
            bets_with_clv=2,
        ),
    )

    exit_code = main(
        [
            "paper-trading-monitoring-summary",
            str(output_path),
            str(paper_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["summary"]["report_count"] == 1
    assert Path(summary["summary_path"]).exists()
