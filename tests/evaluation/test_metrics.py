import pytest

from horse_lab.evaluation import PerformanceSummary, compute_max_drawdown, summarize_performance


def test_compute_max_drawdown_uses_peak_to_trough_fraction():
    assert compute_max_drawdown([10_000, 12_000, 10_800, 13_000]) == pytest.approx(0.1)


def test_compute_max_drawdown_returns_zero_without_drawdown():
    assert compute_max_drawdown([10_000, 11_000, 12_000]) == 0.0
    assert compute_max_drawdown([]) == 0.0


def test_summarize_performance_computes_roi_hit_rate_turnover_and_profit():
    summary = summarize_performance(
        initial_bankroll_jpy=10_000,
        final_bankroll_jpy=10_800,
        stakes_jpy=[1_000, 1_200],
        payouts_jpy=[3_000, 0],
        wins=[True, False],
        bankroll_curve_jpy=[10_000, 12_000, 10_800],
    )

    assert isinstance(summary, PerformanceSummary)
    assert summary.total_bets == 2
    assert summary.wins == 1
    assert summary.total_staked_jpy == 2_200
    assert summary.total_payout_jpy == 3_000
    assert summary.net_profit_jpy == 800
    assert summary.roi == pytest.approx(800 / 2_200)
    assert summary.hit_rate == pytest.approx(0.5)
    assert summary.turnover == pytest.approx(2_200 / 10_000)
    assert summary.max_drawdown == pytest.approx(0.1)
    assert summary.final_bankroll_jpy == 10_800


def test_summarize_performance_handles_zero_bets():
    summary = summarize_performance(
        initial_bankroll_jpy=10_000,
        final_bankroll_jpy=10_000,
        stakes_jpy=[],
        payouts_jpy=[],
        wins=[],
        bankroll_curve_jpy=[10_000],
    )

    assert summary.total_bets == 0
    assert summary.roi == 0.0
    assert summary.hit_rate == 0.0
    assert summary.turnover == 0.0
