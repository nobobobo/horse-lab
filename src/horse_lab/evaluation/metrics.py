"""Performance metric helpers for betting backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PerformanceSummary:
    total_bets: int
    wins: int
    total_staked_jpy: int
    total_payout_jpy: int
    net_profit_jpy: int
    roi: float
    hit_rate: float
    turnover: float
    max_drawdown: float
    final_bankroll_jpy: int


def compute_max_drawdown(bankroll_curve_jpy: Sequence[int]) -> float:
    if not bankroll_curve_jpy:
        return 0.0

    peak = float(bankroll_curve_jpy[0])
    max_drawdown = 0.0
    for value in bankroll_curve_jpy:
        value_float = float(value)
        if value_float > peak:
            peak = value_float
        if peak <= 0.0:
            continue
        drawdown = (peak - value_float) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def summarize_performance(
    *,
    initial_bankroll_jpy: int,
    final_bankroll_jpy: int,
    stakes_jpy: Sequence[int],
    payouts_jpy: Sequence[int],
    wins: Sequence[bool],
    bankroll_curve_jpy: Sequence[int],
) -> PerformanceSummary:
    total_bets = len(stakes_jpy)
    total_staked_jpy = sum(stakes_jpy)
    total_payout_jpy = sum(payouts_jpy)
    net_profit_jpy = final_bankroll_jpy - initial_bankroll_jpy
    win_count = sum(1 for won in wins if won)

    roi = net_profit_jpy / total_staked_jpy if total_staked_jpy else 0.0
    hit_rate = win_count / total_bets if total_bets else 0.0
    turnover = total_staked_jpy / initial_bankroll_jpy if initial_bankroll_jpy else 0.0

    return PerformanceSummary(
        total_bets=total_bets,
        wins=win_count,
        total_staked_jpy=total_staked_jpy,
        total_payout_jpy=total_payout_jpy,
        net_profit_jpy=net_profit_jpy,
        roi=roi,
        hit_rate=hit_rate,
        turnover=turnover,
        max_drawdown=compute_max_drawdown(bankroll_curve_jpy),
        final_bankroll_jpy=final_bankroll_jpy,
    )
