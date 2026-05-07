"""Minimal backtest simulator classes for import wiring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    initial_bankroll_jpy: int = 100_000


@dataclass(frozen=True)
class BacktestResult:
    final_bankroll_jpy: int


class BacktestSimulator:
    pass
