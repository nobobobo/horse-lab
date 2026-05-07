"""Performance summary primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSummary:
    total_bets: int
