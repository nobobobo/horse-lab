"""Kelly staking primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KellyConfig:
    fractional_kelly: float = 0.25
    max_stake_fraction: float = 0.02
    minimum_edge: float = 0.02
    stake_unit_jpy: int = 100


@dataclass(frozen=True)
class StakeDecision:
    probability: float
    odds: float
    edge: float
    full_kelly_fraction: float
    stake_fraction: float
    stake_jpy: int


def calculate_kelly_stake(
    *,
    probability: float,
    odds: float,
    bankroll_jpy: int,
    config: KellyConfig = KellyConfig(),
) -> StakeDecision:
    return StakeDecision(
        probability=probability,
        odds=odds,
        edge=0.0,
        full_kelly_fraction=0.0,
        stake_fraction=0.0,
        stake_jpy=0,
    )
