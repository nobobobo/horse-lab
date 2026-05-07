"""Kelly staking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class KellyConfig:
    fractional_kelly: float = 0.25
    max_stake_fraction: float = 0.02
    minimum_edge: float = 0.02
    stake_unit_jpy: int = 100

    def __post_init__(self) -> None:
        if self.fractional_kelly < 0.0:
            raise ValueError("fractional_kelly must be non-negative")
        if self.max_stake_fraction < 0.0:
            raise ValueError("max_stake_fraction must be non-negative")
        if self.minimum_edge < 0.0:
            raise ValueError("minimum_edge must be non-negative")
        if self.stake_unit_jpy <= 0:
            raise ValueError("stake_unit_jpy must be positive")


@dataclass(frozen=True)
class StakeDecision:
    probability: float
    odds: float
    edge: float
    full_kelly_fraction: float
    stake_fraction: float
    stake_jpy: int


def calculate_edge(*, probability: float, odds: float) -> float:
    _validate_probability(probability)
    _validate_odds(odds)
    return probability * odds - 1.0


def calculate_kelly_stake(
    *,
    probability: float,
    odds: float,
    bankroll_jpy: int,
    config: KellyConfig = KellyConfig(),
) -> StakeDecision:
    _validate_probability(probability)
    _validate_odds(odds)
    if bankroll_jpy <= 0:
        raise ValueError("bankroll_jpy must be positive")

    edge = calculate_edge(probability=probability, odds=odds)
    if edge < config.minimum_edge:
        return StakeDecision(
            probability=probability,
            odds=odds,
            edge=edge,
            full_kelly_fraction=0.0,
            stake_fraction=0.0,
            stake_jpy=0,
        )

    full_kelly_fraction = max(edge / (odds - 1.0), 0.0)
    stake_fraction = min(
        full_kelly_fraction * config.fractional_kelly,
        config.max_stake_fraction,
    )
    raw_stake_jpy = bankroll_jpy * stake_fraction
    stake_jpy = floor(raw_stake_jpy / config.stake_unit_jpy) * config.stake_unit_jpy

    return StakeDecision(
        probability=probability,
        odds=odds,
        edge=edge,
        full_kelly_fraction=full_kelly_fraction,
        stake_fraction=stake_fraction,
        stake_jpy=stake_jpy,
    )


def _validate_probability(probability: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0.0 and 1.0")


def _validate_odds(odds: float) -> None:
    if odds <= 1.0:
        raise ValueError("odds must be greater than 1.0")
