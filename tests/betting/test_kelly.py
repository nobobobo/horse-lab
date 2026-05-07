import pytest

from horse_lab.betting import KellyConfig, calculate_edge, calculate_kelly_stake


def test_calculate_edge_uses_probability_times_odds_minus_one():
    assert calculate_edge(probability=0.4, odds=3.0) == pytest.approx(0.2)


def test_kelly_returns_zero_for_negative_edge():
    decision = calculate_kelly_stake(
        probability=0.2,
        odds=3.0,
        bankroll_jpy=100_000,
        config=KellyConfig(minimum_edge=0.0),
    )

    assert decision.edge == pytest.approx(-0.4)
    assert decision.full_kelly_fraction == 0.0
    assert decision.stake_fraction == 0.0
    assert decision.stake_jpy == 0


def test_kelly_returns_zero_below_minimum_edge():
    decision = calculate_kelly_stake(
        probability=0.34,
        odds=3.0,
        bankroll_jpy=100_000,
        config=KellyConfig(minimum_edge=0.03),
    )

    assert decision.edge == pytest.approx(0.02)
    assert decision.stake_fraction == 0.0
    assert decision.stake_jpy == 0


def test_kelly_caps_fraction_and_rounds_to_stake_unit():
    decision = calculate_kelly_stake(
        probability=0.6,
        odds=3.0,
        bankroll_jpy=101_000,
        config=KellyConfig(
            fractional_kelly=1.0,
            max_stake_fraction=0.05,
            minimum_edge=0.0,
            stake_unit_jpy=100,
        ),
    )

    assert decision.edge == pytest.approx(0.8)
    assert decision.full_kelly_fraction == pytest.approx(0.4)
    assert decision.stake_fraction == pytest.approx(0.05)
    assert decision.stake_jpy == 5_000


def test_kelly_validates_probability_odds_bankroll_and_config():
    with pytest.raises(ValueError, match="probability"):
        calculate_edge(probability=-0.1, odds=2.0)

    with pytest.raises(ValueError, match="odds"):
        calculate_edge(probability=0.5, odds=1.0)

    with pytest.raises(ValueError, match="probability"):
        calculate_kelly_stake(probability=1.1, odds=2.0, bankroll_jpy=10_000)

    with pytest.raises(ValueError, match="odds"):
        calculate_kelly_stake(probability=0.5, odds=1.0, bankroll_jpy=10_000)

    with pytest.raises(ValueError, match="bankroll"):
        calculate_kelly_stake(probability=0.5, odds=2.0, bankroll_jpy=0)

    with pytest.raises(ValueError, match="fractional_kelly"):
        KellyConfig(fractional_kelly=-0.1)

    with pytest.raises(ValueError, match="max_stake_fraction"):
        KellyConfig(max_stake_fraction=-0.1)

    with pytest.raises(ValueError, match="minimum_edge"):
        KellyConfig(minimum_edge=-0.1)

    with pytest.raises(ValueError, match="stake_unit_jpy"):
        KellyConfig(stake_unit_jpy=0)
