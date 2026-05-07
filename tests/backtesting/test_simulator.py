import datetime as dt

import pytest

from horse_lab.backtesting import BacktestConfig, BacktestSimulator
from horse_lab.betting import KellyConfig
from horse_lab.schemas import (
    BetType,
    ModelName,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    RaceId,
    Result,
    RunnerId,
)


def _prediction(runner_id: str, probability: float, as_of: dt.datetime) -> ModelPrediction:
    return ModelPrediction(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        model_name=ModelName("test_model"),
        model_version="test",
        target=PredictionTarget.WIN_PROBABILITY,
        probability=probability,
        as_of=as_of,
    )


def _quote(runner_id: str, odds: float, captured_at: dt.datetime) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        bet_type=BetType.WIN,
        captured_at=captured_at,
        odds=odds,
    )


def _result(runner_id: str, finish_position: int) -> Result:
    return Result(
        race_id=RaceId("race-1"),
        runner_id=RunnerId(runner_id),
        finish_position=finish_position,
    )


def test_backtest_simulator_accounts_for_one_win_and_one_loss():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(
                fractional_kelly=1.0,
                max_stake_fraction=0.10,
                minimum_edge=0.0,
                stake_unit_jpy=1,
            ),
        )
    )

    result = simulator.run(
        predictions=[
            _prediction("runner-1", probability=0.6, as_of=as_of),
            _prediction("runner-2", probability=0.6, as_of=as_of),
        ],
        odds=[
            _quote("runner-1", odds=3.0, captured_at=as_of),
            _quote("runner-2", odds=2.0, captured_at=as_of),
        ],
        results=[
            _result("runner-1", finish_position=1),
            _result("runner-2", finish_position=2),
        ],
    )

    assert len(result.records) == 2
    assert result.records[0].stake_jpy == 1_000
    assert result.records[0].payout_jpy == 3_000
    assert result.records[0].bankroll_after_jpy == 12_000
    assert result.records[1].stake_jpy == 1_200
    assert result.records[1].payout_jpy == 0
    assert result.records[1].bankroll_after_jpy == 10_800
    assert result.final_bankroll_jpy == 10_800
    assert result.summary.total_bets == 2
    assert result.summary.wins == 1
    assert result.summary.roi == pytest.approx(800 / 2_200)
    assert result.summary.max_drawdown == pytest.approx(0.1)


def test_backtest_simulator_skips_zero_stake_predictions():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(minimum_edge=0.05),
        )
    )

    result = simulator.run(
        predictions=[_prediction("runner-1", probability=0.5, as_of=as_of)],
        odds=[_quote("runner-1", odds=2.0, captured_at=as_of)],
        results=[_result("runner-1", finish_position=1)],
    )

    assert result.records == ()
    assert result.final_bankroll_jpy == 10_000
    assert result.summary.total_bets == 0


def test_backtest_simulator_skips_zero_stake_without_requiring_result():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(minimum_edge=0.05),
        )
    )

    result = simulator.run(
        predictions=[_prediction("runner-1", probability=0.5, as_of=as_of)],
        odds=[_quote("runner-1", odds=2.0, captured_at=as_of)],
        results=[],
    )

    assert result.records == ()
    assert result.final_bankroll_jpy == 10_000
    assert result.summary.total_bets == 0


def test_backtest_simulator_rounds_winning_payouts_to_nearest_yen():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(
                fractional_kelly=1.0,
                max_stake_fraction=0.01,
                minimum_edge=0.0,
                stake_unit_jpy=1,
            ),
        )
    )

    result = simulator.run(
        predictions=[_prediction("runner-1", probability=0.9, as_of=as_of)],
        odds=[_quote("runner-1", odds=2.3, captured_at=as_of)],
        results=[_result("runner-1", finish_position=1)],
    )

    assert result.records[0].stake_jpy == 100
    assert result.records[0].payout_jpy == 230
    assert result.records[0].bankroll_after_jpy == 10_130


def test_backtest_simulator_uses_latest_prior_odds_and_ignores_future_quotes():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(
                fractional_kelly=1.0,
                max_stake_fraction=0.10,
                minimum_edge=0.0,
                stake_unit_jpy=1,
            ),
        )
    )

    result = simulator.run(
        predictions=[_prediction("runner-1", probability=0.6, as_of=as_of)],
        odds=[
            _quote("runner-1", odds=2.0, captured_at=as_of - dt.timedelta(minutes=10)),
            _quote("runner-1", odds=3.0, captured_at=as_of - dt.timedelta(minutes=1)),
            _quote("runner-1", odds=10.0, captured_at=as_of + dt.timedelta(minutes=1)),
        ],
        results=[_result("runner-1", finish_position=1)],
    )

    assert result.records[0].odds == 3.0
    assert result.records[0].stake_jpy == 1_000
    assert result.records[0].payout_jpy == 3_000


def test_backtest_simulator_requires_matching_odds_and_results():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    simulator = BacktestSimulator(config=BacktestConfig(initial_bankroll_jpy=10_000))

    with pytest.raises(ValueError, match="Missing odds"):
        simulator.run(
            predictions=[_prediction("runner-1", probability=0.6, as_of=as_of)],
            odds=[],
            results=[_result("runner-1", finish_position=1)],
        )

    with pytest.raises(ValueError, match="Missing result"):
        simulator.run(
            predictions=[_prediction("runner-1", probability=0.6, as_of=as_of)],
            odds=[_quote("runner-1", odds=3.0, captured_at=as_of)],
            results=[],
        )
