import datetime as dt

import pytest

from horse_lab.backtesting import BacktestConfig, BacktestSimulator
from horse_lab.betting import KellyConfig
from horse_lab.schemas import (
    BetType,
    CourseDirection,
    ModelName,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    Race,
    RaceId,
    Result,
    RunnerId,
    Surface,
    TrackCondition,
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


def _race(
    race_id: str,
    *,
    race_number: int,
    start_time: dt.datetime,
) -> Race:
    return Race(
        race_id=RaceId(race_id),
        race_date=start_time.date(),
        venue="Tokyo",
        race_number=race_number,
        name=None,
        surface=Surface.TURF,
        distance_m=1200,
        direction=CourseDirection.LEFT,
        track_condition=TrackCondition.FIRM,
        start_time=start_time,
        field_size=1,
    )


def _race_prediction(
    race_id: str,
    runner_id: str,
    *,
    probability: float,
    as_of: dt.datetime,
) -> ModelPrediction:
    return ModelPrediction(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        model_name=ModelName("test_model"),
        model_version="test",
        target=PredictionTarget.WIN_PROBABILITY,
        probability=probability,
        as_of=as_of,
    )


def _race_quote(
    race_id: str,
    runner_id: str,
    *,
    odds: float,
    captured_at: dt.datetime,
) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        bet_type=BetType.WIN,
        captured_at=captured_at,
        odds=odds,
    )


def _race_result(race_id: str, runner_id: str, *, finish_position: int) -> Result:
    return Result(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        finish_position=finish_position,
    )


def test_backtest_simulator_settles_same_race_bets_without_intra_race_resizing_when_as_of_differs():
    first_as_of = dt.datetime(2026, 5, 7, 14, 55)
    second_as_of = dt.datetime(2026, 5, 7, 14, 56)
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
            _prediction("runner-1", probability=0.6, as_of=first_as_of),
            _prediction("runner-2", probability=0.6, as_of=second_as_of),
        ],
        odds=[
            _quote("runner-1", odds=3.0, captured_at=first_as_of),
            _quote("runner-2", odds=2.0, captured_at=second_as_of),
        ],
        results=[
            _result("runner-1", finish_position=1),
            _result("runner-2", finish_position=2),
        ],
    )

    assert len(result.records) == 2
    assert result.records[0].stake_jpy == 1_000
    assert result.records[0].payout_jpy == 3_000
    assert result.records[0].bankroll_after_jpy == 11_000
    assert result.records[1].stake_jpy == 1_000
    assert result.records[1].payout_jpy == 0
    assert result.records[1].bankroll_after_jpy == 11_000
    assert result.final_bankroll_jpy == 11_000
    assert result.summary.total_bets == 2
    assert result.summary.wins == 1
    assert result.summary.roi == pytest.approx(1_000 / 2_000)
    assert result.summary.max_drawdown == pytest.approx(0.0)
    assert result.bankroll_curve_jpy == (10_000, 11_000)


def test_backtest_simulator_orders_races_by_start_time_when_races_are_provided():
    as_of = dt.datetime(2026, 5, 7, 9, 55)
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
            _race_prediction("race-a", "runner-a", probability=0.6, as_of=as_of),
            _race_prediction("race-z", "runner-z", probability=0.6, as_of=as_of),
        ],
        odds=[
            _race_quote("race-a", "runner-a", odds=2.0, captured_at=as_of),
            _race_quote("race-z", "runner-z", odds=2.0, captured_at=as_of),
        ],
        results=[
            _race_result("race-a", "runner-a", finish_position=2),
            _race_result("race-z", "runner-z", finish_position=1),
        ],
        races=[
            _race("race-a", race_number=2, start_time=dt.datetime(2026, 5, 7, 11, 0)),
            _race("race-z", race_number=1, start_time=dt.datetime(2026, 5, 7, 10, 0)),
        ],
    )

    assert [record.race_id for record in result.records] == [
        RaceId("race-z"),
        RaceId("race-a"),
    ]
    assert result.bankroll_curve_jpy == (10_000, 11_000, 9_900)


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
    assert len(result.decisions) == 1
    assert result.decisions[0].did_bet is False
    assert result.decisions[0].skip_reason == "edge_below_minimum"
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


def test_backtest_simulator_selects_minutes_before_start_odds_and_records_clv():
    as_of = dt.datetime(2026, 5, 7, 23, 59)
    race_start = dt.datetime(2026, 5, 7, 11, 0)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(
                fractional_kelly=1.0,
                max_stake_fraction=0.10,
                minimum_edge=0.0,
                stake_unit_jpy=1,
            ),
            odds_timing="minutes_before_start",
            odds_minutes_before_start=30,
        )
    )

    result = simulator.run(
        predictions=[_prediction("runner-1", probability=0.7, as_of=as_of)],
        odds=[
            _quote(
                "runner-1",
                odds=2.0,
                captured_at=race_start - dt.timedelta(minutes=40),
            ),
            _quote(
                "runner-1",
                odds=3.0,
                captured_at=race_start - dt.timedelta(minutes=5),
            ),
        ],
        results=[_result("runner-1", finish_position=1)],
        races=[_race("race-1", race_number=1, start_time=race_start)],
    )

    assert result.records[0].odds == 2.0
    assert result.records[0].closing_odds == 3.0
    assert result.records[0].clv_odds_delta == pytest.approx(-1.0)
    assert result.records[0].clv_implied_probability_delta == pytest.approx(
        (1.0 / 3.0) - (1.0 / 2.0)
    )
    assert result.decisions[0].odds_timing.value == "minutes_before_start"


def test_backtest_simulator_caps_total_stake_per_race():
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
            max_stake_per_race_jpy=1_500,
        )
    )

    result = simulator.run(
        predictions=[
            _prediction("runner-1", probability=0.6, as_of=as_of),
            _prediction("runner-2", probability=0.6, as_of=as_of),
        ],
        odds=[
            _quote("runner-1", odds=3.0, captured_at=as_of),
            _quote("runner-2", odds=3.0, captured_at=as_of),
        ],
        results=[
            _result("runner-1", finish_position=2),
            _result("runner-2", finish_position=2),
        ],
    )

    assert [record.stake_jpy for record in result.records] == [1_000, 500]
    assert sum(record.stake_jpy for record in result.records) == 1_500


def test_backtest_simulator_applies_daily_stop_loss_after_completed_race():
    as_of = dt.datetime(2026, 5, 7, 9, 55)
    simulator = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=10_000,
            kelly_config=KellyConfig(
                fractional_kelly=1.0,
                max_stake_fraction=0.10,
                minimum_edge=0.0,
                stake_unit_jpy=1,
            ),
            max_daily_loss_jpy=500,
        )
    )

    result = simulator.run(
        predictions=[
            _race_prediction("race-a", "runner-a", probability=0.6, as_of=as_of),
            _race_prediction("race-b", "runner-b", probability=0.6, as_of=as_of),
        ],
        odds=[
            _race_quote("race-a", "runner-a", odds=2.0, captured_at=as_of),
            _race_quote("race-b", "runner-b", odds=2.0, captured_at=as_of),
        ],
        results=[
            _race_result("race-a", "runner-a", finish_position=2),
            _race_result("race-b", "runner-b", finish_position=1),
        ],
        races=[
            _race("race-a", race_number=1, start_time=dt.datetime(2026, 5, 7, 10, 0)),
            _race("race-b", race_number=2, start_time=dt.datetime(2026, 5, 7, 11, 0)),
        ],
    )

    assert len(result.records) == 1
    assert result.records[0].race_id == RaceId("race-a")
    assert result.final_bankroll_jpy == 9_000
    assert result.decisions[1].did_bet is False
    assert result.decisions[1].skip_reason == "daily_stop_loss"


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
