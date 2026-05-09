import csv
import datetime as dt
import json

from horse_lab.backtesting import (
    BacktestConfig,
    BacktestSimulator,
    write_backtest_artifacts,
)
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


def test_write_backtest_artifacts_writes_config_and_decision_report(tmp_path):
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    config = BacktestConfig(
        initial_bankroll_jpy=10_000,
        kelly_config=KellyConfig(
            fractional_kelly=0.5,
            max_stake_fraction=0.05,
            minimum_edge=0.0,
            stake_unit_jpy=100,
        ),
        min_odds=1.5,
        max_stake_per_race_jpy=1_000,
    )
    result = BacktestSimulator(config=config).run(
        predictions=[
            ModelPrediction(
                race_id=RaceId("race-1"),
                runner_id=RunnerId("runner-1"),
                model_name=ModelName("test"),
                model_version="test",
                target=PredictionTarget.WIN_PROBABILITY,
                probability=0.6,
                as_of=as_of,
            )
        ],
        odds=[
            OddsQuote(
                race_id=RaceId("race-1"),
                runner_id=RunnerId("runner-1"),
                bet_type=BetType.WIN,
                captured_at=as_of,
                odds=3.0,
            )
        ],
        results=[
            Result(
                race_id=RaceId("race-1"),
                runner_id=RunnerId("runner-1"),
                finish_position=1,
            )
        ],
    )

    paths = write_backtest_artifacts(
        result=result,
        config=config,
        output_dir=tmp_path,
    )

    config_payload = json.loads(paths.config_path.read_text(encoding="utf-8"))
    assert config_payload["initial_bankroll_jpy"] == 10_000
    assert config_payload["kelly_config"]["fractional_kelly"] == 0.5
    assert config_payload["min_odds"] == 1.5
    assert config_payload["odds_timing"] == "latest_available"

    rows = list(csv.DictReader(paths.decision_report_path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["race_id"] == "race-1"
    assert rows[0]["runner_id"] == "runner-1"
    assert rows[0]["did_bet"] == "True"
    assert rows[0]["skip_reason"] == ""
