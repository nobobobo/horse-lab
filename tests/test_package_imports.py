from horse_lab.schemas import BetType, PredictionTarget


def test_core_package_imports():
    assert BetType.WIN.value == "win"
    assert PredictionTarget.WIN_PROBABILITY.value == "win_probability"


def test_cli_imports():
    from horse_lab.cli import build_parser, main

    assert callable(build_parser)
    assert callable(main)


def test_baseline_modules_import():
    from horse_lab.betting import KellyConfig, calculate_kelly_stake
    from horse_lab.backtesting import BacktestConfig, BacktestSimulator
    from horse_lab.evaluation import PerformanceSummary
    from horse_lab.models import MarketImpliedProbabilityModel

    assert KellyConfig.__name__ == "KellyConfig"
    assert callable(calculate_kelly_stake)
    assert BacktestConfig.__name__ == "BacktestConfig"
    assert BacktestSimulator.__name__ == "BacktestSimulator"
    assert PerformanceSummary.__name__ == "PerformanceSummary"
    assert MarketImpliedProbabilityModel.__name__ == "MarketImpliedProbabilityModel"


def test_storage_and_feature_protocols_import():
    from horse_lab.data import RaceRepository, ResultRepository
    from horse_lab.features import FeatureBuilder

    assert RaceRepository.__name__ == "RaceRepository"
    assert ResultRepository.__name__ == "ResultRepository"
    assert FeatureBuilder.__name__ == "FeatureBuilder"


def test_pipeline_imports():
    from horse_lab.pipelines import ReplayResult, run_market_replay

    assert ReplayResult.__name__ == "ReplayResult"
    assert callable(run_market_replay)


def test_jravan_ingestion_helpers_import():
    from horse_lab.data.jravan import (
        JvDataRecord,
        build_jravan_runner_id,
        ingest_jvdata_file_to_staging,
        map_jvdata_records,
        map_o1_record_to_odds_quote,
        odds_quote_to_csv_row,
        race_to_csv_row,
        map_ra_record_to_race,
        map_se_record_to_entry,
        map_se_record_to_result,
        parse_jvdata_record,
        write_staging_csvs,
    )

    assert JvDataRecord.__name__ == "JvDataRecord"
    assert callable(build_jravan_runner_id)
    assert callable(ingest_jvdata_file_to_staging)
    assert callable(map_jvdata_records)
    assert callable(map_o1_record_to_odds_quote)
    assert callable(odds_quote_to_csv_row)
    assert callable(race_to_csv_row)
    assert callable(map_ra_record_to_race)
    assert callable(map_se_record_to_entry)
    assert callable(map_se_record_to_result)
    assert callable(parse_jvdata_record)
    assert callable(write_staging_csvs)
