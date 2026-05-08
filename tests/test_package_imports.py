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
    from horse_lab.evaluation import (
        PerformanceSummary,
        ProbabilitySummary,
        summarize_win_probability_predictions,
    )
    from horse_lab.models import LightGBMWinProbabilityModel, MarketImpliedProbabilityModel

    assert KellyConfig.__name__ == "KellyConfig"
    assert callable(calculate_kelly_stake)
    assert BacktestConfig.__name__ == "BacktestConfig"
    assert BacktestSimulator.__name__ == "BacktestSimulator"
    assert PerformanceSummary.__name__ == "PerformanceSummary"
    assert ProbabilitySummary.__name__ == "ProbabilitySummary"
    assert callable(summarize_win_probability_predictions)
    assert LightGBMWinProbabilityModel.__name__ == "LightGBMWinProbabilityModel"
    assert MarketImpliedProbabilityModel.__name__ == "MarketImpliedProbabilityModel"


def test_storage_and_feature_protocols_import():
    from horse_lab.data import RaceRepository, ResultRepository
    from horse_lab.features import (
        FeatureBuilder,
        PastPerformanceFeatureBuilder,
        build_past_performance_features,
    )

    assert RaceRepository.__name__ == "RaceRepository"
    assert ResultRepository.__name__ == "ResultRepository"
    assert FeatureBuilder.__name__ == "FeatureBuilder"
    assert PastPerformanceFeatureBuilder.__name__ == "PastPerformanceFeatureBuilder"
    assert callable(build_past_performance_features)


def test_pipeline_imports():
    from horse_lab.pipelines import ReplayResult, run_market_replay

    assert ReplayResult.__name__ == "ReplayResult"
    assert callable(run_market_replay)


def test_jravan_ingestion_helpers_import():
    from horse_lab.data.jravan import (
        JvDataRecord,
        build_replay_dataset_from_staging,
        build_jravan_runner_id,
        build_jravan_s3_raw_sync_plan,
        ingest_jvdata_file_to_staging,
        map_jvdata_records,
        map_o1_record_to_odds_quote,
        map_o1_record_to_odds_quotes,
        odds_quote_to_csv_row,
        race_to_csv_row,
        map_ra_record_to_race,
        map_se_record_to_entry,
        map_se_record_to_result,
        parse_jvdata_record,
        render_sync_command,
        replay_dataset_report_to_dict,
        sync_jravan_raw_from_s3,
        write_feature_rows_csv,
        write_staging_csvs,
    )

    assert JvDataRecord.__name__ == "JvDataRecord"
    assert callable(build_replay_dataset_from_staging)
    assert callable(build_jravan_runner_id)
    assert callable(build_jravan_s3_raw_sync_plan)
    assert callable(ingest_jvdata_file_to_staging)
    assert callable(map_jvdata_records)
    assert callable(map_o1_record_to_odds_quote)
    assert callable(map_o1_record_to_odds_quotes)
    assert callable(odds_quote_to_csv_row)
    assert callable(race_to_csv_row)
    assert callable(map_ra_record_to_race)
    assert callable(map_se_record_to_entry)
    assert callable(map_se_record_to_result)
    assert callable(parse_jvdata_record)
    assert callable(render_sync_command)
    assert callable(replay_dataset_report_to_dict)
    assert callable(sync_jravan_raw_from_s3)
    assert callable(write_feature_rows_csv)
    assert callable(write_staging_csvs)
