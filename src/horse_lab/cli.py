"""Command-line tools for local horse_lab workflows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from horse_lab.analysis import build_replay_data_quality_report
from horse_lab.backtesting import (
    BacktestConfig,
    OddsTiming,
    QuinellaSimulationConfig,
    QuinellaStrategy,
    backtest_config_to_dict,
    run_quinella_simulation_from_csv,
    write_backtest_artifacts,
)
from horse_lab.betting import KellyConfig
from horse_lab.data import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.data.jravan import (
    DEFAULT_JRAVAN_S3_BUCKET,
    DEFAULT_JRAVAN_S3_RAW_PREFIX,
    DEFAULT_REPLAY_FEATURE_VERSION,
    build_replay_dataset_from_staging,
    build_quinella_replay_dataset_from_raw,
    build_quinella_replay_dataset_from_staging,
    build_jravan_s3_raw_sync_plan,
    ingest_jvdata_directory_to_staging,
    ingest_jvdata_file_to_staging,
    quinella_replay_dataset_report_to_dict,
    render_sync_command,
    replay_dataset_report_to_dict,
    sync_jravan_raw_from_s3,
    write_jvdata_utf8_preview,
)
from horse_lab.data.jravan.raw import JV_DATA_ENCODING
from horse_lab.evaluation import PerformanceSummary, ProbabilitySummary
from horse_lab.mlops import (
    build_phase4_model_registry_from_report,
    model_registry_to_dict,
    paper_trading_monitoring_summary_result_to_dict,
    summarize_paper_trading_reports,
)
from horse_lab.pipelines import (
    daily_paper_trading_result_to_dict,
    lightgbm_training_result_to_dict,
    oof_run_result_to_dict,
    paper_trading_result_to_dict,
    run_daily_paper_trading_from_csv,
    run_level0_oof_from_csv,
    run_lightgbm_ablation_from_csv,
    run_lightgbm_training_from_csv,
    run_market_replay,
    run_paper_trading_from_csv,
)
from horse_lab.stacking import (
    blend_search_result_to_dict,
    build_meta_dataset_from_csv,
    meta_dataset_build_result_to_dict,
    meta_learner_training_result_to_dict,
    phase4_study_result_to_dict,
    run_phase4_study_from_csv,
    search_convex_blend_from_csv,
    train_logistic_meta_learner_from_csv,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = args.handler(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="horse-lab",
        description="Horse racing modeling and data pipeline utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "jravan-ingest",
        help="Convert a CP932 JV-Data raw dump into canonical staging CSVs.",
    )
    ingest_parser.add_argument("raw_path", type=Path)
    ingest_parser.add_argument("staging_dir", type=Path)
    ingest_parser.add_argument("--encoding", default=JV_DATA_ENCODING)
    ingest_parser.add_argument(
        "--strict-unknown",
        action="store_true",
        help="Fail when unsupported JV-Data record types are present.",
    )
    ingest_parser.set_defaults(handler=_handle_jravan_ingest)

    ingest_dir_parser = subparsers.add_parser(
        "jravan-ingest-dir",
        help="Convert a directory of CP932 JV-Data dumps into staging CSVs.",
    )
    ingest_dir_parser.add_argument("raw_dir", type=Path)
    ingest_dir_parser.add_argument("staging_dir", type=Path)
    ingest_dir_parser.add_argument("--pattern", default="*.txt")
    ingest_dir_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the raw directory itself, not nested directories.",
    )
    ingest_dir_parser.add_argument("--encoding", default=JV_DATA_ENCODING)
    ingest_dir_parser.add_argument(
        "--strict-unknown",
        action="store_true",
        help="Fail when unsupported JV-Data record types are present.",
    )
    ingest_dir_parser.set_defaults(handler=_handle_jravan_ingest_dir)

    s3_pull_parser = subparsers.add_parser(
        "jravan-s3-pull-raw",
        help="Sync one JRA-VAN raw run from S3 to the local raw directory.",
    )
    s3_pull_parser.add_argument("run_id")
    s3_pull_parser.add_argument(
        "local_raw_root",
        type=Path,
        help="Local root directory, usually data/raw/jravan.",
    )
    s3_pull_parser.add_argument("--bucket", default=DEFAULT_JRAVAN_S3_BUCKET)
    s3_pull_parser.add_argument("--prefix", default=DEFAULT_JRAVAN_S3_RAW_PREFIX)
    s3_pull_parser.add_argument("--aws-cli", default="aws")
    s3_pull_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Return the planned aws command without executing it.",
    )
    s3_pull_parser.set_defaults(handler=_handle_jravan_s3_pull_raw)

    replay_dataset_parser = subparsers.add_parser(
        "jravan-build-replay-dataset",
        help="Build complete replay-ready CSVs from JRA-VAN staging CSVs.",
    )
    replay_dataset_parser.add_argument("staging_dir", type=Path)
    replay_dataset_parser.add_argument("output_dir", type=Path)
    replay_dataset_parser.add_argument(
        "--feature-version",
        default=DEFAULT_REPLAY_FEATURE_VERSION,
        help="Feature version stamped into generated feature rows.",
    )
    replay_dataset_parser.add_argument(
        "--max-odds-captured-at",
        type=_parse_cli_datetime,
        default=None,
        help="Ignore odds quotes captured after this ISO timestamp.",
    )
    replay_dataset_parser.add_argument(
        "--odds-staging-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Additional staging directory to merge odds.csv from. "
            "Can be supplied multiple times."
        ),
    )
    replay_dataset_parser.set_defaults(handler=_handle_jravan_build_replay_dataset)

    quinella_dataset_parser = subparsers.add_parser(
        "jravan-build-quinella-replay-dataset",
        help="Build a pair-level quinella replay dataset from O2 staging odds.",
    )
    quinella_dataset_parser.add_argument("odds_staging_dir", type=Path)
    quinella_dataset_parser.add_argument("payouts_csv", type=Path)
    quinella_dataset_parser.add_argument("output_dir", type=Path)
    quinella_dataset_parser.add_argument("--start-date", type=_parse_cli_date)
    quinella_dataset_parser.add_argument("--end-date", type=_parse_cli_date)
    quinella_dataset_parser.set_defaults(
        handler=_handle_jravan_build_quinella_replay_dataset
    )

    quinella_raw_dataset_parser = subparsers.add_parser(
        "jravan-build-quinella-replay-dataset-raw",
        help="Build a compact quinella replay dataset directly from O2 raw files.",
    )
    quinella_raw_dataset_parser.add_argument("raw_dir", type=Path)
    quinella_raw_dataset_parser.add_argument("payouts_csv", type=Path)
    quinella_raw_dataset_parser.add_argument("output_dir", type=Path)
    quinella_raw_dataset_parser.add_argument("--start-date", type=_parse_cli_date)
    quinella_raw_dataset_parser.add_argument("--end-date", type=_parse_cli_date)
    quinella_raw_dataset_parser.add_argument("--pattern", default="0B42_jvgets.txt")
    quinella_raw_dataset_parser.add_argument(
        "--no-recursive",
        action="store_true",
    )
    quinella_raw_dataset_parser.add_argument("--encoding", default=JV_DATA_ENCODING)
    quinella_raw_dataset_parser.add_argument(
        "--write-timeseries",
        action="store_true",
        help="Also materialize full O2 odds_timeseries.csv. Omit for compact eval.",
    )
    quinella_raw_dataset_parser.set_defaults(
        handler=_handle_jravan_build_quinella_replay_dataset_raw
    )

    data_qa_parser = subparsers.add_parser(
        "jravan-data-qa",
        help="Build a coverage and integrity report for a replay dataset.",
    )
    data_qa_parser.add_argument("dataset_dir", type=Path)
    data_qa_parser.add_argument("output_path", type=Path)
    data_qa_parser.set_defaults(handler=_handle_jravan_data_qa)

    market_replay_parser = subparsers.add_parser(
        "market-replay",
        help="Run the market-implied baseline over a replay-ready CSV dataset.",
    )
    market_replay_parser.add_argument("dataset_dir", type=Path)
    market_replay_parser.add_argument("--start-date", type=_parse_cli_date, required=True)
    market_replay_parser.add_argument("--end-date", type=_parse_cli_date, required=True)
    market_replay_parser.add_argument("--as-of", type=_parse_cli_datetime, required=True)
    market_replay_parser.add_argument(
        "--feature-version",
        default=DEFAULT_REPLAY_FEATURE_VERSION,
        help="Feature version to read from features.csv.",
    )
    market_replay_parser.add_argument(
        "--initial-bankroll-jpy",
        type=int,
        default=100_000,
        help="Initial bankroll used by the Kelly backtest.",
    )
    market_replay_parser.add_argument(
        "--backtest-report-dir",
        type=Path,
        default=None,
        help="Optional directory for backtest_config.json and bet_decisions.csv.",
    )
    _add_backtest_options(market_replay_parser)
    market_replay_parser.set_defaults(handler=_handle_market_replay)

    quinella_parser = subparsers.add_parser(
        "quinella-sim",
        help="Run a quinella simulation from O2 odds and official payouts CSVs.",
    )
    quinella_parser.add_argument("odds_csv", type=Path)
    quinella_parser.add_argument("payouts_csv", type=Path)
    quinella_parser.add_argument("artifact_dir", type=Path)
    quinella_parser.add_argument("--start-date", type=_parse_cli_date, required=True)
    quinella_parser.add_argument("--end-date", type=_parse_cli_date, required=True)
    quinella_parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in QuinellaStrategy],
        default=QuinellaStrategy.FAVORITE.value,
    )
    quinella_parser.add_argument("--initial-bankroll-jpy", type=int, default=100_000)
    quinella_parser.add_argument("--stake-jpy", type=int, default=100)
    quinella_parser.add_argument("--minimum-edge", type=float, default=0.0)
    quinella_parser.add_argument("--max-bets-per-race", type=int, default=1)
    quinella_parser.add_argument(
        "--require-payout-for-race",
        action="store_true",
        help="Skip races without an official quinella payout row.",
    )
    quinella_parser.set_defaults(handler=_handle_quinella_sim)

    lightgbm_parser = subparsers.add_parser(
        "lightgbm-train",
        help="Train and validate the LightGBM win-probability baseline.",
    )
    lightgbm_parser.add_argument("dataset_dir", type=Path)
    lightgbm_parser.add_argument("artifact_dir", type=Path)
    lightgbm_parser.add_argument(
        "--train-end-date",
        type=_parse_cli_date,
        required=True,
    )
    lightgbm_parser.add_argument(
        "--valid-start-date",
        type=_parse_cli_date,
        required=True,
    )
    lightgbm_parser.add_argument(
        "--valid-end-date",
        type=_parse_cli_date,
        required=True,
    )
    lightgbm_parser.add_argument("--as-of", type=_parse_cli_datetime, required=True)
    lightgbm_parser.add_argument(
        "--feature-version",
        default=DEFAULT_REPLAY_FEATURE_VERSION,
        help="Feature version to read from features.csv.",
    )
    lightgbm_parser.add_argument("--random-seed", type=int, default=42)
    lightgbm_parser.add_argument("--model-version", default="lightgbm-win-v1")
    lightgbm_parser.add_argument(
        "--exclude-feature",
        action="append",
        default=(),
        help="Feature name to exclude from training. Can be supplied multiple times.",
    )
    lightgbm_parser.set_defaults(handler=_handle_lightgbm_train)

    lightgbm_ablation_parser = subparsers.add_parser(
        "lightgbm-ablation",
        help="Run full/no-market/no-movement LightGBM ablation scenarios.",
    )
    lightgbm_ablation_parser.add_argument("dataset_dir", type=Path)
    lightgbm_ablation_parser.add_argument("artifact_dir", type=Path)
    lightgbm_ablation_parser.add_argument(
        "--train-end-date",
        type=_parse_cli_date,
        required=True,
    )
    lightgbm_ablation_parser.add_argument(
        "--valid-start-date",
        type=_parse_cli_date,
        required=True,
    )
    lightgbm_ablation_parser.add_argument(
        "--valid-end-date",
        type=_parse_cli_date,
        required=True,
    )
    lightgbm_ablation_parser.add_argument(
        "--as-of",
        type=_parse_cli_datetime,
        required=True,
    )
    lightgbm_ablation_parser.add_argument(
        "--feature-version",
        default=DEFAULT_REPLAY_FEATURE_VERSION,
    )
    lightgbm_ablation_parser.add_argument("--random-seed", type=int, default=42)
    lightgbm_ablation_parser.add_argument("--model-version", default="lightgbm-win-v1")
    lightgbm_ablation_parser.set_defaults(handler=_handle_lightgbm_ablation)

    oof_parser = subparsers.add_parser(
        "level0-oof",
        help="Generate monthly out-of-fold Level 0 predictions for stacking.",
    )
    oof_parser.add_argument("dataset_dir", type=Path)
    oof_parser.add_argument("artifact_dir", type=Path)
    oof_parser.add_argument(
        "--validation-start-date",
        type=_parse_cli_date,
        required=True,
    )
    oof_parser.add_argument(
        "--validation-end-date",
        type=_parse_cli_date,
        required=True,
    )
    oof_parser.add_argument("--as-of", type=_parse_cli_datetime, required=True)
    oof_parser.add_argument(
        "--feature-version",
        default=DEFAULT_REPLAY_FEATURE_VERSION,
        help="Feature version to read from features.csv.",
    )
    oof_parser.add_argument("--random-seed", type=int, default=42)
    oof_parser.add_argument(
        "--use-odds-timeseries",
        action="store_true",
        help="Read odds_timeseries.csv instead of the compact latest odds.csv.",
    )
    oof_parser.add_argument(
        "--model-key",
        action="append",
        default=None,
        choices=("market", "lightgbm_full", "lightgbm_no_market"),
        help="Level 0 model key to include. Defaults to all supported models.",
    )
    oof_parser.set_defaults(handler=_handle_level0_oof)

    meta_dataset_parser = subparsers.add_parser(
        "stacking-build-meta-dataset",
        help="Pivot stored Level 0 predictions into a meta-learner dataset.",
    )
    meta_dataset_parser.add_argument("predictions_csv", type=Path)
    meta_dataset_parser.add_argument("results_csv", type=Path)
    meta_dataset_parser.add_argument("output_dir", type=Path)
    meta_dataset_parser.add_argument(
        "--prediction-role",
        default="oof",
        choices=("oof", "holdout", "live", "backfill"),
    )
    meta_dataset_parser.add_argument(
        "--target",
        default="win_probability",
        choices=("win_probability", "place_probability", "show_probability"),
    )
    meta_dataset_parser.add_argument(
        "--keep-incomplete-rows",
        action="store_true",
        help="Keep rows that are missing one or more model prediction columns.",
    )
    meta_dataset_parser.set_defaults(handler=_handle_stacking_build_meta_dataset)

    meta_train_parser = subparsers.add_parser(
        "stacking-train-meta",
        help="Train a logistic Level 1 meta learner from meta_features.csv.",
    )
    meta_train_parser.add_argument("meta_features_csv", type=Path)
    meta_train_parser.add_argument("artifact_dir", type=Path)
    meta_train_parser.add_argument(
        "--holdout-fold-id",
        default=None,
        help="Fold held out for validation. Defaults to the latest fold_id.",
    )
    meta_train_parser.add_argument(
        "--feature-column",
        action="append",
        default=None,
        help="Prediction column to use. Defaults to every pred__ column.",
    )
    meta_train_parser.add_argument(
        "--model-version",
        default="logistic-meta-v1",
    )
    meta_train_parser.add_argument("--learning-rate", type=float, default=0.05)
    meta_train_parser.add_argument("--max-iterations", type=int, default=2000)
    meta_train_parser.add_argument("--l2", type=float, default=1e-3)
    meta_train_parser.set_defaults(handler=_handle_stacking_train_meta)

    blend_parser = subparsers.add_parser(
        "stacking-search-blend",
        help="Search convex Level 0 blending weights using temporal holdout.",
    )
    blend_parser.add_argument("meta_features_csv", type=Path)
    blend_parser.add_argument("artifact_dir", type=Path)
    blend_parser.add_argument(
        "--holdout-fold-id",
        default=None,
        help="Fold held out for validation. Defaults to the latest fold_id.",
    )
    blend_parser.add_argument(
        "--feature-column",
        action="append",
        default=None,
        help="Prediction column to use. Defaults to every pred__ column.",
    )
    blend_parser.add_argument("--model-version", default="convex-blend-v1")
    blend_parser.add_argument("--grid-step", type=float, default=0.05)
    blend_parser.set_defaults(handler=_handle_stacking_search_blend)

    phase4_parser = subparsers.add_parser(
        "stacking-phase4-study",
        help="Run walk-forward Phase 4 stacking and segment studies.",
    )
    phase4_parser.add_argument("meta_features_csv", type=Path)
    phase4_parser.add_argument("races_csv", type=Path)
    phase4_parser.add_argument("artifact_dir", type=Path)
    phase4_parser.add_argument("--min-train-folds", type=int, default=3)
    phase4_parser.add_argument("--blend-grid-step", type=float, default=0.05)
    phase4_parser.add_argument("--logistic-learning-rate", type=float, default=0.05)
    phase4_parser.add_argument("--logistic-max-iterations", type=int, default=2000)
    phase4_parser.add_argument("--logistic-l2", type=float, default=1e-3)
    phase4_parser.set_defaults(handler=_handle_stacking_phase4_study)

    registry_parser = subparsers.add_parser(
        "model-registry-register-phase4",
        help="Register a Phase 4 candidate model for Phase 5 paper trading.",
    )
    registry_parser.add_argument("phase4_report_path", type=Path)
    registry_parser.add_argument("output_path", type=Path)
    registry_parser.add_argument("--candidate-method", default="convex_blend")
    registry_parser.add_argument(
        "--model-version",
        default="convex-blend-phase5-v1",
    )
    registry_parser.add_argument(
        "--minimum-log-loss-improvement",
        type=float,
        default=0.0,
    )
    registry_parser.set_defaults(handler=_handle_model_registry_register_phase4)

    paper_parser = subparsers.add_parser(
        "paper-trading-run",
        help="Replay a registered/candidate prediction method as paper trading.",
    )
    paper_parser.add_argument("predictions_csv", type=Path)
    paper_parser.add_argument("dataset_dir", type=Path)
    paper_parser.add_argument("artifact_dir", type=Path)
    paper_parser.add_argument("--method", required=True)
    paper_parser.add_argument("--as-of", type=_parse_cli_datetime, required=True)
    paper_parser.add_argument("--model-version", default="paper-trading-v1")
    paper_parser.add_argument(
        "--use-compact-odds",
        action="store_true",
        help="Use odds.csv instead of odds_timeseries.csv when both exist.",
    )
    paper_parser.add_argument("--initial-bankroll-jpy", type=int, default=100_000)
    _add_backtest_options(paper_parser)
    paper_parser.set_defaults(handler=_handle_paper_trading_run)

    daily_paper_parser = subparsers.add_parser(
        "daily-paper-trading-run",
        help=(
            "Generate same-day Level 0 predictions from historical data, blend the "
            "registered candidate, and replay it as paper trading."
        ),
    )
    daily_paper_parser.add_argument("dataset_dir", type=Path)
    daily_paper_parser.add_argument("model_registry_path", type=Path)
    daily_paper_parser.add_argument("artifact_dir", type=Path)
    daily_paper_parser.add_argument("--start-date", type=_parse_cli_date, required=True)
    daily_paper_parser.add_argument("--end-date", type=_parse_cli_date, required=True)
    daily_paper_parser.add_argument("--as-of", type=_parse_cli_datetime, required=True)
    daily_paper_parser.add_argument(
        "--feature-version",
        default=DEFAULT_REPLAY_FEATURE_VERSION,
        help="Feature version to read from features.csv.",
    )
    daily_paper_parser.add_argument(
        "--train-end-date",
        type=_parse_cli_date,
        default=None,
        help="Latest race date allowed for training. Defaults to day before start-date.",
    )
    daily_paper_parser.add_argument(
        "--candidate-method",
        default=None,
        help="Override the candidate method stored in the registry.",
    )
    daily_paper_parser.add_argument(
        "--use-compact-odds",
        action="store_true",
        help="Use odds.csv instead of odds_timeseries.csv when both exist.",
    )
    daily_paper_parser.add_argument(
        "--allow-unapproved-registry",
        action="store_true",
        help="Run even if the registry candidate did not clear the paper gate.",
    )
    daily_paper_parser.add_argument("--initial-bankroll-jpy", type=int, default=100_000)
    _add_backtest_options(daily_paper_parser)
    daily_paper_parser.set_defaults(handler=_handle_daily_paper_trading_run)

    monitoring_parser = subparsers.add_parser(
        "paper-trading-monitoring-summary",
        help="Aggregate Phase 5/6 paper-trading reports into one monitoring summary.",
    )
    monitoring_parser.add_argument("output_path", type=Path)
    monitoring_parser.add_argument("report_path", nargs="+", type=Path)
    monitoring_parser.set_defaults(handler=_handle_paper_trading_monitoring_summary)

    daily_replay_parser = subparsers.add_parser(
        "jravan-daily-market-replay",
        help="Run the local daily JRA-VAN raw-to-market-replay workflow.",
    )
    daily_replay_parser.add_argument("run_id")
    daily_replay_parser.add_argument("--workspace-root", type=Path, default=Path("data"))
    daily_replay_parser.add_argument("--start-date", type=_parse_cli_date, required=True)
    daily_replay_parser.add_argument("--end-date", type=_parse_cli_date, required=True)
    daily_replay_parser.add_argument("--as-of", type=_parse_cli_datetime, required=True)
    daily_replay_parser.add_argument(
        "--feature-version",
        default=DEFAULT_REPLAY_FEATURE_VERSION,
    )
    daily_replay_parser.add_argument(
        "--initial-bankroll-jpy",
        type=int,
        default=100_000,
    )
    daily_replay_parser.add_argument(
        "--backtest-report-dir",
        type=Path,
        default=None,
        help="Optional override for daily backtest artifacts.",
    )
    _add_backtest_options(daily_replay_parser)
    daily_replay_parser.add_argument("--bucket", default=DEFAULT_JRAVAN_S3_BUCKET)
    daily_replay_parser.add_argument("--prefix", default=DEFAULT_JRAVAN_S3_RAW_PREFIX)
    daily_replay_parser.add_argument("--aws-cli", default="aws")
    daily_replay_parser.add_argument(
        "--skip-s3-pull",
        action="store_true",
        help="Use an existing local raw directory instead of syncing from S3.",
    )
    daily_replay_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Return planned paths and commands without executing the workflow.",
    )
    daily_replay_parser.set_defaults(handler=_handle_jravan_daily_market_replay)

    preview_parser = subparsers.add_parser(
        "jravan-preview",
        help="Write a UTF-8 inspection copy of a CP932 JV-Data raw dump.",
    )
    preview_parser.add_argument("raw_path", type=Path)
    preview_parser.add_argument("output_path", type=Path)
    preview_parser.add_argument("--encoding", default=JV_DATA_ENCODING)
    preview_parser.add_argument(
        "--keep-empty-lines",
        action="store_true",
        help="Preserve blank lines from the raw dump in the preview copy.",
    )
    preview_parser.set_defaults(handler=_handle_jravan_preview)

    return parser


def _add_backtest_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--max-stake-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-edge", type=float, default=0.02)
    parser.add_argument("--stake-unit-jpy", type=int, default=100)
    parser.add_argument("--min-odds", type=float, default=None)
    parser.add_argument("--max-odds", type=float, default=None)
    parser.add_argument("--max-stake-per-race-jpy", type=int, default=None)
    parser.add_argument("--max-daily-loss-jpy", type=int, default=None)
    parser.add_argument(
        "--odds-timing",
        choices=[timing.value for timing in OddsTiming],
        default=OddsTiming.LATEST_AVAILABLE.value,
    )
    parser.add_argument("--odds-minutes-before-start", type=int, default=None)


def _handle_jravan_ingest(args: argparse.Namespace) -> dict[str, object]:
    export = ingest_jvdata_file_to_staging(
        args.raw_path,
        args.staging_dir,
        encoding=args.encoding,
        skip_unknown_records=not args.strict_unknown,
    )
    dataset = export.dataset
    return {
        "raw_path": str(args.raw_path),
        "staging_dir": str(args.staging_dir),
        "counts": {
            "races": len(dataset.races),
            "entries": len(dataset.entries),
            "results": len(dataset.results),
            "odds": len(dataset.odds),
            "payouts": len(dataset.payouts),
            "skipped_records": len(dataset.skipped_records),
        },
        "csv_paths": {name: str(path) for name, path in export.csv_paths.items()},
    }


def _handle_jravan_ingest_dir(args: argparse.Namespace) -> dict[str, object]:
    export = ingest_jvdata_directory_to_staging(
        args.raw_dir,
        args.staging_dir,
        pattern=args.pattern,
        recursive=not args.no_recursive,
        encoding=args.encoding,
        skip_unknown_records=not args.strict_unknown,
    )
    dataset = export.dataset
    return {
        "raw_dir": str(args.raw_dir),
        "staging_dir": str(args.staging_dir),
        "pattern": args.pattern,
        "recursive": not args.no_recursive,
        "counts": {
            "races": len(dataset.races),
            "entries": len(dataset.entries),
            "results": len(dataset.results),
            "odds": len(dataset.odds),
            "payouts": len(dataset.payouts),
            "skipped_records": len(dataset.skipped_records),
        },
        "csv_paths": {name: str(path) for name, path in export.csv_paths.items()},
    }


def _handle_jravan_s3_pull_raw(args: argparse.Namespace) -> dict[str, object]:
    plan = build_jravan_s3_raw_sync_plan(
        run_id=args.run_id,
        local_raw_root=args.local_raw_root,
        bucket=args.bucket,
        prefix=args.prefix,
        aws_cli=args.aws_cli,
    )
    result = sync_jravan_raw_from_s3(plan, dry_run=args.dry_run)
    return {
        "bucket": plan.bucket,
        "prefix": plan.prefix,
        "run_id": plan.run_id,
        "s3_uri": plan.s3_uri,
        "local_dir": str(plan.local_dir),
        "command": render_sync_command(plan.command),
        "executed": result.executed,
        "returncode": result.returncode,
    }


def _handle_jravan_build_replay_dataset(
    args: argparse.Namespace,
) -> dict[str, object]:
    export = build_replay_dataset_from_staging(
        args.staging_dir,
        args.output_dir,
        feature_version=args.feature_version,
        max_odds_captured_at=args.max_odds_captured_at,
        odds_staging_dirs=args.odds_staging_dir,
    )
    return {
        "staging_dir": str(args.staging_dir),
        "output_dir": str(args.output_dir),
        "csv_paths": {name: str(path) for name, path in export.csv_paths.items()},
        "report_path": str(export.report_path),
        "report": replay_dataset_report_to_dict(export.report),
    }


def _handle_jravan_build_quinella_replay_dataset(
    args: argparse.Namespace,
) -> dict[str, object]:
    export = build_quinella_replay_dataset_from_staging(
        args.odds_staging_dir,
        args.payouts_csv,
        args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return {
        "odds_staging_dir": str(args.odds_staging_dir),
        "payouts_csv": str(args.payouts_csv),
        "output_dir": str(args.output_dir),
        "csv_paths": {name: str(path) for name, path in export.csv_paths.items()},
        "report_path": str(export.report_path),
        "report": quinella_replay_dataset_report_to_dict(export.report),
    }


def _handle_jravan_build_quinella_replay_dataset_raw(
    args: argparse.Namespace,
) -> dict[str, object]:
    export = build_quinella_replay_dataset_from_raw(
        args.raw_dir,
        args.payouts_csv,
        args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        pattern=args.pattern,
        recursive=not args.no_recursive,
        encoding=args.encoding,
        write_timeseries=args.write_timeseries,
    )
    return {
        "raw_dir": str(args.raw_dir),
        "payouts_csv": str(args.payouts_csv),
        "output_dir": str(args.output_dir),
        "pattern": args.pattern,
        "recursive": not args.no_recursive,
        "write_timeseries": args.write_timeseries,
        "csv_paths": {name: str(path) for name, path in export.csv_paths.items()},
        "report_path": str(export.report_path),
        "report": quinella_replay_dataset_report_to_dict(export.report),
    }


def _handle_jravan_data_qa(args: argparse.Namespace) -> dict[str, object]:
    return build_replay_data_quality_report(args.dataset_dir, args.output_path)


def _handle_market_replay(args: argparse.Namespace) -> dict[str, object]:
    dataset_dir = args.dataset_dir
    backtest_config = _backtest_config_from_args(args)
    result = run_market_replay(
        race_repository=CsvRaceRepository(dataset_dir / "races.csv"),
        odds_repository=CsvOddsRepository(_replay_odds_csv_path(dataset_dir)),
        result_repository=CsvResultRepository(dataset_dir / "results.csv"),
        feature_repository=CsvFeatureRepository(dataset_dir / "features.csv"),
        start_date=args.start_date,
        end_date=args.end_date,
        as_of=args.as_of,
        feature_version=args.feature_version,
        backtest_config=backtest_config,
    )
    summary = {
        "dataset_dir": str(dataset_dir),
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "as_of": args.as_of.isoformat(),
        "feature_version": args.feature_version,
        "backtest_config": backtest_config_to_dict(backtest_config),
        "counts": {
            "races": len(result.races),
            "feature_rows": len(result.feature_rows),
            "odds": len(result.odds),
            "results": len(result.results),
            "predictions": len(result.predictions),
            "bet_records": len(result.backtest_result.records),
            "bet_decisions": len(result.backtest_result.decisions),
        },
        "backtest": _performance_summary_to_dict(result.summary),
        "probability": _probability_summary_to_dict(result.probability_summary),
    }
    if args.backtest_report_dir is not None:
        paths = write_backtest_artifacts(
            result=result.backtest_result,
            config=backtest_config,
            output_dir=args.backtest_report_dir,
        )
        summary["backtest_artifacts"] = {
            "config_path": str(paths.config_path),
            "decision_report_path": str(paths.decision_report_path),
        }
    return summary


def _handle_quinella_sim(args: argparse.Namespace) -> dict[str, object]:
    result = run_quinella_simulation_from_csv(
        args.odds_csv,
        args.payouts_csv,
        args.artifact_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        config=QuinellaSimulationConfig(
            initial_bankroll_jpy=args.initial_bankroll_jpy,
            stake_jpy=args.stake_jpy,
            strategy=args.strategy,
            minimum_edge=args.minimum_edge,
            max_bets_per_race=args.max_bets_per_race,
            require_payout_for_race=args.require_payout_for_race,
        ),
    )
    return {
        "summary": result.summary,
        "summary_path": str(result.summary_path),
        "decisions_path": str(result.decisions_path),
    }


def _replay_odds_csv_path(dataset_dir: Path) -> Path:
    odds_timeseries_path = dataset_dir / "odds_timeseries.csv"
    if odds_timeseries_path.exists():
        return odds_timeseries_path
    return dataset_dir / "odds.csv"


def _handle_lightgbm_train(args: argparse.Namespace) -> dict[str, object]:
    result = run_lightgbm_training_from_csv(
        args.dataset_dir,
        args.artifact_dir,
        train_end_date=args.train_end_date,
        valid_start_date=args.valid_start_date,
        valid_end_date=args.valid_end_date,
        as_of=args.as_of,
        feature_version=args.feature_version,
        random_seed=args.random_seed,
        model_version=args.model_version,
        exclude_feature_names=args.exclude_feature,
    )
    summary = lightgbm_training_result_to_dict(result)
    summary["dataset_dir"] = str(args.dataset_dir)
    summary["artifact_dir"] = str(args.artifact_dir)
    summary["split"] = {
        "train_end_date": args.train_end_date.isoformat(),
        "valid_start_date": args.valid_start_date.isoformat(),
        "valid_end_date": args.valid_end_date.isoformat(),
        "as_of": args.as_of.isoformat(),
    }
    return summary


def _handle_lightgbm_ablation(args: argparse.Namespace) -> dict[str, object]:
    return run_lightgbm_ablation_from_csv(
        args.dataset_dir,
        args.artifact_dir,
        train_end_date=args.train_end_date,
        valid_start_date=args.valid_start_date,
        valid_end_date=args.valid_end_date,
        as_of=args.as_of,
        feature_version=args.feature_version,
        random_seed=args.random_seed,
        model_version=args.model_version,
    )


def _handle_level0_oof(args: argparse.Namespace) -> dict[str, object]:
    result = run_level0_oof_from_csv(
        args.dataset_dir,
        args.artifact_dir,
        validation_start_date=args.validation_start_date,
        validation_end_date=args.validation_end_date,
        as_of=args.as_of,
        feature_version=args.feature_version,
        model_keys=args.model_key,
        random_seed=args.random_seed,
        use_odds_timeseries=args.use_odds_timeseries,
    )
    summary = oof_run_result_to_dict(result)
    summary["dataset_dir"] = str(args.dataset_dir)
    summary["artifact_dir"] = str(args.artifact_dir)
    return summary


def _handle_stacking_build_meta_dataset(
    args: argparse.Namespace,
) -> dict[str, object]:
    result = build_meta_dataset_from_csv(
        args.predictions_csv,
        args.results_csv,
        args.output_dir,
        prediction_role=args.prediction_role,
        target=args.target,
        drop_incomplete_rows=not args.keep_incomplete_rows,
    )
    return meta_dataset_build_result_to_dict(result)


def _handle_stacking_train_meta(args: argparse.Namespace) -> dict[str, object]:
    result = train_logistic_meta_learner_from_csv(
        args.meta_features_csv,
        args.artifact_dir,
        holdout_fold_id=args.holdout_fold_id,
        feature_columns=args.feature_column,
        model_version=args.model_version,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        l2=args.l2,
    )
    return meta_learner_training_result_to_dict(result)


def _handle_stacking_search_blend(args: argparse.Namespace) -> dict[str, object]:
    result = search_convex_blend_from_csv(
        args.meta_features_csv,
        args.artifact_dir,
        holdout_fold_id=args.holdout_fold_id,
        feature_columns=args.feature_column,
        model_version=args.model_version,
        grid_step=args.grid_step,
    )
    return blend_search_result_to_dict(result)


def _handle_stacking_phase4_study(args: argparse.Namespace) -> dict[str, object]:
    result = run_phase4_study_from_csv(
        args.meta_features_csv,
        args.races_csv,
        args.artifact_dir,
        min_train_folds=args.min_train_folds,
        blend_grid_step=args.blend_grid_step,
        logistic_learning_rate=args.logistic_learning_rate,
        logistic_max_iterations=args.logistic_max_iterations,
        logistic_l2=args.logistic_l2,
    )
    return phase4_study_result_to_dict(result)


def _handle_model_registry_register_phase4(
    args: argparse.Namespace,
) -> dict[str, object]:
    result = build_phase4_model_registry_from_report(
        args.phase4_report_path,
        args.output_path,
        candidate_method=args.candidate_method,
        model_version=args.model_version,
        minimum_log_loss_improvement=args.minimum_log_loss_improvement,
    )
    return model_registry_to_dict(result)


def _handle_paper_trading_run(args: argparse.Namespace) -> dict[str, object]:
    result = run_paper_trading_from_csv(
        args.predictions_csv,
        args.dataset_dir,
        args.artifact_dir,
        method=args.method,
        as_of=args.as_of,
        model_version=args.model_version,
        backtest_config=_backtest_config_from_args(args),
        use_odds_timeseries=not args.use_compact_odds,
    )
    return paper_trading_result_to_dict(result)


def _handle_daily_paper_trading_run(args: argparse.Namespace) -> dict[str, object]:
    result = run_daily_paper_trading_from_csv(
        args.dataset_dir,
        args.model_registry_path,
        args.artifact_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        as_of=args.as_of,
        feature_version=args.feature_version,
        train_end_date=args.train_end_date,
        candidate_method=args.candidate_method,
        backtest_config=_backtest_config_from_args(args),
        use_odds_timeseries=not args.use_compact_odds,
        require_registry_approval=not args.allow_unapproved_registry,
    )
    return daily_paper_trading_result_to_dict(result)


def _handle_paper_trading_monitoring_summary(
    args: argparse.Namespace,
) -> dict[str, object]:
    result = summarize_paper_trading_reports(
        report_paths=args.report_path,
        output_path=args.output_path,
    )
    return paper_trading_monitoring_summary_result_to_dict(result)


def _handle_jravan_daily_market_replay(args: argparse.Namespace) -> dict[str, object]:
    paths = _daily_paths(args.workspace_root, args.run_id)
    backtest_config = _backtest_config_from_args(args)
    s3_plan = build_jravan_s3_raw_sync_plan(
        run_id=args.run_id,
        local_raw_root=paths["local_raw_root"],
        bucket=args.bucket,
        prefix=args.prefix,
        aws_cli=args.aws_cli,
    )

    summary: dict[str, object] = {
        "run_id": args.run_id,
        "workspace_root": str(args.workspace_root),
        "paths": {name: str(path) for name, path in paths.items()},
        "s3_pull": {
            "skipped": args.skip_s3_pull,
            "s3_uri": s3_plan.s3_uri,
            "local_dir": str(s3_plan.local_dir),
            "command": render_sync_command(s3_plan.command),
            "executed": False,
            "returncode": None,
        },
        "dry_run": args.dry_run,
        "backtest_config": backtest_config_to_dict(backtest_config),
    }

    if args.dry_run:
        return summary

    if not args.skip_s3_pull:
        s3_result = sync_jravan_raw_from_s3(s3_plan)
        summary["s3_pull"] = {
            "skipped": False,
            "s3_uri": s3_plan.s3_uri,
            "local_dir": str(s3_plan.local_dir),
            "command": render_sync_command(s3_plan.command),
            "executed": s3_result.executed,
            "returncode": s3_result.returncode,
        }

    ingest_export = ingest_jvdata_directory_to_staging(
        paths["raw_dir"],
        paths["staging_dir"],
    )
    summary["ingest"] = {
        "counts": {
            "races": len(ingest_export.dataset.races),
            "entries": len(ingest_export.dataset.entries),
            "results": len(ingest_export.dataset.results),
            "odds": len(ingest_export.dataset.odds),
            "payouts": len(ingest_export.dataset.payouts),
            "skipped_records": len(ingest_export.dataset.skipped_records),
        },
        "csv_paths": {
            name: str(path) for name, path in ingest_export.csv_paths.items()
        },
    }

    replay_export = build_replay_dataset_from_staging(
        paths["staging_dir"],
        paths["replay_dir"],
        feature_version=args.feature_version,
    )
    summary["replay_dataset"] = {
        "csv_paths": {
            name: str(path) for name, path in replay_export.csv_paths.items()
        },
        "report_path": str(replay_export.report_path),
        "report": replay_dataset_report_to_dict(replay_export.report),
    }

    replay_result = run_market_replay(
        race_repository=CsvRaceRepository(paths["replay_dir"] / "races.csv"),
        odds_repository=CsvOddsRepository(paths["replay_dir"] / "odds.csv"),
        result_repository=CsvResultRepository(paths["replay_dir"] / "results.csv"),
        feature_repository=CsvFeatureRepository(paths["replay_dir"] / "features.csv"),
        start_date=args.start_date,
        end_date=args.end_date,
        as_of=args.as_of,
        feature_version=args.feature_version,
        backtest_config=backtest_config,
    )
    summary["market_replay"] = {
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "as_of": args.as_of.isoformat(),
        "feature_version": args.feature_version,
        "counts": {
            "races": len(replay_result.races),
            "feature_rows": len(replay_result.feature_rows),
            "odds": len(replay_result.odds),
            "results": len(replay_result.results),
            "predictions": len(replay_result.predictions),
            "bet_records": len(replay_result.backtest_result.records),
            "bet_decisions": len(replay_result.backtest_result.decisions),
        },
        "backtest": _performance_summary_to_dict(replay_result.summary),
        "probability": _probability_summary_to_dict(
            replay_result.probability_summary
        ),
    }
    artifact_dir = args.backtest_report_dir or paths["backtest_dir"]
    artifact_paths = write_backtest_artifacts(
        result=replay_result.backtest_result,
        config=backtest_config,
        output_dir=artifact_dir,
    )
    summary["backtest_artifacts"] = {
        "config_path": str(artifact_paths.config_path),
        "decision_report_path": str(artifact_paths.decision_report_path),
    }
    _write_json_file(paths["market_report_path"], summary)
    summary["report_path"] = str(paths["market_report_path"])
    return summary


def _handle_jravan_preview(args: argparse.Namespace) -> dict[str, object]:
    lines_written = write_jvdata_utf8_preview(
        args.raw_path,
        args.output_path,
        encoding=args.encoding,
        drop_empty_lines=not args.keep_empty_lines,
    )
    return {
        "raw_path": str(args.raw_path),
        "output_path": str(args.output_path),
        "encoding": args.encoding,
        "lines_written": lines_written,
    }


def _daily_paths(workspace_root: Path, run_id: str) -> dict[str, Path]:
    return {
        "local_raw_root": workspace_root / "raw" / "jravan",
        "raw_dir": workspace_root / "raw" / "jravan" / run_id,
        "staging_dir": workspace_root / "interim" / "jravan" / run_id,
        "replay_dir": workspace_root / "processed" / "jravan" / run_id / "replay",
        "backtest_dir": workspace_root / "processed" / "jravan" / run_id / "backtest",
        "market_report_path": (
            workspace_root
            / "processed"
            / "jravan"
            / run_id
            / "market_replay_report.json"
        ),
    }


def _parse_cli_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected ISO date, got {value!r}") from exc


def _parse_cli_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected ISO datetime, got {value!r}"
        ) from exc


def _backtest_config_from_args(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_bankroll_jpy=args.initial_bankroll_jpy,
        kelly_config=KellyConfig(
            fractional_kelly=args.fractional_kelly,
            max_stake_fraction=args.max_stake_fraction,
            minimum_edge=args.minimum_edge,
            stake_unit_jpy=args.stake_unit_jpy,
        ),
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        max_stake_per_race_jpy=args.max_stake_per_race_jpy,
        max_daily_loss_jpy=args.max_daily_loss_jpy,
        odds_timing=args.odds_timing,
        odds_minutes_before_start=args.odds_minutes_before_start,
    )


def _performance_summary_to_dict(summary: PerformanceSummary) -> dict[str, object]:
    return {
        "total_bets": summary.total_bets,
        "wins": summary.wins,
        "total_staked_jpy": summary.total_staked_jpy,
        "total_payout_jpy": summary.total_payout_jpy,
        "net_profit_jpy": summary.net_profit_jpy,
        "roi": summary.roi,
        "hit_rate": summary.hit_rate,
        "turnover": summary.turnover,
        "max_drawdown": summary.max_drawdown,
        "final_bankroll_jpy": summary.final_bankroll_jpy,
    }


def _probability_summary_to_dict(summary: ProbabilitySummary) -> dict[str, object]:
    return {
        "observations": summary.observations,
        "positives": summary.positives,
        "mean_predicted_probability": summary.mean_predicted_probability,
        "empirical_rate": summary.empirical_rate,
        "log_loss": summary.log_loss,
        "brier_score": summary.brier_score,
        "expected_calibration_error": summary.expected_calibration_error,
        "bins": [
            {
                "lower_bound": calibration_bin.lower_bound,
                "upper_bound": calibration_bin.upper_bound,
                "count": calibration_bin.count,
                "positives": calibration_bin.positives,
                "mean_predicted_probability": calibration_bin.mean_predicted_probability,
                "empirical_rate": calibration_bin.empirical_rate,
                "absolute_error": calibration_bin.absolute_error,
            }
            for calibration_bin in summary.bins
        ],
    }


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
