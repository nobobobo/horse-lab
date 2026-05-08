"""Command-line tools for local horse_lab workflows."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

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
    build_jravan_s3_raw_sync_plan,
    ingest_jvdata_directory_to_staging,
    ingest_jvdata_file_to_staging,
    render_sync_command,
    replay_dataset_report_to_dict,
    sync_jravan_raw_from_s3,
    write_jvdata_utf8_preview,
)
from horse_lab.data.jravan.raw import JV_DATA_ENCODING
from horse_lab.evaluation import PerformanceSummary, ProbabilitySummary
from horse_lab.pipelines import (
    lightgbm_training_result_to_dict,
    run_lightgbm_training_from_csv,
    run_market_replay,
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
    replay_dataset_parser.set_defaults(handler=_handle_jravan_build_replay_dataset)

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
    market_replay_parser.set_defaults(handler=_handle_market_replay)

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
    lightgbm_parser.set_defaults(handler=_handle_lightgbm_train)

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
    )
    return {
        "staging_dir": str(args.staging_dir),
        "output_dir": str(args.output_dir),
        "csv_paths": {name: str(path) for name, path in export.csv_paths.items()},
        "report_path": str(export.report_path),
        "report": replay_dataset_report_to_dict(export.report),
    }


def _handle_market_replay(args: argparse.Namespace) -> dict[str, object]:
    dataset_dir = args.dataset_dir
    result = run_market_replay(
        race_repository=CsvRaceRepository(dataset_dir / "races.csv"),
        odds_repository=CsvOddsRepository(dataset_dir / "odds.csv"),
        result_repository=CsvResultRepository(dataset_dir / "results.csv"),
        feature_repository=CsvFeatureRepository(dataset_dir / "features.csv"),
        start_date=args.start_date,
        end_date=args.end_date,
        as_of=args.as_of,
        feature_version=args.feature_version,
        initial_bankroll_jpy=args.initial_bankroll_jpy,
    )
    return {
        "dataset_dir": str(dataset_dir),
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "as_of": args.as_of.isoformat(),
        "feature_version": args.feature_version,
        "counts": {
            "races": len(result.races),
            "feature_rows": len(result.feature_rows),
            "odds": len(result.odds),
            "results": len(result.results),
            "predictions": len(result.predictions),
            "bet_records": len(result.backtest_result.records),
        },
        "backtest": _performance_summary_to_dict(result.summary),
        "probability": _probability_summary_to_dict(result.probability_summary),
    }


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


def _handle_jravan_daily_market_replay(args: argparse.Namespace) -> dict[str, object]:
    paths = _daily_paths(args.workspace_root, args.run_id)
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
        initial_bankroll_jpy=args.initial_bankroll_jpy,
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
        },
        "backtest": _performance_summary_to_dict(replay_result.summary),
        "probability": _probability_summary_to_dict(
            replay_result.probability_summary
        ),
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
