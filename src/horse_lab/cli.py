"""Command-line tools for local horse_lab workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from horse_lab.data.jravan import (
    ingest_jvdata_file_to_staging,
    write_jvdata_utf8_preview,
)
from horse_lab.data.jravan.raw import JV_DATA_ENCODING


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


if __name__ == "__main__":
    raise SystemExit(main())
