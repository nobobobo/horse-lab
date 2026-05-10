"""Build quinella replay datasets from O2 odds staging and HR payouts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

from horse_lab.data.csv_parsing import parse_odds_quote_row
from horse_lab.data.jravan.exporters import (
    ODDS_CSV_FIELDS,
    PAYOUT_CSV_FIELDS,
    odds_quote_to_csv_row,
    write_odds_csv,
)
from horse_lab.schemas import BetType, OddsQuote


QUINELLA_REPLAY_REPORT_FILENAME = "quinella_replay_dataset_report.json"


@dataclass(frozen=True)
class QuinellaReplayDatasetReport:
    start_date: date | None
    end_date: date | None
    input_odds: int
    input_payouts: int
    races_written: int
    odds_written: int
    odds_timeseries_written: int
    payouts_written: int


@dataclass(frozen=True)
class QuinellaReplayDatasetExport:
    csv_paths: Mapping[str, Path]
    report_path: Path
    report: QuinellaReplayDatasetReport


def build_quinella_replay_dataset_from_staging(
    odds_staging_dir: Path | str,
    payouts_csv_path: Path | str,
    output_dir: Path | str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> QuinellaReplayDatasetExport:
    """Create a pair-level replay dataset for quinella settlement tests."""

    odds_source = Path(odds_staging_dir) / "odds.csv"
    payout_source = Path(payouts_csv_path)
    target = Path(output_dir)

    csv_paths = {
        "odds": target / "odds.csv",
        "odds_timeseries": target / "odds_timeseries.csv",
        "payouts": target / "payouts.csv",
    }
    latest_odds, input_odds, odds_timeseries_written = _write_timeseries_and_latest(
        odds_source,
        csv_paths["odds_timeseries"],
        start_date=start_date,
        end_date=end_date,
    )
    race_ids = {str(quote.race_id) for quote in latest_odds}
    input_payouts, payouts_written = _write_selected_payouts(
        payout_source,
        csv_paths["payouts"],
        race_ids=race_ids,
        start_date=start_date,
        end_date=end_date,
    )
    write_odds_csv(csv_paths["odds"], latest_odds)

    report = QuinellaReplayDatasetReport(
        start_date=start_date,
        end_date=end_date,
        input_odds=input_odds,
        input_payouts=input_payouts,
        races_written=len(race_ids),
        odds_written=len(latest_odds),
        odds_timeseries_written=odds_timeseries_written,
        payouts_written=payouts_written,
    )
    report_path = target / QUINELLA_REPLAY_REPORT_FILENAME
    _write_report(report_path, report)
    return QuinellaReplayDatasetExport(
        csv_paths=csv_paths,
        report_path=report_path,
        report=report,
    )


def quinella_replay_dataset_report_to_dict(
    report: QuinellaReplayDatasetReport,
) -> dict[str, object]:
    return {
        "start_date": report.start_date.isoformat()
        if report.start_date is not None
        else None,
        "end_date": report.end_date.isoformat()
        if report.end_date is not None
        else None,
        "input_counts": {
            "odds": report.input_odds,
            "payouts": report.input_payouts,
        },
        "output_counts": {
            "races": report.races_written,
            "odds": report.odds_written,
            "odds_timeseries": report.odds_timeseries_written,
            "payouts": report.payouts_written,
        },
    }


def _write_timeseries_and_latest(
    source: Path,
    destination: Path,
    *,
    start_date: date | None,
    end_date: date | None,
) -> tuple[tuple[OddsQuote, ...], int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    latest: dict[tuple[str, str], OddsQuote] = {}
    input_odds = 0
    written = 0
    with source.open(newline="", encoding="utf-8") as source_handle:
        with destination.open("w", newline="", encoding="utf-8") as destination_handle:
            reader = csv.DictReader(source_handle)
            writer = csv.DictWriter(destination_handle, fieldnames=ODDS_CSV_FIELDS)
            writer.writeheader()
            for row in reader:
                input_odds += 1
                if row.get("bet_type") != BetType.QUINELLA.value:
                    continue
                if not _is_in_date_window(
                    row["race_id"],
                    start_date=start_date,
                    end_date=end_date,
                ):
                    continue
                quote = parse_odds_quote_row(row)
                writer.writerow(odds_quote_to_csv_row(quote))
                written += 1
                key = (str(quote.race_id), str(quote.runner_id))
                previous = latest.get(key)
                if previous is None or quote.captured_at > previous.captured_at:
                    latest[key] = quote
    return (
        tuple(
            sorted(
                latest.values(),
                key=lambda quote: (
                    str(quote.race_id),
                    str(quote.runner_id),
                    quote.captured_at,
                ),
            )
        ),
        input_odds,
        written,
    )


def _write_selected_payouts(
    source: Path,
    destination: Path,
    *,
    race_ids: set[str],
    start_date: date | None,
    end_date: date | None,
) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        _write_empty_csv(destination, PAYOUT_CSV_FIELDS)
        return 0, 0
    input_payouts = 0
    written = 0
    with source.open(newline="", encoding="utf-8") as source_handle:
        with destination.open("w", newline="", encoding="utf-8") as destination_handle:
            reader = csv.DictReader(source_handle)
            writer = csv.DictWriter(destination_handle, fieldnames=PAYOUT_CSV_FIELDS)
            writer.writeheader()
            for row in reader:
                input_payouts += 1
                if row.get("bet_type") != BetType.QUINELLA.value:
                    continue
                if row.get("race_id") not in race_ids:
                    continue
                if not _is_in_date_window(
                    row["race_id"],
                    start_date=start_date,
                    end_date=end_date,
                ):
                    continue
                writer.writerow(
                    {field: row.get(field, "") for field in PAYOUT_CSV_FIELDS}
                )
                written += 1
    return input_payouts, written


def _write_empty_csv(path: Path, fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _is_in_date_window(
    race_id: str,
    *,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    race_date = _race_date_from_id(race_id)
    if start_date is not None and race_date < start_date:
        return False
    if end_date is not None and race_date > end_date:
        return False
    return True


def _race_date_from_id(race_id: str) -> date:
    return date(
        int(race_id[0:4]),
        int(race_id[4:6]),
        int(race_id[6:8]),
    )


def _write_report(path: Path, report: QuinellaReplayDatasetReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            quinella_replay_dataset_report_to_dict(report),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
