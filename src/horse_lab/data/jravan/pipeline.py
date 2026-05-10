"""JRA-VAN raw dump to canonical staging pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, TypeVar

from horse_lab.data.jravan.exporters import write_staging_csvs
from horse_lab.data.jravan.layouts import (
    parse_minimal_h1_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.mappers import (
    build_jravan_race_id,
    map_h1_record_to_pool_sizes,
    map_hr_record_to_payout_rows,
    map_o1_record_to_odds_quotes,
    map_o2_record_to_odds_quotes,
    map_ra_record_to_race,
    map_se_record_to_entry,
    map_se_record_to_result,
)
from horse_lab.data.jravan.raw import (
    JV_DATA_ENCODING,
    JvDataRecord,
    read_jvdata_records,
)
from horse_lab.schemas import BetType, Entry, OddsQuote, Race, Result


MappedRecord = TypeVar("MappedRecord")


@dataclass(frozen=True)
class SkippedJvDataRecord:
    """A raw JV-Data record intentionally ignored by the minimal ingest slice."""

    record_type: str
    line_number: int | None
    source_path: Path | None
    reason: str


@dataclass(frozen=True)
class JraVanMappedDataset:
    """Canonical objects produced from a JV-Data raw dump."""

    races: tuple[Race, ...]
    entries: tuple[Entry, ...]
    results: tuple[Result, ...]
    odds: tuple[OddsQuote, ...]
    payouts: tuple[Mapping[str, object], ...] = ()
    skipped_records: tuple[SkippedJvDataRecord, ...] = ()


@dataclass(frozen=True)
class JraVanStagingExport:
    """Result of writing a mapped JRA-VAN dataset to staging CSV files."""

    dataset: JraVanMappedDataset
    csv_paths: Mapping[str, Path]


def ingest_jvdata_file_to_staging(
    raw_path: Path | str,
    staging_dir: Path | str,
    *,
    encoding: str = JV_DATA_ENCODING,
    skip_unknown_records: bool = True,
) -> JraVanStagingExport:
    """Read a JV-Data text dump and export mapped canonical staging CSVs."""

    records = read_jvdata_records(Path(raw_path), encoding=encoding)
    dataset = map_jvdata_records(
        records,
        skip_unknown_records=skip_unknown_records,
    )
    csv_paths = write_staging_csvs(
        staging_dir,
        races=dataset.races,
        entries=dataset.entries,
        results=dataset.results,
        odds=dataset.odds,
        payouts=dataset.payouts,
    )
    return JraVanStagingExport(dataset=dataset, csv_paths=csv_paths)


def ingest_jvdata_files_to_staging(
    raw_paths: Iterable[Path | str],
    staging_dir: Path | str,
    *,
    encoding: str = JV_DATA_ENCODING,
    skip_unknown_records: bool = True,
) -> JraVanStagingExport:
    """Read multiple JV-Data text dumps and export one combined staging dataset."""

    paths = tuple(Path(path) for path in raw_paths)
    if not paths:
        raise ValueError("At least one JV-Data raw path is required")

    records: list[JvDataRecord] = []
    for path in paths:
        records.extend(read_jvdata_records(path, encoding=encoding))

    dataset = map_jvdata_records(
        records,
        skip_unknown_records=skip_unknown_records,
    )
    csv_paths = write_staging_csvs(
        staging_dir,
        races=dataset.races,
        entries=dataset.entries,
        results=dataset.results,
        odds=dataset.odds,
        payouts=dataset.payouts,
    )
    return JraVanStagingExport(dataset=dataset, csv_paths=csv_paths)


def ingest_jvdata_directory_to_staging(
    raw_dir: Path | str,
    staging_dir: Path | str,
    *,
    pattern: str = "*.txt",
    recursive: bool = True,
    encoding: str = JV_DATA_ENCODING,
    skip_unknown_records: bool = True,
) -> JraVanStagingExport:
    """Read a directory of JV-Data dumps and export one combined staging dataset."""

    raw_directory = Path(raw_dir)
    paths = tuple(_iter_raw_dump_paths(raw_directory, pattern, recursive=recursive))
    if not paths:
        raise ValueError(
            f"No JV-Data raw files found in {raw_directory} with pattern {pattern!r}"
        )

    return ingest_jvdata_files_to_staging(
        paths,
        staging_dir,
        encoding=encoding,
        skip_unknown_records=skip_unknown_records,
    )


def write_jvdata_utf8_preview(
    raw_path: Path | str,
    output_path: Path | str,
    *,
    encoding: str = JV_DATA_ENCODING,
    drop_empty_lines: bool = True,
) -> int:
    """Write a human-readable UTF-8 copy of a CP932 JV-Data text dump.

    The preview is for inspection only. Keep the original CP932 dump as the
    source of truth because JV-Data fixed-width offsets are byte-oriented.
    """

    raw = Path(raw_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines_written = 0
    with raw.open(encoding=encoding, newline="") as source:
        with output.open("w", encoding="utf-8", newline="\n") as destination:
            for line in source:
                normalized = line.rstrip("\r\n")
                if drop_empty_lines and not normalized:
                    continue
                destination.write(normalized)
                destination.write("\n")
                lines_written += 1

    return lines_written


def map_jvdata_records(
    records: Iterable[JvDataRecord],
    *,
    skip_unknown_records: bool = True,
) -> JraVanMappedDataset:
    """Map supported JV-Data records into canonical schemas.

    The minimal slice supports RA, SE, O1, and O2 records. Duplicate keys use
    last-record-wins semantics so a dump containing later JRA-VAN updates can
    supersede earlier snapshots deterministically.
    """

    races_by_id: dict[str, Race] = {}
    entries_by_runner_id: dict[str, Entry] = {}
    results_by_entry_key: dict[tuple[str, str], Result] = {}
    odds_by_quote_key: dict[tuple[str, str, str, str], OddsQuote] = {}
    payouts_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    pool_sizes_by_race_bet_type: dict[tuple[str, BetType], int] = {}
    skipped_records: list[SkippedJvDataRecord] = []

    for record in records:
        if record.record_type == "RA":
            race = _map_record(record, map_ra_record_to_race)
            races_by_id[str(race.race_id)] = race
            continue

        if record.record_type == "SE":
            if _is_unassigned_se_record(record):
                skipped_records.append(
                    SkippedJvDataRecord(
                        record_type=record.record_type,
                        line_number=record.line_number,
                        source_path=record.source_path,
                        reason="unassigned_runner_key",
                    )
                )
                continue

            entry = _map_record(record, map_se_record_to_entry)
            result_key = (str(entry.race_id), str(entry.runner_id))
            if _is_deleted_se_record(record):
                entries_by_runner_id.pop(str(entry.runner_id), None)
                results_by_entry_key.pop(result_key, None)
                skipped_records.append(
                    SkippedJvDataRecord(
                        record_type=record.record_type,
                        line_number=record.line_number,
                        source_path=record.source_path,
                        reason="deleted_runner_record",
                    )
                )
                continue

            result = _map_record(record, map_se_record_to_result)
            entries_by_runner_id[str(entry.runner_id)] = entry
            if result is None:
                results_by_entry_key.pop(result_key, None)
            else:
                results_by_entry_key[result_key] = result
            continue

        if record.record_type == "O1":
            quotes = _map_record(record, map_o1_record_to_odds_quotes)
            for quote in quotes:
                quote_key = (
                    str(quote.race_id),
                    str(quote.runner_id),
                    quote.bet_type.value,
                    quote.captured_at.isoformat(),
                )
                odds_by_quote_key[quote_key] = quote
            continue

        if record.record_type == "O2":
            quotes = _map_record(record, map_o2_record_to_odds_quotes)
            for quote in quotes:
                quote_key = (
                    str(quote.race_id),
                    str(quote.runner_id),
                    quote.bet_type.value,
                    quote.captured_at.isoformat(),
                )
                odds_by_quote_key[quote_key] = quote
            continue

        if record.record_type == "H1":
            fields = _map_record(record, parse_minimal_h1_fields)
            race_id = build_jravan_race_id(fields)
            pools = _map_record(record, map_h1_record_to_pool_sizes)
            for bet_type, pool_size_jpy in pools.items():
                pool_sizes_by_race_bet_type[(str(race_id), bet_type)] = pool_size_jpy
            continue

        if record.record_type == "HR":
            payout_rows = _map_record(record, map_hr_record_to_payout_rows)
            for row in payout_rows:
                payout_key = (
                    str(row["race_id"]),
                    str(row["runner_id"]),
                    str(row["bet_type"]),
                )
                payouts_by_key[payout_key] = dict(row)
            continue

        if skip_unknown_records:
            skipped_records.append(
                SkippedJvDataRecord(
                    record_type=record.record_type,
                    line_number=record.line_number,
                    source_path=record.source_path,
                    reason="unsupported_record_type",
                )
            )
            continue

        raise ValueError(
            "Unsupported JV-Data record type "
            f"{record.record_type!r} at {_format_record_location(record)}"
        )

    payouts = _fill_payout_pool_sizes(
        payouts_by_key.values(),
        pool_sizes_by_race_bet_type,
    )

    return JraVanMappedDataset(
        races=tuple(
            sorted(
                races_by_id.values(),
                key=lambda race: (race.race_date, race.venue, race.race_number),
            )
        ),
        entries=tuple(
            sorted(
                entries_by_runner_id.values(),
                key=lambda entry: (str(entry.race_id), entry.horse_number),
            )
        ),
        results=tuple(
            sorted(
                results_by_entry_key.values(),
                key=lambda result: (str(result.race_id), str(result.runner_id)),
            )
        ),
        odds=tuple(
            sorted(
                odds_by_quote_key.values(),
                key=lambda quote: (
                    str(quote.race_id),
                    str(quote.runner_id),
                    quote.captured_at,
                    quote.bet_type.value,
                ),
            )
        ),
        payouts=tuple(
            sorted(
                payouts,
                key=lambda row: (
                    str(row.get("race_id", "")),
                    str(row.get("bet_type", "")),
                    str(row.get("runner_id", "")),
                ),
            )
        ),
        skipped_records=tuple(skipped_records),
    )


def _map_record(
    record: JvDataRecord,
    mapper: Callable[[JvDataRecord], MappedRecord],
) -> MappedRecord:
    try:
        return mapper(record)
    except ValueError as exc:
        raise ValueError(
            f"Failed to map {record.record_type} record at "
            f"{_format_record_location(record)}: {exc}"
        ) from exc


def _format_record_location(record: JvDataRecord) -> str:
    source = str(record.source_path) if record.source_path is not None else "<memory>"
    if record.line_number is None:
        return source
    return f"{source}:{record.line_number}"


def _fill_payout_pool_sizes(
    payouts: Iterable[dict[str, object]],
    pool_sizes_by_race_bet_type: Mapping[tuple[str, BetType], int],
) -> list[dict[str, object]]:
    filled: list[dict[str, object]] = []
    for row in payouts:
        next_row = dict(row)
        if next_row.get("pool_size_jpy") in {None, ""}:
            bet_type = BetType(str(next_row["bet_type"]))
            next_row["pool_size_jpy"] = pool_sizes_by_race_bet_type.get(
                (str(next_row["race_id"]), bet_type)
            )
        filled.append(next_row)
    return filled


def _is_unassigned_se_record(record: JvDataRecord) -> bool:
    fields = parse_minimal_se_fields(record)
    return fields.get("horse_number", "").strip() in {"", "00"}


def _is_deleted_se_record(record: JvDataRecord) -> bool:
    fields = parse_minimal_se_fields(record)
    return fields.get("data_kubun", "").strip() == "9"


def _iter_raw_dump_paths(
    raw_dir: Path,
    pattern: str,
    *,
    recursive: bool,
) -> Iterable[Path]:
    iterator = raw_dir.rglob(pattern) if recursive else raw_dir.glob(pattern)
    yield from sorted(
        (
            path
            for path in iterator
            if path.is_file() and not _is_inspection_artifact(path)
        ),
        key=lambda path: path.as_posix(),
    )


def _is_inspection_artifact(path: Path) -> bool:
    return path.name.endswith((".utf8.txt", ".stdout.txt", ".stderr.txt"))
