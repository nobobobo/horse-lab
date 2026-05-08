"""Build replay-ready CSV datasets from JRA-VAN staging exports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from horse_lab.data.csv_parsing import (
    parse_entry_row,
    parse_odds_quote_row,
    parse_race_row,
    parse_result_row,
    read_csv_rows,
)
from horse_lab.data.jravan.exporters import (
    write_odds_csv,
    write_races_csv,
    write_results_csv,
)
from horse_lab.schemas import (
    BetType,
    Entry,
    FeatureName,
    FeatureRow,
    OddsQuote,
    Race,
    RaceId,
    Result,
    RunnerId,
)


DEFAULT_REPLAY_FEATURE_VERSION = "jravan-replay-v1"
REPLAY_REPORT_FILENAME = "replay_dataset_report.json"

FEATURE_CSV_BASE_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "as_of",
    "feature_version",
)
FEATURE_NAMES: tuple[FeatureName, ...] = (
    FeatureName("horse_number"),
    FeatureName("gate_number"),
    FeatureName("carried_weight_kg"),
    FeatureName("age"),
    FeatureName("body_weight_kg"),
    FeatureName("starter"),
)
FEATURE_CSV_FIELDS: tuple[str, ...] = FEATURE_CSV_BASE_FIELDS + tuple(
    f"feature__{name}" for name in FEATURE_NAMES
)


@dataclass(frozen=True)
class SkippedReplayRace:
    race_id: RaceId
    reason: str
    field_size: int | None
    entries: int
    results: int
    win_odds: int


@dataclass(frozen=True)
class ReplayDatasetReport:
    feature_version: str
    odds_policy: str
    max_odds_captured_at: datetime | None
    input_races: int
    input_entries: int
    input_results: int
    input_odds: int
    races_written: int
    feature_rows_written: int
    results_written: int
    odds_written: int
    skipped_races: tuple[SkippedReplayRace, ...]


@dataclass(frozen=True)
class ReplayDatasetExport:
    csv_paths: Mapping[str, Path]
    report_path: Path
    report: ReplayDatasetReport


def build_replay_dataset_from_staging(
    staging_dir: Path | str,
    output_dir: Path | str,
    *,
    feature_version: str = DEFAULT_REPLAY_FEATURE_VERSION,
    max_odds_captured_at: datetime | None = None,
) -> ReplayDatasetExport:
    """Create a replay-ready dataset from canonical JRA-VAN staging CSVs.

    The output intentionally keeps only the latest win quote per runner. That
    makes the generated dataset a deterministic final-odds replay fixture, while
    richer point-in-time odds replay can be layered on top later.
    """

    source = Path(staging_dir)
    target = Path(output_dir)

    races = tuple(parse_race_row(row) for row in read_csv_rows(source / "races.csv"))
    entries = tuple(
        parse_entry_row(row) for row in read_csv_rows(source / "entries.csv")
    )
    results = tuple(
        parse_result_row(row) for row in read_csv_rows(source / "results.csv")
    )
    odds = tuple(
        parse_odds_quote_row(row) for row in read_csv_rows(source / "odds.csv")
    )

    entries_by_race = _group_entries_by_race(entries)
    results_by_race = _group_results_by_race(results)
    win_odds_by_race = _group_latest_win_odds_by_race(
        odds,
        max_captured_at=max_odds_captured_at,
    )

    selected_races: list[Race] = []
    selected_results: list[Result] = []
    selected_odds: list[OddsQuote] = []
    feature_rows: list[FeatureRow] = []
    skipped_races: list[SkippedReplayRace] = []

    for race in sorted(
        races,
        key=lambda item: (item.race_date, item.venue, item.race_number, item.race_id),
    ):
        race_entries = tuple(
            entry
            for entry in entries_by_race.get(race.race_id, ())
            if not entry.is_scratched
        )
        result_by_runner = results_by_race.get(race.race_id, {})
        odds_by_runner = win_odds_by_race.get(race.race_id, {})
        entry_runner_ids = {entry.runner_id for entry in race_entries}
        result_runner_ids = set(result_by_runner)
        odds_runner_ids = set(odds_by_runner)

        skip_reason = _complete_replay_skip_reason(
            race=race,
            entries=race_entries,
            entry_runner_ids=entry_runner_ids,
            result_runner_ids=result_runner_ids,
            odds_runner_ids=odds_runner_ids,
        )
        if skip_reason is not None:
            skipped_races.append(
                SkippedReplayRace(
                    race_id=race.race_id,
                    reason=skip_reason,
                    field_size=race.field_size,
                    entries=len(race_entries),
                    results=len(result_runner_ids),
                    win_odds=len(odds_runner_ids),
                )
            )
            continue

        race_feature_as_of = max(
            odds_by_runner[runner_id].captured_at for runner_id in entry_runner_ids
        )
        selected_races.append(replace(race, field_size=len(race_entries)))
        for entry in sorted(race_entries, key=lambda item: item.horse_number):
            selected_results.append(result_by_runner[entry.runner_id])
            selected_odds.append(odds_by_runner[entry.runner_id])
            feature_rows.append(
                _feature_row_from_entry(
                    entry,
                    as_of=race_feature_as_of,
                    feature_version=feature_version,
                )
            )

    csv_paths = {
        "races": target / "races.csv",
        "features": target / "features.csv",
        "results": target / "results.csv",
        "odds": target / "odds.csv",
    }
    write_races_csv(csv_paths["races"], selected_races)
    write_feature_rows_csv(csv_paths["features"], feature_rows)
    write_results_csv(csv_paths["results"], selected_results)
    write_odds_csv(csv_paths["odds"], selected_odds)

    report = ReplayDatasetReport(
        feature_version=feature_version,
        odds_policy="latest_win_quote_per_runner",
        max_odds_captured_at=max_odds_captured_at,
        input_races=len(races),
        input_entries=len(entries),
        input_results=len(results),
        input_odds=len(odds),
        races_written=len(selected_races),
        feature_rows_written=len(feature_rows),
        results_written=len(selected_results),
        odds_written=len(selected_odds),
        skipped_races=tuple(skipped_races),
    )
    report_path = target / REPLAY_REPORT_FILENAME
    _write_report(report_path, report)

    return ReplayDatasetExport(
        csv_paths=csv_paths,
        report_path=report_path,
        report=report,
    )


def write_feature_rows_csv(
    path: Path | str,
    feature_rows: Sequence[FeatureRow],
) -> None:
    sorted_rows = sorted(
        feature_rows,
        key=lambda row: (str(row.race_id), str(row.runner_id), row.as_of),
    )
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_feature_row_to_csv_row(row) for row in sorted_rows)


def replay_dataset_report_to_dict(
    report: ReplayDatasetReport,
) -> dict[str, object]:
    return {
        "feature_version": report.feature_version,
        "odds_policy": report.odds_policy,
        "max_odds_captured_at": _render_optional_datetime(
            report.max_odds_captured_at
        ),
        "input_counts": {
            "races": report.input_races,
            "entries": report.input_entries,
            "results": report.input_results,
            "odds": report.input_odds,
        },
        "output_counts": {
            "races": report.races_written,
            "feature_rows": report.feature_rows_written,
            "results": report.results_written,
            "odds": report.odds_written,
        },
        "skipped_races": [
            {
                "race_id": str(skipped.race_id),
                "reason": skipped.reason,
                "field_size": skipped.field_size,
                "entries": skipped.entries,
                "results": skipped.results,
                "win_odds": skipped.win_odds,
            }
            for skipped in report.skipped_races
        ],
    }


def _group_entries_by_race(entries: Sequence[Entry]) -> dict[RaceId, tuple[Entry, ...]]:
    grouped: dict[RaceId, list[Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.race_id, []).append(entry)
    return {race_id: tuple(values) for race_id, values in grouped.items()}


def _group_results_by_race(
    results: Sequence[Result],
) -> dict[RaceId, dict[RunnerId, Result]]:
    grouped: dict[RaceId, dict[RunnerId, Result]] = {}
    for result in results:
        grouped.setdefault(result.race_id, {})[result.runner_id] = result
    return grouped


def _group_latest_win_odds_by_race(
    odds: Sequence[OddsQuote],
    *,
    max_captured_at: datetime | None,
) -> dict[RaceId, dict[RunnerId, OddsQuote]]:
    grouped: dict[RaceId, dict[RunnerId, OddsQuote]] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN:
            continue
        if max_captured_at is not None and quote.captured_at > max_captured_at:
            continue
        by_runner = grouped.setdefault(quote.race_id, {})
        previous = by_runner.get(quote.runner_id)
        if previous is None or quote.captured_at > previous.captured_at:
            by_runner[quote.runner_id] = quote
    return grouped


def _complete_replay_skip_reason(
    *,
    race: Race,
    entries: tuple[Entry, ...],
    entry_runner_ids: set[RunnerId],
    result_runner_ids: set[RunnerId],
    odds_runner_ids: set[RunnerId],
) -> str | None:
    if not entries:
        return "missing_entries"
    if race.field_size is not None and len(entries) != race.field_size:
        return "field_size_mismatch"
    if result_runner_ids != entry_runner_ids:
        return "result_runner_mismatch"
    if odds_runner_ids != entry_runner_ids:
        return "win_odds_runner_mismatch"
    return None


def _feature_row_from_entry(
    entry: Entry,
    *,
    as_of: datetime,
    feature_version: str,
) -> FeatureRow:
    return FeatureRow(
        race_id=entry.race_id,
        runner_id=entry.runner_id,
        as_of=as_of,
        feature_version=feature_version,
        values={
            FeatureName("horse_number"): entry.horse_number,
            FeatureName("gate_number"): entry.gate_number,
            FeatureName("carried_weight_kg"): entry.carried_weight_kg,
            FeatureName("age"): entry.age,
            FeatureName("body_weight_kg"): entry.body_weight_kg,
            FeatureName("starter"): True,
        },
    )


def _feature_row_to_csv_row(row: FeatureRow) -> dict[str, str]:
    rendered = {
        "race_id": str(row.race_id),
        "runner_id": str(row.runner_id),
        "as_of": row.as_of.isoformat(),
        "feature_version": row.feature_version,
    }
    for feature_name in FEATURE_NAMES:
        rendered[f"feature__{feature_name}"] = _render_feature_value(
            row.values.get(feature_name)
        )
    return rendered


def _render_feature_value(value: float | int | bool | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_report(path: Path, report: ReplayDatasetReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(replay_dataset_report_to_dict(report), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _render_optional_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
