"""CSV staging exporters for mapped JRA-VAN canonical objects."""

from __future__ import annotations

import csv
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from horse_lab.schemas import Entry, OddsQuote, Race, Result


RACE_CSV_FIELDS: tuple[str, ...] = (
    "race_id",
    "race_date",
    "venue",
    "race_number",
    "name",
    "surface",
    "distance_m",
    "direction",
    "track_condition",
    "weather",
    "grade",
    "start_time",
    "field_size",
)

ENTRY_CSV_FIELDS: tuple[str, ...] = (
    "runner_id",
    "race_id",
    "horse_id",
    "horse_name",
    "horse_symbol_code",
    "horse_number",
    "gate_number",
    "jockey_id",
    "trainer_id",
    "carried_weight_kg",
    "body_weight_kg",
    "body_weight_diff_kg",
    "age",
    "sex",
    "sex_code",
    "breed_code",
    "coat_color_code",
    "trainer_affiliation_code",
    "entry_win_odds",
    "entry_popularity_rank",
    "is_scratched",
)

RESULT_CSV_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "finish_position",
    "is_disqualified",
    "is_dead_heat",
    "final_time_seconds",
    "prize_jpy",
)

ODDS_CSV_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "bet_type",
    "captured_at",
    "odds",
    "popularity_rank",
    "pool_size_jpy",
    "source",
)

PAYOUT_CSV_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "bet_type",
    "finish_position",
    "is_win",
    "payout_jpy_per_100",
    "odds",
    "pool_size_jpy",
    "source",
)


def race_to_csv_row(race: Race) -> dict[str, str]:
    return {
        "race_id": _render_csv_value(race.race_id),
        "race_date": _render_csv_value(race.race_date),
        "venue": _render_csv_value(race.venue),
        "race_number": _render_csv_value(race.race_number),
        "name": _render_csv_value(race.name),
        "surface": _render_csv_value(race.surface),
        "distance_m": _render_csv_value(race.distance_m),
        "direction": _render_csv_value(race.direction),
        "track_condition": _render_csv_value(race.track_condition),
        "weather": _render_csv_value(race.weather),
        "grade": _render_csv_value(race.grade),
        "start_time": _render_csv_value(race.start_time),
        "field_size": _render_csv_value(race.field_size),
    }


def entry_to_csv_row(entry: Entry) -> dict[str, str]:
    return {
        "runner_id": _render_csv_value(entry.runner_id),
        "race_id": _render_csv_value(entry.race_id),
        "horse_id": _render_csv_value(entry.horse_id),
        "horse_name": _render_csv_value(entry.metadata.get("horse_name")),
        "horse_symbol_code": _render_csv_value(
            entry.metadata.get("horse_symbol_code")
        ),
        "horse_number": _render_csv_value(entry.horse_number),
        "gate_number": _render_csv_value(entry.gate_number),
        "jockey_id": _render_csv_value(entry.jockey_id),
        "trainer_id": _render_csv_value(entry.trainer_id),
        "carried_weight_kg": _render_csv_value(entry.carried_weight_kg),
        "body_weight_kg": _render_csv_value(entry.body_weight_kg),
        "body_weight_diff_kg": _render_csv_value(entry.body_weight_diff_kg),
        "age": _render_csv_value(entry.age),
        "sex": _render_csv_value(entry.metadata.get("sex")),
        "sex_code": _render_csv_value(entry.metadata.get("sex_code")),
        "breed_code": _render_csv_value(entry.metadata.get("breed_code")),
        "coat_color_code": _render_csv_value(entry.metadata.get("coat_color_code")),
        "trainer_affiliation_code": _render_csv_value(
            entry.metadata.get("trainer_affiliation_code")
        ),
        "entry_win_odds": _render_csv_value(entry.metadata.get("entry_win_odds")),
        "entry_popularity_rank": _render_csv_value(
            entry.metadata.get("entry_popularity_rank")
        ),
        "is_scratched": _render_csv_value(entry.is_scratched),
    }


def result_to_csv_row(result: Result) -> dict[str, str]:
    return {
        "race_id": _render_csv_value(result.race_id),
        "runner_id": _render_csv_value(result.runner_id),
        "finish_position": _render_csv_value(result.finish_position),
        "is_disqualified": _render_csv_value(result.is_disqualified),
        "is_dead_heat": _render_csv_value(result.is_dead_heat),
        "final_time_seconds": _render_csv_value(result.final_time_seconds),
        "prize_jpy": _render_csv_value(result.prize_jpy),
    }


def odds_quote_to_csv_row(quote: OddsQuote) -> dict[str, str]:
    return {
        "race_id": _render_csv_value(quote.race_id),
        "runner_id": _render_csv_value(quote.runner_id),
        "bet_type": _render_csv_value(quote.bet_type),
        "captured_at": _render_csv_value(quote.captured_at),
        "odds": _render_csv_value(quote.odds),
        "popularity_rank": _render_csv_value(quote.popularity_rank),
        "pool_size_jpy": _render_csv_value(quote.pool_size_jpy),
        "source": _render_csv_value(quote.source),
    }


def write_races_csv(path: Path | str, races: Sequence[Race]) -> None:
    sorted_races = sorted(
        races,
        key=lambda race: (race.race_date, race.venue, race.race_number),
    )
    _write_csv(
        path,
        RACE_CSV_FIELDS,
        (race_to_csv_row(race) for race in sorted_races),
    )


def write_entries_csv(path: Path | str, entries: Sequence[Entry]) -> None:
    sorted_entries = sorted(
        entries,
        key=lambda entry: (entry.race_id, entry.horse_number),
    )
    _write_csv(
        path,
        ENTRY_CSV_FIELDS,
        (entry_to_csv_row(entry) for entry in sorted_entries),
    )


def write_results_csv(path: Path | str, results: Sequence[Result]) -> None:
    sorted_results = sorted(
        results,
        key=lambda result: (result.race_id, result.runner_id),
    )
    _write_csv(
        path,
        RESULT_CSV_FIELDS,
        (result_to_csv_row(result) for result in sorted_results),
    )


def write_odds_csv(path: Path | str, odds: Sequence[OddsQuote]) -> None:
    sorted_odds = sorted(
        odds,
        key=lambda quote: (
            str(quote.race_id),
            str(quote.runner_id),
            quote.captured_at,
            quote.bet_type.value,
        ),
    )
    _write_csv(
        path,
        ODDS_CSV_FIELDS,
        (odds_quote_to_csv_row(quote) for quote in sorted_odds),
    )


def write_payouts_csv(
    path: Path | str,
    payouts: Sequence[dict[str, object]],
) -> None:
    _write_csv(
        path,
        PAYOUT_CSV_FIELDS,
        (
            {field: _render_csv_value(row.get(field)) for field in PAYOUT_CSV_FIELDS}
            for row in payouts
        ),
    )


def write_staging_csvs(
    directory: Path | str,
    *,
    races: Sequence[Race],
    entries: Sequence[Entry],
    results: Sequence[Result],
    odds: Sequence[OddsQuote] = (),
    payouts: Sequence[dict[str, object]] | None = None,
) -> dict[str, Path]:
    target = Path(directory)
    paths = {
        "races": target / "races.csv",
        "entries": target / "entries.csv",
        "results": target / "results.csv",
        "odds": target / "odds.csv",
    }
    if payouts is not None:
        paths["payouts"] = target / "payouts.csv"
    write_races_csv(paths["races"], races)
    write_entries_csv(paths["entries"], entries)
    write_results_csv(paths["results"], results)
    write_odds_csv(paths["odds"], odds)
    if payouts is not None:
        write_payouts_csv(paths["payouts"], payouts)
    return paths


def _write_csv(
    path: Path | str,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, str]],
) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)
