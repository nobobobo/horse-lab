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
from horse_lab.data.master import (
    HorseMasterRecord,
    HorseRatingRecord,
    index_rating_history,
    latest_rating_as_of,
    read_horse_master_csv,
    read_horse_rating_history_csv,
)
from horse_lab.data.jravan.exporters import (
    write_entries_csv,
    write_odds_csv,
    write_payouts_csv,
    write_races_csv,
    write_results_csv,
)
from horse_lab.features import (
    PAST_PERFORMANCE_FEATURE_NAMES,
    build_past_performance_features,
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


DEFAULT_REPLAY_FEATURE_VERSION = "jravan-replay-v2"
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
    FeatureName("sex"),
    FeatureName("horse_symbol_code"),
    FeatureName("breed_code"),
    FeatureName("coat_color_code"),
    FeatureName("trainer_affiliation_code"),
    FeatureName("pedigree_sire_id"),
    FeatureName("pedigree_dam_id"),
    FeatureName("pedigree_damsire_id"),
    FeatureName("horse_birth_year"),
    FeatureName("horse_age_days_from_birth"),
    FeatureName("horse_rating"),
    FeatureName("horse_rating_delta_to_field_mean"),
    FeatureName("horse_rating_rank_in_race"),
    FeatureName("horse_rating_source"),
    FeatureName("jockey_id"),
    FeatureName("trainer_id"),
    FeatureName("body_weight_kg"),
    FeatureName("body_weight_diff_kg"),
    FeatureName("entry_win_odds"),
    FeatureName("entry_popularity_rank"),
    FeatureName("race_venue"),
    FeatureName("race_surface"),
    FeatureName("race_distance_m"),
    FeatureName("race_direction"),
    FeatureName("race_track_condition"),
    FeatureName("race_weather"),
    FeatureName("race_grade"),
    FeatureName("race_grade_group"),
    FeatureName("race_title_type"),
    FeatureName("race_has_title"),
    FeatureName("race_field_size"),
    FeatureName("starter"),
    FeatureName("odds_open"),
    FeatureName("odds_latest"),
    FeatureName("odds_min"),
    FeatureName("odds_max"),
    FeatureName("odds_snapshot_count"),
    FeatureName("odds_change_open_to_latest"),
    FeatureName("implied_probability_change_open_to_latest"),
    FeatureName("pool_size_latest_jpy"),
    *PAST_PERFORMANCE_FEATURE_NAMES,
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
    input_payouts: int
    races_written: int
    entries_written: int
    feature_rows_written: int
    results_written: int
    odds_written: int
    odds_timeseries_written: int
    payouts_written: int
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
    odds_staging_dirs: Path | str | Sequence[Path | str] | None = None,
    horse_master_csv: Path | str | None = None,
    rating_history_csv: Path | str | None = None,
) -> ReplayDatasetExport:
    """Create a replay-ready dataset from canonical JRA-VAN staging CSVs.

    ``staging_dir`` is the canonical race source and must contain races,
    entries, results, and odds. Optional ``odds_staging_dirs`` are merged only
    for odds snapshots, which lets RACE staging be enriched with realtime 0B41
    O1 odds staging without changing the race/entry/result source of truth.

    The compatibility ``odds.csv`` output intentionally keeps only the latest win
    quote per runner. The enriched ``odds_timeseries.csv`` output keeps all
    selected win snapshots so market-movement and point-in-time backtests can be
    built from the same replay artifact.
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
    odds = _read_merged_odds(
        source,
        odds_staging_dirs=odds_staging_dirs,
    )
    official_payouts = _read_payout_rows(source / "payouts.csv")
    official_win_payouts_by_key = _index_win_payout_rows(official_payouts)
    official_non_win_payouts_by_race = _group_non_win_payout_rows_by_race(
        official_payouts
    )
    horse_master_by_id = (
        read_horse_master_csv(horse_master_csv)
        if horse_master_csv is not None
        else {}
    )
    rating_history_by_horse = (
        index_rating_history(read_horse_rating_history_csv(rating_history_csv))
        if rating_history_csv is not None
        else {}
    )

    entries_by_race = _group_entries_by_race(entries)
    results_by_race = _group_results_by_race(results)
    win_odds_by_race = _group_latest_win_odds_by_race(
        odds,
        max_captured_at=max_odds_captured_at,
    )
    win_odds_timeseries_by_runner = _group_win_odds_timeseries_by_runner(
        odds,
        max_captured_at=max_odds_captured_at,
    )

    selected_races: list[Race] = []
    selected_entries: list[Entry] = []
    selected_results: list[Result] = []
    selected_odds: list[OddsQuote] = []
    selected_odds_timeseries: list[OddsQuote] = []
    selected_payouts: list[dict[str, object]] = []
    entry_feature_rows: list[FeatureRow] = []
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
        latest_rating_by_runner = {
            entry.runner_id: latest_rating_as_of(
                rating_history_by_horse.get(entry.horse_id, ()),
                as_of=race_feature_as_of,
            )
            for entry in race_entries
        }
        rating_context = _rating_context(latest_rating_by_runner)
        selected_race = replace(race, field_size=len(race_entries))
        selected_races.append(selected_race)
        for entry in sorted(race_entries, key=lambda item: item.horse_number):
            selected_entries.append(entry)
            selected_results.append(result_by_runner[entry.runner_id])
            result = result_by_runner[entry.runner_id]
            latest_quote = odds_by_runner[entry.runner_id]
            odds_timeseries = win_odds_timeseries_by_runner.get(entry.runner_id, ())
            official_win_payout = official_win_payouts_by_key.get(
                (
                    str(entry.race_id),
                    str(entry.runner_id),
                    BetType.WIN.value,
                )
            )
            selected_odds.append(latest_quote)
            selected_odds_timeseries.extend(odds_timeseries)
            selected_payouts.append(
                _official_or_proxy_win_payout_row(
                    official_row=official_win_payout,
                    result=result,
                    latest_quote=latest_quote,
                )
            )
            entry_feature_rows.append(
                _feature_row_from_entry(
                    entry,
                    race=selected_race,
                    odds_timeseries=odds_timeseries,
                    horse_master=horse_master_by_id.get(entry.horse_id),
                    rating_record=latest_rating_by_runner.get(entry.runner_id),
                    rating_context=rating_context,
                    as_of=race_feature_as_of,
                    feature_version=feature_version,
                )
            )
        selected_payouts.extend(
            official_non_win_payouts_by_race.get(str(race.race_id), ())
        )

    past_features_by_runner = {
        row.runner_id: row
        for row in build_past_performance_features(
            entries=entries,
            races=races,
            results=results,
            odds=odds,
            target_entries=selected_entries,
            target_races=selected_races,
            feature_version=feature_version,
        )
    }
    feature_rows = tuple(
        _merge_feature_rows(
            row,
            past_features_by_runner.get(row.runner_id),
        )
        for row in entry_feature_rows
    )

    csv_paths = {
        "races": target / "races.csv",
        "entries": target / "entries.csv",
        "features": target / "features.csv",
        "results": target / "results.csv",
        "odds": target / "odds.csv",
        "odds_timeseries": target / "odds_timeseries.csv",
        "payouts": target / "payouts.csv",
    }
    write_races_csv(csv_paths["races"], selected_races)
    write_entries_csv(csv_paths["entries"], selected_entries)
    write_feature_rows_csv(csv_paths["features"], feature_rows)
    write_results_csv(csv_paths["results"], selected_results)
    write_odds_csv(csv_paths["odds"], selected_odds)
    write_odds_csv(csv_paths["odds_timeseries"], selected_odds_timeseries)
    write_payouts_csv(csv_paths["payouts"], selected_payouts)

    report = ReplayDatasetReport(
        feature_version=feature_version,
        odds_policy="latest_win_quote_per_runner",
        max_odds_captured_at=max_odds_captured_at,
        input_races=len(races),
        input_entries=len(entries),
        input_results=len(results),
        input_odds=len(odds),
        input_payouts=len(official_payouts),
        races_written=len(selected_races),
        entries_written=len(selected_entries),
        feature_rows_written=len(feature_rows),
        results_written=len(selected_results),
        odds_written=len(selected_odds),
        odds_timeseries_written=len(selected_odds_timeseries),
        payouts_written=len(selected_payouts),
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
            "payouts": report.input_payouts,
        },
        "output_counts": {
            "races": report.races_written,
            "entries": report.entries_written,
            "feature_rows": report.feature_rows_written,
            "results": report.results_written,
            "odds": report.odds_written,
            "odds_timeseries": report.odds_timeseries_written,
            "payouts": report.payouts_written,
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


def _read_merged_odds(
    staging_dir: Path,
    *,
    odds_staging_dirs: Path | str | Sequence[Path | str] | None,
) -> tuple[OddsQuote, ...]:
    sources = (staging_dir, *_normalize_odds_staging_dirs(odds_staging_dirs))
    odds_by_key: dict[tuple[RaceId, RunnerId, BetType, datetime], OddsQuote] = {}
    for source in sources:
        for row in read_csv_rows(source / "odds.csv"):
            if row.get("bet_type") != BetType.WIN.value:
                continue
            quote = parse_odds_quote_row(row)
            odds_by_key[
                (quote.race_id, quote.runner_id, quote.bet_type, quote.captured_at)
            ] = quote
    return tuple(
        sorted(
            odds_by_key.values(),
            key=lambda quote: (
                str(quote.race_id),
                str(quote.runner_id),
                quote.captured_at,
                quote.bet_type.value,
            ),
        )
    )


def _normalize_odds_staging_dirs(
    odds_staging_dirs: Path | str | Sequence[Path | str] | None,
) -> tuple[Path, ...]:
    if odds_staging_dirs is None:
        return ()
    if isinstance(odds_staging_dirs, (str, Path)):
        return (Path(odds_staging_dirs),)
    return tuple(Path(source) for source in odds_staging_dirs)


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


def _group_win_odds_timeseries_by_runner(
    odds: Sequence[OddsQuote],
    *,
    max_captured_at: datetime | None,
) -> dict[RunnerId, tuple[OddsQuote, ...]]:
    grouped: dict[RunnerId, list[OddsQuote]] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN:
            continue
        if max_captured_at is not None and quote.captured_at > max_captured_at:
            continue
        grouped.setdefault(quote.runner_id, []).append(quote)
    return {
        runner_id: tuple(sorted(values, key=lambda quote: quote.captured_at))
        for runner_id, values in grouped.items()
    }


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
    race: Race,
    odds_timeseries: tuple[OddsQuote, ...],
    horse_master: HorseMasterRecord | None = None,
    rating_record: HorseRatingRecord | None = None,
    rating_context: Mapping[RunnerId, tuple[float | None, int | None]] | None = None,
    as_of: datetime,
    feature_version: str,
) -> FeatureRow:
    market_features = _market_movement_features(odds_timeseries)
    rating_delta, rating_rank = (
        rating_context.get(entry.runner_id, (None, None))
        if rating_context is not None
        else (None, None)
    )
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
            FeatureName("sex"): entry.metadata.get("sex"),
            FeatureName("horse_symbol_code"): _prefixed_metadata_value(
                entry,
                "horse_symbol_code",
                prefix="horse_symbol",
            ),
            FeatureName("breed_code"): _prefixed_metadata_value(
                entry,
                "breed_code",
                prefix="breed",
            ),
            FeatureName("coat_color_code"): _prefixed_metadata_value(
                entry,
                "coat_color_code",
                prefix="coat",
            ),
            FeatureName("trainer_affiliation_code"): _prefixed_metadata_value(
                entry,
                "trainer_affiliation_code",
                prefix="trainer_affiliation",
            ),
            FeatureName("pedigree_sire_id"): _prefixed_horse_id(
                horse_master.sire_id if horse_master is not None else None,
                prefix="sire",
            ),
            FeatureName("pedigree_dam_id"): _prefixed_horse_id(
                horse_master.dam_id if horse_master is not None else None,
                prefix="dam",
            ),
            FeatureName("pedigree_damsire_id"): _prefixed_horse_id(
                horse_master.damsire_id if horse_master is not None else None,
                prefix="damsire",
            ),
            FeatureName("horse_birth_year"): (
                horse_master.birth_date.year
                if horse_master is not None and horse_master.birth_date is not None
                else None
            ),
            FeatureName("horse_age_days_from_birth"): (
                (race.race_date - horse_master.birth_date).days
                if horse_master is not None and horse_master.birth_date is not None
                else None
            ),
            FeatureName("horse_rating"): (
                rating_record.rating if rating_record is not None else None
            ),
            FeatureName("horse_rating_delta_to_field_mean"): rating_delta,
            FeatureName("horse_rating_rank_in_race"): rating_rank,
            FeatureName("horse_rating_source"): (
                f"rating_source:{rating_record.source}"
                if rating_record is not None and rating_record.source
                else None
            ),
            FeatureName("jockey_id"): f"jockey:{entry.jockey_id}"
            if entry.jockey_id is not None
            else None,
            FeatureName("trainer_id"): f"trainer:{entry.trainer_id}"
            if entry.trainer_id is not None
            else None,
            FeatureName("body_weight_kg"): entry.body_weight_kg,
            FeatureName("body_weight_diff_kg"): entry.body_weight_diff_kg,
            FeatureName("entry_win_odds"): entry.metadata.get("entry_win_odds"),
            FeatureName("entry_popularity_rank"): entry.metadata.get(
                "entry_popularity_rank"
            ),
            FeatureName("race_venue"): race.venue,
            FeatureName("race_surface"): race.surface.value,
            FeatureName("race_distance_m"): race.distance_m,
            FeatureName("race_direction"): race.direction.value,
            FeatureName("race_track_condition"): race.track_condition.value,
            FeatureName("race_weather"): race.weather,
            FeatureName("race_grade"): race.grade,
            FeatureName("race_grade_group"): _race_grade_group(race.grade),
            FeatureName("race_title_type"): _race_title_type(race.name),
            FeatureName("race_has_title"): bool(race.name),
            FeatureName("race_field_size"): race.field_size,
            FeatureName("starter"): True,
            **market_features,
        },
    )


def _merge_feature_rows(
    base: FeatureRow,
    extra: FeatureRow | None,
) -> FeatureRow:
    if extra is None:
        return base
    return FeatureRow(
        race_id=base.race_id,
        runner_id=base.runner_id,
        as_of=base.as_of,
        feature_version=base.feature_version,
        values={**base.values, **extra.values},
        metadata=base.metadata,
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


def _prefixed_metadata_value(
    entry: Entry,
    key: str,
    *,
    prefix: str,
) -> str | None:
    value = entry.metadata.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return f"{prefix}:{normalized}"


def _prefixed_horse_id(horse_id: object | None, *, prefix: str) -> str | None:
    if horse_id is None:
        return None
    normalized = str(horse_id).strip()
    if not normalized:
        return None
    return f"{prefix}:{normalized}"


def _rating_context(
    rating_by_runner: Mapping[RunnerId, HorseRatingRecord | None],
) -> dict[RunnerId, tuple[float | None, int | None]]:
    available = [
        (runner_id, record)
        for runner_id, record in rating_by_runner.items()
        if record is not None
    ]
    if not available:
        return {}
    mean_rating = sum(record.rating for _, record in available) / len(available)
    ranked = sorted(
        available,
        key=lambda item: (-item[1].rating, str(item[0])),
    )
    rank_by_runner = {
        runner_id: rank for rank, (runner_id, _) in enumerate(ranked, start=1)
    }
    return {
        runner_id: (record.rating - mean_rating, rank_by_runner[runner_id])
        for runner_id, record in available
    }


def _race_grade_group(grade: str | None) -> str:
    if grade is None or not str(grade).strip():
        return "ordinary"
    grade_code = str(grade).strip()
    if grade_code in {"A", "B", "C"}:
        return "graded"
    if grade_code == "L":
        return "listed"
    return f"grade_code:{grade_code}"


def _race_title_type(name: str | None) -> str:
    if name is None or not name.strip():
        return "untitled"
    normalized = name.strip()
    if "ステークス" in normalized:
        return "stakes"
    if "カップ" in normalized:
        return "cup"
    if "特別" in normalized:
        return "tokubetsu"
    if "賞" in normalized:
        return "sho"
    return "named"


def _market_movement_features(
    odds_timeseries: tuple[OddsQuote, ...],
) -> dict[FeatureName, float | int | None]:
    if not odds_timeseries:
        return {
            FeatureName("odds_open"): None,
            FeatureName("odds_latest"): None,
            FeatureName("odds_min"): None,
            FeatureName("odds_max"): None,
            FeatureName("odds_snapshot_count"): 0,
            FeatureName("odds_change_open_to_latest"): None,
            FeatureName("implied_probability_change_open_to_latest"): None,
            FeatureName("pool_size_latest_jpy"): None,
        }

    odds_values = tuple(quote.odds for quote in odds_timeseries)
    open_odds = odds_timeseries[0].odds
    latest = odds_timeseries[-1]
    latest_odds = latest.odds
    return {
        FeatureName("odds_open"): open_odds,
        FeatureName("odds_latest"): latest_odds,
        FeatureName("odds_min"): min(odds_values),
        FeatureName("odds_max"): max(odds_values),
        FeatureName("odds_snapshot_count"): len(odds_timeseries),
        FeatureName("odds_change_open_to_latest"): latest_odds - open_odds,
        FeatureName("implied_probability_change_open_to_latest"): (
            (1.0 / latest_odds) - (1.0 / open_odds)
        ),
        FeatureName("pool_size_latest_jpy"): latest.pool_size_jpy,
    }


def _payout_proxy_row(
    *,
    result: Result,
    latest_quote: OddsQuote,
) -> dict[str, object]:
    return {
        "race_id": result.race_id,
        "runner_id": result.runner_id,
        "bet_type": latest_quote.bet_type,
        "finish_position": result.finish_position,
        "is_win": result.did_win,
        "payout_jpy_per_100": int(round(latest_quote.odds * 100))
        if result.did_win
        else 0,
        "odds": latest_quote.odds,
        "pool_size_jpy": latest_quote.pool_size_jpy,
        "source": "derived_from_latest_win_odds",
    }


def _official_or_proxy_win_payout_row(
    *,
    official_row: Mapping[str, object] | None,
    result: Result,
    latest_quote: OddsQuote,
) -> dict[str, object]:
    if official_row is None:
        return _payout_proxy_row(result=result, latest_quote=latest_quote)

    row = dict(official_row)
    row["finish_position"] = result.finish_position
    row["is_win"] = result.did_win
    if row.get("pool_size_jpy") in {None, ""}:
        row["pool_size_jpy"] = latest_quote.pool_size_jpy
    return row


def _read_payout_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    return tuple(_normalize_payout_row(row) for row in read_csv_rows(path))


def _normalize_payout_row(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "race_id": row.get("race_id", ""),
        "runner_id": row.get("runner_id", ""),
        "bet_type": row.get("bet_type", ""),
        "finish_position": _parse_optional_int(row.get("finish_position")),
        "is_win": _parse_optional_bool(row.get("is_win")),
        "payout_jpy_per_100": _parse_optional_int(row.get("payout_jpy_per_100")),
        "odds": _parse_optional_float(row.get("odds")),
        "pool_size_jpy": _parse_optional_int(row.get("pool_size_jpy")),
        "source": row.get("source", ""),
    }


def _index_win_payout_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    indexed: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in rows:
        if row.get("bet_type") != BetType.WIN.value:
            continue
        indexed[
            (
                str(row.get("race_id", "")),
                str(row.get("runner_id", "")),
                BetType.WIN.value,
            )
        ] = row
    return indexed


def _group_non_win_payout_rows_by_race(
    rows: Sequence[dict[str, object]],
) -> dict[str, tuple[dict[str, object], ...]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("bet_type") == BetType.WIN.value:
            continue
        grouped.setdefault(str(row.get("race_id", "")), []).append(dict(row))
    return {
        race_id: tuple(
            sorted(
                values,
                key=lambda row: (
                    str(row.get("bet_type", "")),
                    str(row.get("runner_id", "")),
                ),
            )
        )
        for race_id, values in grouped.items()
    }


def _parse_optional_int(value: object) -> int | None:
    normalized = "" if value is None else str(value).strip()
    return int(normalized) if normalized else None


def _parse_optional_float(value: object) -> float | None:
    normalized = "" if value is None else str(value).strip()
    return float(normalized) if normalized else None


def _parse_optional_bool(value: object) -> bool | None:
    normalized = "" if value is None else str(value).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean payout value: {value!r}")


def _write_report(path: Path, report: ReplayDatasetReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(replay_dataset_report_to_dict(report), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _render_optional_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
