"""CSV parsing helpers for local historical fixtures."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

from horse_lab.schemas import (
    BetType,
    CourseDirection,
    Entry,
    FeatureName,
    FeatureRow,
    HorseId,
    OddsQuote,
    PersonId,
    Race,
    RaceId,
    Result,
    RunnerId,
    Surface,
    TrackCondition,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"", "false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_int_or_none(value: str | None) -> int | None:
    normalized = (value or "").strip()
    return int(normalized) if normalized else None


def parse_float_or_none(value: str | None) -> float | None:
    normalized = (value or "").strip()
    return float(normalized) if normalized else None


def parse_str_or_none(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


def parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def parse_datetime_or_none(value: str | None) -> datetime | None:
    normalized = (value or "").strip()
    return datetime.fromisoformat(normalized) if normalized else None


def parse_race_row(row: Mapping[str, str]) -> Race:
    return Race(
        race_id=RaceId(row["race_id"]),
        race_date=parse_date(row["race_date"]),
        venue=row["venue"],
        race_number=int(row["race_number"]),
        name=parse_str_or_none(row.get("name")),
        surface=Surface(row.get("surface") or Surface.UNKNOWN.value),
        distance_m=int(row["distance_m"]),
        direction=CourseDirection(row.get("direction") or CourseDirection.UNKNOWN.value),
        track_condition=TrackCondition(
            row.get("track_condition") or TrackCondition.UNKNOWN.value
        ),
        weather=parse_str_or_none(row.get("weather")),
        grade=parse_str_or_none(row.get("grade")),
        start_time=parse_datetime_or_none(row.get("start_time")),
        field_size=parse_int_or_none(row.get("field_size")),
    )


def parse_entry_row(row: Mapping[str, str]) -> Entry:
    jockey_id = parse_str_or_none(row.get("jockey_id"))
    trainer_id = parse_str_or_none(row.get("trainer_id"))
    return Entry(
        runner_id=RunnerId(row["runner_id"]),
        race_id=RaceId(row["race_id"]),
        horse_id=HorseId(row["horse_id"]),
        horse_number=int(row["horse_number"]),
        gate_number=parse_int_or_none(row.get("gate_number")),
        jockey_id=PersonId(jockey_id) if jockey_id is not None else None,
        trainer_id=PersonId(trainer_id) if trainer_id is not None else None,
        carried_weight_kg=parse_float_or_none(row.get("carried_weight_kg")),
        body_weight_kg=parse_int_or_none(row.get("body_weight_kg")),
        body_weight_diff_kg=parse_int_or_none(row.get("body_weight_diff_kg")),
        age=parse_int_or_none(row.get("age")),
        is_scratched=parse_bool(row.get("is_scratched")),
    )


def parse_result_row(row: Mapping[str, str]) -> Result:
    return Result(
        race_id=RaceId(row["race_id"]),
        runner_id=RunnerId(row["runner_id"]),
        finish_position=parse_int_or_none(row.get("finish_position")),
        is_disqualified=parse_bool(row.get("is_disqualified")),
        is_dead_heat=parse_bool(row.get("is_dead_heat")),
        final_time_seconds=parse_float_or_none(row.get("final_time_seconds")),
        prize_jpy=parse_int_or_none(row.get("prize_jpy")),
    )


def parse_odds_quote_row(row: Mapping[str, str]) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId(row["race_id"]),
        runner_id=RunnerId(row["runner_id"]),
        bet_type=BetType(row["bet_type"]),
        captured_at=parse_datetime(row["captured_at"]),
        odds=float(row["odds"]),
        popularity_rank=parse_int_or_none(row.get("popularity_rank")),
        pool_size_jpy=parse_int_or_none(row.get("pool_size_jpy")),
        source=parse_str_or_none(row.get("source")),
    )


def parse_feature_row(row: Mapping[str, str]) -> FeatureRow:
    values = {
        FeatureName(key.removeprefix("feature__")): parse_feature_value(value)
        for key, value in row.items()
        if key.startswith("feature__")
    }
    return FeatureRow(
        race_id=RaceId(row["race_id"]),
        runner_id=RunnerId(row["runner_id"]),
        as_of=parse_datetime(row["as_of"]),
        feature_version=row["feature_version"],
        values=values,
    )


def parse_feature_value(value: str | None) -> float | int | bool | str | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    lower = normalized.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(normalized)
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError:
        return normalized
