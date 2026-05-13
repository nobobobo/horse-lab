"""External horse master and rating-history CSV ingestion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Mapping, Sequence

from horse_lab.schemas import HorseId


@dataclass(frozen=True)
class HorseMasterRecord:
    horse_id: HorseId
    horse_name: str | None = None
    birth_date: date | None = None
    sire_id: HorseId | None = None
    dam_id: HorseId | None = None
    damsire_id: HorseId | None = None
    metadata: Mapping[str, str] | None = None


@dataclass(frozen=True)
class HorseRatingRecord:
    horse_id: HorseId
    as_of: datetime
    rating: float
    source: str | None = None
    metadata: Mapping[str, str] | None = None


def read_horse_master_csv(path: Path | str) -> dict[HorseId, HorseMasterRecord]:
    """Read a horse-level pedigree/profile master keyed by ``horse_id``.

    Expected columns are deliberately small and source-agnostic:
    ``horse_id``, optional ``horse_name``, optional ISO ``birth_date``,
    optional ``sire_id``, ``dam_id``, and ``damsire_id``. Unknown columns are
    preserved in ``metadata`` for lineage.
    """

    records: dict[HorseId, HorseMasterRecord] = {}
    for row in _read_csv_rows(path):
        horse_id = HorseId(_required(row, "horse_id"))
        records[horse_id] = HorseMasterRecord(
            horse_id=horse_id,
            horse_name=_optional(row, "horse_name"),
            birth_date=_parse_optional_date(row.get("birth_date")),
            sire_id=_optional_horse_id(row.get("sire_id")),
            dam_id=_optional_horse_id(row.get("dam_id")),
            damsire_id=_optional_horse_id(row.get("damsire_id")),
            metadata=_extra_metadata(
                row,
                known_fields={
                    "horse_id",
                    "horse_name",
                    "birth_date",
                    "sire_id",
                    "dam_id",
                    "damsire_id",
                },
            ),
        )
    return records


def read_horse_rating_history_csv(path: Path | str) -> tuple[HorseRatingRecord, ...]:
    """Read point-in-time horse ratings.

    Required columns are ``horse_id``, ``as_of``, and ``rating``. ``as_of`` may
    be an ISO date or datetime. Date-only values are interpreted as midnight,
    so callers should provide the last known pre-race timestamp when available.
    """

    rows = []
    for row in _read_csv_rows(path):
        horse_id = HorseId(_required(row, "horse_id"))
        rows.append(
            HorseRatingRecord(
                horse_id=horse_id,
                as_of=_parse_required_datetime(_required(row, "as_of")),
                rating=float(_required(row, "rating")),
                source=_optional(row, "source"),
                metadata=_extra_metadata(
                    row,
                    known_fields={"horse_id", "as_of", "rating", "source"},
                ),
            )
        )
    return tuple(sorted(rows, key=lambda record: (record.horse_id, record.as_of)))


def index_rating_history(
    records: Sequence[HorseRatingRecord],
) -> dict[HorseId, tuple[HorseRatingRecord, ...]]:
    grouped: dict[HorseId, list[HorseRatingRecord]] = {}
    for record in records:
        grouped.setdefault(record.horse_id, []).append(record)
    return {
        horse_id: tuple(sorted(values, key=lambda record: record.as_of))
        for horse_id, values in grouped.items()
    }


def latest_rating_as_of(
    records: Sequence[HorseRatingRecord],
    *,
    as_of: datetime,
) -> HorseRatingRecord | None:
    latest: HorseRatingRecord | None = None
    for record in records:
        if record.as_of <= as_of:
            latest = record
        else:
            break
    return latest


def _read_csv_rows(path: Path | str) -> tuple[dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _required(row: Mapping[str, str], key: str) -> str:
    value = _optional(row, key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _optional(row: Mapping[str, str], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _optional_horse_id(value: str | None) -> HorseId | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return HorseId(normalized)


def _parse_optional_date(value: str | None) -> date | None:
    if value is None or not value.strip():
        return None
    return date.fromisoformat(value.strip())


def _parse_required_datetime(value: str) -> datetime:
    normalized = value.strip()
    if "T" in normalized:
        return datetime.fromisoformat(normalized)
    return datetime.combine(date.fromisoformat(normalized), time.min)


def _extra_metadata(
    row: Mapping[str, str],
    *,
    known_fields: set[str],
) -> dict[str, str]:
    return {
        key: value
        for key, value in row.items()
        if key not in known_fields and value.strip()
    }
