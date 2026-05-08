"""Minimal synthetic JRA-VAN RA/SE fixed-width layouts."""

from __future__ import annotations

from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
)


JRAVAN_MINIMAL_RA_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("race_date", start=4, length=8),
    FixedWidthField("venue_code", start=12, length=2),
    FixedWidthField("kaiji", start=14, length=2),
    FixedWidthField("nichiji", start=16, length=2),
    FixedWidthField("race_number", start=18, length=2),
    FixedWidthField("race_name", start=20, length=60),
    FixedWidthField("surface_code", start=80, length=1),
    FixedWidthField("distance_m", start=81, length=4),
    FixedWidthField("direction_code", start=85, length=1),
    FixedWidthField("track_condition_code", start=86, length=1),
    FixedWidthField("weather_code", start=87, length=1),
    FixedWidthField("grade_code", start=88, length=2),
    FixedWidthField("start_time", start=90, length=4),
    FixedWidthField("field_size", start=94, length=2),
)

JRAVAN_MINIMAL_SE_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("race_date", start=4, length=8),
    FixedWidthField("venue_code", start=12, length=2),
    FixedWidthField("kaiji", start=14, length=2),
    FixedWidthField("nichiji", start=16, length=2),
    FixedWidthField("race_number", start=18, length=2),
    FixedWidthField("horse_number", start=20, length=2),
    FixedWidthField("gate_number", start=22, length=2),
    FixedWidthField("horse_id", start=24, length=10),
    FixedWidthField("horse_name", start=34, length=40),
    FixedWidthField("sex_code", start=74, length=1),
    FixedWidthField("age", start=75, length=2),
    FixedWidthField("trainer_id", start=77, length=6),
    FixedWidthField("jockey_id", start=83, length=6),
    FixedWidthField("carried_weight", start=89, length=3),
    FixedWidthField("body_weight", start=92, length=3),
    FixedWidthField("body_weight_diff_sign", start=95, length=1),
    FixedWidthField("body_weight_diff", start=96, length=2),
    FixedWidthField("finish_position", start=98, length=2),
    FixedWidthField("is_disqualified", start=100, length=1),
    FixedWidthField("is_dead_heat", start=101, length=1),
    FixedWidthField("final_time_seconds", start=102, length=5),
    FixedWidthField("prize_jpy", start=107, length=8),
)


def parse_minimal_ra_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "RA")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_RA_FIELDS)


def parse_minimal_se_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "SE")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_SE_FIELDS)


def _require_record_type(record: JvDataRecord, expected: str) -> None:
    if record.record_type != expected:
        raise ValueError(f"Expected {expected} record, got {record.record_type!r}")
