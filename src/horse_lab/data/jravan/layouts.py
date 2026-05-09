"""Minimal JRA-VAN RA/SE/O1/O2 fixed-width layouts."""

from __future__ import annotations

from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
)


JRAVAN_MINIMAL_RA_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("data_created_date", start=4, length=8),
    FixedWidthField("race_date", start=12, length=8),
    FixedWidthField("venue_code", start=20, length=2),
    FixedWidthField("kaiji", start=22, length=2),
    FixedWidthField("nichiji", start=24, length=2),
    FixedWidthField("race_number", start=26, length=2),
    FixedWidthField("weekday_code", start=28, length=1),
    FixedWidthField("special_race_number", start=29, length=4),
    FixedWidthField("race_name", start=33, length=60),
    FixedWidthField("grade_code", start=615, length=1),
    FixedWidthField("distance_m", start=698, length=4),
    FixedWidthField("track_code", start=706, length=2),
    FixedWidthField("start_time", start=874, length=4),
    FixedWidthField("registered_horse_count", start=882, length=2),
    FixedWidthField("starter_count", start=884, length=2),
    FixedWidthField("weather_code", start=888, length=1),
    FixedWidthField("turf_track_condition_code", start=889, length=1),
    FixedWidthField("dirt_track_condition_code", start=890, length=1),
)

JRAVAN_MINIMAL_SE_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("data_created_date", start=4, length=8),
    FixedWidthField("race_date", start=12, length=8),
    FixedWidthField("venue_code", start=20, length=2),
    FixedWidthField("kaiji", start=22, length=2),
    FixedWidthField("nichiji", start=24, length=2),
    FixedWidthField("race_number", start=26, length=2),
    FixedWidthField("gate_number", start=28, length=1),
    FixedWidthField("horse_number", start=29, length=2),
    FixedWidthField("horse_id", start=31, length=10),
    FixedWidthField("horse_name", start=41, length=36),
    FixedWidthField("horse_symbol_code", start=77, length=2),
    FixedWidthField("sex_code", start=79, length=1),
    FixedWidthField("breed_code", start=80, length=1),
    FixedWidthField("coat_color_code", start=81, length=2),
    FixedWidthField("age", start=83, length=2),
    FixedWidthField("trainer_affiliation_code", start=85, length=1),
    FixedWidthField("trainer_id", start=86, length=5),
    FixedWidthField("carried_weight", start=289, length=3),
    FixedWidthField("jockey_id", start=297, length=5),
    FixedWidthField("body_weight", start=325, length=3),
    FixedWidthField("body_weight_diff_sign", start=328, length=1),
    FixedWidthField("body_weight_diff", start=329, length=3),
    FixedWidthField("abnormal_code", start=332, length=1),
    FixedWidthField("finish_position", start=335, length=2),
    FixedWidthField("is_dead_heat", start=337, length=1),
    FixedWidthField("final_time_seconds", start=339, length=4),
    FixedWidthField("win_odds", start=360, length=4),
    FixedWidthField("popularity_rank", start=364, length=2),
    FixedWidthField("prize_jpy_x100", start=366, length=8),
)

JRAVAN_MINIMAL_O1_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("data_created_date", start=4, length=8),
    FixedWidthField("race_date", start=12, length=8),
    FixedWidthField("venue_code", start=20, length=2),
    FixedWidthField("kaiji", start=22, length=2),
    FixedWidthField("nichiji", start=24, length=2),
    FixedWidthField("race_number", start=26, length=2),
    FixedWidthField("captured_month_day_time", start=28, length=8),
    FixedWidthField("registered_horse_count", start=36, length=2),
    FixedWidthField("starter_count", start=38, length=2),
    FixedWidthField("win_sale_flag", start=40, length=1),
    FixedWidthField("place_sale_flag", start=41, length=1),
    FixedWidthField("bracket_quinella_sale_flag", start=42, length=1),
    FixedWidthField("place_payout_key", start=43, length=1),
    FixedWidthField("win_odds_entries", start=44, length=224),
    FixedWidthField("win_pool_size_jpy_x100", start=928, length=11),
)

JRAVAN_MINIMAL_O2_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("data_created_date", start=4, length=8),
    FixedWidthField("race_date", start=12, length=8),
    FixedWidthField("venue_code", start=20, length=2),
    FixedWidthField("kaiji", start=22, length=2),
    FixedWidthField("nichiji", start=24, length=2),
    FixedWidthField("race_number", start=26, length=2),
    FixedWidthField("captured_month_day_time", start=28, length=8),
    FixedWidthField("registered_horse_count", start=36, length=2),
    FixedWidthField("starter_count", start=38, length=2),
    FixedWidthField("quinella_sale_flag", start=40, length=1),
    FixedWidthField("quinella_odds_entries", start=41, length=1989),
    FixedWidthField("quinella_pool_size_jpy_x100", start=2030, length=11),
)


def parse_minimal_ra_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "RA")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_RA_FIELDS)


def parse_minimal_se_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "SE")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_SE_FIELDS)


def parse_minimal_o1_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "O1")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_O1_FIELDS)


def parse_minimal_o2_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "O2")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_O2_FIELDS)


def _require_record_type(record: JvDataRecord, expected: str) -> None:
    if record.record_type != expected:
        raise ValueError(f"Expected {expected} record, got {record.record_type!r}")
