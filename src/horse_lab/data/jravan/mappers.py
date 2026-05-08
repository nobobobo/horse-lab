"""Canonical schema mappers for minimal JRA-VAN RA and SE records."""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping

from horse_lab.data.jravan.layouts import (
    parse_minimal_ra_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.raw import JvDataRecord
from horse_lab.schemas import (
    CourseDirection,
    Entry,
    HorseId,
    PersonId,
    Race,
    RaceId,
    Result,
    RunnerId,
    Surface,
    TrackCondition,
)


VENUE_BY_CODE = {
    "01": "Sapporo",
    "02": "Hakodate",
    "03": "Fukushima",
    "04": "Niigata",
    "05": "Tokyo",
    "06": "Nakayama",
    "07": "Chukyo",
    "08": "Kyoto",
    "09": "Hanshin",
    "10": "Kokura",
}

SURFACE_BY_CODE = {
    "1": Surface.TURF,
    "2": Surface.DIRT,
    "3": Surface.JUMP,
}

DIRECTION_BY_CODE = {
    "1": CourseDirection.RIGHT,
    "2": CourseDirection.LEFT,
    "3": CourseDirection.STRAIGHT,
}

TRACK_CONDITION_BY_CODE = {
    "1": TrackCondition.FIRM,
    "2": TrackCondition.GOOD,
    "3": TrackCondition.YIELDING,
    "4": TrackCondition.HEAVY,
}

WEATHER_BY_CODE = {
    "1": "sunny",
    "2": "cloudy",
    "3": "rain",
    "4": "snow",
}


def build_jravan_race_id(fields: Mapping[str, str]) -> RaceId:
    race_date = _parse_jravan_race_id_date(_require(fields, "race_date"))
    venue_code = _require_two_digit_component(fields, "venue_code")
    kaiji = _require_two_digit_component(fields, "kaiji")
    nichiji = _require_two_digit_component(fields, "nichiji")
    race_number = _require_two_digit_component(fields, "race_number")
    return RaceId(f"{race_date}{venue_code}{kaiji}{nichiji}{race_number}")


def build_jravan_runner_id(fields: Mapping[str, str]) -> RunnerId:
    horse_number = _require_two_digit_component(fields, "horse_number")
    if _parse_int(horse_number, "horse_number") <= 0:
        raise ValueError(
            f"Invalid horse_number: expected positive, got {horse_number!r}"
        )
    return RunnerId(f"{build_jravan_race_id(fields)}-{horse_number}")


def map_ra_record_to_race(record: JvDataRecord) -> Race:
    fields = parse_minimal_ra_fields(record)
    race_date = _parse_yyyymmdd(_require(fields, "race_date"))
    venue_code = _require(fields, "venue_code")
    surface_code = _optional_str(fields.get("surface_code"))
    direction_code = _optional_str(fields.get("direction_code"))
    track_condition_code = _optional_str(fields.get("track_condition_code"))
    weather_code = _optional_str(fields.get("weather_code"))
    grade_code = _optional_str(fields.get("grade_code"))

    return Race(
        race_id=build_jravan_race_id(fields),
        race_date=race_date,
        venue=VENUE_BY_CODE.get(venue_code, f"unknown:{venue_code}"),
        race_number=_parse_int(_require(fields, "race_number"), "race_number"),
        name=_optional_str(fields.get("race_name")),
        surface=SURFACE_BY_CODE.get(surface_code or "", Surface.UNKNOWN),
        distance_m=_parse_int(_require(fields, "distance_m"), "distance_m"),
        direction=DIRECTION_BY_CODE.get(direction_code or "", CourseDirection.UNKNOWN),
        track_condition=TRACK_CONDITION_BY_CODE.get(
            track_condition_code or "",
            TrackCondition.UNKNOWN,
        ),
        weather=WEATHER_BY_CODE.get(weather_code or "", _unknown_or_none(weather_code)),
        grade=grade_code,
        start_time=_parse_hhmm(race_date, fields.get("start_time", "")),
        field_size=_parse_optional_int(fields.get("field_size", ""), "field_size"),
        metadata={
            "source": "jravan_minimal",
            "data_kubun": fields.get("data_kubun", ""),
            "venue_code": venue_code,
            "kaiji": fields.get("kaiji", ""),
            "nichiji": fields.get("nichiji", ""),
            "surface_code": surface_code,
            "direction_code": direction_code,
            "track_condition_code": track_condition_code,
            "weather_code": weather_code,
            "grade_code": grade_code,
        },
    )


def map_se_record_to_entry(record: JvDataRecord) -> Entry:
    fields = parse_minimal_se_fields(record)
    jockey_id = _optional_str(fields.get("jockey_id"))
    trainer_id = _optional_str(fields.get("trainer_id"))

    return Entry(
        runner_id=build_jravan_runner_id(fields),
        race_id=build_jravan_race_id(fields),
        horse_id=HorseId(_require(fields, "horse_id")),
        horse_number=_parse_int(_require(fields, "horse_number"), "horse_number"),
        gate_number=_parse_optional_int(fields.get("gate_number"), "gate_number"),
        jockey_id=PersonId(jockey_id) if jockey_id is not None else None,
        trainer_id=PersonId(trainer_id) if trainer_id is not None else None,
        carried_weight_kg=_parse_deci_number(
            fields.get("carried_weight"),
            "carried_weight",
        ),
        body_weight_kg=_parse_optional_int(fields.get("body_weight"), "body_weight"),
        body_weight_diff_kg=_parse_body_weight_diff(
            fields.get("body_weight_diff_sign"),
            fields.get("body_weight_diff"),
        ),
        age=_parse_optional_int(fields.get("age"), "age"),
        is_scratched=False,
        metadata={
            "source": "jravan_minimal",
            "data_kubun": fields.get("data_kubun", ""),
            "horse_name": _optional_str(fields.get("horse_name")),
            "sex_code": _optional_str(fields.get("sex_code")),
        },
    )


def map_se_record_to_result(record: JvDataRecord) -> Result | None:
    fields = parse_minimal_se_fields(record)
    finish_position = _parse_optional_int(
        fields.get("finish_position"),
        "finish_position",
    )
    if finish_position is None:
        return None
    if finish_position <= 0:
        raise ValueError(
            f"Invalid finish_position: expected positive, got {finish_position!r}"
        )

    return Result(
        race_id=build_jravan_race_id(fields),
        runner_id=build_jravan_runner_id(fields),
        finish_position=finish_position,
        is_disqualified=_parse_flag(fields.get("is_disqualified")),
        is_dead_heat=_parse_flag(fields.get("is_dead_heat")),
        final_time_seconds=_parse_deci_number(
            fields.get("final_time_seconds"),
            "final_time_seconds",
        ),
        prize_jpy=_parse_optional_int(fields.get("prize_jpy"), "prize_jpy"),
    )


def _require(fields: Mapping[str, str], name: str) -> str:
    value = _optional_str(fields.get(name))
    if value is None:
        raise ValueError(f"Missing required JRA-VAN field: {name}")
    return value


def _require_two_digit_component(fields: Mapping[str, str], name: str) -> str:
    value = _require(fields, name)
    if len(value) != 2 or not value.isdigit():
        raise ValueError(f"Invalid {name}: expected exactly 2 digits, got {value!r}")
    return value


def _optional_str(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {field_name}: {value!r}") from exc


def _parse_optional_int(value: str | None, field_name: str) -> int | None:
    normalized = _optional_str(value)
    return _parse_int(normalized, field_name) if normalized is not None else None


def _parse_yyyymmdd(value: str) -> date:
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid race_date: {value!r}")
    try:
        return date(int(value[:4]), int(value[4:6]), int(value[6:8]))
    except ValueError as exc:
        raise ValueError(f"Invalid race_date: {value!r}") from exc


def _parse_jravan_race_id_date(value: str) -> str:
    _parse_yyyymmdd(value)
    return value


def _parse_hhmm(race_date: date, value: str | None) -> datetime | None:
    normalized = _optional_str(value)
    if normalized is None:
        return None
    if len(normalized) != 4 or not normalized.isdigit():
        raise ValueError(f"Invalid start_time: {value!r}")
    return datetime(
        race_date.year,
        race_date.month,
        race_date.day,
        int(normalized[:2]),
        int(normalized[2:4]),
    )


def _parse_deci_number(value: str | None, field_name: str) -> float | None:
    normalized = _optional_str(value)
    return _parse_int(normalized, field_name) / 10.0 if normalized is not None else None


def _parse_body_weight_diff(sign: str | None, value: str | None) -> int | None:
    diff = _parse_optional_int(value, "body_weight_diff")
    if diff is None:
        return None
    return -diff if (sign or "").strip() == "-" else diff


def _parse_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "y", "yes", "true"}


def _unknown_or_none(value: str | None) -> str | None:
    return f"unknown:{value}" if value else None
