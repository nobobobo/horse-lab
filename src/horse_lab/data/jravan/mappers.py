"""Canonical schema mappers for minimal JRA-VAN RA records."""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping

from horse_lab.data.jravan.layouts import parse_minimal_ra_fields
from horse_lab.data.jravan.raw import JvDataRecord
from horse_lab.schemas import (
    CourseDirection,
    Race,
    RaceId,
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
    race_date = _require(fields, "race_date")
    venue_code = _require(fields, "venue_code")
    kaiji = _require(fields, "kaiji")
    nichiji = _require(fields, "nichiji")
    race_number = _require(fields, "race_number")
    return RaceId(f"{race_date}{venue_code}{kaiji}{nichiji}{race_number}")


def map_ra_record_to_race(record: JvDataRecord) -> Race:
    fields = parse_minimal_ra_fields(record)
    race_date = _parse_yyyymmdd(_require(fields, "race_date"))
    venue_code = _require(fields, "venue_code")
    surface_code = _optional_str(fields.get("surface_code"))
    direction_code = _optional_str(fields.get("direction_code"))
    track_condition_code = _optional_str(fields.get("track_condition_code"))
    weather_code = _optional_str(fields.get("weather_code"))

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
        grade=_optional_str(fields.get("grade_code")),
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
        },
    )


def _require(fields: Mapping[str, str], name: str) -> str:
    value = _optional_str(fields.get(name))
    if value is None:
        raise ValueError(f"Missing required JRA-VAN field: {name}")
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
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


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


def _unknown_or_none(value: str | None) -> str | None:
    return f"unknown:{value}" if value else None
