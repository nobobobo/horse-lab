"""Canonical schema mappers for minimal JRA-VAN RA, SE, and O1 records."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Mapping

from horse_lab.data.jravan.layouts import (
    parse_minimal_o1_fields,
    parse_minimal_ra_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.raw import JvDataRecord
from horse_lab.schemas import (
    BetType,
    CourseDirection,
    Entry,
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

TURF_TRACK_CODES = {
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
}

DIRT_TRACK_CODES = {"23", "24", "25", "26", "29"}

JUMP_TRACK_CODES = {
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "57",
    "58",
    "59",
}

LEFT_TRACK_CODES = {
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "23",
    "25",
    "27",
    "53",
}

RIGHT_TRACK_CODES = {
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "24",
    "26",
    "28",
}

STRAIGHT_TRACK_CODES = {"10", "29"}

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
    "4": "drizzle",
    "5": "snow",
    "6": "light_snow",
}


def build_jravan_race_id(fields: Mapping[str, str]) -> RaceId:
    race_date = _parse_jravan_race_id_date(_require(fields, "race_date"))
    venue_code = _require_two_digit_component(fields, "venue_code")
    kaiji = _require_two_digit_component(fields, "kaiji")
    nichiji = _require_two_digit_component(fields, "nichiji")
    race_number = _require_two_digit_component(fields, "race_number")
    if _parse_int(race_number, "race_number") <= 0:
        raise ValueError(
            f"Invalid race_number: expected positive, got {race_number!r}"
        )
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
    track_code = _optional_str(fields.get("track_code"))
    surface = _surface_from_track_code(track_code)
    direction = _direction_from_track_code(track_code)
    track_condition_code = _track_condition_code_for_surface(fields, surface)
    weather_code = _optional_str(fields.get("weather_code"))
    grade_code = _optional_str(fields.get("grade_code"))

    return Race(
        race_id=build_jravan_race_id(fields),
        race_date=race_date,
        venue=VENUE_BY_CODE.get(venue_code, f"unknown:{venue_code}"),
        race_number=_parse_int(_require(fields, "race_number"), "race_number"),
        name=_optional_str(fields.get("race_name")),
        surface=surface,
        distance_m=_parse_int(_require(fields, "distance_m"), "distance_m"),
        direction=direction,
        track_condition=TRACK_CONDITION_BY_CODE.get(
            track_condition_code or "",
            TrackCondition.UNKNOWN,
        ),
        weather=WEATHER_BY_CODE.get(weather_code or "", _unknown_or_none(weather_code)),
        grade=grade_code,
        start_time=_parse_hhmm(race_date, fields.get("start_time", "")),
        field_size=_parse_field_size(fields),
        metadata={
            "source": "jravan_minimal",
            "data_kubun": fields.get("data_kubun", ""),
            "data_created_date": fields.get("data_created_date", ""),
            "venue_code": venue_code,
            "kaiji": fields.get("kaiji", ""),
            "nichiji": fields.get("nichiji", ""),
            "track_code": track_code,
            "track_condition_code": track_condition_code,
            "turf_track_condition_code": _optional_str(
                fields.get("turf_track_condition_code")
            ),
            "dirt_track_condition_code": _optional_str(
                fields.get("dirt_track_condition_code")
            ),
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
    data_kubun = _optional_str(fields.get("data_kubun"))
    if data_kubun in {"1", "2"}:
        return None

    finish_position = _parse_optional_int(
        fields.get("finish_position"),
        "finish_position",
    )
    if finish_position is None:
        return None
    if finish_position <= 0:
        if _is_disqualified(fields.get("abnormal_code")):
            return Result(
                race_id=build_jravan_race_id(fields),
                runner_id=build_jravan_runner_id(fields),
                finish_position=None,
                is_disqualified=True,
                is_dead_heat=False,
            )
        raise ValueError(
            f"Invalid finish_position: expected positive, got {finish_position!r}"
        )

    return Result(
        race_id=build_jravan_race_id(fields),
        runner_id=build_jravan_runner_id(fields),
        finish_position=finish_position,
        is_disqualified=_is_disqualified(fields.get("abnormal_code")),
        is_dead_heat=_parse_flag(fields.get("is_dead_heat"), "is_dead_heat"),
        final_time_seconds=_parse_deci_number(
            fields.get("final_time_seconds"),
            "final_time_seconds",
        ),
        prize_jpy=_parse_prize_jpy(fields.get("prize_jpy_x100")),
    )


def map_o1_record_to_odds_quotes(record: JvDataRecord) -> tuple[OddsQuote, ...]:
    fields = parse_minimal_o1_fields(record)
    race_id = build_jravan_race_id(fields)
    captured_at = _parse_o1_captured_at(fields)
    pool_size_jpy = _parse_pool_size_jpy_x100(fields.get("win_pool_size_jpy_x100"))
    quotes: list[OddsQuote] = []

    for horse_number, odds, popularity_rank in _iter_o1_win_odds_entries(fields):
        quote = OddsQuote(
            race_id=race_id,
            runner_id=RunnerId(f"{race_id}-{horse_number}"),
            bet_type=BetType.WIN,
            captured_at=captured_at,
            odds=odds,
            popularity_rank=popularity_rank,
            pool_size_jpy=pool_size_jpy,
            source="jravan_o1_win",
        )
        quotes.append(quote)

    return tuple(quotes)


def map_o1_record_to_odds_quote(record: JvDataRecord) -> OddsQuote:
    quotes = map_o1_record_to_odds_quotes(record)
    if not quotes:
        raise ValueError("O1 record contains no supported win odds quotes")
    return quotes[0]


def _iter_o1_win_odds_entries(
    fields: Mapping[str, str],
) -> Iterable[tuple[str, float, int | None]]:
    block = fields.get("win_odds_entries", "")
    raw_bytes = block.encode("cp932")
    for index in range(28):
        entry_bytes = raw_bytes[index * 8 : (index + 1) * 8]
        if not entry_bytes.strip():
            continue
        entry = entry_bytes.decode("cp932")
        horse_number = entry[0:2].strip()
        odds = _parse_o1_odds_or_none(entry[2:6])
        if odds is None:
            continue
        if len(horse_number) != 2 or not horse_number.isdigit() or horse_number == "00":
            raise ValueError(f"Invalid O1 horse_number: {horse_number!r}")
        popularity_rank = _parse_o1_rank_or_none(entry[6:8])
        yield horse_number, odds, popularity_rank


def _parse_o1_captured_at(fields: Mapping[str, str]) -> datetime:
    race_date = _parse_yyyymmdd(_require(fields, "race_date"))
    data_created_date = _parse_yyyymmdd(_require(fields, "data_created_date"))
    captured_month_day_time = _optional_str(fields.get("captured_month_day_time"))
    if captured_month_day_time is None or captured_month_day_time == "00000000":
        return datetime(
            data_created_date.year,
            data_created_date.month,
            data_created_date.day,
        )

    if len(captured_month_day_time) != 8 or not captured_month_day_time.isdigit():
        raise ValueError(
            f"Invalid captured_month_day_time: {captured_month_day_time!r}"
        )

    month = int(captured_month_day_time[0:2])
    day = int(captured_month_day_time[2:4])
    hour = int(captured_month_day_time[4:6])
    minute = int(captured_month_day_time[6:8])
    try:
        return datetime(race_date.year, month, day, hour, minute)
    except ValueError as exc:
        raise ValueError(
            f"Invalid captured_month_day_time: {captured_month_day_time!r}"
        ) from exc


def _parse_o1_odds_or_none(value: str) -> float | None:
    normalized = value.strip()
    if normalized in {"", "0000", "----", "****"}:
        return None
    if not normalized.isdigit():
        raise ValueError(f"Invalid O1 win odds: {value!r}")
    odds = int(normalized) / 10.0
    return odds if odds > 1.0 else None


def _parse_o1_rank_or_none(value: str) -> int | None:
    normalized = value.strip()
    if normalized in {"", "--", "**"}:
        return None
    return _parse_int(normalized, "popularity_rank")


def _parse_pool_size_jpy_x100(value: str | None) -> int | None:
    pool_size_x100 = _parse_optional_int(value, "win_pool_size_jpy_x100")
    return pool_size_x100 * 100 if pool_size_x100 is not None else None


def _surface_from_track_code(track_code: str | None) -> Surface:
    if track_code in TURF_TRACK_CODES:
        return Surface.TURF
    if track_code in DIRT_TRACK_CODES:
        return Surface.DIRT
    if track_code in JUMP_TRACK_CODES:
        return Surface.JUMP
    return Surface.UNKNOWN


def _direction_from_track_code(track_code: str | None) -> CourseDirection:
    if track_code in LEFT_TRACK_CODES:
        return CourseDirection.LEFT
    if track_code in RIGHT_TRACK_CODES:
        return CourseDirection.RIGHT
    if track_code in STRAIGHT_TRACK_CODES:
        return CourseDirection.STRAIGHT
    return CourseDirection.UNKNOWN


def _track_condition_code_for_surface(
    fields: Mapping[str, str],
    surface: Surface,
) -> str | None:
    if surface == Surface.DIRT:
        return _optional_str(fields.get("dirt_track_condition_code"))
    return _optional_str(fields.get("turf_track_condition_code"))


def _parse_field_size(fields: Mapping[str, str]) -> int | None:
    starter_count = _parse_optional_int(fields.get("starter_count"), "starter_count")
    if starter_count is not None and starter_count > 0:
        return starter_count
    registered_count = _parse_optional_int(
        fields.get("registered_horse_count"),
        "registered_horse_count",
    )
    if registered_count is not None and registered_count > 0:
        return registered_count
    return None


def _is_disqualified(abnormal_code: str | None) -> bool:
    normalized = (abnormal_code or "").strip()
    if normalized in {"", "0"}:
        return False
    # JV-Data abnormal codes represent non-normal outcomes. The MVP result
    # schema only has one coarse flag, so preserve the detailed code in later
    # schema work and use this as a conservative boolean for now.
    return True


def _parse_prize_jpy(value_x100: str | None) -> int | None:
    prize_x100 = _parse_optional_int(value_x100, "prize_jpy_x100")
    return prize_x100 * 100 if prize_x100 is not None else None


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


def _parse_hhmm(
    race_date: date,
    value: str | None,
    *,
    field_name: str = "start_time",
) -> datetime | None:
    normalized = _optional_str(value)
    if normalized is None:
        return None
    if len(normalized) != 4 or not normalized.isdigit():
        raise ValueError(f"Invalid {field_name}: {value!r}")
    try:
        return datetime(
            race_date.year,
            race_date.month,
            race_date.day,
            int(normalized[:2]),
            int(normalized[2:4]),
        )
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc


def _parse_required_hhmm(
    race_date: date,
    value: str | None,
    *,
    field_name: str,
) -> datetime:
    parsed = _parse_hhmm(race_date, value, field_name=field_name)
    if parsed is None:
        raise ValueError(f"Missing required JRA-VAN field: {field_name}")
    return parsed


def _parse_deci_number(value: str | None, field_name: str) -> float | None:
    normalized = _optional_str(value)
    return _parse_int(normalized, field_name) / 10.0 if normalized is not None else None


def _parse_required_deci_number(value: str | None, field_name: str) -> float:
    parsed = _parse_deci_number(value, field_name)
    if parsed is None:
        raise ValueError(f"Missing required JRA-VAN field: {field_name}")
    return parsed


def _parse_body_weight_diff(sign: str | None, value: str | None) -> int | None:
    diff = _parse_optional_int(value, "body_weight_diff")
    if diff is None:
        return None
    return -diff if (sign or "").strip() == "-" else diff


def _parse_flag(value: str | None, field_name: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"", "0", "n", "no", "false"}:
        return False
    if normalized in {"1", "y", "yes", "true"}:
        return True
    raise ValueError(f"Invalid flag for {field_name}: {value!r}")


def _unknown_or_none(value: str | None) -> str | None:
    return f"unknown:{value}" if value else None
