import datetime as dt

import pytest

from horse_lab.data.jravan import parse_jvdata_record
from horse_lab.data.jravan.layouts import (
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
    parse_minimal_ra_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.mappers import (
    build_jravan_race_id,
    build_jravan_runner_id,
    map_ra_record_to_race,
    map_se_record_to_entry,
    map_se_record_to_result,
)
from horse_lab.data.jravan.raw import FixedWidthField
from horse_lab.schemas import (
    CourseDirection,
    HorseId,
    PersonId,
    RaceId,
    RunnerId,
    Surface,
    TrackCondition,
)


def _fixed_width_text(
    fields: tuple[FixedWidthField, ...],
    values: dict[str, str],
) -> str:
    total_length = max(field.start + field.length - 1 for field in fields)
    raw = bytearray(b" " * total_length)
    for field in fields:
        encoded = values.get(field.name, "").encode("cp932")
        assert len(encoded) <= field.length, field.name
        start_index = field.start - 1
        raw[start_index : start_index + field.length] = encoded.ljust(
            field.length,
            b" ",
        )
    return bytes(raw).decode("cp932")


def _ra_record(**overrides: str):
    values = {
        "record_type": "RA",
        "data_kubun": "7",
        "race_date": "20260508",
        "venue_code": "05",
        "kaiji": "01",
        "nichiji": "01",
        "race_number": "01",
        "race_name": "若葉ステークス",
        "surface_code": "1",
        "distance_m": "2000",
        "direction_code": "2",
        "track_condition_code": "1",
        "weather_code": "1",
        "grade_code": "G2",
        "start_time": "1005",
        "field_size": "16",
    }
    values.update(overrides)
    return parse_jvdata_record(_fixed_width_text(JRAVAN_MINIMAL_RA_FIELDS, values))


def _se_record(**overrides: str):
    values = {
        "record_type": "SE",
        "data_kubun": "7",
        "race_date": "20260508",
        "venue_code": "05",
        "kaiji": "01",
        "nichiji": "01",
        "race_number": "01",
        "horse_number": "07",
        "gate_number": "03",
        "horse_id": "2020123456",
        "horse_name": "テストホース",
        "sex_code": "2",
        "age": "04",
        "trainer_id": "040506",
        "jockey_id": "010203",
        "carried_weight": "565",
        "body_weight": "486",
        "body_weight_diff_sign": "-",
        "body_weight_diff": "08",
        "finish_position": "01",
        "is_disqualified": "0",
        "is_dead_heat": "1",
        "final_time_seconds": "00705",
        "prize_jpy": "10000000",
    }
    values.update(overrides)
    return parse_jvdata_record(_fixed_width_text(JRAVAN_MINIMAL_SE_FIELDS, values))


def test_minimal_ra_layout_extracts_cp932_fixed_width_fields():
    fields = parse_minimal_ra_fields(_ra_record())

    assert fields["record_type"] == "RA"
    assert fields["race_date"] == "20260508"
    assert fields["venue_code"] == "05"
    assert fields["race_name"] == "若葉ステークス"
    assert fields["surface_code"] == "1"
    assert fields["start_time"] == "1005"


def test_minimal_se_layout_extracts_cp932_fixed_width_fields():
    fields = parse_minimal_se_fields(_se_record())

    assert fields["record_type"] == "SE"
    assert fields["race_date"] == "20260508"
    assert fields["horse_number"] == "07"
    assert fields["horse_name"] == "テストホース"
    assert fields["carried_weight"] == "565"
    assert fields["final_time_seconds"] == "00705"


def test_minimal_layout_helpers_reject_wrong_record_types():
    with pytest.raises(ValueError, match="Expected RA"):
        parse_minimal_ra_fields(_se_record())

    with pytest.raises(ValueError, match="Expected SE"):
        parse_minimal_se_fields(_ra_record())


def test_build_jravan_race_id_uses_date_venue_meeting_day_and_race_number():
    fields = parse_minimal_ra_fields(_ra_record())

    assert build_jravan_race_id(fields) == RaceId("2026050805010101")


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("kaiji", "1"),
        ("race_number", "1"),
    ),
)
def test_build_jravan_race_id_rejects_malformed_two_digit_components(
    field_name: str,
    value: str,
):
    fields = parse_minimal_ra_fields(_ra_record(**{field_name: value}))

    with pytest.raises(ValueError, match=field_name):
        build_jravan_race_id(fields)


def test_build_jravan_race_id_rejects_invalid_race_date():
    fields = parse_minimal_ra_fields(_ra_record(race_date="20260230"))

    with pytest.raises(ValueError, match="race_date"):
        build_jravan_race_id(fields)


def test_map_ra_record_to_race_maps_minimal_race_schema():
    race = map_ra_record_to_race(_ra_record())

    assert race.race_id == RaceId("2026050805010101")
    assert race.race_date == dt.date(2026, 5, 8)
    assert race.venue == "Tokyo"
    assert race.race_number == 1
    assert race.name == "若葉ステークス"
    assert race.surface == Surface.TURF
    assert race.distance_m == 2000
    assert race.direction == CourseDirection.LEFT
    assert race.track_condition == TrackCondition.FIRM
    assert race.weather == "sunny"
    assert race.grade == "G2"
    assert race.start_time == dt.datetime(2026, 5, 8, 10, 5)
    assert race.field_size == 16
    assert race.metadata["venue_code"] == "05"
    assert race.metadata["kaiji"] == "01"
    assert race.metadata["nichiji"] == "01"
    assert race.metadata["data_kubun"] == "7"
    assert race.metadata["grade_code"] == "G2"


def test_map_ra_record_to_race_preserves_unknown_codes_in_metadata():
    race = map_ra_record_to_race(
        _ra_record(
            venue_code="99",
            surface_code="9",
            direction_code="9",
            track_condition_code="9",
            weather_code="9",
        )
    )

    assert race.venue == "unknown:99"
    assert race.surface == Surface.UNKNOWN
    assert race.direction == CourseDirection.UNKNOWN
    assert race.track_condition == TrackCondition.UNKNOWN
    assert race.weather == "unknown:9"
    assert race.metadata["surface_code"] == "9"
    assert race.metadata["direction_code"] == "9"
    assert race.metadata["track_condition_code"] == "9"


def test_map_ra_record_to_race_rejects_non_ra_record():
    with pytest.raises(ValueError, match="Expected RA"):
        map_ra_record_to_race(_se_record())


def test_build_jravan_runner_id_adds_horse_number_suffix():
    fields = parse_minimal_se_fields(_se_record())

    assert build_jravan_runner_id(fields) == RunnerId("2026050805010101-07")


def test_build_jravan_runner_id_rejects_malformed_horse_number():
    fields = parse_minimal_se_fields(_se_record(horse_number="7"))

    with pytest.raises(ValueError, match="horse_number"):
        build_jravan_runner_id(fields)


def test_build_jravan_runner_id_rejects_zero_horse_number():
    fields = parse_minimal_se_fields(_se_record(horse_number="00"))

    with pytest.raises(ValueError, match="horse_number"):
        build_jravan_runner_id(fields)


def test_map_se_record_to_entry_maps_minimal_entry_schema():
    entry = map_se_record_to_entry(_se_record())

    assert entry.runner_id == RunnerId("2026050805010101-07")
    assert entry.race_id == RaceId("2026050805010101")
    assert entry.horse_id == HorseId("2020123456")
    assert entry.horse_number == 7
    assert entry.gate_number == 3
    assert entry.jockey_id == PersonId("010203")
    assert entry.trainer_id == PersonId("040506")
    assert entry.carried_weight_kg == 56.5
    assert entry.body_weight_kg == 486
    assert entry.body_weight_diff_kg == -8
    assert entry.age == 4
    assert entry.is_scratched is False
    assert entry.metadata["horse_name"] == "テストホース"
    assert entry.metadata["sex_code"] == "2"
    assert entry.metadata["data_kubun"] == "7"


def test_map_se_record_to_entry_maps_blank_body_weight_diff_to_none():
    entry = map_se_record_to_entry(
        _se_record(body_weight_diff_sign="", body_weight_diff="")
    )

    assert entry.body_weight_diff_kg is None


def test_map_se_record_to_entry_maps_unsigned_body_weight_diff_as_non_negative():
    entry = map_se_record_to_entry(
        _se_record(body_weight_diff_sign="", body_weight_diff="08")
    )

    assert entry.body_weight_diff_kg == 8


def test_map_se_record_to_result_maps_populated_result_fields():
    result = map_se_record_to_result(_se_record())

    assert result is not None
    assert result.race_id == RaceId("2026050805010101")
    assert result.runner_id == RunnerId("2026050805010101-07")
    assert result.finish_position == 1
    assert result.is_disqualified is False
    assert result.is_dead_heat is True
    assert result.final_time_seconds == 70.5
    assert result.prize_jpy == 10_000_000
    assert result.did_win is True


def test_map_se_record_to_result_returns_none_when_finish_position_is_blank():
    result = map_se_record_to_result(
        _se_record(
            finish_position="",
            is_disqualified="",
            is_dead_heat="",
            final_time_seconds="",
            prize_jpy="",
        )
    )

    assert result is None


def test_map_se_record_to_entry_rejects_non_se_record():
    with pytest.raises(ValueError, match="Expected SE"):
        map_se_record_to_entry(_ra_record())


def test_map_se_record_to_result_rejects_invalid_finish_position():
    with pytest.raises(ValueError, match="Invalid integer for finish_position"):
        map_se_record_to_result(_se_record(finish_position="XX"))


def test_map_se_record_to_result_rejects_zero_finish_position():
    with pytest.raises(ValueError, match="finish_position"):
        map_se_record_to_result(_se_record(finish_position="00"))
