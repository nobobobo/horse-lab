import datetime as dt

import pytest

from horse_lab.data.csv_parsing import (
    parse_bool,
    parse_date,
    parse_datetime,
    parse_datetime_or_none,
    parse_entry_row,
    parse_feature_value,
    parse_feature_row,
    parse_float_or_none,
    parse_int_or_none,
    parse_odds_quote_row,
    parse_race_row,
    parse_result_row,
    parse_str_or_none,
)
from horse_lab.schemas import (
    BetType,
    CourseDirection,
    FeatureName,
    RaceId,
    RunnerId,
    Surface,
    TrackCondition,
)


def test_parse_bool_accepts_common_csv_values():
    assert parse_bool("true") is True
    assert parse_bool("1") is True
    assert parse_bool("false") is False
    assert parse_bool("0") is False
    assert parse_bool("") is False

    with pytest.raises(ValueError, match="boolean"):
        parse_bool("maybe")


def test_optional_number_parsers_return_none_for_blank_values():
    assert parse_int_or_none("") is None
    assert parse_int_or_none("42") == 42
    assert parse_float_or_none("") is None
    assert parse_float_or_none("56.5") == 56.5


def test_optional_string_parser_strips_blank_values():
    assert parse_str_or_none("") is None
    assert parse_str_or_none("   ") is None
    assert parse_str_or_none(" Tokyo ") == "Tokyo"


def test_date_and_datetime_parsers_use_iso_values():
    assert parse_date("2026-05-08") == dt.date(2026, 5, 8)
    assert parse_datetime("2026-05-08T10:00:00") == dt.datetime(2026, 5, 8, 10, 0)
    assert parse_datetime_or_none("") is None
    assert parse_datetime_or_none("   ") is None
    assert parse_datetime_or_none("2026-05-08T09:55:00") == dt.datetime(
        2026, 5, 8, 9, 55
    )


@pytest.mark.parametrize(
    ("parser", "value"),
    [
        (parse_date, "2026-99-99"),
        (parse_datetime, "not-a-datetime"),
        (parse_datetime_or_none, "not-a-datetime"),
        (parse_int_or_none, "12.5"),
        (parse_float_or_none, "not-a-number"),
    ],
)
def test_invalid_date_datetime_and_numeric_inputs_raise_value_error(parser, value):
    with pytest.raises(ValueError):
        parser(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", None),
        ("   ", None),
        ("false", False),
        ("FALSE", False),
        ("72", 72),
        ("0.1", 0.1),
        ("fixture", "fixture"),
    ],
)
def test_parse_feature_value_coerces_supported_scalar_values(value, expected):
    assert parse_feature_value(value) == expected


def test_parse_race_row_builds_race_schema():
    race = parse_race_row(
        {
            "race_id": "202605080101",
            "race_date": "2026-05-08",
            "venue": "Tokyo",
            "race_number": "1",
            "name": "Fixture Sprint",
            "surface": "turf",
            "distance_m": "1200",
            "direction": "left",
            "track_condition": "firm",
            "weather": "Sunny",
            "grade": "",
            "start_time": "2026-05-08T10:00:00",
            "field_size": "2",
        }
    )

    assert race.race_id == RaceId("202605080101")
    assert race.race_date == dt.date(2026, 5, 8)
    assert race.surface == Surface.TURF
    assert race.direction == CourseDirection.LEFT
    assert race.track_condition == TrackCondition.FIRM
    assert race.start_time == dt.datetime(2026, 5, 8, 10, 0)
    assert race.field_size == 2
    assert race.grade is None


def test_parse_race_row_defaults_blank_enum_fields_to_unknown():
    race = parse_race_row(
        {
            "race_id": "202605080101",
            "race_date": "2026-05-08",
            "venue": "Tokyo",
            "race_number": "1",
            "name": "Fixture Sprint",
            "surface": "   ",
            "distance_m": "1200",
            "direction": "   ",
            "track_condition": "   ",
            "weather": "Sunny",
            "grade": "",
            "start_time": "2026-05-08T10:00:00",
            "field_size": "2",
        }
    )

    assert race.surface == Surface.UNKNOWN
    assert race.direction == CourseDirection.UNKNOWN
    assert race.track_condition == TrackCondition.UNKNOWN


def test_parse_entry_row_builds_entry_schema():
    entry = parse_entry_row(
        {
            "runner_id": "202605080101-01",
            "race_id": "202605080101",
            "horse_id": "horse-001",
            "horse_number": "1",
            "gate_number": "1",
            "jockey_id": "jockey-001",
            "trainer_id": "trainer-001",
            "carried_weight_kg": "56.0",
            "body_weight_kg": "480",
            "body_weight_diff_kg": "2",
            "age": "4",
            "is_scratched": "false",
        }
    )

    assert entry.runner_id == RunnerId("202605080101-01")
    assert entry.gate_number == 1
    assert entry.carried_weight_kg == 56.0
    assert entry.is_scratched is False


def test_parse_result_row_builds_result_schema():
    result = parse_result_row(
        {
            "race_id": "202605080101",
            "runner_id": "202605080101-01",
            "finish_position": "1",
            "is_disqualified": "false",
            "is_dead_heat": "false",
            "final_time_seconds": "70.5",
            "prize_jpy": "10000000",
        }
    )

    assert result.did_win is True
    assert result.final_time_seconds == 70.5
    assert result.prize_jpy == 10_000_000


def test_parse_odds_quote_row_builds_odds_schema():
    quote = parse_odds_quote_row(
        {
            "race_id": "202605080101",
            "runner_id": "202605080101-01",
            "bet_type": "win",
            "captured_at": "2026-05-08T09:50:00",
            "odds": "3.0",
            "popularity_rank": "1",
            "pool_size_jpy": "123456",
            "source": "fixture",
        }
    )

    assert quote.bet_type == BetType.WIN
    assert quote.captured_at == dt.datetime(2026, 5, 8, 9, 50)
    assert quote.odds == 3.0
    assert quote.pool_size_jpy == 123_456


def test_parse_feature_row_strips_feature_prefix_and_coerces_values():
    feature = parse_feature_row(
        {
            "race_id": "202605080101",
            "runner_id": "202605080101-01",
            "as_of": "2026-05-08T09:55:00",
            "feature_version": "fixture-v1",
            "feature__recent_speed": "72",
            "feature__gate_bias": "0.1",
            "feature__is_favorite": "true",
            "notes": "ignored",
        }
    )

    assert feature.race_id == RaceId("202605080101")
    assert feature.as_of == dt.datetime(2026, 5, 8, 9, 55)
    assert feature.values[FeatureName("recent_speed")] == 72
    assert feature.values[FeatureName("gate_bias")] == 0.1
    assert feature.values[FeatureName("is_favorite")] is True
    assert FeatureName("notes") not in feature.values
