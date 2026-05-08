import datetime as dt

from horse_lab.data.csv_parsing import (
    parse_entry_row,
    parse_race_row,
    parse_result_row,
    read_csv_rows,
)
from horse_lab.data.jravan import parse_jvdata_record
from horse_lab.data.jravan.exporters import (
    entry_to_csv_row,
    result_to_csv_row,
    write_staging_csvs,
)
from horse_lab.data.jravan.layouts import (
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
)
from horse_lab.data.jravan.mappers import (
    map_ra_record_to_race,
    map_se_record_to_entry,
    map_se_record_to_result,
)
from horse_lab.data.jravan.raw import FixedWidthField
from horse_lab.schemas import Entry, HorseId, RaceId, Result, RunnerId


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


def test_jravan_mapped_objects_round_trip_through_staging_csvs(tmp_path):
    race = map_ra_record_to_race(_ra_record())
    entry = map_se_record_to_entry(_se_record())
    result = map_se_record_to_result(_se_record())
    assert result is not None

    paths = write_staging_csvs(
        tmp_path / "staging",
        races=[race],
        entries=[entry],
        results=[result],
    )

    parsed_race = parse_race_row(read_csv_rows(paths["races"])[0])
    parsed_entry = parse_entry_row(read_csv_rows(paths["entries"])[0])
    parsed_result = parse_result_row(read_csv_rows(paths["results"])[0])

    assert parsed_race.race_id == race.race_id
    assert parsed_race.race_date == dt.date(2026, 5, 8)
    assert parsed_race.name == "若葉ステークス"
    assert parsed_race.surface == race.surface
    assert parsed_race.start_time == dt.datetime(2026, 5, 8, 10, 5)
    assert parsed_entry.runner_id == entry.runner_id
    assert parsed_entry.horse_id == entry.horse_id
    assert parsed_entry.carried_weight_kg == 56.5
    assert parsed_entry.body_weight_diff_kg == -8
    assert parsed_result.runner_id == result.runner_id
    assert parsed_result.is_dead_heat is True
    assert parsed_result.final_time_seconds == 70.5
    assert parsed_result.prize_jpy == 10_000_000


def test_csv_row_helpers_render_none_and_booleans_consistently():
    entry = Entry(
        runner_id=RunnerId("2026050805010101-01"),
        race_id=RaceId("2026050805010101"),
        horse_id=HorseId("2020123456"),
        horse_number=1,
        gate_number=None,
        is_scratched=True,
    )
    result = Result(
        race_id=RaceId("2026050805010101"),
        runner_id=RunnerId("2026050805010101-01"),
        finish_position=None,
        is_disqualified=False,
        is_dead_heat=True,
    )

    entry_row = entry_to_csv_row(entry)
    result_row = result_to_csv_row(result)

    assert entry_row["gate_number"] == ""
    assert entry_row["jockey_id"] == ""
    assert entry_row["carried_weight_kg"] == ""
    assert entry_row["is_scratched"] == "true"
    assert result_row["finish_position"] == ""
    assert result_row["is_disqualified"] == "false"
    assert result_row["is_dead_heat"] == "true"
    assert result_row["final_time_seconds"] == ""


def test_write_staging_csvs_orders_rows_deterministically(tmp_path):
    tokyo_race_2 = map_ra_record_to_race(_ra_record(race_number="02"))
    kyoto_race_1 = map_ra_record_to_race(_ra_record(venue_code="08"))
    tokyo_later_date_race_1 = map_ra_record_to_race(
        _ra_record(race_date="20260509")
    )

    tokyo_horse_7 = map_se_record_to_entry(_se_record(horse_number="07"))
    tokyo_horse_2 = map_se_record_to_entry(
        _se_record(horse_number="02", horse_id="2020123452")
    )
    kyoto_horse_1 = map_se_record_to_entry(
        _se_record(venue_code="08", horse_number="01", horse_id="2020123451")
    )

    tokyo_result_7 = map_se_record_to_result(_se_record(horse_number="07"))
    tokyo_result_2 = map_se_record_to_result(
        _se_record(horse_number="02", horse_id="2020123452")
    )
    kyoto_result_1 = map_se_record_to_result(
        _se_record(venue_code="08", horse_number="01", horse_id="2020123451")
    )
    assert tokyo_result_7 is not None
    assert tokyo_result_2 is not None
    assert kyoto_result_1 is not None

    paths = write_staging_csvs(
        tmp_path,
        races=[tokyo_later_date_race_1, tokyo_race_2, kyoto_race_1],
        entries=[tokyo_horse_7, kyoto_horse_1, tokyo_horse_2],
        results=[tokyo_result_7, kyoto_result_1, tokyo_result_2],
    )

    assert [
        (row["race_date"], row["venue"], row["race_number"])
        for row in read_csv_rows(paths["races"])
    ] == [
        ("2026-05-08", "Kyoto", "1"),
        ("2026-05-08", "Tokyo", "2"),
        ("2026-05-09", "Tokyo", "1"),
    ]
    assert [
        (row["race_id"], row["horse_number"])
        for row in read_csv_rows(paths["entries"])
    ] == [
        ("2026050805010101", "2"),
        ("2026050805010101", "7"),
        ("2026050808010101", "1"),
    ]
    assert [
        (row["race_id"], row["runner_id"])
        for row in read_csv_rows(paths["results"])
    ] == [
        ("2026050805010101", "2026050805010101-02"),
        ("2026050805010101", "2026050805010101-07"),
        ("2026050808010101", "2026050808010101-01"),
    ]


def test_write_staging_csvs_returns_expected_paths_and_creates_parent_directory(
    tmp_path,
):
    target = tmp_path / "nested" / "jravan"

    paths = write_staging_csvs(target, races=[], entries=[], results=[])

    assert paths == {
        "races": target / "races.csv",
        "entries": target / "entries.csv",
        "results": target / "results.csv",
    }
    assert (target / "races.csv").exists()
    assert (target / "entries.csv").exists()
    assert (target / "results.csv").exists()
