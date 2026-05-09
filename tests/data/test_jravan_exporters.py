import datetime as dt

from horse_lab.data.csv_parsing import (
    parse_entry_row,
    parse_odds_quote_row,
    parse_race_row,
    parse_result_row,
    read_csv_rows,
)
from horse_lab.data.jravan import parse_jvdata_record
from horse_lab.data.jravan.exporters import (
    entry_to_csv_row,
    odds_quote_to_csv_row,
    result_to_csv_row,
    write_payouts_csv,
    write_staging_csvs,
)
from horse_lab.data.jravan.layouts import (
    JRAVAN_MINIMAL_O1_FIELDS,
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
)
from horse_lab.data.jravan.mappers import (
    map_o1_record_to_odds_quote,
    map_ra_record_to_race,
    map_se_record_to_entry,
    map_se_record_to_result,
)
from horse_lab.data.jravan.raw import FixedWidthField
from horse_lab.schemas import BetType, Entry, HorseId, OddsQuote, RaceId, Result, RunnerId


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
        "data_created_date": "20260507",
        "race_date": "20260508",
        "venue_code": "05",
        "kaiji": "01",
        "nichiji": "01",
        "race_number": "01",
        "race_name": "若葉ステークス",
        "distance_m": "2000",
        "track_code": "11",
        "turf_track_condition_code": "1",
        "dirt_track_condition_code": "0",
        "weather_code": "1",
        "grade_code": "B",
        "start_time": "1005",
        "registered_horse_count": "16",
        "starter_count": "16",
    }
    values.update(overrides)
    return parse_jvdata_record(_fixed_width_text(JRAVAN_MINIMAL_RA_FIELDS, values))


def _se_record(**overrides: str):
    values = {
        "record_type": "SE",
        "data_kubun": "7",
        "data_created_date": "20260507",
        "race_date": "20260508",
        "venue_code": "05",
        "kaiji": "01",
        "nichiji": "01",
        "race_number": "01",
        "horse_number": "07",
        "gate_number": "3",
        "horse_id": "2020123456",
        "horse_name": "テストホース",
        "sex_code": "2",
        "age": "04",
        "trainer_id": "04050",
        "jockey_id": "01020",
        "carried_weight": "565",
        "body_weight": "486",
        "body_weight_diff_sign": "-",
        "body_weight_diff": "008",
        "abnormal_code": "0",
        "finish_position": "01",
        "is_dead_heat": "1",
        "final_time_seconds": "0705",
        "win_odds": "0035",
        "popularity_rank": "02",
        "prize_jpy_x100": "00100000",
    }
    values.update(overrides)
    return parse_jvdata_record(_fixed_width_text(JRAVAN_MINIMAL_SE_FIELDS, values))


def _o1_record(**overrides: str):
    values = {
        "record_type": "O1",
        "data_kubun": "7",
        "data_created_date": "20260508",
        "race_date": "20260508",
        "venue_code": "05",
        "kaiji": "01",
        "nichiji": "01",
        "race_number": "01",
        "captured_month_day_time": "05080950",
        "registered_horse_count": "16",
        "starter_count": "16",
        "win_sale_flag": "7",
        "place_sale_flag": "7",
        "bracket_quinella_sale_flag": "7",
        "place_payout_key": "3",
        "win_odds_entries": _o1_win_entries(("07", "0035", "02")),
        "win_pool_size_jpy_x100": "0000012345",
    }
    values.update(overrides)
    return parse_jvdata_record(_fixed_width_text(JRAVAN_MINIMAL_O1_FIELDS, values))


def _o1_win_entries(*entries: tuple[str, str, str]) -> str:
    encoded = "".join(
        f"{horse_number:>2}{odds:>4}{popularity_rank:>2}"
        for horse_number, odds, popularity_rank in entries
    )
    return encoded.ljust(224)


def test_jravan_mapped_objects_round_trip_through_staging_csvs(tmp_path):
    race = map_ra_record_to_race(_ra_record())
    entry = map_se_record_to_entry(_se_record())
    result = map_se_record_to_result(_se_record())
    quote = map_o1_record_to_odds_quote(_o1_record())
    assert result is not None

    paths = write_staging_csvs(
        tmp_path / "staging",
        races=[race],
        entries=[entry],
        results=[result],
        odds=[quote],
    )

    parsed_race = parse_race_row(read_csv_rows(paths["races"])[0])
    parsed_entry = parse_entry_row(read_csv_rows(paths["entries"])[0])
    parsed_result = parse_result_row(read_csv_rows(paths["results"])[0])
    parsed_quote = parse_odds_quote_row(read_csv_rows(paths["odds"])[0])

    assert parsed_race.race_id == race.race_id
    assert parsed_race.race_date == dt.date(2026, 5, 8)
    assert parsed_race.name == "若葉ステークス"
    assert parsed_race.surface == race.surface
    assert parsed_race.start_time == dt.datetime(2026, 5, 8, 10, 5)
    assert parsed_entry.runner_id == entry.runner_id
    assert parsed_entry.horse_id == entry.horse_id
    assert parsed_entry.carried_weight_kg == 56.5
    assert parsed_entry.body_weight_diff_kg == -8
    assert parsed_entry.metadata["horse_name"] == "テストホース"
    assert parsed_entry.metadata["sex"] == "female"
    assert parsed_entry.metadata["entry_win_odds"] == 3.5
    assert parsed_entry.metadata["entry_popularity_rank"] == 2
    assert parsed_result.runner_id == result.runner_id
    assert parsed_result.is_dead_heat is True
    assert parsed_result.final_time_seconds == 70.5
    assert parsed_result.prize_jpy == 10_000_000
    assert parsed_quote.runner_id == quote.runner_id
    assert parsed_quote.bet_type == BetType.WIN
    assert parsed_quote.captured_at == dt.datetime(2026, 5, 8, 9, 50)
    assert parsed_quote.odds == 3.5


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
    quote = OddsQuote(
        race_id=RaceId("2026050805010101"),
        runner_id=RunnerId("2026050805010101-01"),
        bet_type=BetType.WIN,
        captured_at=dt.datetime(2026, 5, 8, 9, 50),
        odds=3.5,
    )

    entry_row = entry_to_csv_row(entry)
    result_row = result_to_csv_row(result)
    odds_row = odds_quote_to_csv_row(quote)

    assert entry_row["gate_number"] == ""
    assert entry_row["horse_name"] == ""
    assert entry_row["jockey_id"] == ""
    assert entry_row["carried_weight_kg"] == ""
    assert entry_row["entry_win_odds"] == ""
    assert entry_row["is_scratched"] == "true"
    assert result_row["finish_position"] == ""
    assert result_row["is_disqualified"] == "false"
    assert result_row["is_dead_heat"] == "true"
    assert result_row["final_time_seconds"] == ""
    assert odds_row["bet_type"] == "win"
    assert odds_row["popularity_rank"] == ""
    assert odds_row["pool_size_jpy"] == ""


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
    tokyo_quote_7_0950 = map_o1_record_to_odds_quote(_o1_record())
    tokyo_quote_7_0940 = map_o1_record_to_odds_quote(
        _o1_record(
            captured_month_day_time="05080940",
            win_odds_entries=_o1_win_entries(("07", "0040", "02")),
        )
    )
    tokyo_quote_2 = map_o1_record_to_odds_quote(
        _o1_record(win_odds_entries=_o1_win_entries(("02", "0025", "01")))
    )
    assert tokyo_result_7 is not None
    assert tokyo_result_2 is not None
    assert kyoto_result_1 is not None

    paths = write_staging_csvs(
        tmp_path,
        races=[tokyo_later_date_race_1, tokyo_race_2, kyoto_race_1],
        entries=[tokyo_horse_7, kyoto_horse_1, tokyo_horse_2],
        results=[tokyo_result_7, kyoto_result_1, tokyo_result_2],
        odds=[tokyo_quote_7_0950, tokyo_quote_2, tokyo_quote_7_0940],
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
    assert [
        (row["runner_id"], row["captured_at"], row["odds"])
        for row in read_csv_rows(paths["odds"])
    ] == [
        ("2026050805010101-02", "2026-05-08T09:50:00", "2.5"),
        ("2026050805010101-07", "2026-05-08T09:40:00", "4.0"),
        ("2026050805010101-07", "2026-05-08T09:50:00", "3.5"),
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
        "odds": target / "odds.csv",
    }
    assert (target / "races.csv").exists()
    assert (target / "entries.csv").exists()
    assert (target / "results.csv").exists()
    assert (target / "odds.csv").exists()


def test_write_payouts_csv_renders_proxy_rows(tmp_path):
    path = tmp_path / "replay" / "payouts.csv"

    write_payouts_csv(
        path,
        [
            {
                "race_id": RaceId("2026050805010101"),
                "runner_id": RunnerId("2026050805010101-07"),
                "bet_type": BetType.WIN,
                "finish_position": 1,
                "is_win": True,
                "payout_jpy_per_100": 350,
                "odds": 3.5,
                "pool_size_jpy": None,
                "source": "derived_from_latest_win_odds",
            }
        ],
    )

    assert read_csv_rows(path) == [
        {
            "race_id": "2026050805010101",
            "runner_id": "2026050805010101-07",
            "bet_type": "win",
            "finish_position": "1",
            "is_win": "true",
            "payout_jpy_per_100": "350",
            "odds": "3.5",
            "pool_size_jpy": "",
            "source": "derived_from_latest_win_odds",
        }
    ]
