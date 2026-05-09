from pathlib import Path

import pytest

from horse_lab.data.csv_parsing import (
    parse_entry_row,
    parse_odds_quote_row,
    parse_race_row,
    parse_result_row,
    read_csv_rows,
)
from horse_lab.data.jravan import (
    JRAVAN_MINIMAL_O1_FIELDS,
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
    FixedWidthField,
    ingest_jvdata_directory_to_staging,
    ingest_jvdata_file_to_staging,
    ingest_jvdata_files_to_staging,
    map_jvdata_records,
    parse_jvdata_record,
)
from horse_lab.schemas import RaceId, RunnerId


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


def _ra_text(**overrides: str) -> str:
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
    return _fixed_width_text(JRAVAN_MINIMAL_RA_FIELDS, values)


def _se_text(**overrides: str) -> str:
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
        "is_dead_heat": "0",
        "final_time_seconds": "0705",
        "prize_jpy_x100": "00100000",
    }
    values.update(overrides)
    return _fixed_width_text(JRAVAN_MINIMAL_SE_FIELDS, values)


def _o1_text(**overrides: str) -> str:
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
    return _fixed_width_text(JRAVAN_MINIMAL_O1_FIELDS, values)


def _o1_win_entries(*entries: tuple[str, str, str]) -> str:
    encoded = "".join(
        f"{horse_number:>2}{odds:>4}{popularity_rank:>2}"
        for horse_number, odds, popularity_rank in entries
    )
    return encoded.ljust(224)


def _write_raw_file(path: Path, *lines: str) -> None:
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932"))


def test_ingest_jvdata_file_to_staging_maps_ra_se_and_skips_unknown_records(
    tmp_path: Path,
):
    raw_path = tmp_path / "jvdata.txt"
    _write_raw_file(raw_path, _ra_text(), _se_text(), _o1_text(), "ZZignored")

    export = ingest_jvdata_file_to_staging(raw_path, tmp_path / "staging")

    assert len(export.dataset.races) == 1
    assert len(export.dataset.entries) == 1
    assert len(export.dataset.results) == 1
    assert len(export.dataset.odds) == 1
    assert export.dataset.skipped_records[0].record_type == "ZZ"
    assert export.dataset.skipped_records[0].line_number == 4
    assert export.dataset.skipped_records[0].reason == "unsupported_record_type"

    race = parse_race_row(read_csv_rows(export.csv_paths["races"])[0])
    entry = parse_entry_row(read_csv_rows(export.csv_paths["entries"])[0])
    result = parse_result_row(read_csv_rows(export.csv_paths["results"])[0])
    quote = parse_odds_quote_row(read_csv_rows(export.csv_paths["odds"])[0])

    assert race.race_id == RaceId("2026050805010101")
    assert race.name == "若葉ステークス"
    assert entry.runner_id == RunnerId("2026050805010101-07")
    assert entry.body_weight_diff_kg == -8
    assert result.runner_id == entry.runner_id
    assert result.did_win is True
    assert quote.runner_id == entry.runner_id
    assert quote.odds == 3.5


def test_map_jvdata_records_skips_unassigned_se_records():
    records = [
        parse_jvdata_record(
            _se_text(data_kubun="1", gate_number="0", horse_number="00"),
            line_number=1,
        ),
        parse_jvdata_record(_se_text(), line_number=2),
    ]

    dataset = map_jvdata_records(records)

    assert len(dataset.entries) == 1
    assert len(dataset.results) == 1
    assert dataset.skipped_records[0].record_type == "SE"
    assert dataset.skipped_records[0].line_number == 1
    assert dataset.skipped_records[0].reason == "unassigned_runner_key"


def test_ingest_jvdata_files_to_staging_combines_multiple_raw_dumps(
    tmp_path: Path,
):
    race_path = tmp_path / "20260508_RACE.txt"
    odds_path = tmp_path / "20260508_0B31_jvgets.txt"
    _write_raw_file(race_path, _ra_text(), _se_text())
    _write_raw_file(odds_path, _o1_text())

    export = ingest_jvdata_files_to_staging(
        [race_path, odds_path],
        tmp_path / "staging",
    )

    assert len(export.dataset.races) == 1
    assert len(export.dataset.entries) == 1
    assert len(export.dataset.results) == 1
    assert len(export.dataset.odds) == 1
    assert read_csv_rows(export.csv_paths["odds"])[0]["odds"] == "3.5"


def test_ingest_jvdata_directory_to_staging_scans_nested_raw_dumps(
    tmp_path: Path,
):
    raw_dir = tmp_path / "raw"
    nested_dir = raw_dir / "rt"
    nested_dir.mkdir(parents=True)
    _write_raw_file(raw_dir / "20260508_RACE.txt", _ra_text(), _se_text())
    _write_raw_file(nested_dir / "20260508_0B31_jvgets.txt", _o1_text())
    (raw_dir / "20260508_RACE.utf8.txt").write_text(
        _ra_text(race_name="検査用プレビュー"),
        encoding="utf-8",
    )
    (nested_dir / "20260508_0B31_jvgets.stdout.txt").write_text(
        '{"outputPath":"not raw"}',
        encoding="utf-8",
    )

    export = ingest_jvdata_directory_to_staging(raw_dir, tmp_path / "staging")

    assert len(export.dataset.races) == 1
    assert len(export.dataset.entries) == 1
    assert len(export.dataset.results) == 1
    assert len(export.dataset.odds) == 1
    assert export.dataset.races[0].name == "若葉ステークス"


def test_ingest_jvdata_directory_to_staging_rejects_empty_matches(tmp_path: Path):
    with pytest.raises(ValueError, match="No JV-Data raw files found"):
        ingest_jvdata_directory_to_staging(tmp_path / "missing", tmp_path / "staging")


def test_map_jvdata_records_keeps_last_duplicate_update():
    records = [
        parse_jvdata_record(_ra_text(race_name="初期レース"), line_number=1),
        parse_jvdata_record(_ra_text(race_name="更新レース"), line_number=2),
        parse_jvdata_record(
            _se_text(
                horse_name="出走前ホース",
                finish_position="",
                abnormal_code="",
                is_dead_heat="",
                final_time_seconds="",
                prize_jpy_x100="",
            ),
            line_number=3,
        ),
        parse_jvdata_record(_se_text(horse_name="確定ホース"), line_number=4),
        parse_jvdata_record(
            _o1_text(win_odds_entries=_o1_win_entries(("07", "0040", "02"))),
            line_number=5,
        ),
        parse_jvdata_record(
            _o1_text(win_odds_entries=_o1_win_entries(("07", "0035", "02"))),
            line_number=6,
        ),
        parse_jvdata_record(
            _o1_text(
                captured_month_day_time="05081000",
                win_odds_entries=_o1_win_entries(("07", "0030", "01")),
            ),
            line_number=7,
        ),
    ]

    dataset = map_jvdata_records(records)

    assert [race.name for race in dataset.races] == ["更新レース"]
    assert [entry.metadata["horse_name"] for entry in dataset.entries] == [
        "確定ホース"
    ]
    assert [result.finish_position for result in dataset.results] == [1]
    assert [quote.odds for quote in dataset.odds] == [3.5, 3.0]


def test_map_jvdata_records_applies_deleted_se_record():
    records = [
        parse_jvdata_record(_ra_text(), line_number=1),
        parse_jvdata_record(_se_text(), line_number=2),
        parse_jvdata_record(
            _se_text(
                data_kubun="9",
                finish_position="00",
                abnormal_code="0",
                final_time_seconds="0000",
                prize_jpy_x100="00000000",
            ),
            line_number=3,
        ),
    ]

    dataset = map_jvdata_records(records)

    assert len(dataset.races) == 1
    assert dataset.entries == ()
    assert dataset.results == ()
    assert dataset.skipped_records[-1].record_type == "SE"
    assert dataset.skipped_records[-1].line_number == 3
    assert dataset.skipped_records[-1].reason == "deleted_runner_record"


def test_map_jvdata_records_can_reject_unknown_record_types():
    records = [parse_jvdata_record("ZZignored", line_number=9)]

    with pytest.raises(ValueError, match="Unsupported JV-Data record type"):
        map_jvdata_records(records, skip_unknown_records=False)


def test_ingest_jvdata_file_to_staging_wraps_mapping_errors_with_source_context(
    tmp_path: Path,
):
    raw_path = tmp_path / "bad-jvdata.txt"
    _write_raw_file(raw_path, _ra_text(), _se_text(horse_number="XX"))

    with pytest.raises(ValueError, match=r"bad-jvdata\.txt:2.*horse_number"):
        ingest_jvdata_file_to_staging(raw_path, tmp_path / "staging")
