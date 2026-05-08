from pathlib import Path

import pytest

from horse_lab.data.jravan import (
    FixedWidthField,
    parse_fixed_width_fields,
    parse_jvdata_record,
    read_jvdata_records,
)


def test_parse_jvdata_record_wraps_raw_line_without_record_specific_assumptions():
    record = parse_jvdata_record("RA20260508東京01\n", line_number=7)

    assert record.record_type == "RA"
    assert record.text == "RA20260508東京01"
    assert record.line_number == 7


def test_read_jvdata_records_uses_cp932_and_preserves_source_metadata(tmp_path: Path):
    path = tmp_path / "jvdata.txt"
    path.write_bytes("RA20260508東京01\r\n\nSE20260508010101馬名\r\n".encode("cp932"))

    records = read_jvdata_records(path)

    assert [record.record_type for record in records] == ["RA", "SE"]
    assert [record.line_number for record in records] == [1, 3]
    assert {record.source_path for record in records} == {path}
    assert records[0].text.endswith("東京01")


def test_parse_fixed_width_fields_uses_byte_offsets_for_japanese_text():
    raw = b"RA" + b"20260508" + "東京".encode("cp932") + b"  " + b"01"

    values = parse_fixed_width_fields(
        raw,
        fields=[
            FixedWidthField("record_type", start=1, length=2),
            FixedWidthField("race_date", start=3, length=8),
            FixedWidthField("venue", start=11, length=6),
            FixedWidthField("race_number", start=17, length=2),
        ],
    )

    assert values == {
        "record_type": "RA",
        "race_date": "20260508",
        "venue": "東京",
        "race_number": "01",
    }


def test_fixed_width_field_validates_byte_positions():
    with pytest.raises(ValueError, match="start"):
        FixedWidthField("bad", start=0, length=1)

    with pytest.raises(ValueError, match="length"):
        FixedWidthField("bad", start=1, length=0)
