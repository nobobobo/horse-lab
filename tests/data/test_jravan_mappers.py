import pytest

from horse_lab.data.jravan import parse_jvdata_record
from horse_lab.data.jravan.layouts import (
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
    parse_minimal_ra_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.raw import FixedWidthField


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
