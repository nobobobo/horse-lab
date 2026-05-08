# JRA-VAN Minimal RA/SE Mapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map synthetic JRA-VAN `RA` and `SE` raw records into Horse Lab canonical `Race`, `Entry`, and `Result` schemas without requiring Windows or JV-Link locally.

**Architecture:** Keep low-level JV-Data record parsing in `raw.py`, add minimal fixed-width field profiles in `layouts.py`, and add canonical schema conversion in `mappers.py`. This slice deliberately uses synthetic CP932 fixtures and a minimal layout profile rather than full official JV-Data coverage.

**Tech Stack:** Python 3.9 standard library, dataclasses/NewType domain schemas, pytest, CP932 fixed-width byte parsing.

---

## File Structure

- Create `src/horse_lab/data/jravan/layouts.py`
  - Owns minimal RA/SE `FixedWidthField` definitions and field extraction helpers.
- Create `src/horse_lab/data/jravan/mappers.py`
  - Converts parsed RA/SE field dicts into `Race`, `Entry`, and `Result`.
- Modify `src/horse_lab/data/jravan/__init__.py`
  - Exports layout and mapper APIs.
- Create `tests/data/test_jravan_mappers.py`
  - Synthetic CP932 fixed-width fixtures and mapper behavior tests.
- Modify `tests/test_package_imports.py`
  - Adds import assertion for public JRA-VAN mapper helpers.

---

### Task 1: Minimal RA/SE Layout Parsing

**Files:**
- Create: `src/horse_lab/data/jravan/layouts.py`
- Create: `tests/data/test_jravan_mappers.py`
- Modify: `src/horse_lab/data/jravan/__init__.py`

- [ ] **Step 1: Write failing layout tests**

Create `tests/data/test_jravan_mappers.py`:

```python
import datetime as dt

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
        raw[start_index:start_index + field.length] = encoded.ljust(field.length, b" ")
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
```

- [ ] **Step 2: Run layout tests and verify RED**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_jravan_mappers.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'horse_lab.data.jravan.layouts'`.

- [ ] **Step 3: Implement minimal layouts**

Create `src/horse_lab/data/jravan/layouts.py`:

```python
"""Minimal synthetic JRA-VAN RA/SE fixed-width layouts."""

from __future__ import annotations

from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
)


JRAVAN_MINIMAL_RA_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("race_date", start=4, length=8),
    FixedWidthField("venue_code", start=12, length=2),
    FixedWidthField("kaiji", start=14, length=2),
    FixedWidthField("nichiji", start=16, length=2),
    FixedWidthField("race_number", start=18, length=2),
    FixedWidthField("race_name", start=20, length=60),
    FixedWidthField("surface_code", start=80, length=1),
    FixedWidthField("distance_m", start=81, length=4),
    FixedWidthField("direction_code", start=85, length=1),
    FixedWidthField("track_condition_code", start=86, length=1),
    FixedWidthField("weather_code", start=87, length=1),
    FixedWidthField("grade_code", start=88, length=2),
    FixedWidthField("start_time", start=90, length=4),
    FixedWidthField("field_size", start=94, length=2),
)

JRAVAN_MINIMAL_SE_FIELDS: tuple[FixedWidthField, ...] = (
    FixedWidthField("record_type", start=1, length=2),
    FixedWidthField("data_kubun", start=3, length=1),
    FixedWidthField("race_date", start=4, length=8),
    FixedWidthField("venue_code", start=12, length=2),
    FixedWidthField("kaiji", start=14, length=2),
    FixedWidthField("nichiji", start=16, length=2),
    FixedWidthField("race_number", start=18, length=2),
    FixedWidthField("horse_number", start=20, length=2),
    FixedWidthField("gate_number", start=22, length=2),
    FixedWidthField("horse_id", start=24, length=10),
    FixedWidthField("horse_name", start=34, length=40),
    FixedWidthField("sex_code", start=74, length=1),
    FixedWidthField("age", start=75, length=2),
    FixedWidthField("trainer_id", start=77, length=6),
    FixedWidthField("jockey_id", start=83, length=6),
    FixedWidthField("carried_weight", start=89, length=3),
    FixedWidthField("body_weight", start=92, length=3),
    FixedWidthField("body_weight_diff_sign", start=95, length=1),
    FixedWidthField("body_weight_diff", start=96, length=2),
    FixedWidthField("finish_position", start=98, length=2),
    FixedWidthField("is_disqualified", start=100, length=1),
    FixedWidthField("is_dead_heat", start=101, length=1),
    FixedWidthField("final_time_seconds", start=102, length=5),
    FixedWidthField("prize_jpy", start=107, length=8),
)


def parse_minimal_ra_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "RA")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_RA_FIELDS)


def parse_minimal_se_fields(record: JvDataRecord) -> dict[str, str]:
    _require_record_type(record, "SE")
    return parse_fixed_width_fields(record.text, JRAVAN_MINIMAL_SE_FIELDS)


def _require_record_type(record: JvDataRecord, expected: str) -> None:
    if record.record_type != expected:
        raise ValueError(
            f"Expected {expected} record, got {record.record_type!r}"
        )
```

Update `src/horse_lab/data/jravan/__init__.py`:

```python
"""JRA-VAN raw-data ingestion helpers.

The JV-Link extraction process itself is expected to run on Windows. This
package keeps the platform-independent parsing and normalization boundary in
Python so the rest of horse_lab can run on macOS/Linux.
"""

from horse_lab.data.jravan.layouts import (
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
    parse_minimal_ra_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
    parse_jvdata_record,
    read_jvdata_records,
)

__all__ = [
    "FixedWidthField",
    "JRAVAN_MINIMAL_RA_FIELDS",
    "JRAVAN_MINIMAL_SE_FIELDS",
    "JvDataRecord",
    "parse_fixed_width_fields",
    "parse_jvdata_record",
    "parse_minimal_ra_fields",
    "parse_minimal_se_fields",
    "read_jvdata_records",
]
```

- [ ] **Step 4: Run layout tests and verify GREEN**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_jravan_mappers.py tests/data/test_jravan_raw.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit layouts**

Run:

```bash
git add src/horse_lab/data/jravan/__init__.py src/horse_lab/data/jravan/layouts.py tests/data/test_jravan_mappers.py
git commit -m "feat: add jravan minimal layouts"
```

---

### Task 2: RA Record To Race Mapper

**Files:**
- Create: `src/horse_lab/data/jravan/mappers.py`
- Modify: `tests/data/test_jravan_mappers.py`
- Modify: `src/horse_lab/data/jravan/__init__.py`

- [ ] **Step 1: Extend tests for RA mapping**

Append to `tests/data/test_jravan_mappers.py`:

```python
from horse_lab.data.jravan.mappers import (
    build_jravan_race_id,
    map_ra_record_to_race,
)
from horse_lab.schemas import (
    CourseDirection,
    RaceId,
    Surface,
    TrackCondition,
)


def test_build_jravan_race_id_uses_date_venue_meeting_day_and_race_number():
    fields = parse_minimal_ra_fields(_ra_record())

    assert build_jravan_race_id(fields) == RaceId("2026050805010101")


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
```

- [ ] **Step 2: Run RA mapper tests and verify RED**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_jravan_mappers.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'horse_lab.data.jravan.mappers'`.

- [ ] **Step 3: Implement RA mapper**

Create `src/horse_lab/data/jravan/mappers.py`:

```python
"""Canonical schema mappers for minimal JRA-VAN RA/SE records."""

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
```

Update `src/horse_lab/data/jravan/__init__.py`:

```python
"""JRA-VAN raw-data ingestion helpers.

The JV-Link extraction process itself is expected to run on Windows. This
package keeps the platform-independent parsing and normalization boundary in
Python so the rest of horse_lab can run on macOS/Linux.
"""

from horse_lab.data.jravan.layouts import (
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
    parse_minimal_ra_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.mappers import (
    build_jravan_race_id,
    map_ra_record_to_race,
)
from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
    parse_jvdata_record,
    read_jvdata_records,
)

__all__ = [
    "FixedWidthField",
    "JRAVAN_MINIMAL_RA_FIELDS",
    "JRAVAN_MINIMAL_SE_FIELDS",
    "JvDataRecord",
    "build_jravan_race_id",
    "map_ra_record_to_race",
    "parse_fixed_width_fields",
    "parse_jvdata_record",
    "parse_minimal_ra_fields",
    "parse_minimal_se_fields",
    "read_jvdata_records",
]
```

- [ ] **Step 4: Run RA mapper tests and verify GREEN**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_jravan_mappers.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit RA mapper**

Run:

```bash
git add src/horse_lab/data/jravan/__init__.py src/horse_lab/data/jravan/mappers.py tests/data/test_jravan_mappers.py
git commit -m "feat: map jravan ra records to races"
```

---

### Task 3: SE Record To Entry And Result Mappers

**Files:**
- Modify: `src/horse_lab/data/jravan/mappers.py`
- Modify: `src/horse_lab/data/jravan/__init__.py`
- Modify: `tests/data/test_jravan_mappers.py`
- Modify: `tests/test_package_imports.py`

- [ ] **Step 1: Extend tests for SE mapping**

Append to `tests/data/test_jravan_mappers.py`:

```python
from horse_lab.data.jravan.mappers import (
    build_jravan_runner_id,
    map_se_record_to_entry,
    map_se_record_to_result,
)
from horse_lab.schemas import HorseId, PersonId, RunnerId


def test_build_jravan_runner_id_adds_horse_number_suffix():
    fields = parse_minimal_se_fields(_se_record())

    assert build_jravan_runner_id(fields) == RunnerId("2026050805010101-07")


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
```

Extend `tests/test_package_imports.py` by replacing `test_jravan_ingestion_helpers_import` with:

```python
def test_jravan_ingestion_helpers_import():
    from horse_lab.data.jravan import (
        JvDataRecord,
        map_ra_record_to_race,
        map_se_record_to_entry,
        parse_jvdata_record,
    )

    assert JvDataRecord.__name__ == "JvDataRecord"
    assert callable(parse_jvdata_record)
    assert callable(map_ra_record_to_race)
    assert callable(map_se_record_to_entry)
```

- [ ] **Step 2: Run SE mapper tests and verify RED**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_jravan_mappers.py tests/test_package_imports.py -v
```

Expected: FAIL with `ImportError` or `AttributeError` for missing SE mapper functions.

- [ ] **Step 3: Implement SE mappers**

Replace `src/horse_lab/data/jravan/mappers.py` with:

```python
"""Canonical schema mappers for minimal JRA-VAN RA/SE records."""

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
    race_date = _require(fields, "race_date")
    venue_code = _require(fields, "venue_code")
    kaiji = _require(fields, "kaiji")
    nichiji = _require(fields, "nichiji")
    race_number = _require(fields, "race_number")
    return RaceId(f"{race_date}{venue_code}{kaiji}{nichiji}{race_number}")


def build_jravan_runner_id(fields: Mapping[str, str]) -> RunnerId:
    horse_number = _parse_int(_require(fields, "horse_number"), "horse_number")
    return RunnerId(f"{build_jravan_race_id(fields)}-{horse_number:02d}")


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


def _parse_deci_number(value: str | None, field_name: str) -> float | None:
    normalized = _optional_str(value)
    if normalized is None:
        return None
    return _parse_int(normalized, field_name) / 10.0


def _parse_body_weight_diff(sign: str | None, value: str | None) -> int | None:
    diff = _parse_optional_int(value, "body_weight_diff")
    if diff is None:
        return None
    return -diff if (sign or "").strip() == "-" else diff


def _parse_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "y", "yes", "true"}


def _unknown_or_none(value: str | None) -> str | None:
    return f"unknown:{value}" if value else None
```

Update `src/horse_lab/data/jravan/__init__.py`:

```python
"""JRA-VAN raw-data ingestion helpers.

The JV-Link extraction process itself is expected to run on Windows. This
package keeps the platform-independent parsing and normalization boundary in
Python so the rest of horse_lab can run on macOS/Linux.
"""

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
from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
    parse_jvdata_record,
    read_jvdata_records,
)

__all__ = [
    "FixedWidthField",
    "JRAVAN_MINIMAL_RA_FIELDS",
    "JRAVAN_MINIMAL_SE_FIELDS",
    "JvDataRecord",
    "build_jravan_race_id",
    "build_jravan_runner_id",
    "map_ra_record_to_race",
    "map_se_record_to_entry",
    "map_se_record_to_result",
    "parse_fixed_width_fields",
    "parse_jvdata_record",
    "parse_minimal_ra_fields",
    "parse_minimal_se_fields",
    "read_jvdata_records",
]
```

- [ ] **Step 4: Run SE mapper tests and verify GREEN**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_jravan_mappers.py tests/test_package_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full verification**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/horse-lab-pycache PYTHONPATH=src python3 -m pytest -v
PYTHONPYCACHEPREFIX=/private/tmp/horse-lab-pycache python3 -m compileall src tests
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit SE mappers**

Run:

```bash
git add src/horse_lab/data/jravan/__init__.py src/horse_lab/data/jravan/mappers.py tests/data/test_jravan_mappers.py tests/test_package_imports.py
git commit -m "feat: map jravan se records to entries and results"
```

---

## Self-Review Checklist

- Spec coverage:
  - Minimal RA layout is covered by Task 1.
  - Minimal SE layout is covered by Task 1.
  - RA to `Race` mapping is covered by Task 2.
  - SE to `Entry` and `Result` mapping is covered by Task 3.
  - Wrong record type validation is covered in Tasks 1, 2, and 3.
  - Synthetic CP932 fixture behavior is covered by the `_fixed_width_text` helper and layout tests.
- Scope guard:
  - No JV-Link calls.
  - No real JRA-VAN data.
  - No odds, payout, master, CSV export, or web scraping.
- Verification:
  - Each task has a focused pytest command.
  - Task 3 has full pytest and compileall checks before commit.
