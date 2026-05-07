# Historical Dataset Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replayable local CSV historical-data loop that loads fixture races, feature rows, odds, and results, runs the market-implied baseline, and produces deterministic backtest metrics.

**Architecture:** Keep JRA-VAN integration behind the repository boundary. Implement CSV parsers and read-only CSV repositories now, then add a replay pipeline that coordinates repositories, `MarketImpliedProbabilityModel`, and `BacktestSimulator`.

**Tech Stack:** Python 3.9+, standard library `csv`/`dataclasses`/`pathlib`, existing Horse Lab schemas and models, pytest.

---

## File Structure

- Create: `src/horse_lab/data/csv_parsing.py`
  - Parses CSV row dictionaries into domain schema objects.
- Create: `src/horse_lab/data/csv_repositories.py`
  - Implements read-only CSV repositories for races, odds, results, and feature rows.
- Modify: `src/horse_lab/data/__init__.py`
  - Exports CSV repository classes.
- Create: `src/horse_lab/pipelines/__init__.py`
  - Exports replay pipeline API.
- Create: `src/horse_lab/pipelines/replay.py`
  - Coordinates repository loading, market prediction, and backtesting.
- Create: `sample_data/README.md`
  - Documents fixture purpose and CSV conventions.
- Create: `sample_data/races.csv`
  - Two deterministic fixture races.
- Create: `sample_data/entries.csv`
  - Fixture race entries for schema coverage.
- Create: `sample_data/results.csv`
  - Fixture official results.
- Create: `sample_data/odds.csv`
  - Fixture odds snapshots including one future quote.
- Create: `sample_data/features.csv`
  - Fixture runner-level feature rows including an older duplicate row.
- Create: `tests/data/test_csv_parsing.py`
  - Unit tests for parser conversions.
- Create: `tests/data/test_csv_repositories.py`
  - Unit tests for repository filtering and ordering.
- Create: `tests/data/test_sample_data.py`
  - Ensures committed sample data is loadable and shaped as expected.
- Create: `tests/pipelines/test_replay.py`
  - End-to-end replay tests over `sample_data/`.
- Modify: `docs/architecture.md`
  - Adds a short Phase 1A section for CSV replay and JRA-VAN adapter boundary.

---

### Task 1: CSV Parsing Primitives

**Files:**
- Create: `src/horse_lab/data/csv_parsing.py`
- Create: `tests/data/test_csv_parsing.py`

- [ ] **Step 1: Write failing parser tests**

Create `tests/data/test_csv_parsing.py`:

```python
import datetime as dt

import pytest

from horse_lab.data.csv_parsing import (
    parse_bool,
    parse_entry_row,
    parse_feature_row,
    parse_float_or_none,
    parse_int_or_none,
    parse_odds_quote_row,
    parse_race_row,
    parse_result_row,
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
```

- [ ] **Step 2: Run parser tests to verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_csv_parsing.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'horse_lab.data.csv_parsing'`.

- [ ] **Step 3: Implement CSV parsing**

Create `src/horse_lab/data/csv_parsing.py`:

```python
"""CSV parsing helpers for local historical fixtures."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

from horse_lab.schemas import (
    BetType,
    CourseDirection,
    Entry,
    FeatureName,
    FeatureRow,
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"", "false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_int_or_none(value: str | None) -> int | None:
    normalized = (value or "").strip()
    return int(normalized) if normalized else None


def parse_float_or_none(value: str | None) -> float | None:
    normalized = (value or "").strip()
    return float(normalized) if normalized else None


def parse_str_or_none(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


def parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def parse_datetime_or_none(value: str | None) -> datetime | None:
    normalized = (value or "").strip()
    return datetime.fromisoformat(normalized) if normalized else None


def parse_race_row(row: Mapping[str, str]) -> Race:
    return Race(
        race_id=RaceId(row["race_id"]),
        race_date=parse_date(row["race_date"]),
        venue=row["venue"],
        race_number=int(row["race_number"]),
        name=parse_str_or_none(row.get("name")),
        surface=Surface(row.get("surface") or Surface.UNKNOWN.value),
        distance_m=int(row["distance_m"]),
        direction=CourseDirection(row.get("direction") or CourseDirection.UNKNOWN.value),
        track_condition=TrackCondition(
            row.get("track_condition") or TrackCondition.UNKNOWN.value
        ),
        weather=parse_str_or_none(row.get("weather")),
        grade=parse_str_or_none(row.get("grade")),
        start_time=parse_datetime_or_none(row.get("start_time")),
        field_size=parse_int_or_none(row.get("field_size")),
    )


def parse_entry_row(row: Mapping[str, str]) -> Entry:
    jockey_id = parse_str_or_none(row.get("jockey_id"))
    trainer_id = parse_str_or_none(row.get("trainer_id"))
    return Entry(
        runner_id=RunnerId(row["runner_id"]),
        race_id=RaceId(row["race_id"]),
        horse_id=HorseId(row["horse_id"]),
        horse_number=int(row["horse_number"]),
        gate_number=parse_int_or_none(row.get("gate_number")),
        jockey_id=PersonId(jockey_id) if jockey_id is not None else None,
        trainer_id=PersonId(trainer_id) if trainer_id is not None else None,
        carried_weight_kg=parse_float_or_none(row.get("carried_weight_kg")),
        body_weight_kg=parse_int_or_none(row.get("body_weight_kg")),
        body_weight_diff_kg=parse_int_or_none(row.get("body_weight_diff_kg")),
        age=parse_int_or_none(row.get("age")),
        is_scratched=parse_bool(row.get("is_scratched")),
    )


def parse_result_row(row: Mapping[str, str]) -> Result:
    return Result(
        race_id=RaceId(row["race_id"]),
        runner_id=RunnerId(row["runner_id"]),
        finish_position=parse_int_or_none(row.get("finish_position")),
        is_disqualified=parse_bool(row.get("is_disqualified")),
        is_dead_heat=parse_bool(row.get("is_dead_heat")),
        final_time_seconds=parse_float_or_none(row.get("final_time_seconds")),
        prize_jpy=parse_int_or_none(row.get("prize_jpy")),
    )


def parse_odds_quote_row(row: Mapping[str, str]) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId(row["race_id"]),
        runner_id=RunnerId(row["runner_id"]),
        bet_type=BetType(row["bet_type"]),
        captured_at=parse_datetime(row["captured_at"]),
        odds=float(row["odds"]),
        popularity_rank=parse_int_or_none(row.get("popularity_rank")),
        pool_size_jpy=parse_int_or_none(row.get("pool_size_jpy")),
        source=parse_str_or_none(row.get("source")),
    )


def parse_feature_row(row: Mapping[str, str]) -> FeatureRow:
    values = {
        FeatureName(key.removeprefix("feature__")): parse_feature_value(value)
        for key, value in row.items()
        if key.startswith("feature__")
    }
    return FeatureRow(
        race_id=RaceId(row["race_id"]),
        runner_id=RunnerId(row["runner_id"]),
        as_of=parse_datetime(row["as_of"]),
        feature_version=row["feature_version"],
        values=values,
    )


def parse_feature_value(value: str | None) -> float | int | bool | str | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    lower = normalized.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return int(normalized)
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError:
        return normalized
```

- [ ] **Step 4: Run parser tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_csv_parsing.py -v
```

Expected: PASS with `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/horse_lab/data/csv_parsing.py tests/data/test_csv_parsing.py
git commit -m "feat: add csv parsing helpers"
```

---

### Task 2: CSV Repository Implementations

**Files:**
- Create: `src/horse_lab/data/csv_repositories.py`
- Modify: `src/horse_lab/data/__init__.py`
- Create: `tests/data/test_csv_repositories.py`

- [ ] **Step 1: Write failing repository tests**

Create `tests/data/test_csv_repositories.py`:

```python
import datetime as dt

from horse_lab.data import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.schemas import FeatureName, RaceId


def _write(path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def test_csv_race_repository_filters_date_range_and_orders(tmp_path):
    path = tmp_path / "races.csv"
    _write(
        path,
        """
race_id,race_date,venue,race_number,name,surface,distance_m,direction,track_condition,weather,grade,start_time,field_size
race-2,2026-05-09,Tokyo,2,Race 2,dirt,1600,left,good,Cloudy,,2026-05-09T11:00:00,2
race-1,2026-05-08,Tokyo,1,Race 1,turf,1200,left,firm,Sunny,,2026-05-08T10:00:00,2
race-0,2026-05-07,Tokyo,1,Race 0,turf,1200,left,firm,Sunny,,2026-05-07T10:00:00,2
""",
    )

    races = CsvRaceRepository(path).list_races(
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 9),
    )

    assert [race.race_id for race in races] == [RaceId("race-1"), RaceId("race-2")]


def test_csv_odds_repository_filters_races_and_future_quotes(tmp_path):
    path = tmp_path / "odds.csv"
    _write(
        path,
        """
race_id,runner_id,bet_type,captured_at,odds,popularity_rank,pool_size_jpy,source
race-1,runner-1,win,2026-05-08T09:50:00,3.0,1,1000,fixture
race-1,runner-1,win,2026-05-08T09:56:00,10.0,1,1000,fixture
race-2,runner-9,win,2026-05-08T09:50:00,2.0,1,1000,fixture
""",
    )

    odds = CsvOddsRepository(path).list_odds(
        race_ids=[RaceId("race-1")],
        captured_at_or_before=dt.datetime(2026, 5, 8, 9, 55),
    )

    assert len(odds) == 1
    assert odds[0].odds == 3.0


def test_csv_result_repository_filters_by_race_id(tmp_path):
    path = tmp_path / "results.csv"
    _write(
        path,
        """
race_id,runner_id,finish_position,is_disqualified,is_dead_heat,final_time_seconds,prize_jpy
race-1,runner-1,1,false,false,70.5,100
race-2,runner-2,2,false,false,71.0,0
""",
    )

    results = CsvResultRepository(path).list_results(race_ids=[RaceId("race-1")])

    assert len(results) == 1
    assert results[0].runner_id == "runner-1"


def test_csv_feature_repository_returns_latest_feature_row_at_or_before_as_of(tmp_path):
    path = tmp_path / "features.csv"
    _write(
        path,
        """
race_id,runner_id,as_of,feature_version,feature__recent_speed
race-1,runner-1,2026-05-08T09:45:00,fixture-v1,70
race-1,runner-1,2026-05-08T09:55:00,fixture-v1,72
race-1,runner-1,2026-05-08T09:56:00,fixture-v1,99
race-1,runner-2,2026-05-08T09:55:00,fixture-v2,80
""",
    )

    rows = CsvFeatureRepository(path).list_feature_rows(
        race_ids=[RaceId("race-1")],
        feature_version="fixture-v1",
        as_of=dt.datetime(2026, 5, 8, 9, 55),
    )

    assert len(rows) == 1
    assert rows[0].values[FeatureName("recent_speed")] == 72
```

- [ ] **Step 2: Run repository tests to verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_csv_repositories.py -v
```

Expected: FAIL with `ImportError` for `CsvRaceRepository`.

- [ ] **Step 3: Implement CSV repositories**

Create `src/horse_lab/data/csv_repositories.py`:

```python
"""Read-only CSV repository implementations."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from horse_lab.data.csv_parsing import (
    parse_feature_row,
    parse_odds_quote_row,
    parse_race_row,
    parse_result_row,
    read_csv_rows,
)
from horse_lab.schemas import FeatureRow, OddsQuote, Race, RaceId, Result


class CsvRaceRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def list_races(self, *, start_date: date, end_date: date) -> Sequence[Race]:
        races = [
            parse_race_row(row)
            for row in read_csv_rows(self.path)
        ]
        selected = [
            race for race in races
            if start_date <= race.race_date <= end_date
        ]
        return tuple(
            sorted(
                selected,
                key=lambda race: (race.race_date, race.venue, race.race_number),
            )
        )


class CsvOddsRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def list_odds(
        self,
        *,
        race_ids: Sequence[RaceId],
        captured_at_or_before: datetime,
    ) -> Sequence[OddsQuote]:
        race_id_set = set(race_ids)
        odds = [
            parse_odds_quote_row(row)
            for row in read_csv_rows(self.path)
        ]
        selected = [
            quote for quote in odds
            if quote.race_id in race_id_set
            and quote.captured_at <= captured_at_or_before
        ]
        return tuple(
            sorted(
                selected,
                key=lambda quote: (str(quote.race_id), str(quote.runner_id), quote.captured_at),
            )
        )


class CsvResultRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def list_results(self, *, race_ids: Sequence[RaceId]) -> Sequence[Result]:
        race_id_set = set(race_ids)
        results = [
            parse_result_row(row)
            for row in read_csv_rows(self.path)
        ]
        selected = [result for result in results if result.race_id in race_id_set]
        return tuple(
            sorted(
                selected,
                key=lambda result: (str(result.race_id), str(result.runner_id)),
            )
        )


class CsvFeatureRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def list_feature_rows(
        self,
        *,
        race_ids: Sequence[RaceId],
        feature_version: str,
        as_of: datetime,
    ) -> Sequence[FeatureRow]:
        race_id_set = set(race_ids)
        latest_by_runner: dict[tuple[RaceId, str], FeatureRow] = {}
        for row in read_csv_rows(self.path):
            feature_row = parse_feature_row(row)
            if feature_row.race_id not in race_id_set:
                continue
            if feature_row.feature_version != feature_version:
                continue
            if feature_row.as_of > as_of:
                continue
            key = (feature_row.race_id, str(feature_row.runner_id))
            previous = latest_by_runner.get(key)
            if previous is None or feature_row.as_of > previous.as_of:
                latest_by_runner[key] = feature_row

        return tuple(
            sorted(
                latest_by_runner.values(),
                key=lambda row: (str(row.race_id), str(row.runner_id)),
            )
        )
```

Replace `src/horse_lab/data/__init__.py` with:

```python
"""Data access protocols and repository implementations."""

from horse_lab.data.csv_repositories import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.data.repositories import (
    FeatureRepository,
    OddsRepository,
    RaceRepository,
    ResultRepository,
)

__all__ = [
    "CsvFeatureRepository",
    "CsvOddsRepository",
    "CsvRaceRepository",
    "CsvResultRepository",
    "FeatureRepository",
    "OddsRepository",
    "RaceRepository",
    "ResultRepository",
]
```

- [ ] **Step 4: Run repository and parser tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_csv_parsing.py tests/data/test_csv_repositories.py -v
```

Expected: PASS with `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/horse_lab/data/csv_repositories.py src/horse_lab/data/__init__.py tests/data/test_csv_repositories.py
git commit -m "feat: add csv repositories"
```

---

### Task 3: Sample Historical Fixture Data

**Files:**
- Create: `sample_data/README.md`
- Create: `sample_data/races.csv`
- Create: `sample_data/entries.csv`
- Create: `sample_data/results.csv`
- Create: `sample_data/odds.csv`
- Create: `sample_data/features.csv`
- Create: `tests/data/test_sample_data.py`

- [ ] **Step 1: Write failing sample-data test**

Create `tests/data/test_sample_data.py`:

```python
import datetime as dt
from pathlib import Path

from horse_lab.data import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.data.csv_parsing import parse_entry_row, read_csv_rows
from horse_lab.schemas import FeatureName, RaceId


SAMPLE_DATA = Path("sample_data")
AS_OF = dt.datetime(2026, 5, 8, 9, 55)


def test_sample_data_files_are_loadable():
    races = CsvRaceRepository(SAMPLE_DATA / "races.csv").list_races(
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
    )
    race_ids = [race.race_id for race in races]
    odds = CsvOddsRepository(SAMPLE_DATA / "odds.csv").list_odds(
        race_ids=race_ids,
        captured_at_or_before=AS_OF,
    )
    results = CsvResultRepository(SAMPLE_DATA / "results.csv").list_results(
        race_ids=race_ids,
    )
    features = CsvFeatureRepository(SAMPLE_DATA / "features.csv").list_feature_rows(
        race_ids=race_ids,
        feature_version="fixture-v1",
        as_of=AS_OF,
    )
    entries = [parse_entry_row(row) for row in read_csv_rows(SAMPLE_DATA / "entries.csv")]

    assert race_ids == [RaceId("202605080101"), RaceId("202605080102")]
    assert len(entries) == 4
    assert len(results) == 4
    assert len(features) == 4
    assert len(odds) == 5
    assert all(quote.odds != 10.0 for quote in odds)
    assert features[0].values[FeatureName("recent_speed")] == 72
```

- [ ] **Step 2: Run sample-data test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data/test_sample_data.py -v
```

Expected: FAIL because `sample_data/races.csv` does not exist.

- [ ] **Step 3: Add sample data README**

Create `sample_data/README.md`:

```markdown
# Sample Data

These CSV files are tiny deterministic fixtures for the historical replay tests.

They are not real JRA records. The field names mirror the domain schemas so a future JRA-VAN adapter can map official records into the same repository contracts.

The fixture intentionally includes:

- two races on May 8, 2026;
- four runners;
- one future odds quote after replay `as_of`;
- two feature rows for the same runner so latest-at-or-before logic is tested;
- same-race win/loss settlement behavior for the backtest simulator.
```

- [ ] **Step 4: Add sample CSV fixtures**

Create `sample_data/races.csv`:

```csv
race_id,race_date,venue,race_number,name,surface,distance_m,direction,track_condition,weather,grade,start_time,field_size
202605080101,2026-05-08,Tokyo,1,Fixture Sprint,turf,1200,left,firm,Sunny,,2026-05-08T10:00:00,2
202605080102,2026-05-08,Tokyo,2,Fixture Mile,dirt,1600,left,good,Cloudy,,2026-05-08T11:00:00,2
```

Create `sample_data/entries.csv`:

```csv
runner_id,race_id,horse_id,horse_number,gate_number,jockey_id,trainer_id,carried_weight_kg,body_weight_kg,body_weight_diff_kg,age,is_scratched
202605080101-01,202605080101,horse-001,1,1,jockey-001,trainer-001,56.0,480,2,4,false
202605080101-02,202605080101,horse-002,2,2,jockey-002,trainer-002,56.0,472,-4,5,false
202605080102-01,202605080102,horse-003,1,1,jockey-003,trainer-003,57.0,500,0,4,false
202605080102-02,202605080102,horse-004,2,2,jockey-004,trainer-004,55.0,458,6,3,false
```

Create `sample_data/results.csv`:

```csv
race_id,runner_id,finish_position,is_disqualified,is_dead_heat,final_time_seconds,prize_jpy
202605080101,202605080101-01,1,false,false,70.5,10000000
202605080101,202605080101-02,2,false,false,71.0,4000000
202605080102,202605080102-01,2,false,false,98.2,4000000
202605080102,202605080102-02,1,false,false,97.8,10000000
```

Create `sample_data/odds.csv`:

```csv
race_id,runner_id,bet_type,captured_at,odds,popularity_rank,pool_size_jpy,source
202605080101,202605080101-01,win,2026-05-08T09:40:00,4.0,2,900000,fixture
202605080101,202605080101-01,win,2026-05-08T09:50:00,3.0,1,1000000,fixture
202605080101,202605080101-01,win,2026-05-08T09:56:00,10.0,2,1000000,fixture
202605080101,202605080101-02,win,2026-05-08T09:50:00,3.0,1,1000000,fixture
202605080102,202605080102-01,win,2026-05-08T09:50:00,2.0,1,1100000,fixture
202605080102,202605080102-02,win,2026-05-08T09:50:00,5.0,2,1100000,fixture
```

Create `sample_data/features.csv`:

```csv
race_id,runner_id,as_of,feature_version,feature__recent_speed,feature__gate_bias
202605080101,202605080101-01,2026-05-08T09:45:00,fixture-v1,70,0.0
202605080101,202605080101-01,2026-05-08T09:55:00,fixture-v1,72,0.1
202605080101,202605080101-02,2026-05-08T09:55:00,fixture-v1,68,-0.1
202605080102,202605080102-01,2026-05-08T09:55:00,fixture-v1,80,0.2
202605080102,202605080102-02,2026-05-08T09:55:00,fixture-v1,74,-0.2
```

- [ ] **Step 5: Run sample-data and repository tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data -v
```

Expected: PASS with `12 passed`.

- [ ] **Step 6: Commit**

```bash
git add sample_data tests/data/test_sample_data.py
git commit -m "test: add historical replay sample data"
```

---

### Task 4: Replay Pipeline

**Files:**
- Create: `src/horse_lab/pipelines/__init__.py`
- Create: `src/horse_lab/pipelines/replay.py`
- Create: `tests/pipelines/test_replay.py`
- Modify: `tests/test_package_imports.py`

- [ ] **Step 1: Write failing replay tests**

Create `tests/pipelines/test_replay.py`:

```python
import datetime as dt
from pathlib import Path

import pytest

from horse_lab.betting import KellyConfig
from horse_lab.data import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.pipelines import run_market_replay


SAMPLE_DATA = Path("sample_data")


def test_run_market_replay_returns_deterministic_sample_metrics():
    result = run_market_replay(
        race_repository=CsvRaceRepository(SAMPLE_DATA / "races.csv"),
        odds_repository=CsvOddsRepository(SAMPLE_DATA / "odds.csv"),
        result_repository=CsvResultRepository(SAMPLE_DATA / "results.csv"),
        feature_repository=CsvFeatureRepository(SAMPLE_DATA / "features.csv"),
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 9, 55),
        feature_version="fixture-v1",
        initial_bankroll_jpy=100_000,
        kelly_config=KellyConfig(),
    )

    assert len(result.races) == 2
    assert len(result.feature_rows) == 4
    assert len(result.odds) == 5
    assert len(result.predictions) == 4
    assert result.summary.final_bankroll_jpy == 108_000
    assert result.summary.total_bets == 4
    assert result.summary.wins == 2
    assert result.summary.total_staked_jpy == 8_000
    assert result.summary.total_payout_jpy == 16_000
    assert result.summary.net_profit_jpy == 8_000
    assert result.summary.roi == pytest.approx(1.0)
    assert result.summary.hit_rate == pytest.approx(0.5)
    assert result.summary.turnover == pytest.approx(0.08)
    assert result.summary.max_drawdown == 0.0
    assert result.backtest_result.bankroll_curve_jpy == (100_000, 102_000, 108_000)


def test_run_market_replay_ignores_future_odds_from_repository():
    result = run_market_replay(
        race_repository=CsvRaceRepository(SAMPLE_DATA / "races.csv"),
        odds_repository=CsvOddsRepository(SAMPLE_DATA / "odds.csv"),
        result_repository=CsvResultRepository(SAMPLE_DATA / "results.csv"),
        feature_repository=CsvFeatureRepository(SAMPLE_DATA / "features.csv"),
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
        as_of=dt.datetime(2026, 5, 8, 9, 55),
        feature_version="fixture-v1",
        initial_bankroll_jpy=100_000,
        kelly_config=KellyConfig(),
    )

    race1_runner1 = [
        prediction for prediction in result.predictions
        if prediction.runner_id == "202605080101-01"
    ][0]

    assert race1_runner1.metadata["market_odds"] == 3.0
    assert race1_runner1.metadata["market_odds"] != 10.0
```

Replace `tests/test_package_imports.py` with the current tests plus a pipeline import assertion:

```python
from horse_lab.schemas import BetType, PredictionTarget


def test_core_package_imports():
    assert BetType.WIN.value == "win"
    assert PredictionTarget.WIN_PROBABILITY.value == "win_probability"


def test_baseline_modules_import():
    from horse_lab.betting import KellyConfig, calculate_kelly_stake
    from horse_lab.backtesting import BacktestConfig, BacktestSimulator
    from horse_lab.evaluation import PerformanceSummary
    from horse_lab.models import MarketImpliedProbabilityModel

    assert KellyConfig.__name__ == "KellyConfig"
    assert callable(calculate_kelly_stake)
    assert BacktestConfig.__name__ == "BacktestConfig"
    assert BacktestSimulator.__name__ == "BacktestSimulator"
    assert PerformanceSummary.__name__ == "PerformanceSummary"
    assert MarketImpliedProbabilityModel.__name__ == "MarketImpliedProbabilityModel"


def test_storage_and_feature_protocols_import():
    from horse_lab.data import RaceRepository, ResultRepository
    from horse_lab.features import FeatureBuilder

    assert RaceRepository.__name__ == "RaceRepository"
    assert ResultRepository.__name__ == "ResultRepository"
    assert FeatureBuilder.__name__ == "FeatureBuilder"


def test_replay_pipeline_import():
    from horse_lab.pipelines import ReplayResult, run_market_replay

    assert ReplayResult.__name__ == "ReplayResult"
    assert callable(run_market_replay)
```

- [ ] **Step 2: Run replay tests to verify they fail**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/pipelines/test_replay.py tests/test_package_imports.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'horse_lab.pipelines'`.

- [ ] **Step 3: Implement replay pipeline**

Create `src/horse_lab/pipelines/replay.py`:

```python
"""Replay pipelines for local historical data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from horse_lab.backtesting import BacktestResult, BacktestSimulator, BacktestConfig
from horse_lab.betting import KellyConfig
from horse_lab.data.repositories import (
    FeatureRepository,
    OddsRepository,
    RaceRepository,
    ResultRepository,
)
from horse_lab.evaluation import PerformanceSummary
from horse_lab.models import InferenceContext, MarketImpliedProbabilityModel
from horse_lab.schemas import FeatureRow, ModelPrediction, OddsQuote, Race, Result


@dataclass(frozen=True)
class ReplayResult:
    races: Sequence[Race]
    feature_rows: Sequence[FeatureRow]
    odds: Sequence[OddsQuote]
    results: Sequence[Result]
    predictions: Sequence[ModelPrediction]
    backtest_result: BacktestResult
    summary: PerformanceSummary


def run_market_replay(
    *,
    race_repository: RaceRepository,
    odds_repository: OddsRepository,
    result_repository: ResultRepository,
    feature_repository: FeatureRepository,
    start_date: date,
    end_date: date,
    as_of: datetime,
    feature_version: str,
    initial_bankroll_jpy: int = 100_000,
    kelly_config: KellyConfig = KellyConfig(),
    model_version: str = "market-implied-v1",
) -> ReplayResult:
    races = tuple(
        race_repository.list_races(start_date=start_date, end_date=end_date)
    )
    race_ids = tuple(race.race_id for race in races)
    feature_rows = tuple(
        feature_repository.list_feature_rows(
            race_ids=race_ids,
            feature_version=feature_version,
            as_of=as_of,
        )
    )
    odds = tuple(
        odds_repository.list_odds(
            race_ids=race_ids,
            captured_at_or_before=as_of,
        )
    )
    results = tuple(result_repository.list_results(race_ids=race_ids))

    model = MarketImpliedProbabilityModel(model_version=model_version)
    predictions = tuple(
        model.predict(
            feature_rows,
            context=InferenceContext(
                as_of=as_of,
                feature_version=feature_version,
                races=races,
                odds=odds,
            ),
        )
    )
    backtest_result = BacktestSimulator(
        config=BacktestConfig(
            initial_bankroll_jpy=initial_bankroll_jpy,
            kelly_config=kelly_config,
        )
    ).run(predictions=predictions, odds=odds, results=results)

    return ReplayResult(
        races=races,
        feature_rows=feature_rows,
        odds=odds,
        results=results,
        predictions=predictions,
        backtest_result=backtest_result,
        summary=backtest_result.summary,
    )
```

Create `src/horse_lab/pipelines/__init__.py`:

```python
"""Pipeline orchestration helpers."""

from horse_lab.pipelines.replay import ReplayResult, run_market_replay

__all__ = [
    "ReplayResult",
    "run_market_replay",
]
```

- [ ] **Step 4: Run replay and import tests to verify they pass**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/pipelines/test_replay.py tests/test_package_imports.py -v
```

Expected: PASS with `6 passed`.

- [ ] **Step 5: Run all focused tests**

Run:

```bash
PYTHONPATH=src python3 -m pytest tests/data tests/pipelines tests/backtesting tests/evaluation tests/models tests/betting tests/test_package_imports.py -v
```

Expected: PASS with all tests passing.

- [ ] **Step 6: Commit**

```bash
git add src/horse_lab/pipelines tests/pipelines/test_replay.py tests/test_package_imports.py
git commit -m "feat: add market replay pipeline"
```

---

### Task 5: Architecture Documentation And Verification

**Files:**
- Modify: `docs/architecture.md`

- [ ] **Step 1: Update architecture documentation**

Append this section to `docs/architecture.md`:

```markdown

## Implemented Historical Replay

The first historical replay layer uses local CSV fixtures and read-only CSV repositories.

- `horse_lab.data.csv_parsing` converts CSV rows into domain schemas.
- `horse_lab.data.csv_repositories` implements race, odds, result, and feature repositories over local files.
- `sample_data/` provides deterministic fixtures for two races, four runners, multiple odds timestamps, and point-in-time feature rows.
- `horse_lab.pipelines.replay.run_market_replay` coordinates repositories, the market-implied baseline, and the Kelly backtest simulator.

JRA-VAN Data Lab. remains the intended production source for JRA data. Future JRA-VAN adapters should implement the same repository protocols so modeling and backtesting logic do not change when the data source changes.
```

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/horse-lab-pycache PYTHONPATH=src python3 -m pytest -v
```

Expected: PASS with all tests passing.

- [ ] **Step 3: Run compile verification**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/horse-lab-pycache python3 -m compileall src tests
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: shows only the intentional `docs/architecture.md` modification before commit.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: document historical replay layer"
```

---

## Self-Review Checklist

- Spec coverage:
  - CSV fixture schemas: Tasks 1 and 3.
  - CSV-backed repositories: Task 2.
  - Replay orchestration: Task 4.
  - Deterministic sample data: Task 3.
  - Point-in-time odds and feature safety: Tasks 2, 3, and 4.
  - JRA-VAN adapter boundary documentation: Task 5.
- Type consistency:
  - CSV parsers return existing `horse_lab.schemas` dataclasses.
  - CSV repositories conform to existing keyword-only repository protocols.
  - Replay uses existing `MarketImpliedProbabilityModel`, `InferenceContext`, `KellyConfig`, and `BacktestSimulator`.
- Verification:
  - Parser and repository tests prove local data loading behavior.
  - Replay tests prove deterministic metrics from `sample_data/`.
  - Final verification runs pytest and compileall.
