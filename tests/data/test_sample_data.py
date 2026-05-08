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
