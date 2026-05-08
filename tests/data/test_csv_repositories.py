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
