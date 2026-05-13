from __future__ import annotations

import datetime as dt

import pytest

from horse_lab.features import PastPerformanceFeatureBuilder
from horse_lab.schemas import (
    BetType,
    Entry,
    FeatureName,
    HorseId,
    OddsQuote,
    PersonId,
    Race,
    RaceId,
    Result,
    RunnerId,
    Surface,
)


def _race(
    race_id: str,
    race_date: dt.date,
    *,
    surface: Surface = Surface.TURF,
    distance_m: int = 1600,
    start_time: dt.datetime | None = None,
    grade: str | None = None,
) -> Race:
    return Race(
        race_id=RaceId(race_id),
        race_date=race_date,
        venue="Tokyo",
        race_number=1,
        name="Fixture",
        surface=surface,
        distance_m=distance_m,
        start_time=start_time,
        grade=grade,
    )


def _entry(
    race_id: str,
    horse_id: str,
    runner_suffix: str = "01",
    *,
    jockey_id: str | None = None,
    trainer_id: str | None = None,
    body_weight_kg: int | None = None,
    body_weight_diff_kg: int | None = None,
) -> Entry:
    return Entry(
        race_id=RaceId(race_id),
        runner_id=RunnerId(f"{race_id}-{runner_suffix}"),
        horse_id=HorseId(horse_id),
        horse_number=int(runner_suffix),
        gate_number=int(runner_suffix),
        jockey_id=PersonId(jockey_id) if jockey_id else None,
        trainer_id=PersonId(trainer_id) if trainer_id else None,
        body_weight_kg=body_weight_kg,
        body_weight_diff_kg=body_weight_diff_kg,
    )


def _result(
    entry: Entry,
    finish_position: int | None,
    *,
    final_time_seconds: float | None = None,
    prize_jpy: int | None = None,
) -> Result:
    return Result(
        race_id=entry.race_id,
        runner_id=entry.runner_id,
        finish_position=finish_position,
        final_time_seconds=final_time_seconds,
        prize_jpy=prize_jpy,
    )


def _quote(
    entry: Entry,
    odds: float,
    captured_at: dt.datetime,
    *,
    bet_type: BetType = BetType.WIN,
) -> OddsQuote:
    return OddsQuote(
        race_id=entry.race_id,
        runner_id=entry.runner_id,
        bet_type=bet_type,
        captured_at=captured_at,
        odds=odds,
    )


def _values(row):
    return row.values


def test_past_performance_builder_prevents_future_race_leakage():
    past_race = _race(
        "race-1",
        dt.date(2026, 5, 1),
        start_time=dt.datetime(2026, 5, 1, 10, 0),
    )
    target_race = _race(
        "race-2",
        dt.date(2026, 5, 8),
        start_time=dt.datetime(2026, 5, 8, 10, 0),
    )
    future_race = _race(
        "race-3",
        dt.date(2026, 5, 9),
        start_time=dt.datetime(2026, 5, 9, 10, 0),
    )
    past_entry = _entry("race-1", "horse-1")
    target_entry = _entry("race-2", "horse-1")
    future_entry = _entry("race-3", "horse-1")

    rows = PastPerformanceFeatureBuilder().build(
        entries=[past_entry, target_entry, future_entry],
        races=[past_race, target_race, future_race],
        results=[_result(past_entry, 2), _result(future_entry, 1)],
        odds=[
            _quote(past_entry, 4.0, dt.datetime(2026, 5, 1, 9, 40)),
            _quote(future_entry, 2.0, dt.datetime(2026, 5, 9, 9, 40)),
        ],
        target_entries=[target_entry],
        target_races=[target_race],
    )

    assert len(rows) == 1
    values = _values(rows[0])
    assert rows[0].feature_version == "past-performance-v1"
    assert values[FeatureName("past_run_count")] == 1
    assert values[FeatureName("last_finish_position")] == 2
    assert values[FeatureName("win_rate_last5")] == pytest.approx(0.0)


def test_past_performance_builder_aggregates_recent_runs():
    target_race = _race(
        "race-target",
        dt.date(2026, 5, 8),
        surface=Surface.TURF,
        distance_m=1800,
        start_time=dt.datetime(2026, 5, 8, 10, 0),
    )
    history = [
        (
            _race(
                "race-1",
                dt.date(2026, 5, 1),
                surface=Surface.TURF,
                distance_m=1200,
                start_time=dt.datetime(2026, 5, 1, 10, 0),
            ),
            3,
            7.0,
        ),
        (
            _race(
                "race-2",
                dt.date(2026, 5, 2),
                surface=Surface.DIRT,
                distance_m=1400,
                start_time=dt.datetime(2026, 5, 2, 10, 0),
            ),
            1,
            5.0,
        ),
        (
            _race(
                "race-3",
                dt.date(2026, 5, 3),
                surface=Surface.TURF,
                distance_m=1600,
                start_time=dt.datetime(2026, 5, 3, 10, 0),
            ),
            5,
            9.0,
        ),
        (
            _race(
                "race-4",
                dt.date(2026, 5, 4),
                surface=Surface.TURF,
                distance_m=2000,
                start_time=dt.datetime(2026, 5, 4, 10, 0),
            ),
            2,
            3.0,
        ),
    ]
    target_entry = _entry("race-target", "horse-1")
    past_entries = [_entry(race.race_id, "horse-1") for race, _, _ in history]

    rows = PastPerformanceFeatureBuilder().build(
        entries=[*past_entries, target_entry],
        races=[*(race for race, _, _ in history), target_race],
        results=[
            _result(entry, finish_position)
            for entry, (_, finish_position, _) in zip(past_entries, history)
        ],
        odds=[
            _quote(entry, 99.0, dt.datetime(2026, 5, index, 9, 30))
            for index, entry in enumerate(past_entries, start=1)
        ]
        + [
            _quote(entry, odds, dt.datetime(2026, 5, index, 9, 50))
            for index, (entry, (_, _, odds)) in enumerate(
                zip(past_entries, history), start=1
            )
        ]
        + [
            _quote(
                past_entries[-1],
                2.0,
                dt.datetime(2026, 5, 4, 9, 55),
                bet_type=BetType.PLACE,
            )
        ],
        target_entries=[target_entry],
        target_races=[target_race],
    )

    values = _values(rows[0])
    assert values[FeatureName("past_run_count")] == 4
    assert values[FeatureName("days_since_last_run")] == 4
    assert values[FeatureName("avg_finish_position_last3")] == pytest.approx(
        (2 + 5 + 1) / 3
    )
    assert values[FeatureName("best_finish_position_last3")] == 1
    assert values[FeatureName("win_rate_last5")] == pytest.approx(1 / 4)
    assert values[FeatureName("avg_distance_m_last3")] == pytest.approx(
        (2000 + 1600 + 1400) / 3
    )
    assert values[FeatureName("same_surface_run_count")] == 3
    assert values[FeatureName("same_surface_win_rate")] == pytest.approx(0.0)
    assert values[FeatureName("same_venue_run_count")] == 4
    assert values[FeatureName("same_venue_win_rate")] == pytest.approx(1 / 4)
    assert values[FeatureName("same_venue_top3_rate")] == pytest.approx(3 / 4)
    assert values[FeatureName("same_venue_surface_run_count")] == 3
    assert values[FeatureName("same_venue_surface_win_rate")] == pytest.approx(0.0)
    assert values[FeatureName("avg_odds_last3")] == pytest.approx(
        (3.0 + 9.0 + 5.0) / 3
    )
    assert values[FeatureName("last_finish_position")] == 2
    assert values[FeatureName("last_odds")] == 3.0


def test_past_performance_builder_adds_expanded_horse_and_person_features():
    target_race = _race(
        "race-target",
        dt.date(2026, 5, 8),
        distance_m=1800,
        grade="B",
        start_time=dt.datetime(2026, 5, 8, 10, 0),
    )
    past_race_1 = _race(
        "race-1",
        dt.date(2026, 5, 1),
        distance_m=1600,
        grade="C",
        start_time=dt.datetime(2026, 5, 1, 10, 0),
    )
    past_race_2 = _race(
        "race-2",
        dt.date(2026, 5, 2),
        distance_m=1800,
        grade="C",
        start_time=dt.datetime(2026, 5, 2, 10, 0),
    )
    target_entry = _entry(
        "race-target",
        "horse-1",
        jockey_id="01020",
        trainer_id="04050",
    )
    past_entry_1 = _entry(
        "race-1",
        "horse-1",
        jockey_id="01020",
        trainer_id="04050",
        body_weight_kg=480,
        body_weight_diff_kg=2,
    )
    past_entry_2 = _entry(
        "race-2",
        "horse-1",
        jockey_id="99999",
        trainer_id="04050",
        body_weight_kg=484,
        body_weight_diff_kg=4,
    )
    other_jockey_entry = _entry(
        "race-2",
        "horse-2",
        runner_suffix="02",
        jockey_id="01020",
        trainer_id="77777",
    )

    rows = PastPerformanceFeatureBuilder().build(
        entries=[past_entry_1, past_entry_2, other_jockey_entry, target_entry],
        races=[past_race_1, past_race_2, target_race],
        results=[
            _result(past_entry_1, 1, final_time_seconds=96.5, prize_jpy=1000),
            _result(past_entry_2, 3, final_time_seconds=108.0, prize_jpy=200),
            _result(other_jockey_entry, 2, final_time_seconds=109.0, prize_jpy=500),
        ],
        odds=[
            _quote(past_entry_1, 4.0, dt.datetime(2026, 5, 1, 9, 50)),
            _quote(past_entry_2, 6.0, dt.datetime(2026, 5, 2, 9, 50)),
            _quote(other_jockey_entry, 8.0, dt.datetime(2026, 5, 2, 9, 50)),
        ],
        target_entries=[target_entry],
        target_races=[target_race],
    )

    values = _values(rows[0])
    assert values[FeatureName("top3_rate_last5")] == pytest.approx(1.0)
    assert values[FeatureName("avg_prize_jpy_last3")] == pytest.approx(600.0)
    assert values[FeatureName("avg_final_time_seconds_last3")] == pytest.approx(
        (108.0 + 96.5) / 2
    )
    assert values[FeatureName("same_distance_run_count")] == 1
    assert values[FeatureName("same_distance_win_rate")] == pytest.approx(0.0)
    assert values[FeatureName("same_distance_top3_rate")] == pytest.approx(1.0)
    assert values[FeatureName("same_venue_run_count")] == 2
    assert values[FeatureName("same_venue_win_rate")] == pytest.approx(0.5)
    assert values[FeatureName("same_venue_top3_rate")] == pytest.approx(1.0)
    assert values[FeatureName("same_venue_surface_run_count")] == 2
    assert values[FeatureName("same_venue_surface_win_rate")] == pytest.approx(0.5)
    assert values[FeatureName("same_grade_run_count")] == 0
    assert values[FeatureName("same_grade_win_rate")] is None
    assert values[FeatureName("same_grade_top3_rate")] is None
    assert values[FeatureName("distance_delta_from_last")] == 0
    assert values[FeatureName("last_race_distance_m")] == 1800
    assert values[FeatureName("last_race_surface")] == "turf"
    assert values[FeatureName("last_race_grade")] == "C"
    assert values[FeatureName("last_body_weight_kg")] == 484
    assert values[FeatureName("last_body_weight_diff_kg")] == 4
    assert values[FeatureName("jockey_past_run_count")] == 2
    assert values[FeatureName("jockey_past_win_rate")] == pytest.approx(0.5)
    assert values[FeatureName("trainer_past_run_count")] == 2
    assert values[FeatureName("trainer_past_win_rate")] == pytest.approx(0.5)


def test_past_performance_builder_allows_missing_odds():
    target_race = _race(
        "race-target",
        dt.date(2026, 5, 8),
        start_time=dt.datetime(2026, 5, 8, 10, 0),
    )
    past_race = _race(
        "race-1",
        dt.date(2026, 5, 1),
        start_time=dt.datetime(2026, 5, 1, 10, 0),
    )
    past_entry = _entry("race-1", "horse-1")
    target_entry = _entry("race-target", "horse-1")

    rows = PastPerformanceFeatureBuilder().build(
        entries=[past_entry, target_entry],
        races=[past_race, target_race],
        results=[_result(past_entry, 1)],
        odds=[],
        target_entries=[target_entry],
        target_races=[target_race],
    )

    values = _values(rows[0])
    assert values[FeatureName("past_run_count")] == 1
    assert values[FeatureName("avg_odds_last3")] is None
    assert values[FeatureName("last_odds")] is None


def test_past_performance_builder_ignores_odds_after_past_race_start():
    target_race = _race(
        "race-target",
        dt.date(2026, 5, 8),
        start_time=dt.datetime(2026, 5, 8, 10, 0),
    )
    past_race = _race(
        "race-1",
        dt.date(2026, 5, 1),
        start_time=dt.datetime(2026, 5, 1, 10, 0),
    )
    past_entry = _entry("race-1", "horse-1")
    target_entry = _entry("race-target", "horse-1")

    rows = PastPerformanceFeatureBuilder().build(
        entries=[past_entry, target_entry],
        races=[past_race, target_race],
        results=[_result(past_entry, 1)],
        odds=[
            _quote(past_entry, 4.0, dt.datetime(2026, 5, 1, 9, 55)),
            _quote(past_entry, 99.0, dt.datetime(2026, 5, 1, 10, 5)),
        ],
        target_entries=[target_entry],
        target_races=[target_race],
    )

    assert rows[0].values[FeatureName("last_odds")] == 4.0
