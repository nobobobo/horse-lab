"""Past-performance feature builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from statistics import mean
from typing import Sequence

from horse_lab.schemas import (
    BetType,
    Entry,
    FeatureName,
    FeatureRow,
    HorseId,
    OddsQuote,
    Race,
    RaceId,
    Result,
    RunnerId,
)


PAST_PERFORMANCE_FEATURE_VERSION = "past-performance-v1"

PAST_RUN_COUNT = FeatureName("past_run_count")
DAYS_SINCE_LAST_RUN = FeatureName("days_since_last_run")
AVG_FINISH_POSITION_LAST3 = FeatureName("avg_finish_position_last3")
BEST_FINISH_POSITION_LAST3 = FeatureName("best_finish_position_last3")
WIN_RATE_LAST5 = FeatureName("win_rate_last5")
AVG_DISTANCE_M_LAST3 = FeatureName("avg_distance_m_last3")
SAME_SURFACE_RUN_COUNT = FeatureName("same_surface_run_count")
SAME_SURFACE_WIN_RATE = FeatureName("same_surface_win_rate")
SAME_VENUE_RUN_COUNT = FeatureName("same_venue_run_count")
SAME_VENUE_WIN_RATE = FeatureName("same_venue_win_rate")
SAME_VENUE_TOP3_RATE = FeatureName("same_venue_top3_rate")
SAME_VENUE_SURFACE_RUN_COUNT = FeatureName("same_venue_surface_run_count")
SAME_VENUE_SURFACE_WIN_RATE = FeatureName("same_venue_surface_win_rate")
AVG_ODDS_LAST3 = FeatureName("avg_odds_last3")
LAST_FINISH_POSITION = FeatureName("last_finish_position")
LAST_ODDS = FeatureName("last_odds")
TOP3_RATE_LAST5 = FeatureName("top3_rate_last5")
AVG_PRIZE_JPY_LAST3 = FeatureName("avg_prize_jpy_last3")
AVG_FINAL_TIME_SECONDS_LAST3 = FeatureName("avg_final_time_seconds_last3")
SAME_DISTANCE_RUN_COUNT = FeatureName("same_distance_run_count")
SAME_DISTANCE_WIN_RATE = FeatureName("same_distance_win_rate")
SAME_DISTANCE_TOP3_RATE = FeatureName("same_distance_top3_rate")
SAME_GRADE_RUN_COUNT = FeatureName("same_grade_run_count")
SAME_GRADE_WIN_RATE = FeatureName("same_grade_win_rate")
SAME_GRADE_TOP3_RATE = FeatureName("same_grade_top3_rate")
DISTANCE_DELTA_FROM_LAST = FeatureName("distance_delta_from_last")
LAST_RACE_DISTANCE_M = FeatureName("last_race_distance_m")
LAST_RACE_SURFACE = FeatureName("last_race_surface")
LAST_RACE_GRADE = FeatureName("last_race_grade")
LAST_BODY_WEIGHT_KG = FeatureName("last_body_weight_kg")
LAST_BODY_WEIGHT_DIFF_KG = FeatureName("last_body_weight_diff_kg")
JOCKEY_PAST_RUN_COUNT = FeatureName("jockey_past_run_count")
JOCKEY_PAST_WIN_RATE = FeatureName("jockey_past_win_rate")
TRAINER_PAST_RUN_COUNT = FeatureName("trainer_past_run_count")
TRAINER_PAST_WIN_RATE = FeatureName("trainer_past_win_rate")

PAST_PERFORMANCE_FEATURE_NAMES: tuple[FeatureName, ...] = (
    PAST_RUN_COUNT,
    DAYS_SINCE_LAST_RUN,
    AVG_FINISH_POSITION_LAST3,
    BEST_FINISH_POSITION_LAST3,
    WIN_RATE_LAST5,
    AVG_DISTANCE_M_LAST3,
    SAME_SURFACE_RUN_COUNT,
    SAME_SURFACE_WIN_RATE,
    SAME_VENUE_RUN_COUNT,
    SAME_VENUE_WIN_RATE,
    SAME_VENUE_TOP3_RATE,
    SAME_VENUE_SURFACE_RUN_COUNT,
    SAME_VENUE_SURFACE_WIN_RATE,
    AVG_ODDS_LAST3,
    LAST_FINISH_POSITION,
    LAST_ODDS,
    TOP3_RATE_LAST5,
    AVG_PRIZE_JPY_LAST3,
    AVG_FINAL_TIME_SECONDS_LAST3,
    SAME_DISTANCE_RUN_COUNT,
    SAME_DISTANCE_WIN_RATE,
    SAME_DISTANCE_TOP3_RATE,
    SAME_GRADE_RUN_COUNT,
    SAME_GRADE_WIN_RATE,
    SAME_GRADE_TOP3_RATE,
    DISTANCE_DELTA_FROM_LAST,
    LAST_RACE_DISTANCE_M,
    LAST_RACE_SURFACE,
    LAST_RACE_GRADE,
    LAST_BODY_WEIGHT_KG,
    LAST_BODY_WEIGHT_DIFF_KG,
    JOCKEY_PAST_RUN_COUNT,
    JOCKEY_PAST_WIN_RATE,
    TRAINER_PAST_RUN_COUNT,
    TRAINER_PAST_WIN_RATE,
)


@dataclass(frozen=True)
class _PastRun:
    race: Race
    entry: Entry
    result: Result
    odds: float | None


@dataclass(frozen=True)
class PastPerformanceFeatureBuilder:
    """Build runner-level features from races before each target race."""

    feature_version: str = PAST_PERFORMANCE_FEATURE_VERSION

    def build(
        self,
        *,
        entries: Sequence[Entry],
        races: Sequence[Race],
        results: Sequence[Result],
        odds: Sequence[OddsQuote],
        target_entries: Sequence[Entry] | None = None,
        target_races: Sequence[Race] | None = None,
    ) -> Sequence[FeatureRow]:
        """Build point-in-time safe past-performance features."""

        race_by_id = {race.race_id: race for race in races}
        target_races_by_id = {
            race.race_id: race
            for race in (target_races if target_races is not None else races)
        }
        target_race_ids = set(target_races_by_id)
        selected_entries = (
            tuple(target_entries)
            if target_entries is not None
            else tuple(entry for entry in entries if entry.race_id in target_race_ids)
        )

        result_by_runner = {
            (result.race_id, result.runner_id): result for result in results
        }
        entries_by_horse = _entries_by_horse(entries)
        entries_by_jockey = _entries_by_person(entries, role="jockey")
        entries_by_trainer = _entries_by_person(entries, role="trainer")
        win_odds_by_runner = _win_odds_by_runner(odds)

        rows: list[FeatureRow] = []
        for target_entry in selected_entries:
            target_race = target_races_by_id.get(target_entry.race_id)
            if target_race is None:
                continue

            target_as_of = _race_as_of(target_race)
            past_runs = _past_runs_for_horse(
                target_entry=target_entry,
                target_race=target_race,
                target_as_of=target_as_of,
                candidate_entries=entries_by_horse.get(target_entry.horse_id, ()),
                race_by_id=race_by_id,
                result_by_runner=result_by_runner,
                win_odds_by_runner=win_odds_by_runner,
            )
            jockey_runs = _past_runs_for_person(
                person_id=target_entry.jockey_id,
                person_role="jockey",
                target_as_of=target_as_of,
                candidate_entries=entries_by_jockey.get(target_entry.jockey_id, ()),
                race_by_id=race_by_id,
                result_by_runner=result_by_runner,
            )
            trainer_runs = _past_runs_for_person(
                person_id=target_entry.trainer_id,
                person_role="trainer",
                target_as_of=target_as_of,
                candidate_entries=entries_by_trainer.get(target_entry.trainer_id, ()),
                race_by_id=race_by_id,
                result_by_runner=result_by_runner,
            )

            rows.append(
                FeatureRow(
                    race_id=target_entry.race_id,
                    runner_id=target_entry.runner_id,
                    as_of=target_as_of,
                    feature_version=self.feature_version,
                    values=_feature_values(
                        target_race,
                        target_as_of,
                        past_runs,
                        jockey_runs=jockey_runs,
                        trainer_runs=trainer_runs,
                    ),
                )
            )

        return tuple(rows)


def build_past_performance_features(
    *,
    entries: Sequence[Entry],
    races: Sequence[Race],
    results: Sequence[Result],
    odds: Sequence[OddsQuote],
    target_entries: Sequence[Entry] | None = None,
    target_races: Sequence[Race] | None = None,
    feature_version: str = PAST_PERFORMANCE_FEATURE_VERSION,
) -> Sequence[FeatureRow]:
    """Build past-performance feature rows with the default builder."""

    return PastPerformanceFeatureBuilder(feature_version=feature_version).build(
        entries=entries,
        races=races,
        results=results,
        odds=odds,
        target_entries=target_entries,
        target_races=target_races,
    )


def _race_as_of(race: Race) -> datetime:
    if race.start_time is not None:
        return race.start_time
    return datetime.combine(race.race_date, time.min)


def _entries_by_horse(entries: Sequence[Entry]) -> dict[HorseId, tuple[Entry, ...]]:
    grouped: dict[HorseId, list[Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.horse_id, []).append(entry)
    return {horse_id: tuple(values) for horse_id, values in grouped.items()}


def _entries_by_person(
    entries: Sequence[Entry],
    *,
    role: str,
) -> dict[object, tuple[Entry, ...]]:
    grouped: dict[object, list[Entry]] = {}
    for entry in entries:
        person_id = entry.jockey_id if role == "jockey" else entry.trainer_id
        if person_id is None:
            continue
        grouped.setdefault(person_id, []).append(entry)
    return {person_id: tuple(values) for person_id, values in grouped.items()}


def _win_odds_by_runner(
    odds: Sequence[OddsQuote],
) -> dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]]:
    grouped: dict[tuple[RaceId, RunnerId], list[OddsQuote]] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN:
            continue
        key = (quote.race_id, quote.runner_id)
        grouped.setdefault(key, []).append(quote)
    return {
        key: tuple(sorted(values, key=lambda quote: quote.captured_at))
        for key, values in grouped.items()
    }


def _past_runs_for_horse(
    *,
    target_entry: Entry,
    target_race: Race,
    target_as_of: datetime,
    candidate_entries: Sequence[Entry],
    race_by_id: dict[RaceId, Race],
    result_by_runner: dict[tuple[RaceId, RunnerId], Result],
    win_odds_by_runner: dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]],
) -> tuple[_PastRun, ...]:
    past_runs: list[_PastRun] = []
    for entry in candidate_entries:
        if entry.runner_id == target_entry.runner_id:
            continue

        race = race_by_id.get(entry.race_id)
        if race is None or _race_as_of(race) >= target_as_of:
            continue

        result = result_by_runner.get((entry.race_id, entry.runner_id))
        if result is None:
            continue

        quote = _latest_quote_before(
            win_odds_by_runner.get((entry.race_id, entry.runner_id), ()),
            _race_as_of(race),
        )
        past_runs.append(
            _PastRun(
                race=race,
                entry=entry,
                result=result,
                odds=quote.odds if quote is not None else None,
            )
        )

    return tuple(
        sorted(
            past_runs,
            key=lambda run: (
                _race_as_of(run.race),
                run.race.race_id,
                run.entry.runner_id,
            ),
            reverse=True,
        )
    )


def _past_runs_for_person(
    *,
    person_id: object | None,
    person_role: str,
    target_as_of: datetime,
    candidate_entries: Sequence[Entry],
    race_by_id: dict[RaceId, Race],
    result_by_runner: dict[tuple[RaceId, RunnerId], Result],
) -> tuple[Result, ...]:
    if person_id is None:
        return ()

    results: list[Result] = []
    for entry in candidate_entries:
        entry_person_id = (
            entry.jockey_id if person_role == "jockey" else entry.trainer_id
        )
        if entry_person_id != person_id:
            continue

        race = race_by_id.get(entry.race_id)
        if race is None or _race_as_of(race) >= target_as_of:
            continue

        result = result_by_runner.get((entry.race_id, entry.runner_id))
        if result is not None:
            results.append(result)

    return tuple(results)


def _latest_quote_before(
    quotes: Sequence[OddsQuote],
    as_of: datetime,
) -> OddsQuote | None:
    for quote in reversed(quotes):
        if quote.captured_at < as_of:
            return quote
    return None


def _feature_values(
    target_race: Race,
    target_as_of: datetime,
    past_runs: Sequence[_PastRun],
    *,
    jockey_runs: Sequence[Result],
    trainer_runs: Sequence[Result],
) -> dict[FeatureName, float | int | bool | str | None]:
    last3 = tuple(past_runs[:3])
    last5 = tuple(past_runs[:5])
    same_surface = tuple(
        run for run in past_runs if run.race.surface == target_race.surface
    )
    same_venue = tuple(run for run in past_runs if run.race.venue == target_race.venue)
    same_venue_surface = tuple(
        run for run in same_venue if run.race.surface == target_race.surface
    )
    same_distance = tuple(
        run for run in past_runs if run.race.distance_m == target_race.distance_m
    )
    same_grade = tuple(
        run
        for run in past_runs
        if target_race.grade is not None and run.race.grade == target_race.grade
    )
    finish_positions_last3 = tuple(
        run.result.finish_position
        for run in last3
        if run.result.finish_position is not None
    )
    odds_last3 = tuple(run.odds for run in last3 if run.odds is not None)
    prizes_last3 = tuple(
        run.result.prize_jpy for run in last3 if run.result.prize_jpy is not None
    )
    final_times_last3 = tuple(
        run.result.final_time_seconds
        for run in last3
        if run.result.final_time_seconds is not None
    )

    last_run = past_runs[0] if past_runs else None

    return {
        PAST_RUN_COUNT: len(past_runs),
        DAYS_SINCE_LAST_RUN: (
            (target_as_of.date() - _race_as_of(last_run.race).date()).days
            if last_run is not None
            else None
        ),
        AVG_FINISH_POSITION_LAST3: (
            mean(finish_positions_last3) if finish_positions_last3 else None
        ),
        BEST_FINISH_POSITION_LAST3: (
            min(finish_positions_last3) if finish_positions_last3 else None
        ),
        WIN_RATE_LAST5: (
            sum(run.result.did_win for run in last5) / len(last5) if last5 else None
        ),
        AVG_DISTANCE_M_LAST3: mean(run.race.distance_m for run in last3)
        if last3
        else None,
        SAME_SURFACE_RUN_COUNT: len(same_surface),
        SAME_SURFACE_WIN_RATE: (
            sum(run.result.did_win for run in same_surface) / len(same_surface)
            if same_surface
            else None
        ),
        SAME_VENUE_RUN_COUNT: len(same_venue),
        SAME_VENUE_WIN_RATE: _past_win_rate(same_venue),
        SAME_VENUE_TOP3_RATE: _past_top3_rate(same_venue),
        SAME_VENUE_SURFACE_RUN_COUNT: len(same_venue_surface),
        SAME_VENUE_SURFACE_WIN_RATE: _past_win_rate(same_venue_surface),
        AVG_ODDS_LAST3: mean(odds_last3) if odds_last3 else None,
        LAST_FINISH_POSITION: (
            last_run.result.finish_position if last_run is not None else None
        ),
        LAST_ODDS: last_run.odds if last_run is not None else None,
        TOP3_RATE_LAST5: (
            sum(
                1
                for run in last5
                if run.result.finish_position is not None
                and run.result.finish_position <= 3
            )
            / len(last5)
            if last5
            else None
        ),
        AVG_PRIZE_JPY_LAST3: mean(prizes_last3) if prizes_last3 else None,
        AVG_FINAL_TIME_SECONDS_LAST3: (
            mean(final_times_last3) if final_times_last3 else None
        ),
        SAME_DISTANCE_RUN_COUNT: len(same_distance),
        SAME_DISTANCE_WIN_RATE: (
            sum(run.result.did_win for run in same_distance) / len(same_distance)
            if same_distance
            else None
        ),
        SAME_DISTANCE_TOP3_RATE: _past_top3_rate(same_distance),
        SAME_GRADE_RUN_COUNT: len(same_grade),
        SAME_GRADE_WIN_RATE: _past_win_rate(same_grade),
        SAME_GRADE_TOP3_RATE: _past_top3_rate(same_grade),
        DISTANCE_DELTA_FROM_LAST: (
            target_race.distance_m - last_run.race.distance_m
            if last_run is not None
            else None
        ),
        LAST_RACE_DISTANCE_M: last_run.race.distance_m if last_run is not None else None,
        LAST_RACE_SURFACE: (
            last_run.race.surface.value if last_run is not None else None
        ),
        LAST_RACE_GRADE: last_run.race.grade if last_run is not None else None,
        LAST_BODY_WEIGHT_KG: (
            last_run.entry.body_weight_kg if last_run is not None else None
        ),
        LAST_BODY_WEIGHT_DIFF_KG: (
            last_run.entry.body_weight_diff_kg if last_run is not None else None
        ),
        JOCKEY_PAST_RUN_COUNT: len(jockey_runs),
        JOCKEY_PAST_WIN_RATE: _win_rate(jockey_runs),
        TRAINER_PAST_RUN_COUNT: len(trainer_runs),
        TRAINER_PAST_WIN_RATE: _win_rate(trainer_runs),
    }


def _win_rate(results: Sequence[Result]) -> float | None:
    if not results:
        return None
    return sum(result.did_win for result in results) / len(results)


def _past_win_rate(past_runs: Sequence[_PastRun]) -> float | None:
    if not past_runs:
        return None
    return sum(run.result.did_win for run in past_runs) / len(past_runs)


def _past_top3_rate(past_runs: Sequence[_PastRun]) -> float | None:
    if not past_runs:
        return None
    return (
        sum(
            1
            for run in past_runs
            if run.result.finish_position is not None
            and run.result.finish_position <= 3
        )
        / len(past_runs)
    )
