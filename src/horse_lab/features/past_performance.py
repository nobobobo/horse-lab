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
AVG_ODDS_LAST3 = FeatureName("avg_odds_last3")
LAST_FINISH_POSITION = FeatureName("last_finish_position")
LAST_ODDS = FeatureName("last_odds")

PAST_PERFORMANCE_FEATURE_NAMES: tuple[FeatureName, ...] = (
    PAST_RUN_COUNT,
    DAYS_SINCE_LAST_RUN,
    AVG_FINISH_POSITION_LAST3,
    BEST_FINISH_POSITION_LAST3,
    WIN_RATE_LAST5,
    AVG_DISTANCE_M_LAST3,
    SAME_SURFACE_RUN_COUNT,
    SAME_SURFACE_WIN_RATE,
    AVG_ODDS_LAST3,
    LAST_FINISH_POSITION,
    LAST_ODDS,
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

        rows: list[FeatureRow] = []
        for target_entry in selected_entries:
            target_race = target_races_by_id.get(target_entry.race_id)
            if target_race is None:
                continue

            target_as_of = _race_as_of(target_race)
            win_odds_by_runner = _win_odds_by_runner(odds, target_as_of)
            past_runs = _past_runs_for_horse(
                target_entry=target_entry,
                target_race=target_race,
                target_as_of=target_as_of,
                entries=entries,
                race_by_id=race_by_id,
                result_by_runner=result_by_runner,
                win_odds_by_runner=win_odds_by_runner,
            )

            rows.append(
                FeatureRow(
                    race_id=target_entry.race_id,
                    runner_id=target_entry.runner_id,
                    as_of=target_as_of,
                    feature_version=self.feature_version,
                    values=_feature_values(target_race, target_as_of, past_runs),
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


def _win_odds_by_runner(
    odds: Sequence[OddsQuote], as_of: datetime
) -> dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]]:
    grouped: dict[tuple[RaceId, RunnerId], list[OddsQuote]] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN or quote.captured_at >= as_of:
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
    entries: Sequence[Entry],
    race_by_id: dict[RaceId, Race],
    result_by_runner: dict[tuple[RaceId, RunnerId], Result],
    win_odds_by_runner: dict[tuple[RaceId, RunnerId], tuple[OddsQuote, ...]],
) -> tuple[_PastRun, ...]:
    past_runs: list[_PastRun] = []
    for entry in entries:
        if entry.horse_id != target_entry.horse_id:
            continue
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


def _latest_quote_before(
    quotes: Sequence[OddsQuote],
    as_of: datetime,
) -> OddsQuote | None:
    candidates = [quote for quote in quotes if quote.captured_at < as_of]
    return max(candidates, key=lambda quote: quote.captured_at) if candidates else None


def _feature_values(
    target_race: Race, target_as_of: datetime, past_runs: Sequence[_PastRun]
) -> dict[FeatureName, float | int | bool | str | None]:
    last3 = tuple(past_runs[:3])
    last5 = tuple(past_runs[:5])
    same_surface = tuple(
        run for run in past_runs if run.race.surface == target_race.surface
    )
    finish_positions_last3 = tuple(
        run.result.finish_position
        for run in last3
        if run.result.finish_position is not None
    )
    odds_last3 = tuple(run.odds for run in last3 if run.odds is not None)

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
        AVG_ODDS_LAST3: mean(odds_last3) if odds_last3 else None,
        LAST_FINISH_POSITION: (
            last_run.result.finish_position if last_run is not None else None
        ),
        LAST_ODDS: last_run.odds if last_run is not None else None,
    }
