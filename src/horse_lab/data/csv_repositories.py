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
        races = [parse_race_row(row) for row in read_csv_rows(self.path)]
        selected = [
            race for race in races if start_date <= race.race_date <= end_date
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
        odds = [parse_odds_quote_row(row) for row in read_csv_rows(self.path)]
        selected = [
            quote
            for quote in odds
            if quote.race_id in race_id_set
            and quote.captured_at <= captured_at_or_before
        ]
        return tuple(
            sorted(
                selected,
                key=lambda quote: (
                    str(quote.race_id),
                    str(quote.runner_id),
                    quote.captured_at,
                ),
            )
        )


class CsvResultRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def list_results(self, *, race_ids: Sequence[RaceId]) -> Sequence[Result]:
        race_id_set = set(race_ids)
        results = [parse_result_row(row) for row in read_csv_rows(self.path)]
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
