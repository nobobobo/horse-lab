"""Repository protocols for historical racing data."""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, Sequence

from horse_lab.schemas import FeatureRow, OddsQuote, Race, RaceId, Result


class RaceRepository(Protocol):
    def list_races(self, *, start_date: date, end_date: date) -> Sequence[Race]:
        """Return races whose race dates are within the inclusive date range."""


class OddsRepository(Protocol):
    def list_odds(
        self,
        *,
        race_ids: Sequence[RaceId],
        captured_at_or_before: datetime,
    ) -> Sequence[OddsQuote]:
        """Return odds snapshots available at or before the requested time."""


class ResultRepository(Protocol):
    def list_results(self, *, race_ids: Sequence[RaceId]) -> Sequence[Result]:
        """Return official race results for the requested races."""


class FeatureRepository(Protocol):
    def list_feature_rows(
        self,
        *,
        race_ids: Sequence[RaceId],
        feature_version: str,
        as_of: datetime,
    ) -> Sequence[FeatureRow]:
        """Return point-in-time runner-level feature rows."""
