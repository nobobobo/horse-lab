"""Feature builder protocols."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from horse_lab.schemas import Entry, FeatureRow, OddsQuote, Race


class FeatureBuilder(Protocol):
    feature_version: str

    def build(
        self,
        *,
        races: Sequence[Race],
        entries: Sequence[Entry],
        odds: Sequence[OddsQuote],
        as_of: datetime,
    ) -> Sequence[FeatureRow]:
        """Build runner-level features using only information available at as_of."""
