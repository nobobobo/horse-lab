"""Data access protocols and repository implementations."""

from horse_lab.data.csv_repositories import (
    CsvFeatureRepository,
    CsvOddsRepository,
    CsvRaceRepository,
    CsvResultRepository,
)
from horse_lab.data.master import (
    HorseMasterRecord,
    HorseRatingRecord,
    index_rating_history,
    latest_rating_as_of,
    read_horse_master_csv,
    read_horse_rating_history_csv,
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
    "HorseMasterRecord",
    "HorseRatingRecord",
    "OddsRepository",
    "RaceRepository",
    "ResultRepository",
    "index_rating_history",
    "latest_rating_as_of",
    "read_horse_master_csv",
    "read_horse_rating_history_csv",
]
