"""Domain schemas for Japanese horse-racing modeling.

The schemas are deliberately dependency-light. They define stable contracts
between ingestion, feature generation, models, and simulation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, NewType, Sequence


RaceId = NewType("RaceId", str)
RunnerId = NewType("RunnerId", str)
HorseId = NewType("HorseId", str)
PersonId = NewType("PersonId", str)
ModelName = NewType("ModelName", str)
FeatureName = NewType("FeatureName", str)


class Surface(str, Enum):
    TURF = "turf"
    DIRT = "dirt"
    JUMP = "jump"
    UNKNOWN = "unknown"


class TrackCondition(str, Enum):
    FIRM = "firm"
    GOOD = "good"
    YIELDING = "yielding"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class CourseDirection(str, Enum):
    RIGHT = "right"
    LEFT = "left"
    STRAIGHT = "straight"
    UNKNOWN = "unknown"


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    GELDING = "gelding"
    UNKNOWN = "unknown"


class BetType(str, Enum):
    WIN = "win"
    PLACE = "place"
    EXACTA = "exacta"
    QUINELLA = "quinella"
    TRIFECTA = "trifecta"


class PredictionTarget(str, Enum):
    WIN_PROBABILITY = "win_probability"
    PLACE_PROBABILITY = "place_probability"
    SHOW_PROBABILITY = "show_probability"


@dataclass(frozen=True)
class Race:
    race_id: RaceId
    race_date: date
    venue: str
    race_number: int
    name: str | None
    surface: Surface
    distance_m: int
    direction: CourseDirection = CourseDirection.UNKNOWN
    track_condition: TrackCondition = TrackCondition.UNKNOWN
    weather: str | None = None
    grade: str | None = None
    start_time: datetime | None = None
    field_size: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.race_number <= 0:
            raise ValueError("race_number must be positive")
        if self.distance_m <= 0:
            raise ValueError("distance_m must be positive")
        if self.field_size is not None and self.field_size <= 0:
            raise ValueError("field_size must be positive when present")


@dataclass(frozen=True)
class Horse:
    horse_id: HorseId
    name: str
    sex: Sex = Sex.UNKNOWN
    birth_date: date | None = None
    sire_id: HorseId | None = None
    dam_id: HorseId | None = None
    damsire_id: HorseId | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entry:
    runner_id: RunnerId
    race_id: RaceId
    horse_id: HorseId
    horse_number: int
    gate_number: int | None
    jockey_id: PersonId | None = None
    trainer_id: PersonId | None = None
    carried_weight_kg: float | None = None
    body_weight_kg: int | None = None
    body_weight_diff_kg: int | None = None
    age: int | None = None
    is_scratched: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horse_number <= 0:
            raise ValueError("horse_number must be positive")
        if self.gate_number is not None and self.gate_number <= 0:
            raise ValueError("gate_number must be positive when present")


@dataclass(frozen=True)
class Result:
    race_id: RaceId
    runner_id: RunnerId
    finish_position: int | None
    is_disqualified: bool = False
    is_dead_heat: bool = False
    margin_lengths: float | None = None
    final_time_seconds: float | None = None
    corner_positions: Sequence[int] = field(default_factory=tuple)
    prize_jpy: int | None = None

    @property
    def did_win(self) -> bool:
        return self.finish_position == 1 and not self.is_disqualified


@dataclass(frozen=True)
class OddsQuote:
    race_id: RaceId
    runner_id: RunnerId
    bet_type: BetType
    captured_at: datetime
    odds: float
    min_odds: float | None = None
    max_odds: float | None = None
    popularity_rank: int | None = None
    pool_size_jpy: int | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.odds <= 1.0:
            raise ValueError("odds must be greater than 1.0")


@dataclass(frozen=True)
class FeatureRow:
    race_id: RaceId
    runner_id: RunnerId
    as_of: datetime
    feature_version: str
    values: Mapping[FeatureName, float | int | bool | str | None]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingLabel:
    race_id: RaceId
    runner_id: RunnerId
    target: PredictionTarget
    value: float


@dataclass(frozen=True)
class ModelPrediction:
    race_id: RaceId
    runner_id: RunnerId
    model_name: ModelName
    model_version: str
    target: PredictionTarget
    probability: float
    as_of: datetime
    lower_probability: float | None = None
    upper_probability: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")


@dataclass(frozen=True)
class BetRecommendation:
    race_id: RaceId
    runner_id: RunnerId
    bet_type: BetType
    probability: float
    market_odds: float
    fair_odds: float
    expected_value: float
    kelly_fraction: float
    stake_fraction: float
    stake_jpy: int
    model_version: str
    as_of: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.market_odds <= 1.0:
            raise ValueError("market_odds must be greater than 1.0")
        if self.fair_odds <= 1.0:
            raise ValueError("fair_odds must be greater than 1.0")
        if self.stake_fraction < 0.0:
            raise ValueError("stake_fraction must be non-negative")
        if self.stake_jpy < 0:
            raise ValueError("stake_jpy must be non-negative")
