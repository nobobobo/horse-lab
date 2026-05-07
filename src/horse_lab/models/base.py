"""Base interfaces for specialist models and stacking models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.schemas import (
    FeatureRow,
    ModelName,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    Race,
    TrainingLabel,
)


@dataclass(frozen=True)
class TrainingContext:
    train_start: date
    train_end: date
    feature_version: str
    random_seed: int = 42
    validation_start: date | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceContext:
    as_of: datetime
    feature_version: str
    races: Sequence[Race] = field(default_factory=tuple)
    odds: Sequence[OddsQuote] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingDataset:
    feature_rows: Sequence[FeatureRow]
    labels: Sequence[TrainingLabel]
    feature_names: Sequence[str]
    target: PredictionTarget
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_rows:
            raise ValueError("feature_rows must not be empty")
        if not self.labels:
            raise ValueError("labels must not be empty")
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")


@dataclass(frozen=True)
class ModelArtifact:
    model_name: ModelName
    model_version: str
    trained_at: datetime
    feature_version: str
    target: PredictionTarget
    metrics: Mapping[str, float] = field(default_factory=dict)
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseRaceModel(ABC):
    """Common interface for every model used by the platform."""

    model_name: ModelName
    model_version: str
    target: PredictionTarget

    def __init__(
        self,
        *,
        model_name: ModelName,
        model_version: str,
        target: PredictionTarget,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.target = target
        self._artifact: ModelArtifact | None = None

    @property
    def artifact(self) -> ModelArtifact | None:
        return self._artifact

    @property
    def is_fitted(self) -> bool:
        return self._artifact is not None

    @abstractmethod
    def fit(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
    ) -> ModelArtifact:
        """Train the model and return its artifact metadata."""

    @abstractmethod
    def predict(
        self,
        feature_rows: Sequence[FeatureRow],
        *,
        context: InferenceContext,
    ) -> Sequence[ModelPrediction]:
        """Return runner-level probabilities for the configured target."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the model implementation and metadata."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseRaceModel":
        """Load a persisted model implementation."""


class BaseLevel0Model(BaseRaceModel):
    """Specialist model that produces out-of-fold inputs for the meta learner."""

    @abstractmethod
    def fit_predict_oof(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
        n_splits: int,
    ) -> Sequence[ModelPrediction]:
        """Train by time-aware folds and return out-of-fold predictions."""


class BaseMetaModel(BaseRaceModel):
    """Stacking model that combines Level 0 specialist predictions."""

    @abstractmethod
    def fit_from_level0(
        self,
        level0_predictions: Sequence[ModelPrediction],
        labels: Sequence[TrainingLabel],
        *,
        context: TrainingContext,
    ) -> ModelArtifact:
        """Train only on historical out-of-fold specialist predictions."""

    @abstractmethod
    def predict_from_level0(
        self,
        level0_predictions: Sequence[ModelPrediction],
        *,
        context: InferenceContext,
    ) -> Sequence[ModelPrediction]:
        """Combine live specialist predictions into final probabilities."""
