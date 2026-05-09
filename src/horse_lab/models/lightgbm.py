"""LightGBM tabular baseline model."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from horse_lab.models.base import (
    BaseLevel0Model,
    InferenceContext,
    ModelArtifact,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.schemas import (
    FeatureName,
    FeatureRow,
    ModelName,
    ModelPrediction,
    PredictionTarget,
    RaceId,
    Result,
    RunnerId,
)


EstimatorFactory = Callable[[int], Any]


class LightGBMWinProbabilityModel(BaseLevel0Model):
    """Minimal runner-level LightGBM baseline for win probability."""

    def __init__(
        self,
        *,
        model_version: str = "lightgbm-win-v1",
        estimator_factory: EstimatorFactory | None = None,
    ) -> None:
        super().__init__(
            model_name=ModelName("lightgbm_win_probability"),
            model_version=model_version,
            target=PredictionTarget.WIN_PROBABILITY,
        )
        self._estimator_factory = estimator_factory or _default_estimator_factory
        self._estimator: Any | None = None
        self._feature_names: tuple[FeatureName, ...] = ()
        self._numeric_features: tuple[FeatureName, ...] = ()
        self._categorical_features: tuple[FeatureName, ...] = ()
        self._category_maps: dict[FeatureName, dict[str, int]] = {}

    def fit(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
    ) -> ModelArtifact:
        self._feature_names = tuple(dataset.feature_names)
        self._numeric_features, self._categorical_features = _split_features(
            dataset.feature_rows,
            self._feature_names,
        )
        self._category_maps = _fit_category_maps(
            dataset.feature_rows,
            self._categorical_features,
        )

        x_train = self._transform_rows(dataset.feature_rows)
        y_train = _win_labels_from_results(
            dataset.feature_rows,
            dataset.metadata.get("results"),
        )

        estimator = self._estimator_factory(context.random_seed)
        estimator.fit(x_train, y_train)
        self._estimator = estimator

        artifact = ModelArtifact(
            model_name=self.model_name,
            model_version=self.model_version,
            trained_at=datetime.utcnow(),
            feature_version=context.feature_version,
            target=self.target,
            metrics={},
            metadata={
                "rows": float(len(dataset.feature_rows)),
                "numeric_features": tuple(str(name) for name in self._numeric_features),
                "categorical_features": tuple(
                    str(name) for name in self._categorical_features
                ),
            },
        )
        self._artifact = artifact
        return artifact

    def predict(
        self,
        feature_rows: Sequence[FeatureRow],
        *,
        context: InferenceContext,
    ) -> Sequence[ModelPrediction]:
        if self._estimator is None:
            raise ValueError("LightGBMWinProbabilityModel must be fitted before predict")

        raw_probabilities = _positive_class_probabilities(
            self._estimator.predict_proba(self._transform_rows(feature_rows))
        )
        normalized = _normalize_by_race(feature_rows, raw_probabilities)

        return [
            ModelPrediction(
                race_id=row.race_id,
                runner_id=row.runner_id,
                model_name=self.model_name,
                model_version=self.model_version,
                target=self.target,
                probability=probability,
                as_of=context.as_of,
                metadata={"raw_probability": raw_probability},
            )
            for row, probability, raw_probability in zip(
                feature_rows,
                normalized,
                raw_probabilities,
            )
        ]

    def fit_predict_oof(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
        n_splits: int,
    ) -> Sequence[ModelPrediction]:
        raise NotImplementedError("LightGBM OOF training is not implemented yet")

    def save(self, path: Path) -> None:
        if self._estimator is None:
            raise ValueError("LightGBMWinProbabilityModel must be fitted before save")

        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": str(self.model_name),
            "model_version": self.model_version,
            "target": self.target.value,
            "feature_names": tuple(str(name) for name in self._feature_names),
            "numeric_features": tuple(str(name) for name in self._numeric_features),
            "categorical_features": tuple(str(name) for name in self._categorical_features),
            "category_maps": {
                str(name): mapping for name, mapping in self._category_maps.items()
            },
        }
        (path / "lightgbm_model.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        with (path / "lightgbm_estimator.pkl").open("wb") as handle:
            pickle.dump(self._estimator, handle)

    def feature_importances(self) -> tuple[dict[str, Any], ...]:
        """Return trained estimator importances aligned with transformed features."""

        if self._estimator is None:
            raise ValueError(
                "LightGBMWinProbabilityModel must be fitted before feature_importances"
            )

        feature_order = tuple(self._numeric_features) + tuple(self._categorical_features)
        split_values = _importance_values(
            self._estimator,
            importance_type="split",
            expected_length=len(feature_order),
        )
        gain_values = _importance_values(
            self._estimator,
            importance_type="gain",
            expected_length=len(feature_order),
        )
        total_split = sum(split_values)
        total_gain = sum(gain_values)
        numeric_features = set(self._numeric_features)

        rows = [
            {
                "feature_name": str(feature_name),
                "feature_type": (
                    "numeric" if feature_name in numeric_features else "categorical"
                ),
                "split_importance": split_importance,
                "gain_importance": gain_importance,
                "split_fraction": (
                    split_importance / total_split if total_split > 0.0 else 0.0
                ),
                "gain_fraction": gain_importance / total_gain if total_gain > 0.0 else 0.0,
            }
            for feature_name, split_importance, gain_importance in zip(
                feature_order,
                split_values,
                gain_values,
            )
        ]
        ranked = sorted(
            rows,
            key=lambda row: (
                row["gain_importance"],
                row["split_importance"],
                row["feature_name"],
            ),
            reverse=True,
        )
        return tuple(
            {"rank": rank, **row}
            for rank, row in enumerate(ranked, start=1)
        )

    @classmethod
    def load(cls, path: Path) -> "LightGBMWinProbabilityModel":
        payload = json.loads((path / "lightgbm_model.json").read_text(encoding="utf-8"))
        model = cls(model_version=payload["model_version"])
        model._feature_names = tuple(FeatureName(name) for name in payload["feature_names"])
        model._numeric_features = tuple(
            FeatureName(name) for name in payload["numeric_features"]
        )
        model._categorical_features = tuple(
            FeatureName(name) for name in payload["categorical_features"]
        )
        model._category_maps = {
            FeatureName(name): {str(key): int(value) for key, value in mapping.items()}
            for name, mapping in payload["category_maps"].items()
        }
        with (path / "lightgbm_estimator.pkl").open("rb") as handle:
            model._estimator = pickle.load(handle)
        return model

    def _transform_rows(self, feature_rows: Sequence[FeatureRow]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for row in feature_rows:
            values: list[float] = []
            for name in self._numeric_features:
                values.append(_numeric_value(row.values.get(name)))
            for name in self._categorical_features:
                values.append(
                    float(
                        self._category_maps.get(name, {}).get(
                            _category_key(row.values.get(name)),
                            0,
                        )
                    )
                )
            matrix.append(values)
        return matrix


def _default_estimator_factory(random_seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm is an optional dependency. Install horse-lab[ml] to use "
            "LightGBMWinProbabilityModel."
        ) from exc

    return LGBMClassifier(objective="binary", random_state=random_seed)


def _split_features(
    feature_rows: Sequence[FeatureRow],
    feature_names: Sequence[FeatureName],
) -> tuple[tuple[FeatureName, ...], tuple[FeatureName, ...]]:
    numeric: list[FeatureName] = []
    categorical: list[FeatureName] = []
    for name in feature_names:
        non_null_values = [
            row.values.get(name) for row in feature_rows if row.values.get(name) is not None
        ]
        if all(_is_numeric(value) for value in non_null_values):
            numeric.append(name)
        else:
            categorical.append(name)
    return tuple(numeric), tuple(categorical)


def _fit_category_maps(
    feature_rows: Sequence[FeatureRow],
    categorical_features: Sequence[FeatureName],
) -> dict[FeatureName, dict[str, int]]:
    maps: dict[FeatureName, dict[str, int]] = {}
    for name in categorical_features:
        categories = sorted(
            {
                _category_key(row.values.get(name))
                for row in feature_rows
                if row.values.get(name) is not None
            }
        )
        maps[name] = {category: index + 1 for index, category in enumerate(categories)}
    return maps


def _win_labels_from_results(
    feature_rows: Sequence[FeatureRow],
    raw_results: Any,
) -> list[int]:
    if raw_results is None:
        raise ValueError('TrainingDataset.metadata["results"] is required')

    results = tuple(raw_results)
    result_by_runner: dict[tuple[RaceId, RunnerId], Result] = {}
    for result in results:
        if not isinstance(result, Result):
            raise TypeError('TrainingDataset.metadata["results"] must contain Result objects')
        result_by_runner[(result.race_id, result.runner_id)] = result

    labels: list[int] = []
    for row in feature_rows:
        result = result_by_runner.get((row.race_id, row.runner_id))
        if result is None:
            raise ValueError(
                "Missing result for "
                f"race_id={row.race_id!r}, runner_id={row.runner_id!r}"
            )
        labels.append(1 if result.did_win else 0)
    return labels


def _positive_class_probabilities(raw_probabilities: Any) -> list[float]:
    probabilities: list[float] = []
    for row in raw_probabilities:
        if isinstance(row, (float, int)):
            probabilities.append(float(row))
            continue
        probabilities.append(float(row[1]))
    return probabilities


def _importance_values(
    estimator: Any,
    *,
    importance_type: str,
    expected_length: int,
) -> list[float]:
    values = None
    booster = getattr(estimator, "booster_", None)
    if booster is not None and hasattr(booster, "feature_importance"):
        try:
            values = booster.feature_importance(importance_type=importance_type)
        except TypeError:
            values = None
    if values is None and importance_type == "split":
        values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return [0.0] * expected_length

    normalized = [float(value) for value in values]
    if len(normalized) < expected_length:
        normalized.extend([0.0] * (expected_length - len(normalized)))
    return normalized[:expected_length]


def _normalize_by_race(
    feature_rows: Sequence[FeatureRow],
    raw_probabilities: Sequence[float],
) -> list[float]:
    totals: dict[RaceId, float] = defaultdict(float)
    counts: dict[RaceId, int] = defaultdict(int)
    for row, probability in zip(feature_rows, raw_probabilities):
        totals[row.race_id] += max(float(probability), 0.0)
        counts[row.race_id] += 1

    normalized: list[float] = []
    for row, probability in zip(feature_rows, raw_probabilities):
        total = totals[row.race_id]
        if total <= 0.0:
            normalized.append(1.0 / counts[row.race_id])
        else:
            normalized.append(max(float(probability), 0.0) / total)
    return normalized


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, bool)) and not isinstance(value, str)


def _numeric_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric feature value, got {value!r}")


def _category_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
