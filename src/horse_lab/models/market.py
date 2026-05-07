"""Market-implied probability baseline model."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from horse_lab.models.base import (
    BaseLevel0Model,
    InferenceContext,
    ModelArtifact,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.schemas import (
    BetType,
    FeatureRow,
    ModelName,
    ModelPrediction,
    OddsQuote,
    PredictionTarget,
    RaceId,
    RunnerId,
)


class MarketImpliedProbabilityModel(BaseLevel0Model):
    """Baseline model that converts win odds into normalized probabilities."""

    def __init__(self, *, model_version: str = "market-implied-v1") -> None:
        super().__init__(
            model_name=ModelName("market_implied_probability"),
            model_version=model_version,
            target=PredictionTarget.WIN_PROBABILITY,
        )

    def fit(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
    ) -> ModelArtifact:
        artifact = ModelArtifact(
            model_name=self.model_name,
            model_version=self.model_version,
            trained_at=datetime.utcnow(),
            feature_version=context.feature_version,
            target=self.target,
            metrics={},
            metadata={"rows": float(len(dataset.feature_rows))},
        )
        self._artifact = artifact
        return artifact

    def predict(
        self,
        feature_rows: Sequence[FeatureRow],
        *,
        context: InferenceContext,
    ) -> Sequence[ModelPrediction]:
        latest_quotes = _latest_win_quotes_by_runner(context.odds, context.as_of)
        grouped_raw_probabilities: dict[RaceId, list[tuple[FeatureRow, OddsQuote, float]]] = (
            defaultdict(list)
        )

        for row in feature_rows:
            key = (row.race_id, row.runner_id)
            quote = latest_quotes.get(key)
            if quote is None:
                raise ValueError(
                    "Missing win odds for "
                    f"race_id={row.race_id!r}, runner_id={row.runner_id!r}"
                )
            grouped_raw_probabilities[row.race_id].append((row, quote, 1.0 / quote.odds))

        predictions: list[ModelPrediction] = []
        for race_id, rows in grouped_raw_probabilities.items():
            raw_sum = sum(raw_probability for _, _, raw_probability in rows)
            if raw_sum <= 0.0:
                raise ValueError(f"Raw implied probability sum is zero for race_id={race_id!r}")

            for row, quote, raw_probability in rows:
                predictions.append(
                    ModelPrediction(
                        race_id=row.race_id,
                        runner_id=row.runner_id,
                        model_name=self.model_name,
                        model_version=self.model_version,
                        target=self.target,
                        probability=raw_probability / raw_sum,
                        as_of=context.as_of,
                        metadata={
                            "market_odds": quote.odds,
                            "odds_captured_at": quote.captured_at.isoformat(),
                            "raw_implied_probability": raw_probability,
                        },
                    )
                )

        return predictions

    def fit_predict_oof(
        self,
        dataset: TrainingDataset,
        *,
        context: TrainingContext,
        n_splits: int,
    ) -> Sequence[ModelPrediction]:
        raise NotImplementedError(
            "MarketImpliedProbabilityModel requires odds snapshots in "
            "InferenceContext for prediction; OOF support will be added with "
            "the prediction store."
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": str(self.model_name),
            "model_version": self.model_version,
            "target": self.target.value,
        }
        (path / "market_model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MarketImpliedProbabilityModel":
        payload = json.loads((path / "market_model.json").read_text(encoding="utf-8"))
        return cls(model_version=payload["model_version"])


def _latest_win_quotes_by_runner(
    odds: Sequence[OddsQuote],
    as_of: datetime,
) -> Mapping[tuple[RaceId, RunnerId], OddsQuote]:
    latest: dict[tuple[RaceId, RunnerId], OddsQuote] = {}
    for quote in odds:
        if quote.bet_type != BetType.WIN or quote.captured_at > as_of:
            continue
        key = (quote.race_id, quote.runner_id)
        previous = latest.get(key)
        if previous is None or quote.captured_at > previous.captured_at:
            latest[key] = quote
    return latest
