import datetime as dt

import pytest

from horse_lab.models import (
    InferenceContext,
    LightGBMWinProbabilityModel,
    TrainingContext,
    TrainingDataset,
)
from horse_lab.schemas import (
    FeatureName,
    FeatureRow,
    PredictionTarget,
    RaceId,
    Result,
    RunnerId,
    TrainingLabel,
)


class FakeEstimator:
    def __init__(self, probabilities):
        self.probabilities = tuple(probabilities)
        self.fit_x = None
        self.fit_y = None
        self.predict_x = None

    def fit(self, x_train, y_train):
        self.fit_x = x_train
        self.fit_y = y_train

    def predict_proba(self, x_values):
        self.predict_x = x_values
        return [[1.0 - probability, probability] for probability in self.probabilities]


def _feature_row(
    race_id: str,
    runner_id: str,
    *,
    speed: float,
    venue: str,
    is_favorite: bool = False,
) -> FeatureRow:
    return FeatureRow(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        as_of=dt.datetime(2026, 5, 7, 14, 55),
        feature_version="test-v1",
        values={
            FeatureName("speed"): speed,
            FeatureName("venue"): venue,
            FeatureName("missing_numeric"): None,
            FeatureName("is_favorite"): is_favorite,
        },
    )


def _result(race_id: str, runner_id: str, finish_position: int) -> Result:
    return Result(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        finish_position=finish_position,
    )


def _training_label(race_id: str, runner_id: str, value: float) -> TrainingLabel:
    return TrainingLabel(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        target=PredictionTarget.WIN_PROBABILITY,
        value=value,
    )


def _dataset(feature_rows, results) -> TrainingDataset:
    return TrainingDataset(
        feature_rows=feature_rows,
        labels=[
            _training_label(str(row.race_id), str(row.runner_id), 0.0)
            for row in feature_rows
        ],
        feature_names=[
            FeatureName("speed"),
            FeatureName("venue"),
            FeatureName("missing_numeric"),
            FeatureName("is_favorite"),
        ],
        target=PredictionTarget.WIN_PROBABILITY,
        metadata={"results": results},
    )


def test_lightgbm_fit_uses_result_did_win_labels_and_deterministic_features():
    fake = FakeEstimator(probabilities=[0.7, 0.3])
    feature_rows = [
        _feature_row("race-1", "runner-1", speed=80.0, venue="tokyo", is_favorite=True),
        _feature_row("race-1", "runner-2", speed=72.5, venue="kyoto"),
    ]
    model = LightGBMWinProbabilityModel(
        model_version="test",
        estimator_factory=lambda random_seed: fake,
    )

    artifact = model.fit(
        _dataset(
            feature_rows,
            [
                _result("race-1", "runner-1", 1),
                _result("race-1", "runner-2", 2),
            ],
        ),
        context=TrainingContext(
            train_start=dt.date(2026, 5, 1),
            train_end=dt.date(2026, 5, 7),
            feature_version="test-v1",
            random_seed=123,
        ),
    )

    assert artifact.metadata["rows"] == 2.0
    assert fake.fit_y == [1, 0]
    assert fake.fit_x == [
        [80.0, 0.0, 1.0, 2.0],
        [72.5, 0.0, 0.0, 1.0],
    ]


def test_lightgbm_predict_returns_race_normalized_probabilities():
    fake = FakeEstimator(probabilities=[0.8, 0.2, 0.5])
    feature_rows = [
        _feature_row("race-1", "runner-1", speed=80.0, venue="tokyo"),
        _feature_row("race-1", "runner-2", speed=72.5, venue="kyoto"),
        _feature_row("race-2", "runner-3", speed=77.0, venue="tokyo"),
    ]
    model = LightGBMWinProbabilityModel(
        model_version="test",
        estimator_factory=lambda random_seed: fake,
    )
    model.fit(
        _dataset(
            feature_rows,
            [
                _result("race-1", "runner-1", 1),
                _result("race-1", "runner-2", 2),
                _result("race-2", "runner-3", 1),
            ],
        ),
        context=TrainingContext(
            train_start=dt.date(2026, 5, 1),
            train_end=dt.date(2026, 5, 7),
            feature_version="test-v1",
        ),
    )

    predictions = model.predict(
        feature_rows,
        context=InferenceContext(
            as_of=dt.datetime(2026, 5, 7, 14, 55),
            feature_version="test-v1",
        ),
    )

    by_runner = {prediction.runner_id: prediction for prediction in predictions}
    assert by_runner[RunnerId("runner-1")].probability == pytest.approx(0.8)
    assert by_runner[RunnerId("runner-2")].probability == pytest.approx(0.2)
    assert by_runner[RunnerId("runner-3")].probability == pytest.approx(1.0)
    assert sum(
        prediction.probability
        for prediction in predictions
        if prediction.race_id == RaceId("race-1")
    ) == pytest.approx(1.0)
    assert all(
        prediction.target == PredictionTarget.WIN_PROBABILITY
        for prediction in predictions
    )


def test_lightgbm_fit_requires_results_metadata():
    fake = FakeEstimator(probabilities=[0.7])
    feature_rows = [_feature_row("race-1", "runner-1", speed=80.0, venue="tokyo")]
    model = LightGBMWinProbabilityModel(
        model_version="test",
        estimator_factory=lambda random_seed: fake,
    )

    with pytest.raises(ValueError, match='metadata\\["results"\\]'):
        model.fit(
            TrainingDataset(
                feature_rows=feature_rows,
                labels=[_training_label("race-1", "runner-1", 0.0)],
                feature_names=[FeatureName("speed")],
                target=PredictionTarget.WIN_PROBABILITY,
            ),
            context=TrainingContext(
                train_start=dt.date(2026, 5, 1),
                train_end=dt.date(2026, 5, 7),
                feature_version="test-v1",
            ),
        )
