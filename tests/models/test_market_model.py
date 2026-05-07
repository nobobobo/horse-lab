import datetime as dt

import pytest

from horse_lab.models import InferenceContext, MarketImpliedProbabilityModel
from horse_lab.schemas import (
    BetType,
    FeatureName,
    FeatureRow,
    OddsQuote,
    PredictionTarget,
    RaceId,
    RunnerId,
)


def _feature_row(race_id: str, runner_id: str, as_of: dt.datetime) -> FeatureRow:
    return FeatureRow(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        as_of=as_of,
        feature_version="test-v1",
        values={FeatureName("bias"): 0.0},
    )


def _quote(
    race_id: str,
    runner_id: str,
    captured_at: dt.datetime,
    odds: float,
) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId(race_id),
        runner_id=RunnerId(runner_id),
        bet_type=BetType.WIN,
        captured_at=captured_at,
        odds=odds,
    )


def test_market_model_normalizes_implied_probabilities_within_race():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    model = MarketImpliedProbabilityModel(model_version="test")

    predictions = model.predict(
        [
            _feature_row("race-1", "runner-1", as_of),
            _feature_row("race-1", "runner-2", as_of),
        ],
        context=InferenceContext(
            as_of=as_of,
            feature_version="test-v1",
            odds=[
                _quote("race-1", "runner-1", as_of, 2.0),
                _quote("race-1", "runner-2", as_of, 4.0),
            ],
        ),
    )

    by_runner = {prediction.runner_id: prediction for prediction in predictions}

    assert sum(prediction.probability for prediction in predictions) == pytest.approx(1.0)
    assert by_runner[RunnerId("runner-1")].probability == pytest.approx(2 / 3)
    assert by_runner[RunnerId("runner-2")].probability == pytest.approx(1 / 3)
    assert all(
        prediction.target == PredictionTarget.WIN_PROBABILITY
        for prediction in predictions
    )


def test_market_model_uses_latest_quote_at_or_before_as_of():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    model = MarketImpliedProbabilityModel(model_version="test")

    predictions = model.predict(
        [
            _feature_row("race-1", "runner-1", as_of),
            _feature_row("race-1", "runner-2", as_of),
        ],
        context=InferenceContext(
            as_of=as_of,
            feature_version="test-v1",
            odds=[
                _quote("race-1", "runner-1", dt.datetime(2026, 5, 7, 14, 40), 10.0),
                _quote("race-1", "runner-1", dt.datetime(2026, 5, 7, 14, 50), 2.0),
                _quote("race-1", "runner-1", dt.datetime(2026, 5, 7, 15, 0), 100.0),
                _quote("race-1", "runner-2", dt.datetime(2026, 5, 7, 14, 50), 4.0),
            ],
        ),
    )

    by_runner = {prediction.runner_id: prediction for prediction in predictions}

    assert by_runner[RunnerId("runner-1")].metadata["market_odds"] == 2.0
    assert by_runner[RunnerId("runner-1")].probability == pytest.approx(2 / 3)


def test_market_model_requires_matching_win_odds():
    as_of = dt.datetime(2026, 5, 7, 14, 55)
    model = MarketImpliedProbabilityModel(model_version="test")

    with pytest.raises(ValueError, match="Missing win odds"):
        model.predict(
            [_feature_row("race-1", "runner-1", as_of)],
            context=InferenceContext(as_of=as_of, feature_version="test-v1", odds=[]),
        )
