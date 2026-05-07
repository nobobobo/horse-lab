from horse_lab.schemas import BetType, PredictionTarget


def test_core_package_imports():
    assert BetType.WIN.value == "win"
    assert PredictionTarget.WIN_PROBABILITY.value == "win_probability"


def test_baseline_modules_import():
    from horse_lab.betting import KellyConfig, calculate_kelly_stake
    from horse_lab.backtesting import BacktestConfig, BacktestSimulator
    from horse_lab.evaluation import PerformanceSummary
    from horse_lab.models import MarketImpliedProbabilityModel

    assert KellyConfig.__name__ == "KellyConfig"
    assert callable(calculate_kelly_stake)
    assert BacktestConfig.__name__ == "BacktestConfig"
    assert BacktestSimulator.__name__ == "BacktestSimulator"
    assert PerformanceSummary.__name__ == "PerformanceSummary"
    assert MarketImpliedProbabilityModel.__name__ == "MarketImpliedProbabilityModel"


def test_storage_and_feature_protocols_import():
    from horse_lab.data import RaceRepository, ResultRepository
    from horse_lab.features import FeatureBuilder

    assert RaceRepository.__name__ == "RaceRepository"
    assert ResultRepository.__name__ == "ResultRepository"
    assert FeatureBuilder.__name__ == "FeatureBuilder"
