"""Paper-trading replay pipeline for Phase 5 monitoring."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from horse_lab.backtesting import (
    BacktestConfig,
    BacktestResult,
    BacktestSimulator,
    backtest_config_to_dict,
    bet_decision_to_dict,
    write_backtest_artifacts,
)
from horse_lab.data import CsvOddsRepository, CsvRaceRepository, CsvResultRepository
from horse_lab.evaluation import ProbabilitySummary, summarize_win_probability_predictions
from horse_lab.schemas import (
    ModelName,
    ModelPrediction,
    PredictionTarget,
    Race,
    RaceId,
    Result,
    RunnerId,
)


PAPER_PREDICTION_FIELDS: tuple[str, ...] = (
    "race_id",
    "runner_id",
    "method",
    "model_name",
    "model_version",
    "target",
    "probability",
    "as_of",
    "metadata_json",
)


@dataclass(frozen=True)
class PaperTradingResult:
    predictions_path: Path
    report_path: Path
    clv_report_path: Path
    backtest_config_path: Path
    bet_decisions_path: Path
    report: dict[str, Any]
    predictions: tuple[ModelPrediction, ...]
    backtest_result: BacktestResult
    probability_summary: ProbabilitySummary


def run_paper_trading_from_csv(
    predictions_csv: Path | str,
    dataset_dir: Path | str,
    artifact_dir: Path | str,
    *,
    method: str,
    as_of: datetime,
    model_version: str = "paper-trading-v1",
    backtest_config: BacktestConfig = BacktestConfig(),
    use_odds_timeseries: bool = True,
) -> PaperTradingResult:
    dataset_path = Path(dataset_dir)
    predictions = _read_method_predictions(
        predictions_csv,
        method=method,
        as_of=as_of,
        model_version=model_version,
    )
    _require_non_empty(predictions, f"predictions for method={method!r}")
    race_ids = tuple(sorted({prediction.race_id for prediction in predictions}))
    races = _races_for_ids(dataset_path, race_ids)
    results = tuple(
        CsvResultRepository(dataset_path / "results.csv").list_results(
            race_ids=race_ids,
        )
    )
    odds = tuple(
        CsvOddsRepository(
            _replay_odds_csv_path(
                dataset_path,
                use_odds_timeseries=use_odds_timeseries,
            )
        ).list_odds(
            race_ids=race_ids,
            captured_at_or_before=as_of,
        )
    )
    _validate_inputs(
        predictions=predictions,
        races=races,
        results=results,
    )

    backtest_result = BacktestSimulator(config=backtest_config).run(
        predictions=predictions,
        odds=odds,
        results=results,
        races=races,
    )
    probability_summary = summarize_win_probability_predictions(
        predictions=predictions,
        results=results,
    )

    output_path = Path(artifact_dir)
    predictions_path = output_path / "paper_predictions.csv"
    report_path = output_path / "paper_trading_report.json"
    clv_report_path = output_path / "clv_report.csv"
    _write_predictions(predictions_path, method=method, predictions=predictions)
    backtest_paths = write_backtest_artifacts(
        result=backtest_result,
        config=backtest_config,
        output_dir=output_path / "backtest",
    )
    _write_clv_report(clv_report_path, backtest_result)
    report = {
        "method": method,
        "model_version": model_version,
        "as_of": as_of.isoformat(),
        "dataset_dir": str(dataset_path),
        "source_predictions_csv": str(predictions_csv),
        "counts": {
            "races": len(races),
            "predictions": len(predictions),
            "odds": len(odds),
            "results": len(results),
            "bet_decisions": len(backtest_result.decisions),
            "bet_records": len(backtest_result.records),
        },
        "probability": _probability_summary_to_dict(probability_summary),
        "backtest": _performance_summary_to_dict(backtest_result.summary),
        "backtest_config": backtest_config_to_dict(backtest_config),
        "decision_summary": _decision_summary(backtest_result),
        "clv": _clv_summary(backtest_result),
        "artifacts": {
            "predictions_path": str(predictions_path),
            "clv_report_path": str(clv_report_path),
            "backtest_config_path": str(backtest_paths.config_path),
            "bet_decisions_path": str(backtest_paths.decision_report_path),
        },
    }
    _write_json(report_path, report)
    return PaperTradingResult(
        predictions_path=predictions_path,
        report_path=report_path,
        clv_report_path=clv_report_path,
        backtest_config_path=backtest_paths.config_path,
        bet_decisions_path=backtest_paths.decision_report_path,
        report=report,
        predictions=predictions,
        backtest_result=backtest_result,
        probability_summary=probability_summary,
    )


def paper_trading_result_to_dict(result: PaperTradingResult) -> dict[str, Any]:
    return {
        "predictions_path": str(result.predictions_path),
        "report_path": str(result.report_path),
        "clv_report_path": str(result.clv_report_path),
        "backtest_config_path": str(result.backtest_config_path),
        "bet_decisions_path": str(result.bet_decisions_path),
        "report": result.report,
    }


def _read_method_predictions(
    path: Path | str,
    *,
    method: str,
    as_of: datetime,
    model_version: str,
) -> tuple[ModelPrediction, ...]:
    predictions: list[ModelPrediction] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("prediction CSV must contain a header")
        required = {"race_id", "runner_id", "method", "probability"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"prediction CSV is missing fields: {sorted(missing)}")
        for row in reader:
            if row["method"] != method:
                continue
            predictions.append(
                ModelPrediction(
                    race_id=RaceId(row["race_id"]),
                    runner_id=RunnerId(row["runner_id"]),
                    model_name=ModelName(method),
                    model_version=model_version,
                    target=PredictionTarget.WIN_PROBABILITY,
                    probability=float(row["probability"]),
                    as_of=as_of,
                    metadata={
                        "source_method": method,
                        "source_fold_id": row.get("fold_id"),
                    },
                )
            )
    return tuple(
        sorted(
            predictions,
            key=lambda prediction: (
                str(prediction.race_id),
                str(prediction.runner_id),
            ),
        )
    )


def _races_for_ids(
    dataset_path: Path,
    race_ids: Sequence[RaceId],
) -> tuple[Race, ...]:
    race_id_set = set(race_ids)
    races = CsvRaceRepository(dataset_path / "races.csv").list_races(
        start_date=date.min,
        end_date=date.max,
    )
    return tuple(race for race in races if race.race_id in race_id_set)


def _validate_inputs(
    *,
    predictions: tuple[ModelPrediction, ...],
    races: tuple[Race, ...],
    results: tuple[Result, ...],
) -> None:
    prediction_keys = {
        (prediction.race_id, prediction.runner_id)
        for prediction in predictions
    }
    race_ids = {race.race_id for race in races}
    missing_races = {
        race_id for race_id, _ in prediction_keys if race_id not in race_ids
    }
    if missing_races:
        raise ValueError(f"Missing races for prediction race_ids: {sorted(missing_races)}")

    result_keys = {(result.race_id, result.runner_id) for result in results}
    missing_results = prediction_keys - result_keys
    if missing_results:
        race_id, runner_id = sorted(
            missing_results,
            key=lambda item: (str(item[0]), str(item[1])),
        )[0]
        raise ValueError(
            f"Missing result for race_id={race_id!r}, runner_id={runner_id!r}"
        )


def _write_predictions(
    path: Path,
    *,
    method: str,
    predictions: Sequence[ModelPrediction],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                "race_id": str(prediction.race_id),
                "runner_id": str(prediction.runner_id),
                "method": method,
                "model_name": str(prediction.model_name),
                "model_version": prediction.model_version,
                "target": prediction.target.value,
                "probability": str(prediction.probability),
                "as_of": prediction.as_of.isoformat(),
                "metadata_json": json.dumps(
                    prediction.metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            for prediction in predictions
        )


def _write_clv_report(path: Path, result: BacktestResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "race_id",
        "runner_id",
        "did_bet",
        "odds",
        "closing_odds",
        "edge",
        "stake_jpy",
        "clv_odds_delta",
        "clv_implied_probability_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "race_id": str(decision.race_id),
                "runner_id": str(decision.runner_id),
                "did_bet": decision.did_bet,
                "odds": decision.odds,
                "closing_odds": decision.closing_odds,
                "edge": decision.edge,
                "stake_jpy": decision.stake_jpy,
                "clv_odds_delta": decision.clv_odds_delta,
                "clv_implied_probability_delta": (
                    decision.clv_implied_probability_delta
                ),
            }
            for decision in result.decisions
        )


def _decision_summary(result: BacktestResult) -> dict[str, Any]:
    skip_counts: dict[str, int] = {}
    for decision in result.decisions:
        if decision.did_bet:
            continue
        reason = decision.skip_reason or "unknown"
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    positive_edge = [
        decision for decision in result.decisions if decision.edge > 0.0
    ]
    return {
        "decisions": len(result.decisions),
        "did_bet": sum(1 for decision in result.decisions if decision.did_bet),
        "positive_edge_decisions": len(positive_edge),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "mean_edge": (
            sum(decision.edge for decision in result.decisions) / len(result.decisions)
            if result.decisions
            else 0.0
        ),
    }


def _clv_summary(result: BacktestResult) -> dict[str, Any]:
    bet_decisions = [decision for decision in result.decisions if decision.did_bet]
    with_clv = [
        decision
        for decision in bet_decisions
        if decision.clv_implied_probability_delta is not None
    ]
    if not with_clv:
        return {
            "bets_with_clv": 0,
            "mean_clv_odds_delta": 0.0,
            "mean_clv_implied_probability_delta": 0.0,
            "positive_clv_rate": 0.0,
        }
    return {
        "bets_with_clv": len(with_clv),
        "mean_clv_odds_delta": _mean(
            decision.clv_odds_delta or 0.0 for decision in with_clv
        ),
        "mean_clv_implied_probability_delta": _mean(
            decision.clv_implied_probability_delta or 0.0
            for decision in with_clv
        ),
        "positive_clv_rate": (
            sum(
                1
                for decision in with_clv
                if (decision.clv_implied_probability_delta or 0.0) > 0.0
            )
            / len(with_clv)
        ),
    }


def _mean(values: Sequence[float] | Any) -> float:
    materialized = tuple(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _probability_summary_to_dict(summary: ProbabilitySummary) -> dict[str, Any]:
    return {
        "observations": summary.observations,
        "positives": summary.positives,
        "mean_predicted_probability": summary.mean_predicted_probability,
        "empirical_rate": summary.empirical_rate,
        "log_loss": summary.log_loss,
        "brier_score": summary.brier_score,
        "expected_calibration_error": summary.expected_calibration_error,
    }


def _performance_summary_to_dict(summary: Any) -> dict[str, Any]:
    return {
        "total_bets": summary.total_bets,
        "wins": summary.wins,
        "total_staked_jpy": summary.total_staked_jpy,
        "total_payout_jpy": summary.total_payout_jpy,
        "net_profit_jpy": summary.net_profit_jpy,
        "roi": summary.roi,
        "hit_rate": summary.hit_rate,
        "turnover": summary.turnover,
        "max_drawdown": summary.max_drawdown,
        "final_bankroll_jpy": summary.final_bankroll_jpy,
    }


def _replay_odds_csv_path(
    dataset_path: Path,
    *,
    use_odds_timeseries: bool,
) -> Path:
    odds_timeseries_path = dataset_path / "odds_timeseries.csv"
    odds_path = dataset_path / "odds.csv"
    if use_odds_timeseries and odds_timeseries_path.exists():
        return odds_timeseries_path
    if odds_path.exists():
        return odds_path
    return odds_timeseries_path


def _require_non_empty(rows: Sequence[object], label: str) -> None:
    if not rows:
        raise ValueError(f"No {label} found")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
