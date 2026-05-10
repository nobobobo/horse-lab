"""Replay dataset coverage and integrity reports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping


REPLAY_DATASET_FILES: tuple[str, ...] = (
    "races",
    "features",
    "results",
    "odds",
    "odds_timeseries",
    "payouts",
)


def build_replay_data_quality_report(
    dataset_dir: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable QA report for a replay-ready dataset."""

    dataset = Path(dataset_dir)
    file_summary = {
        name: _file_summary(dataset / f"{name}.csv") for name in REPLAY_DATASET_FILES
    }
    replay_report = _read_optional_json(dataset / "replay_dataset_report.json")

    races = _summarize_races(dataset / "races.csv")
    features = _summarize_entity_file(dataset / "features.csv")
    results = _summarize_entity_file(dataset / "results.csv")
    odds_latest = _summarize_entity_file(dataset / "odds.csv")
    odds_timeseries = _summarize_odds_timeseries(dataset / "odds_timeseries.csv")
    payouts = _summarize_payouts(dataset / "payouts.csv")

    coverage = {
        "races": races["rows"],
        "feature_rows": features["rows"],
        "results": results["rows"],
        "odds_latest": odds_latest["rows"],
        "odds_timeseries": odds_timeseries["rows"],
        "payouts": payouts["rows"],
        "races_with_features": features["unique_races"],
        "races_with_results": results["unique_races"],
        "races_with_latest_odds": odds_latest["unique_races"],
        "races_with_odds_timeseries": odds_timeseries["unique_races"],
        "races_with_payouts": payouts["unique_races"],
    }

    report = {
        "dataset_dir": str(dataset),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": file_summary,
        "race_dates": races["race_dates"],
        "coverage": coverage,
        "odds_timeseries": odds_timeseries,
        "payouts": payouts,
        "replay_dataset_report": replay_report,
        "warnings": _quality_warnings(coverage, odds_timeseries, payouts),
    }

    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def _file_summary(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "rows": _count_csv_rows(path) if path.exists() else 0,
    }


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _summarize_races(path: Path) -> dict[str, object]:
    race_dates: list[str] = []
    rows = 0
    for row in _iter_csv_rows(path):
        rows += 1
        race_date = row.get("race_date", "")
        if race_date:
            race_dates.append(race_date)
    return {
        "rows": rows,
        "race_dates": {
            "min": min(race_dates) if race_dates else None,
            "max": max(race_dates) if race_dates else None,
        },
    }


def _summarize_entity_file(path: Path) -> dict[str, object]:
    race_ids: set[str] = set()
    runner_ids: set[str] = set()
    rows = 0
    for row in _iter_csv_rows(path):
        rows += 1
        _add_if_present(race_ids, row.get("race_id"))
        _add_if_present(runner_ids, row.get("runner_id"))
    return {
        "rows": rows,
        "unique_races": len(race_ids),
        "unique_runners": len(runner_ids),
    }


def _summarize_odds_timeseries(path: Path) -> dict[str, object]:
    race_ids: set[str] = set()
    runner_ids: set[str] = set()
    bet_types: Counter[str] = Counter()
    snapshots_by_runner: Counter[str] = Counter()
    snapshots_by_race: Counter[str] = Counter()
    missing_pool_size_rows = 0
    rows = 0

    for row in _iter_csv_rows(path):
        rows += 1
        race_id = row.get("race_id", "")
        runner_id = row.get("runner_id", "")
        _add_if_present(race_ids, race_id)
        _add_if_present(runner_ids, runner_id)
        if row.get("bet_type"):
            bet_types[row["bet_type"]] += 1
        if runner_id:
            snapshots_by_runner[runner_id] += 1
        if race_id:
            snapshots_by_race[race_id] += 1
        if not row.get("pool_size_jpy"):
            missing_pool_size_rows += 1

    return {
        "rows": rows,
        "unique_races": len(race_ids),
        "unique_runners": len(runner_ids),
        "bet_types": dict(sorted(bet_types.items())),
        "missing_pool_size_rows": missing_pool_size_rows,
        "snapshot_count_per_runner": _distribution(snapshots_by_runner.values()),
        "snapshot_count_per_race": _distribution(snapshots_by_race.values()),
    }


def _summarize_payouts(path: Path) -> dict[str, object]:
    race_ids: set[str] = set()
    runner_ids: set[str] = set()
    by_bet_type: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    missing_pool_size_rows = 0
    official_rows = 0
    rows = 0

    for row in _iter_csv_rows(path):
        rows += 1
        _add_if_present(race_ids, row.get("race_id"))
        _add_if_present(runner_ids, row.get("runner_id"))
        if row.get("bet_type"):
            by_bet_type[row["bet_type"]] += 1
        if row.get("source"):
            by_source[row["source"]] += 1
            if str(row["source"]).startswith("jravan_hr"):
                official_rows += 1
        if not row.get("pool_size_jpy"):
            missing_pool_size_rows += 1

    return {
        "rows": rows,
        "unique_races": len(race_ids),
        "unique_runners": len(runner_ids),
        "by_bet_type": dict(sorted(by_bet_type.items())),
        "by_source": dict(sorted(by_source.items())),
        "official_rows": official_rows,
        "missing_pool_size_rows": missing_pool_size_rows,
    }


def _distribution(values: Iterable[int]) -> dict[str, float | int | None]:
    items = sorted(values)
    if not items:
        return {"min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "min": items[0],
        "p50": median(items),
        "p95": _percentile(items, 0.95),
        "max": items[-1],
        "mean": mean(items),
    }


def _percentile(values: list[int], q: float) -> float:
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _quality_warnings(
    coverage: Mapping[str, object],
    odds_timeseries: Mapping[str, object],
    payouts: Mapping[str, object],
) -> list[str]:
    warnings: list[str] = []
    races = int(coverage["races"])
    if races == 0:
        warnings.append("no_races")
    for key in ("feature_rows", "results", "odds_latest"):
        if int(coverage[key]) == 0:
            warnings.append(f"missing_{key}")
    if int(odds_timeseries["rows"]) == 0:
        warnings.append("missing_odds_timeseries")
    if int(payouts["official_rows"]) == 0:
        warnings.append("missing_official_payouts")
    if races and int(coverage["races_with_features"]) != races:
        warnings.append("feature_race_coverage_mismatch")
    return warnings


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        yield from reader


def _add_if_present(values: set[str], value: str | None) -> None:
    if value:
        values.add(value)


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
