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
    "entries",
    "features",
    "results",
    "odds",
    "odds_timeseries",
    "payouts",
)

DATA_SOURCE_CATALOG: tuple[dict[str, object], ...] = (
    {
        "source": "RACE",
        "layer": "accumulated",
        "contains": ["races", "entries", "results", "payouts", "pools"],
        "primary_uses": ["entry_features", "past_performance", "settlement"],
        "point_in_time_notes": "Use only rows known before the target race for features.",
    },
    {
        "source": "0B41",
        "layer": "realtime",
        "contains": ["win_odds", "place_odds", "bracket_quinella_odds"],
        "primary_uses": ["market_features", "odds_movement", "closing_line_value"],
        "point_in_time_notes": "Filter snapshots by captured_at for live-like replay.",
    },
    {
        "source": "0B42",
        "layer": "realtime",
        "contains": ["quinella_pair_odds"],
        "primary_uses": ["pair_probability_model", "quinella_simulation"],
        "point_in_time_notes": "Keep pair odds separate from runner-level win replay.",
    },
    {
        "source": "0B30",
        "layer": "realtime",
        "contains": ["all_bet_type_odds"],
        "primary_uses": ["trio_simulation", "trifecta_simulation"],
        "point_in_time_notes": "Must be accumulated before the one-week retention expires.",
    },
)

FEATURE_PROVENANCE_RULES: tuple[tuple[str, str, str], ...] = (
    ("race_", "RACE", "race_conditions"),
    ("horse_number", "RACE", "entry_details"),
    ("gate_number", "RACE", "entry_details"),
    ("carried_weight_kg", "RACE", "entry_details"),
    ("age", "RACE", "entry_details"),
    ("sex", "RACE", "entry_details"),
    ("horse_symbol_code", "RACE", "entry_details"),
    ("breed_code", "RACE", "entry_details"),
    ("coat_color_code", "RACE", "entry_details"),
    ("trainer_affiliation_code", "RACE", "entry_details"),
    ("jockey_id", "RACE", "entry_details"),
    ("trainer_id", "RACE", "entry_details"),
    ("body_weight", "RACE", "entry_details"),
    ("starter", "RACE", "entry_details"),
    ("entry_", "RACE", "entry_market_snapshot"),
    ("odds_", "0B41", "odds_timeseries"),
    ("implied_probability_", "0B41", "odds_timeseries"),
    ("pool_size_", "0B41/H1", "pool"),
    ("past_", "RACE", "past_performance"),
    ("days_since_last_run", "RACE", "past_performance"),
    ("avg_", "RACE", "past_performance"),
    ("best_", "RACE", "past_performance"),
    ("win_rate_", "RACE", "past_performance"),
    ("same_", "RACE", "past_performance"),
    ("last_", "RACE", "past_performance"),
    ("top3_", "RACE", "past_performance"),
    ("distance_delta_", "RACE", "past_performance"),
    ("jockey_past_", "RACE", "person_oof_stats"),
    ("trainer_past_", "RACE", "person_oof_stats"),
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
    entries = _summarize_entries(dataset / "entries.csv")
    features = _summarize_entity_file(dataset / "features.csv")
    results = _summarize_entity_file(dataset / "results.csv")
    odds_latest = _summarize_entity_file(dataset / "odds.csv")
    odds_timeseries = _summarize_odds_timeseries(dataset / "odds_timeseries.csv")
    payouts = _summarize_payouts(dataset / "payouts.csv")
    feature_columns = _feature_columns(dataset / "features.csv")
    feature_provenance = _summarize_feature_provenance(feature_columns)

    coverage = {
        "races": races["rows"],
        "entries": entries["rows"],
        "feature_rows": features["rows"],
        "results": results["rows"],
        "odds_latest": odds_latest["rows"],
        "odds_timeseries": odds_timeseries["rows"],
        "payouts": payouts["rows"],
        "races_with_features": features["unique_races"],
        "races_with_entries": entries["unique_races"],
        "races_with_results": results["unique_races"],
        "races_with_latest_odds": odds_latest["unique_races"],
        "races_with_odds_timeseries": odds_timeseries["unique_races"],
        "races_with_payouts": payouts["unique_races"],
    }

    report = {
        "dataset_dir": str(dataset),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": file_summary,
        "data_source_catalog": list(DATA_SOURCE_CATALOG),
        "dataset_manifest": _dataset_manifest(
            dataset=dataset,
            races=races,
            entries=entries,
            coverage=coverage,
            odds_timeseries=odds_timeseries,
            payouts=payouts,
            replay_report=replay_report,
            feature_columns=feature_columns,
            feature_provenance=feature_provenance,
        ),
        "race_dates": races["race_dates"],
        "identity": entries["identity"],
        "coverage": coverage,
        "odds_timeseries": odds_timeseries,
        "payouts": payouts,
        "feature_provenance": feature_provenance,
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


def _summarize_entries(path: Path) -> dict[str, object]:
    summary = _summarize_entity_file(path)
    horse_ids: set[str] = set()
    missing_horse_id_rows = 0
    duplicate_runner_ids = 0
    runner_ids: set[str] = set()

    for row in _iter_csv_rows(path):
        runner_id = row.get("runner_id", "")
        if runner_id:
            if runner_id in runner_ids:
                duplicate_runner_ids += 1
            runner_ids.add(runner_id)
        horse_id = row.get("horse_id", "")
        if horse_id:
            horse_ids.add(horse_id)
        else:
            missing_horse_id_rows += 1

    return {
        **summary,
        "identity": {
            "unique_horses": len(horse_ids),
            "missing_horse_id_rows": missing_horse_id_rows,
            "duplicate_runner_ids": duplicate_runner_ids,
            "has_cross_race_horse_identity": len(horse_ids) > 0,
        },
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


def _feature_columns(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return tuple(
        field.removeprefix("feature__")
        for field in header
        if field.startswith("feature__")
    )


def _summarize_feature_provenance(
    feature_columns: tuple[str, ...],
) -> dict[str, object]:
    by_group: dict[str, list[str]] = {}
    by_source: dict[str, list[str]] = {}
    unmatched: list[str] = []

    for feature_name in feature_columns:
        matched = _feature_provenance(feature_name)
        if matched is None:
            unmatched.append(feature_name)
            continue
        source, group = matched
        by_group.setdefault(group, []).append(feature_name)
        by_source.setdefault(source, []).append(feature_name)

    return {
        "feature_count": len(feature_columns),
        "groups": {
            group: {
                "feature_count": len(features),
                "sample_features": sorted(features)[:10],
            }
            for group, features in sorted(by_group.items())
        },
        "sources": {
            source: {
                "feature_count": len(features),
                "sample_features": sorted(features)[:10],
            }
            for source, features in sorted(by_source.items())
        },
        "unmatched_features": sorted(unmatched),
    }


def _feature_provenance(feature_name: str) -> tuple[str, str] | None:
    for prefix, source, group in FEATURE_PROVENANCE_RULES:
        if feature_name.startswith(prefix):
            return source, group
    return None


def _dataset_manifest(
    *,
    dataset: Path,
    races: Mapping[str, object],
    entries: Mapping[str, object],
    coverage: Mapping[str, object],
    odds_timeseries: Mapping[str, object],
    payouts: Mapping[str, object],
    replay_report: Mapping[str, object] | None,
    feature_columns: tuple[str, ...],
    feature_provenance: Mapping[str, object],
) -> dict[str, object]:
    output_counts = (
        replay_report.get("output_counts", {}) if replay_report is not None else {}
    )
    feature_version = (
        replay_report.get("feature_version") if replay_report is not None else None
    )
    return {
        "dataset_dir": str(dataset),
        "dataset_version": feature_version,
        "race_dates": races["race_dates"],
        "row_counts": {
            "races": coverage["races"],
            "entries": coverage["entries"],
            "feature_rows": coverage["feature_rows"],
            "results": coverage["results"],
            "odds_latest": coverage["odds_latest"],
            "odds_timeseries": coverage["odds_timeseries"],
            "payouts": coverage["payouts"],
        },
        "replay_output_counts": output_counts,
        "enrichment_flags": {
            "entry_details": _has_group(feature_provenance, "entry_details"),
            "race_conditions": _has_group(feature_provenance, "race_conditions"),
            "odds_timeseries": int(odds_timeseries["rows"]) > int(coverage["odds_latest"]),
            "market_movement_features": _has_group(feature_provenance, "odds_timeseries"),
            "pool_features": _has_group(feature_provenance, "pool"),
            "official_payouts": int(payouts["official_rows"]) > 0,
            "past_performance": _has_group(feature_provenance, "past_performance"),
            "person_stats": _has_group(feature_provenance, "person_oof_stats"),
            "categorical_person_ids": (
                "jockey_id" in feature_columns or "trainer_id" in feature_columns
            ),
            "runner_horse_identity_map": (
                entries["identity"]["has_cross_race_horse_identity"]
            ),
        },
        "coverage_ratios": _coverage_ratios(coverage),
        "point_in_time_safety": {
            "requires_as_of_filter": True,
            "feature_rows_have_as_of": bool(feature_version),
            "odds_timeseries_filter_required": int(odds_timeseries["rows"]) > 0,
            "known_leakage_risks": [
                "closing odds used as features for pre-race inference",
                "result/payout rows joined before settlement",
                "person stats computed with target race included",
            ],
        },
        "identity": entries["identity"],
    }


def _has_group(feature_provenance: Mapping[str, object], group: str) -> bool:
    groups = feature_provenance.get("groups", {})
    return isinstance(groups, Mapping) and group in groups


def _coverage_ratios(coverage: Mapping[str, object]) -> dict[str, float]:
    races = int(coverage["races"])
    if races <= 0:
        return {
            "features_per_race": 0.0,
            "entries_per_race": 0.0,
            "results_per_race": 0.0,
            "latest_odds_per_race": 0.0,
            "odds_snapshots_per_race": 0.0,
            "payouts_per_race": 0.0,
        }
    return {
        "features_per_race": int(coverage["feature_rows"]) / races,
        "entries_per_race": int(coverage["entries"]) / races,
        "results_per_race": int(coverage["results"]) / races,
        "latest_odds_per_race": int(coverage["odds_latest"]) / races,
        "odds_snapshots_per_race": int(coverage["odds_timeseries"]) / races,
        "payouts_per_race": int(coverage["payouts"]) / races,
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
    for key in ("entries", "feature_rows", "results", "odds_latest"):
        if int(coverage[key]) == 0:
            warnings.append(f"missing_{key}")
    if int(odds_timeseries["rows"]) == 0:
        warnings.append("missing_odds_timeseries")
    if int(payouts["official_rows"]) == 0:
        warnings.append("missing_official_payouts")
    if races and int(coverage["races_with_features"]) != races:
        warnings.append("feature_race_coverage_mismatch")
    if races and int(coverage["races_with_entries"]) != races:
        warnings.append("entry_race_coverage_mismatch")
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
