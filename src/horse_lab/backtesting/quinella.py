"""Quinella replay simulation from O2 odds and official HR payouts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from horse_lab.schemas import BetType


class QuinellaStrategy(str, Enum):
    FAVORITE = "favorite"
    POSITIVE_EDGE = "positive_edge"


@dataclass(frozen=True)
class QuinellaSimulationConfig:
    initial_bankroll_jpy: int = 100_000
    stake_jpy: int = 100
    strategy: QuinellaStrategy | str = QuinellaStrategy.FAVORITE
    minimum_edge: float = 0.0
    max_bets_per_race: int = 1

    def __post_init__(self) -> None:
        if self.initial_bankroll_jpy <= 0:
            raise ValueError("initial_bankroll_jpy must be positive")
        if self.stake_jpy <= 0:
            raise ValueError("stake_jpy must be positive")
        if self.max_bets_per_race <= 0:
            raise ValueError("max_bets_per_race must be positive")
        object.__setattr__(self, "strategy", QuinellaStrategy(self.strategy))


@dataclass(frozen=True)
class QuinellaSimulationResult:
    decisions: tuple[dict[str, object], ...]
    summary: dict[str, object]
    summary_path: Path
    decisions_path: Path


def run_quinella_simulation_from_csv(
    odds_csv_path: Path | str,
    payouts_csv_path: Path | str,
    artifact_dir: Path | str,
    *,
    start_date: date,
    end_date: date,
    config: QuinellaSimulationConfig = QuinellaSimulationConfig(),
) -> QuinellaSimulationResult:
    """Run a settlement-focused quinella replay simulation."""

    odds_by_race = _latest_quinella_odds_by_race(
        Path(odds_csv_path),
        start_date=start_date,
        end_date=end_date,
    )
    payouts_by_key = _official_quinella_payouts(Path(payouts_csv_path))

    bankroll = config.initial_bankroll_jpy
    decisions: list[dict[str, object]] = []
    for race_id in sorted(odds_by_race):
        race_quotes = odds_by_race[race_id]
        candidates = _rank_candidates(race_quotes, config=config)
        for quote in candidates[: config.max_bets_per_race]:
            payout_jpy_per_100 = payouts_by_key.get((race_id, quote["runner_id"]), 0)
            stake = min(config.stake_jpy, bankroll)
            payout_jpy = int(round(stake * payout_jpy_per_100 / 100.0))
            profit_jpy = payout_jpy - stake
            bankroll_after = bankroll + profit_jpy
            decisions.append(
                {
                    "race_id": race_id,
                    "runner_id": quote["runner_id"],
                    "bet_type": BetType.QUINELLA.value,
                    "captured_at": quote["captured_at"],
                    "odds": quote["odds"],
                    "market_probability": quote["market_probability"],
                    "edge": quote["edge"],
                    "stake_jpy": stake,
                    "payout_jpy": payout_jpy,
                    "profit_jpy": profit_jpy,
                    "bankroll_before_jpy": bankroll,
                    "bankroll_after_jpy": bankroll_after,
                    "is_win": payout_jpy_per_100 > 0,
                    "payout_jpy_per_100": payout_jpy_per_100,
                    "strategy": config.strategy.value,
                }
            )
            bankroll = bankroll_after
            if bankroll <= 0:
                break
        if bankroll <= 0:
            break

    summary = _summarize_decisions(
        decisions,
        initial_bankroll_jpy=config.initial_bankroll_jpy,
        final_bankroll_jpy=bankroll,
        races_considered=len(odds_by_race),
    )
    artifact_path = Path(artifact_dir)
    summary_path = artifact_path / "quinella_simulation_summary.json"
    decisions_path = artifact_path / "quinella_bet_decisions.csv"
    _write_decisions_csv(decisions_path, decisions)
    _write_json(summary_path, {"config": _config_to_dict(config), "summary": summary})
    return QuinellaSimulationResult(
        decisions=tuple(decisions),
        summary=summary,
        summary_path=summary_path,
        decisions_path=decisions_path,
    )


def _latest_quinella_odds_by_race(
    path: Path,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, list[dict[str, object]]]:
    latest_by_key: dict[tuple[str, str], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("bet_type") != BetType.QUINELLA.value:
                continue
            race_id = row["race_id"]
            race_date = _race_date_from_id(race_id)
            if race_date < start_date or race_date > end_date:
                continue
            captured_at = datetime.fromisoformat(row["captured_at"])
            key = (race_id, row["runner_id"])
            previous = latest_by_key.get(key)
            if previous is None or captured_at > previous["captured_at_dt"]:
                latest_by_key[key] = {
                    "race_id": race_id,
                    "runner_id": row["runner_id"],
                    "captured_at": captured_at.isoformat(),
                    "captured_at_dt": captured_at,
                    "odds": float(row["odds"]),
                }

    grouped: dict[str, list[dict[str, object]]] = {}
    for quote in latest_by_key.values():
        grouped.setdefault(str(quote["race_id"]), []).append(quote)
    return {
        race_id: _with_market_probabilities(quotes)
        for race_id, quotes in grouped.items()
    }


def _with_market_probabilities(
    quotes: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    inverse_sum = sum(1.0 / float(quote["odds"]) for quote in quotes)
    enriched: list[dict[str, object]] = []
    for quote in quotes:
        odds = float(quote["odds"])
        probability = (1.0 / odds) / inverse_sum if inverse_sum > 0.0 else 0.0
        enriched.append(
            {
                **quote,
                "market_probability": probability,
                "edge": probability * odds - 1.0,
            }
        )
    return enriched


def _rank_candidates(
    quotes: Sequence[dict[str, object]],
    *,
    config: QuinellaSimulationConfig,
) -> list[dict[str, object]]:
    if config.strategy == QuinellaStrategy.FAVORITE:
        return sorted(
            quotes,
            key=lambda quote: (float(quote["odds"]), str(quote["runner_id"])),
        )
    return [
        quote
        for quote in sorted(
            quotes,
            key=lambda quote: (
                float(quote["edge"]),
                float(quote["market_probability"]),
            ),
            reverse=True,
        )
        if float(quote["edge"]) >= config.minimum_edge
    ]


def _official_quinella_payouts(path: Path) -> dict[tuple[str, str], int]:
    payouts: dict[tuple[str, str], int] = {}
    if not path.exists():
        return payouts
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("bet_type") != BetType.QUINELLA.value:
                continue
            payout = _parse_int_or_zero(row.get("payout_jpy_per_100"))
            if payout <= 0:
                continue
            payouts[(row["race_id"], row["runner_id"])] = payout
    return payouts


def _summarize_decisions(
    decisions: Sequence[Mapping[str, object]],
    *,
    initial_bankroll_jpy: int,
    final_bankroll_jpy: int,
    races_considered: int,
) -> dict[str, object]:
    total_stake = sum(int(row["stake_jpy"]) for row in decisions)
    total_payout = sum(int(row["payout_jpy"]) for row in decisions)
    wins = sum(1 for row in decisions if row["is_win"])
    return {
        "races_considered": races_considered,
        "bets": len(decisions),
        "wins": wins,
        "hit_rate": wins / len(decisions) if decisions else 0.0,
        "total_stake_jpy": total_stake,
        "total_payout_jpy": total_payout,
        "profit_jpy": total_payout - total_stake,
        "roi": (total_payout - total_stake) / total_stake if total_stake else 0.0,
        "initial_bankroll_jpy": initial_bankroll_jpy,
        "final_bankroll_jpy": final_bankroll_jpy,
    }


def _write_decisions_csv(
    path: Path,
    decisions: Sequence[Mapping[str, object]],
) -> None:
    fieldnames = (
        "race_id",
        "runner_id",
        "bet_type",
        "captured_at",
        "odds",
        "market_probability",
        "edge",
        "stake_jpy",
        "payout_jpy",
        "profit_jpy",
        "bankroll_before_jpy",
        "bankroll_after_jpy",
        "is_win",
        "payout_jpy_per_100",
        "strategy",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decisions:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _config_to_dict(config: QuinellaSimulationConfig) -> dict[str, object]:
    return {
        "initial_bankroll_jpy": config.initial_bankroll_jpy,
        "stake_jpy": config.stake_jpy,
        "strategy": config.strategy.value,
        "minimum_edge": config.minimum_edge,
        "max_bets_per_race": config.max_bets_per_race,
    }


def _race_date_from_id(race_id: str) -> date:
    return date(int(race_id[:4]), int(race_id[4:6]), int(race_id[6:8]))


def _parse_int_or_zero(value: str | None) -> int:
    normalized = (value or "").strip()
    return int(normalized) if normalized else 0
