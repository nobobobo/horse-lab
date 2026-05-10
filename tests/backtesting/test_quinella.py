import csv
import datetime as dt
import json
from pathlib import Path

from horse_lab.backtesting import (
    QuinellaSimulationConfig,
    QuinellaStrategy,
    run_quinella_simulation_from_csv,
)
from horse_lab.cli import main


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _odds_rows() -> list[dict[str, object]]:
    return [
        {
            "race_id": "2026050805010101",
            "runner_id": "2026050805010101-01_02",
            "bet_type": "quinella",
            "captured_at": "2026-05-08T09:50:00",
            "odds": "8.0",
            "popularity_rank": "1",
            "pool_size_jpy": "10000",
            "source": "fixture",
        },
        {
            "race_id": "2026050805010101",
            "runner_id": "2026050805010101-01_03",
            "bet_type": "quinella",
            "captured_at": "2026-05-08T09:55:00",
            "odds": "12.0",
            "popularity_rank": "2",
            "pool_size_jpy": "10000",
            "source": "fixture",
        },
        {
            "race_id": "2026050805010102",
            "runner_id": "2026050805010102-02_03",
            "bet_type": "quinella",
            "captured_at": "2026-05-08T09:55:00",
            "odds": "5.0",
            "popularity_rank": "1",
            "pool_size_jpy": "10000",
            "source": "fixture",
        },
    ]


def test_quinella_simulation_settles_favorite_strategy_with_official_payouts(tmp_path):
    odds_csv = tmp_path / "odds.csv"
    payouts_csv = tmp_path / "payouts.csv"
    artifact_dir = tmp_path / "artifacts"
    _write_csv(odds_csv, _odds_rows())
    _write_csv(
        payouts_csv,
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "finish_position": "",
                "is_win": "true",
                "payout_jpy_per_100": "800",
                "odds": "8.0",
                "pool_size_jpy": "10000",
                "source": "jravan_hr_official",
            }
        ],
    )

    result = run_quinella_simulation_from_csv(
        odds_csv,
        payouts_csv,
        artifact_dir,
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
        config=QuinellaSimulationConfig(stake_jpy=100),
    )

    assert result.summary["races_considered"] == 2
    assert result.summary["bets"] == 2
    assert result.summary["wins"] == 1
    assert result.summary["total_stake_jpy"] == 200
    assert result.summary["total_payout_jpy"] == 800
    assert result.summary_path.exists()
    assert result.decisions_path.exists()


def test_quinella_positive_edge_strategy_can_make_zero_bets(tmp_path):
    odds_csv = tmp_path / "odds.csv"
    payouts_csv = tmp_path / "payouts.csv"
    artifact_dir = tmp_path / "artifacts"
    _write_csv(odds_csv, _odds_rows())
    _write_csv(
        payouts_csv,
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "payout_jpy_per_100": "800",
            }
        ],
    )

    result = run_quinella_simulation_from_csv(
        odds_csv,
        payouts_csv,
        artifact_dir,
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
        config=QuinellaSimulationConfig(
            strategy=QuinellaStrategy.POSITIVE_EDGE,
            minimum_edge=10.0,
        ),
    )

    assert result.summary["bets"] == 0
    assert result.summary["final_bankroll_jpy"] == 100_000


def test_quinella_sim_cli_writes_artifacts(tmp_path, capsys):
    odds_csv = tmp_path / "odds.csv"
    payouts_csv = tmp_path / "payouts.csv"
    artifact_dir = tmp_path / "artifacts"
    _write_csv(odds_csv, _odds_rows())
    _write_csv(
        payouts_csv,
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "payout_jpy_per_100": "800",
            }
        ],
    )

    assert (
        main(
            [
                "quinella-sim",
                str(odds_csv),
                str(payouts_csv),
                str(artifact_dir),
                "--start-date",
                "2026-05-08",
                "--end-date",
                "2026-05-08",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["summary"]["bets"] == 2
    assert Path(summary["summary_path"]).exists()
    assert Path(summary["decisions_path"]).exists()
