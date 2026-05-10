import csv
import json
from pathlib import Path

from horse_lab.analysis import build_replay_data_quality_report
from horse_lab.cli import main


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0]) if rows else ("race_id",)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_replay_data_quality_report_summarizes_core_coverage(tmp_path):
    dataset_dir = tmp_path / "replay"
    _write_csv(
        dataset_dir / "races.csv",
        [
            {"race_id": "2026050805010101", "race_date": "2026-05-08"},
            {"race_id": "2026050905010101", "race_date": "2026-05-09"},
        ],
    )
    _write_csv(
        dataset_dir / "features.csv",
        [
            {"race_id": "2026050805010101", "runner_id": "r1"},
            {"race_id": "2026050805010101", "runner_id": "r2"},
        ],
    )
    _write_csv(
        dataset_dir / "results.csv",
        [{"race_id": "2026050805010101", "runner_id": "r1"}],
    )
    _write_csv(
        dataset_dir / "odds.csv",
        [{"race_id": "2026050805010101", "runner_id": "r1"}],
    )
    _write_csv(
        dataset_dir / "odds_timeseries.csv",
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "r1",
                "bet_type": "win",
                "pool_size_jpy": "1000",
            },
            {
                "race_id": "2026050805010101",
                "runner_id": "r1",
                "bet_type": "win",
                "pool_size_jpy": "",
            },
            {
                "race_id": "2026050805010101",
                "runner_id": "r2",
                "bet_type": "win",
                "pool_size_jpy": "1000",
            },
        ],
    )
    _write_csv(
        dataset_dir / "payouts.csv",
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "r1",
                "bet_type": "win",
                "pool_size_jpy": "1000",
                "source": "jravan_hr_official",
            }
        ],
    )

    report = build_replay_data_quality_report(dataset_dir)

    assert report["coverage"]["races"] == 2
    assert report["race_dates"] == {"min": "2026-05-08", "max": "2026-05-09"}
    assert report["odds_timeseries"]["snapshot_count_per_runner"]["max"] == 2
    assert report["odds_timeseries"]["missing_pool_size_rows"] == 1
    assert report["payouts"]["official_rows"] == 1
    assert "feature_race_coverage_mismatch" in report["warnings"]


def test_jravan_data_qa_cli_writes_report(tmp_path, capsys):
    dataset_dir = tmp_path / "replay"
    output_path = tmp_path / "qa" / "report.json"
    _write_csv(dataset_dir / "races.csv", [{"race_id": "r1", "race_date": ""}])
    _write_csv(dataset_dir / "features.csv", [{"race_id": "r1", "runner_id": "h1"}])
    _write_csv(dataset_dir / "results.csv", [{"race_id": "r1", "runner_id": "h1"}])
    _write_csv(dataset_dir / "odds.csv", [{"race_id": "r1", "runner_id": "h1"}])
    _write_csv(dataset_dir / "odds_timeseries.csv", [{"race_id": "r1", "runner_id": "h1"}])
    _write_csv(
        dataset_dir / "payouts.csv",
        [{"race_id": "r1", "runner_id": "h1", "source": ""}],
    )

    assert main(["jravan-data-qa", str(dataset_dir), str(output_path)]) == 0

    stdout_report = json.loads(capsys.readouterr().out)
    disk_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_report["coverage"]["races"] == 1
    assert disk_report["coverage"]["races"] == 1
