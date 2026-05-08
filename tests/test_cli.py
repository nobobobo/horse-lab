import json
from pathlib import Path

from horse_lab.cli import main
from horse_lab.data.csv_parsing import parse_entry_row, parse_race_row, read_csv_rows
from horse_lab.data.jravan import (
    FixedWidthField,
    JRAVAN_MINIMAL_O1_FIELDS,
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
)
from horse_lab.schemas import RaceId, RunnerId


def _fixed_width_text(
    fields: tuple[FixedWidthField, ...],
    values: dict[str, str],
) -> str:
    total_length = max(field.start + field.length - 1 for field in fields)
    raw = bytearray(b" " * total_length)
    for field in fields:
        encoded = values.get(field.name, "").encode("cp932")
        assert len(encoded) <= field.length, field.name
        start_index = field.start - 1
        raw[start_index : start_index + field.length] = encoded.ljust(
            field.length,
            b" ",
        )
    return bytes(raw).decode("cp932")


def _ra_text(**overrides: str) -> str:
    values = {
            "record_type": "RA",
            "data_kubun": "7",
            "data_created_date": "20260507",
            "race_date": "20260508",
            "venue_code": "05",
            "kaiji": "01",
            "nichiji": "01",
            "race_number": "01",
            "race_name": "若葉ステークス",
            "distance_m": "2000",
            "track_code": "11",
            "turf_track_condition_code": "1",
            "dirt_track_condition_code": "0",
            "weather_code": "1",
            "grade_code": "B",
            "start_time": "1005",
            "registered_horse_count": "16",
            "starter_count": "16",
    }
    values.update(overrides)
    return _fixed_width_text(JRAVAN_MINIMAL_RA_FIELDS, values)


def _se_text(**overrides: str) -> str:
    values = {
            "record_type": "SE",
            "data_kubun": "7",
            "data_created_date": "20260507",
            "race_date": "20260508",
            "venue_code": "05",
            "kaiji": "01",
            "nichiji": "01",
            "race_number": "01",
            "gate_number": "3",
            "horse_number": "07",
            "horse_id": "2020123456",
            "horse_name": "テストホース",
            "sex_code": "2",
            "age": "04",
            "trainer_id": "04050",
            "jockey_id": "01020",
            "carried_weight": "565",
            "body_weight": "486",
            "body_weight_diff_sign": "-",
            "body_weight_diff": "008",
            "abnormal_code": "0",
            "finish_position": "01",
            "is_dead_heat": "0",
            "final_time_seconds": "0705",
            "prize_jpy_x100": "00100000",
    }
    values.update(overrides)
    return _fixed_width_text(JRAVAN_MINIMAL_SE_FIELDS, values)


def _o1_text(**overrides: str) -> str:
    values = {
        "record_type": "O1",
        "data_kubun": "7",
        "data_created_date": "20260508",
        "race_date": "20260508",
        "venue_code": "05",
        "kaiji": "01",
        "nichiji": "01",
        "race_number": "01",
        "captured_month_day_time": "05080950",
        "registered_horse_count": "01",
        "starter_count": "01",
        "win_sale_flag": "7",
        "place_sale_flag": "7",
        "bracket_quinella_sale_flag": "7",
        "place_payout_key": "3",
        "win_odds_entries": _o1_win_entries(("07", "0035", "01")),
        "win_pool_size_jpy_x100": "0000012345",
    }
    values.update(overrides)
    return _fixed_width_text(JRAVAN_MINIMAL_O1_FIELDS, values)


def _o1_win_entries(*entries: tuple[str, str, str]) -> str:
    encoded = "".join(
        f"{horse_number:>2}{odds:>4}{popularity_rank:>2}"
        for horse_number, odds, popularity_rank in entries
    )
    return encoded.ljust(224)


def _write_raw_file(path: Path, *lines: str) -> None:
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932"))


def test_jravan_ingest_cli_writes_staging_csvs(
    tmp_path: Path,
    capsys,
):
    raw_path = tmp_path / "jvdata.txt"
    staging_dir = tmp_path / "staging"
    _write_raw_file(raw_path, _ra_text(), _se_text(), "ZZignored")

    assert main(["jravan-ingest", str(raw_path), str(staging_dir)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"] == {
        "races": 1,
        "entries": 1,
        "results": 1,
        "odds": 0,
        "skipped_records": 1,
    }

    race = parse_race_row(read_csv_rows(staging_dir / "races.csv")[0])
    entry = parse_entry_row(read_csv_rows(staging_dir / "entries.csv")[0])

    assert race.race_id == RaceId("2026050805010101")
    assert race.name == "若葉ステークス"
    assert entry.runner_id == RunnerId("2026050805010101-07")
    assert entry.horse_id == "2020123456"
    assert entry.carried_weight_kg == 56.5


def test_jravan_preview_cli_writes_utf8_copy_without_blank_lines(
    tmp_path: Path,
    capsys,
):
    raw_path = tmp_path / "raw.txt"
    preview_path = tmp_path / "preview.txt"
    raw_path.write_bytes("JGテスト\r\n\r\nRA若葉\r\n".encode("cp932"))

    assert main(["jravan-preview", str(raw_path), str(preview_path)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["lines_written"] == 2
    assert preview_path.read_text(encoding="utf-8") == "JGテスト\nRA若葉\n"


def test_jravan_preview_cli_can_preserve_blank_lines(
    tmp_path: Path,
    capsys,
):
    raw_path = tmp_path / "raw.txt"
    preview_path = tmp_path / "preview.txt"
    raw_path.write_bytes("JGテスト\r\n\r\n".encode("cp932"))

    assert (
        main(
            [
                "jravan-preview",
                str(raw_path),
                str(preview_path),
                "--keep-empty-lines",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["lines_written"] == 2
    assert preview_path.read_text(encoding="utf-8") == "JGテスト\n\n"


def test_jravan_s3_pull_raw_cli_dry_run_returns_sync_plan(tmp_path: Path, capsys):
    local_raw_root = tmp_path / "raw"

    assert (
        main(
            [
                "jravan-s3-pull-raw",
                "run-1",
                str(local_raw_root),
                "--bucket",
                "horse-lab-test",
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["s3_uri"] == "s3://horse-lab-test/raw/jravan/run-1/"
    assert summary["local_dir"] == str(local_raw_root / "run-1")
    assert summary["command"].startswith("aws s3 sync")
    assert summary["executed"] is False
    assert not (local_raw_root / "run-1").exists()


def test_market_replay_cli_runs_replay_ready_csv_dataset(capsys):
    sample_data = Path(__file__).resolve().parents[1] / "sample_data"

    assert (
        main(
            [
                "market-replay",
                str(sample_data),
                "--start-date",
                "2026-05-08",
                "--end-date",
                "2026-05-08",
                "--as-of",
                "2026-05-08T09:55:00",
                "--feature-version",
                "fixture-v1",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"] == {
        "races": 2,
        "feature_rows": 4,
        "odds": 5,
        "results": 4,
        "predictions": 4,
        "bet_records": 4,
    }
    assert summary["backtest"]["final_bankroll_jpy"] == 108_000
    assert summary["probability"]["observations"] == 4
    assert summary["probability"]["brier_score"] > 0.0


def test_jravan_daily_market_replay_cli_runs_local_raw_workflow(
    tmp_path: Path,
    capsys,
):
    run_id = "daily_fixture"
    raw_dir = tmp_path / "raw" / "jravan" / run_id
    raw_dir.mkdir(parents=True)
    _write_raw_file(
        raw_dir / "RACE_20260508.txt",
        _ra_text(registered_horse_count="01", starter_count="01"),
        _se_text(),
        _o1_text(),
    )

    assert (
        main(
            [
                "jravan-daily-market-replay",
                run_id,
                "--workspace-root",
                str(tmp_path),
                "--start-date",
                "2026-05-08",
                "--end-date",
                "2026-05-08",
                "--as-of",
                "2026-05-08T23:59:00",
                "--skip-s3-pull",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["s3_pull"]["skipped"] is True
    assert summary["ingest"]["counts"] == {
        "races": 1,
        "entries": 1,
        "results": 1,
        "odds": 1,
        "skipped_records": 0,
    }
    assert summary["replay_dataset"]["report"]["output_counts"] == {
        "races": 1,
        "feature_rows": 1,
        "results": 1,
        "odds": 1,
    }
    assert summary["market_replay"]["counts"]["predictions"] == 1
    assert summary["market_replay"]["probability"]["observations"] == 1
    assert Path(summary["report_path"]).exists()


def test_jravan_daily_market_replay_cli_dry_run_does_not_create_outputs(
    tmp_path: Path,
    capsys,
):
    assert (
        main(
            [
                "jravan-daily-market-replay",
                "daily_fixture",
                "--workspace-root",
                str(tmp_path),
                "--start-date",
                "2026-05-08",
                "--end-date",
                "2026-05-08",
                "--as-of",
                "2026-05-08T23:59:00",
                "--dry-run",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True
    assert "ingest" not in summary
    assert "market_replay" not in summary
    assert not (tmp_path / "processed").exists()
