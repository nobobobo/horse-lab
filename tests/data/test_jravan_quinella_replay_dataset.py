import csv
import datetime as dt
import json
from pathlib import Path

from horse_lab.cli import main
from horse_lab.data.csv_parsing import parse_odds_quote_row, read_csv_rows
from horse_lab.data.jravan import (
    FixedWidthField,
    JRAVAN_MINIMAL_O2_FIELDS,
    build_quinella_replay_dataset_from_raw,
    build_quinella_replay_dataset_from_staging,
    quinella_replay_dataset_report_to_dict,
    write_payouts_csv,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def _o2_text(**overrides: str) -> str:
    values = {
        "record_type": "O2",
        "data_kubun": "7",
        "data_created_date": "20260508",
        "race_date": "20260508",
        "venue_code": "05",
        "kaiji": "01",
        "nichiji": "01",
        "race_number": "01",
        "captured_month_day_time": "05080950",
        "registered_horse_count": "16",
        "starter_count": "16",
        "quinella_sale_flag": "7",
        "quinella_odds_entries": _o2_quinella_entries(
            ("01", "02", "0000800", "01"),
        ),
        "quinella_pool_size_jpy_x100": "0000000120",
    }
    values.update(overrides)
    return _fixed_width_text(JRAVAN_MINIMAL_O2_FIELDS, values)


def _o2_quinella_entries(*entries: tuple[str, str, str, str]) -> str:
    encoded = "".join(
        f"{horse1:>2}{horse2:>2}{odds:>7}{popularity_rank:>2}"
        for horse1, horse2, odds, popularity_rank in entries
    )
    return encoded.ljust(1989)


def _write_raw_file(path: Path, *lines: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp932"))


def _write_o2_staging(staging_dir: Path) -> None:
    _write_csv(
        staging_dir / "odds.csv",
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "captured_at": "2026-05-08T09:40:00",
                "odds": "9.0",
                "popularity_rank": "2",
                "pool_size_jpy": "1000",
                "source": "fixture",
            },
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "captured_at": "2026-05-08T09:55:00",
                "odds": "8.0",
                "popularity_rank": "1",
                "pool_size_jpy": "1200",
                "source": "fixture",
            },
            {
                "race_id": "2026050905010101",
                "runner_id": "2026050905010101-01_03",
                "bet_type": "quinella",
                "captured_at": "2026-05-09T09:55:00",
                "odds": "12.0",
                "popularity_rank": "3",
                "pool_size_jpy": "1300",
                "source": "fixture",
            },
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01",
                "bet_type": "win",
                "captured_at": "2026-05-08T09:55:00",
                "odds": "2.0",
                "popularity_rank": "1",
                "pool_size_jpy": "900",
                "source": "fixture",
            },
        ],
    )


def test_build_quinella_replay_dataset_keeps_latest_pair_odds_and_payouts(tmp_path):
    staging_dir = tmp_path / "o2_staging"
    payout_path = tmp_path / "payouts.csv"
    output_dir = tmp_path / "quinella_replay"
    _write_o2_staging(staging_dir)
    write_payouts_csv(
        payout_path,
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "is_win": True,
                "payout_jpy_per_100": 800,
                "odds": 8.0,
                "pool_size_jpy": 1200,
                "source": "jravan_hr_official",
            },
            {
                "race_id": "2026050905010101",
                "runner_id": "2026050905010101-01_03",
                "bet_type": "quinella",
                "is_win": True,
                "payout_jpy_per_100": 1200,
                "odds": 12.0,
                "pool_size_jpy": 1300,
                "source": "jravan_hr_official",
            },
        ],
    )

    export = build_quinella_replay_dataset_from_staging(
        staging_dir,
        payout_path,
        output_dir,
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
    )

    latest_odds = [
        parse_odds_quote_row(row) for row in read_csv_rows(output_dir / "odds.csv")
    ]
    payouts = read_csv_rows(output_dir / "payouts.csv")
    report = json.loads(
        (output_dir / "quinella_replay_dataset_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(latest_odds) == 1
    assert latest_odds[0].captured_at == dt.datetime(2026, 5, 8, 9, 55)
    assert latest_odds[0].odds == 8.0
    assert len(payouts) == 1
    assert payouts[0]["payout_jpy_per_100"] == "800"
    assert report == quinella_replay_dataset_report_to_dict(export.report)
    assert report["output_counts"] == {
        "races": 1,
        "odds": 1,
        "odds_timeseries": 2,
        "payouts": 1,
    }


def test_jravan_build_quinella_replay_dataset_cli(tmp_path, capsys):
    staging_dir = tmp_path / "o2_staging"
    payout_path = tmp_path / "payouts.csv"
    output_dir = tmp_path / "quinella_replay"
    _write_o2_staging(staging_dir)
    write_payouts_csv(
        payout_path,
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "payout_jpy_per_100": 800,
            }
        ],
    )

    assert (
        main(
            [
                "jravan-build-quinella-replay-dataset",
                str(staging_dir),
                str(payout_path),
                str(output_dir),
                "--start-date",
                "2026-05-08",
                "--end-date",
                "2026-05-08",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["output_counts"]["odds"] == 1
    assert Path(summary["csv_paths"]["odds"]).exists()


def test_build_quinella_replay_dataset_from_raw_can_skip_full_timeseries(tmp_path):
    raw_dir = tmp_path / "raw"
    payout_path = tmp_path / "payouts.csv"
    output_dir = tmp_path / "quinella_replay"
    _write_raw_file(
        raw_dir / "race" / "0B42_jvgets.txt",
        _o2_text(captured_month_day_time="05080940"),
        _o2_text(captured_month_day_time="05080955"),
    )
    write_payouts_csv(
        payout_path,
        [
            {
                "race_id": "2026050805010101",
                "runner_id": "2026050805010101-01_02",
                "bet_type": "quinella",
                "payout_jpy_per_100": 800,
            }
        ],
    )

    export = build_quinella_replay_dataset_from_raw(
        raw_dir,
        payout_path,
        output_dir,
        start_date=dt.date(2026, 5, 8),
        end_date=dt.date(2026, 5, 8),
    )

    latest_odds = [
        parse_odds_quote_row(row) for row in read_csv_rows(output_dir / "odds.csv")
    ]
    timeseries_rows = read_csv_rows(output_dir / "odds_timeseries.csv")

    assert len(latest_odds) == 1
    assert latest_odds[0].captured_at == dt.datetime(2026, 5, 8, 9, 55)
    assert timeseries_rows == []
    assert export.report.input_raw_files == 1
    assert export.report.input_records == 2
    assert export.report.input_odds == 2
    assert export.report.odds_timeseries_written == 0
