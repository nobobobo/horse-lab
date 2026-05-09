import datetime as dt
import json
from pathlib import Path

from horse_lab.cli import main
from horse_lab.data.csv_parsing import (
    parse_feature_row,
    parse_odds_quote_row,
    parse_race_row,
    read_csv_rows,
)
from horse_lab.data.jravan import (
    REPLAY_REPORT_FILENAME,
    build_replay_dataset_from_staging,
    replay_dataset_report_to_dict,
    write_staging_csvs,
)
from horse_lab.schemas import (
    BetType,
    FeatureName,
    HorseId,
    Entry,
    OddsQuote,
    Race,
    RaceId,
    Result,
    RunnerId,
    Surface,
)


def _race(race_id: str, *, field_size: int = 2) -> Race:
    race_date = dt.date(
        int(race_id[0:4]),
        int(race_id[4:6]),
        int(race_id[6:8]),
    )
    return Race(
        race_id=RaceId(race_id),
        race_date=race_date,
        venue="Tokyo",
        race_number=int(race_id[-2:]),
        name="Fixture",
        surface=Surface.TURF,
        distance_m=1600,
        start_time=dt.datetime.combine(race_date, dt.time(10, 0)),
        field_size=field_size,
    )


def _runner_id(race_id: str, horse_number: int) -> RunnerId:
    return RunnerId(f"{race_id}-{horse_number:02d}")


def _entry(race_id: str, horse_number: int) -> Entry:
    return Entry(
        race_id=RaceId(race_id),
        runner_id=_runner_id(race_id, horse_number),
        horse_id=HorseId(f"horse-{horse_number}"),
        horse_number=horse_number,
        gate_number=horse_number,
        carried_weight_kg=56.0 + horse_number,
        age=4,
        body_weight_kg=480 + horse_number,
    )


def _result(race_id: str, horse_number: int, finish_position: int) -> Result:
    return Result(
        race_id=RaceId(race_id),
        runner_id=_runner_id(race_id, horse_number),
        finish_position=finish_position,
    )


def _quote(
    race_id: str,
    horse_number: int,
    odds: float,
    captured_at: dt.datetime,
    *,
    bet_type: BetType = BetType.WIN,
) -> OddsQuote:
    return OddsQuote(
        race_id=RaceId(race_id),
        runner_id=_runner_id(race_id, horse_number),
        bet_type=bet_type,
        captured_at=captured_at,
        odds=odds,
        popularity_rank=horse_number,
        source="fixture",
    )


def _write_staging_fixture(staging_dir: Path) -> None:
    complete_race = "2026050805010101"
    missing_odds_race = "2026050805010102"
    write_staging_csvs(
        staging_dir,
        races=[_race(complete_race), _race(missing_odds_race)],
        entries=[
            _entry(complete_race, 1),
            _entry(complete_race, 2),
            _entry(missing_odds_race, 1),
            _entry(missing_odds_race, 2),
        ],
        results=[
            _result(complete_race, 1, 2),
            _result(complete_race, 2, 1),
            _result(missing_odds_race, 1, 1),
            _result(missing_odds_race, 2, 2),
        ],
        odds=[
            _quote(
                complete_race,
                1,
                4.0,
                dt.datetime(2026, 5, 8, 9, 40),
            ),
            _quote(
                complete_race,
                1,
                3.0,
                dt.datetime(2026, 5, 8, 9, 50),
            ),
            _quote(
                complete_race,
                1,
                1.8,
                dt.datetime(2026, 5, 8, 9, 50),
                bet_type=BetType.PLACE,
            ),
            _quote(
                complete_race,
                2,
                2.5,
                dt.datetime(2026, 5, 8, 9, 45),
            ),
            _quote(
                missing_odds_race,
                1,
                2.0,
                dt.datetime(2026, 5, 8, 9, 45),
            ),
        ],
    )


def test_build_replay_dataset_keeps_complete_races_and_latest_win_odds(tmp_path):
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "replay"
    _write_staging_fixture(staging_dir)

    export = build_replay_dataset_from_staging(
        staging_dir,
        output_dir,
        feature_version="fixture-replay-v1",
    )

    assert export.report.races_written == 1
    assert export.report.feature_rows_written == 2
    assert export.report.results_written == 2
    assert export.report.odds_written == 2
    assert [skipped.reason for skipped in export.report.skipped_races] == [
        "win_odds_runner_mismatch"
    ]

    races = [parse_race_row(row) for row in read_csv_rows(output_dir / "races.csv")]
    features = [
        parse_feature_row(row) for row in read_csv_rows(output_dir / "features.csv")
    ]
    odds = [
        parse_odds_quote_row(row) for row in read_csv_rows(output_dir / "odds.csv")
    ]
    odds_timeseries = [
        parse_odds_quote_row(row)
        for row in read_csv_rows(output_dir / "odds_timeseries.csv")
    ]
    payouts = read_csv_rows(output_dir / "payouts.csv")

    assert [race.race_id for race in races] == [RaceId("2026050805010101")]
    assert races[0].field_size == 2
    assert len(features) == 2
    assert {row.feature_version for row in features} == {"fixture-replay-v1"}
    assert features[0].values[FeatureName("horse_number")] == 1
    assert features[0].values[FeatureName("race_surface")] == "turf"
    assert features[0].values[FeatureName("race_distance_m")] == 1600
    assert features[0].values[FeatureName("race_field_size")] == 2
    assert features[0].values[FeatureName("past_run_count")] == 0
    assert features[0].values[FeatureName("starter")] is True
    assert {
        (quote.runner_id, quote.bet_type, quote.odds) for quote in odds
    } == {
        (RunnerId("2026050805010101-01"), BetType.WIN, 3.0),
        (RunnerId("2026050805010101-02"), BetType.WIN, 2.5),
    }
    assert {
        (quote.runner_id, quote.captured_at, quote.odds) for quote in odds_timeseries
    } == {
        (
            RunnerId("2026050805010101-01"),
            dt.datetime(2026, 5, 8, 9, 40),
            4.0,
        ),
        (
            RunnerId("2026050805010101-01"),
            dt.datetime(2026, 5, 8, 9, 50),
            3.0,
        ),
        (
            RunnerId("2026050805010101-02"),
            dt.datetime(2026, 5, 8, 9, 45),
            2.5,
        ),
    }
    assert {
        (row["runner_id"], row["is_win"], row["payout_jpy_per_100"])
        for row in payouts
    } == {
        ("2026050805010101-01", "false", "0"),
        ("2026050805010101-02", "true", "250"),
    }
    assert features[0].values[FeatureName("odds_open")] == 4.0
    assert features[0].values[FeatureName("odds_latest")] == 3.0
    assert features[0].values[FeatureName("odds_snapshot_count")] == 2
    assert features[0].values[FeatureName("odds_change_open_to_latest")] == -1.0
    assert features[0].values[
        FeatureName("implied_probability_change_open_to_latest")
    ] == (1 / 3.0) - (1 / 4.0)

    report = json.loads((output_dir / REPLAY_REPORT_FILENAME).read_text())
    assert report == replay_dataset_report_to_dict(export.report)
    assert report["output_counts"]["odds_timeseries"] == 3
    assert report["output_counts"]["payouts"] == 2


def test_build_replay_dataset_respects_max_odds_captured_at(tmp_path):
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "replay"
    _write_staging_fixture(staging_dir)

    export = build_replay_dataset_from_staging(
        staging_dir,
        output_dir,
        max_odds_captured_at=dt.datetime(2026, 5, 8, 9, 47),
    )

    odds = [
        parse_odds_quote_row(row) for row in read_csv_rows(output_dir / "odds.csv")
    ]

    assert export.report.races_written == 1
    assert {
        (quote.runner_id, quote.odds) for quote in odds
    } == {
        (RunnerId("2026050805010101-01"), 4.0),
        (RunnerId("2026050805010101-02"), 2.5),
    }


def test_build_replay_dataset_adds_past_performance_features(tmp_path):
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "replay"
    race1 = "2026050105010101"
    race2 = "2026050805010102"
    write_staging_csvs(
        staging_dir,
        races=[_race(race1, field_size=1), _race(race2, field_size=1)],
        entries=[_entry(race1, 1), _entry(race2, 1)],
        results=[_result(race1, 1, 1), _result(race2, 1, 2)],
        odds=[
            _quote(race1, 1, 4.0, dt.datetime(2026, 5, 1, 9, 50)),
            _quote(race2, 1, 3.0, dt.datetime(2026, 5, 8, 9, 50)),
        ],
    )

    build_replay_dataset_from_staging(staging_dir, output_dir)

    features = [
        parse_feature_row(row) for row in read_csv_rows(output_dir / "features.csv")
    ]
    by_runner = {row.runner_id: row for row in features}

    first_runner = by_runner[RunnerId("2026050105010101-01")]
    second_runner = by_runner[RunnerId("2026050805010102-01")]

    assert first_runner.values[FeatureName("past_run_count")] == 0
    assert second_runner.values[FeatureName("past_run_count")] == 1
    assert second_runner.values[FeatureName("last_finish_position")] == 1
    assert second_runner.values[FeatureName("last_odds")] == 4.0


def test_build_replay_dataset_renders_person_ids_as_categorical_strings(tmp_path):
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "replay"
    race_id = "2026050805010101"
    entry = Entry(
        race_id=RaceId(race_id),
        runner_id=_runner_id(race_id, 1),
        horse_id=HorseId("horse-1"),
        horse_number=1,
        gate_number=1,
        jockey_id="01020",
        trainer_id="04050",
    )
    write_staging_csvs(
        staging_dir,
        races=[_race(race_id, field_size=1)],
        entries=[entry],
        results=[_result(race_id, 1, 1)],
        odds=[_quote(race_id, 1, 3.0, dt.datetime(2026, 5, 8, 9, 50))],
    )

    build_replay_dataset_from_staging(staging_dir, output_dir)

    features = [
        parse_feature_row(row) for row in read_csv_rows(output_dir / "features.csv")
    ]

    assert features[0].values[FeatureName("jockey_id")] == "jockey:01020"
    assert features[0].values[FeatureName("trainer_id")] == "trainer:04050"


def test_jravan_build_replay_dataset_cli_writes_json_summary(tmp_path, capsys):
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "replay"
    _write_staging_fixture(staging_dir)

    assert (
        main(
            [
                "jravan-build-replay-dataset",
                str(staging_dir),
                str(output_dir),
                "--feature-version",
                "fixture-replay-v1",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["report"]["output_counts"]["races"] == 1
    assert summary["report"]["output_counts"]["feature_rows"] == 2
    assert Path(summary["report_path"]).exists()
