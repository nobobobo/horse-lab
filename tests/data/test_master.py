import datetime as dt

from horse_lab.data.master import (
    index_rating_history,
    latest_rating_as_of,
    read_horse_master_csv,
    read_horse_rating_history_csv,
)
from horse_lab.schemas import HorseId


def test_read_horse_master_csv_preserves_pedigree_and_metadata(tmp_path):
    path = tmp_path / "horse_master.csv"
    path.write_text(
        "\n".join(
            [
                "horse_id,horse_name,birth_date,sire_id,dam_id,damsire_id,family",
                "horse-1,Test Horse,2022-03-01,sire-1,dam-1,damsire-1,1-l",
            ]
        ),
        encoding="utf-8",
    )

    records = read_horse_master_csv(path)

    record = records[HorseId("horse-1")]
    assert record.horse_name == "Test Horse"
    assert record.birth_date == dt.date(2022, 3, 1)
    assert record.sire_id == HorseId("sire-1")
    assert record.dam_id == HorseId("dam-1")
    assert record.damsire_id == HorseId("damsire-1")
    assert record.metadata == {"family": "1-l"}


def test_rating_history_uses_latest_record_before_as_of(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text(
        "\n".join(
            [
                "horse_id,as_of,rating,source,note",
                "horse-1,2026-05-07T09:00:00,71.5,official,before",
                "horse-1,2026-05-08T11:00:00,80.0,official,after",
            ]
        ),
        encoding="utf-8",
    )

    ratings = index_rating_history(read_horse_rating_history_csv(path))
    latest = latest_rating_as_of(
        ratings[HorseId("horse-1")],
        as_of=dt.datetime(2026, 5, 8, 9, 50),
    )

    assert latest is not None
    assert latest.rating == 71.5
    assert latest.source == "official"
    assert latest.metadata == {"note": "before"}
