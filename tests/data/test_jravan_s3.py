from pathlib import Path

import pytest

from horse_lab.data.jravan import (
    DEFAULT_JRAVAN_S3_BUCKET,
    build_jravan_s3_raw_sync_plan,
    render_sync_command,
    sync_jravan_raw_from_s3,
)


def test_build_jravan_s3_raw_sync_plan_uses_canonical_prefix():
    plan = build_jravan_s3_raw_sync_plan(
        run_id="finalized_20260504_retry",
        local_raw_root=Path("data/raw/jravan"),
    )

    assert plan.bucket == DEFAULT_JRAVAN_S3_BUCKET
    assert plan.prefix == "raw/jravan"
    assert plan.s3_uri == (
        "s3://horse-lab-jravan-244306245597-apne1/"
        "raw/jravan/finalized_20260504_retry/"
    )
    assert plan.local_dir == Path("data/raw/jravan/finalized_20260504_retry")
    assert plan.command == (
        "aws",
        "s3",
        "sync",
        plan.s3_uri,
        "data/raw/jravan/finalized_20260504_retry",
        "--only-show-errors",
    )


def test_build_jravan_s3_raw_sync_plan_rejects_unsafe_run_id():
    with pytest.raises(ValueError, match="must not contain"):
        build_jravan_s3_raw_sync_plan(
            run_id="../bad",
            local_raw_root=Path("data/raw/jravan"),
        )


def test_sync_jravan_raw_from_s3_dry_run_does_not_create_local_dir(tmp_path):
    plan = build_jravan_s3_raw_sync_plan(
        run_id="run-1",
        local_raw_root=tmp_path / "raw",
    )

    result = sync_jravan_raw_from_s3(plan, dry_run=True)

    assert result.executed is False
    assert result.returncode is None
    assert not plan.local_dir.exists()


def test_render_sync_command_quotes_spaces():
    rendered = render_sync_command(
        ("aws", "s3", "sync", "s3://bucket/raw/run/", "local path/run")
    )

    assert rendered == "aws s3 sync s3://bucket/raw/run/ 'local path/run'"
