"""S3 path helpers for JRA-VAN raw artifact handoff."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_JRAVAN_S3_BUCKET = "horse-lab-jravan-244306245597-apne1"
DEFAULT_JRAVAN_S3_RAW_PREFIX = "raw/jravan"


@dataclass(frozen=True)
class JraVanS3RawSyncPlan:
    bucket: str
    prefix: str
    run_id: str
    s3_uri: str
    local_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class JraVanS3RawSyncResult:
    plan: JraVanS3RawSyncPlan
    executed: bool
    returncode: int | None


def build_jravan_s3_raw_sync_plan(
    *,
    run_id: str,
    local_raw_root: Path | str,
    bucket: str = DEFAULT_JRAVAN_S3_BUCKET,
    prefix: str = DEFAULT_JRAVAN_S3_RAW_PREFIX,
    aws_cli: str = "aws",
) -> JraVanS3RawSyncPlan:
    normalized_run_id = _normalize_s3_path_component(run_id, "run_id")
    normalized_prefix = _normalize_s3_prefix(prefix)
    normalized_bucket = bucket.strip()
    if not normalized_bucket:
        raise ValueError("bucket must not be empty")
    local_dir = Path(local_raw_root) / normalized_run_id
    s3_uri = f"s3://{normalized_bucket}/{normalized_prefix}/{normalized_run_id}/"
    command = (
        aws_cli,
        "s3",
        "sync",
        s3_uri,
        str(local_dir),
        "--only-show-errors",
    )
    return JraVanS3RawSyncPlan(
        bucket=normalized_bucket,
        prefix=normalized_prefix,
        run_id=normalized_run_id,
        s3_uri=s3_uri,
        local_dir=local_dir,
        command=command,
    )


def sync_jravan_raw_from_s3(
    plan: JraVanS3RawSyncPlan,
    *,
    dry_run: bool = False,
) -> JraVanS3RawSyncResult:
    if dry_run:
        return JraVanS3RawSyncResult(
            plan=plan,
            executed=False,
            returncode=None,
        )

    plan.local_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(plan.command, check=True)
    return JraVanS3RawSyncResult(
        plan=plan,
        executed=True,
        returncode=completed.returncode,
    )


def render_sync_command(command: Sequence[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _normalize_s3_prefix(prefix: str) -> str:
    normalized = prefix.strip().strip("/")
    if not normalized:
        raise ValueError("prefix must not be empty")
    return normalized


def _normalize_s3_path_component(value: str, name: str) -> str:
    normalized = value.strip().strip("/")
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if ".." in normalized.split("/"):
        raise ValueError(f"{name} must not contain '..'")
    return normalized


def _shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "/._:-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
