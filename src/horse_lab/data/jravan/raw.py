"""Platform-independent helpers for JRA-VAN JV-Data raw records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


JV_DATA_ENCODING = "cp932"


@dataclass(frozen=True)
class JvDataRecord:
    """One raw JV-Data text record emitted by a Windows JV-Link worker."""

    record_type: str
    text: str
    line_number: int | None = None
    source_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedWidthField:
    """A 1-based byte-position field from a JRA-VAN fixed-width spec table."""

    name: str
    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start <= 0:
            raise ValueError("start must be a 1-based positive byte offset")
        if self.length <= 0:
            raise ValueError("length must be positive")


def parse_jvdata_record(
    line: str,
    *,
    line_number: int | None = None,
    source_path: Path | None = None,
) -> JvDataRecord:
    """Parse one raw JV-Data line into a stable envelope.

    JRA-VAN record-specific field parsing is intentionally separate from this
    envelope parser. Most JV-Data record specs use the first two characters as
    a record identifier, so this function only extracts that common boundary.
    """

    text = line.rstrip("\r\n")
    if len(text) < 2:
        raise ValueError("JV-Data record line must contain at least 2 characters")
    return JvDataRecord(
        record_type=text[:2],
        text=text,
        line_number=line_number,
        source_path=source_path,
    )


def read_jvdata_records(
    path: Path,
    *,
    encoding: str = JV_DATA_ENCODING,
) -> list[JvDataRecord]:
    """Read raw JV-Data records from a text dump produced by JV-Link."""

    records: list[JvDataRecord] = []
    with path.open(encoding=encoding, newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records.append(
                parse_jvdata_record(
                    line,
                    line_number=line_number,
                    source_path=path,
                )
            )
    return records


def parse_fixed_width_fields(
    raw: str | bytes,
    fields: Iterable[FixedWidthField],
    *,
    encoding: str = JV_DATA_ENCODING,
) -> dict[str, str]:
    """Parse 1-based byte-position fixed-width fields.

    JV-Data specs are byte-oriented and often contain Japanese text. Parsing by
    bytes keeps field positions stable even when a decoded string has multibyte
    characters.
    """

    raw_bytes = raw if isinstance(raw, bytes) else raw.encode(encoding)
    values: dict[str, str] = {}
    for field in fields:
        start_index = field.start - 1
        end_index = start_index + field.length
        values[field.name] = raw_bytes[start_index:end_index].decode(encoding).strip()
    return values
