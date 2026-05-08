"""JRA-VAN raw-data ingestion helpers.

The JV-Link extraction process itself is expected to run on Windows. This
package keeps the platform-independent parsing and normalization boundary in
Python so the rest of horse_lab can run on macOS/Linux.
"""

from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
    parse_jvdata_record,
    read_jvdata_records,
)

__all__ = [
    "FixedWidthField",
    "JvDataRecord",
    "parse_fixed_width_fields",
    "parse_jvdata_record",
    "read_jvdata_records",
]
