"""JRA-VAN raw-data ingestion helpers.

The JV-Link extraction process itself is expected to run on Windows. This
package keeps the platform-independent parsing and normalization boundary in
Python so the rest of horse_lab can run on macOS/Linux.
"""

from horse_lab.data.jravan.layouts import (
    JRAVAN_MINIMAL_RA_FIELDS,
    JRAVAN_MINIMAL_SE_FIELDS,
    parse_minimal_ra_fields,
    parse_minimal_se_fields,
)
from horse_lab.data.jravan.mappers import (
    build_jravan_race_id,
    map_ra_record_to_race,
)
from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
    parse_jvdata_record,
    read_jvdata_records,
)

__all__ = [
    "FixedWidthField",
    "JRAVAN_MINIMAL_RA_FIELDS",
    "JRAVAN_MINIMAL_SE_FIELDS",
    "JvDataRecord",
    "build_jravan_race_id",
    "map_ra_record_to_race",
    "parse_fixed_width_fields",
    "parse_jvdata_record",
    "parse_minimal_ra_fields",
    "parse_minimal_se_fields",
    "read_jvdata_records",
]
