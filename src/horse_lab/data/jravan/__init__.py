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
from horse_lab.data.jravan.exporters import (
    ENTRY_CSV_FIELDS,
    RACE_CSV_FIELDS,
    RESULT_CSV_FIELDS,
    entry_to_csv_row,
    race_to_csv_row,
    result_to_csv_row,
    write_entries_csv,
    write_races_csv,
    write_results_csv,
    write_staging_csvs,
)
from horse_lab.data.jravan.mappers import (
    build_jravan_race_id,
    build_jravan_runner_id,
    map_ra_record_to_race,
    map_se_record_to_entry,
    map_se_record_to_result,
)
from horse_lab.data.jravan.raw import (
    FixedWidthField,
    JvDataRecord,
    parse_fixed_width_fields,
    parse_jvdata_record,
    read_jvdata_records,
)

__all__ = [
    "ENTRY_CSV_FIELDS",
    "FixedWidthField",
    "JRAVAN_MINIMAL_RA_FIELDS",
    "JRAVAN_MINIMAL_SE_FIELDS",
    "JvDataRecord",
    "RACE_CSV_FIELDS",
    "RESULT_CSV_FIELDS",
    "build_jravan_race_id",
    "build_jravan_runner_id",
    "entry_to_csv_row",
    "map_ra_record_to_race",
    "map_se_record_to_entry",
    "map_se_record_to_result",
    "parse_fixed_width_fields",
    "parse_jvdata_record",
    "parse_minimal_ra_fields",
    "parse_minimal_se_fields",
    "race_to_csv_row",
    "read_jvdata_records",
    "result_to_csv_row",
    "write_entries_csv",
    "write_races_csv",
    "write_results_csv",
    "write_staging_csvs",
]
