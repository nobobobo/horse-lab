# JRA-VAN Minimal RA/SE Mapper Design

## Purpose

Build the first real-data ingestion slice without requiring JRA-VAN JV-Link to run on this macOS development environment. The implementation should map small raw JV-Data fixture records into Horse Lab canonical schemas so later Windows-based JV-Link extraction can replace fixtures with real dumps.

This phase starts with the minimal stable subset: race detail (`RA`) and runner-level race information (`SE`). It should produce `Race`, `Entry`, and `Result` objects only. Odds, payouts, scratches, horse master data, and full JRA-VAN field coverage stay out of scope.

## Source And Environment Constraints

JRA-VAN Data Lab. remains the intended official production source. JV-Link is a Windows ActiveX COM component, and the current local environment is macOS arm64, so the project should not attempt to call JV-Link directly.

The boundary is:

```text
Windows JV-Link worker
        -> raw JV-Data text dump
        -> horse_lab.data.jravan raw parser and mapper
        -> canonical Race / Entry / Result schemas
        -> existing repositories and replay pipeline
```

Real licensed data must remain outside git. The repository should only include synthetic fixture records.

References:

- JRA-VAN Data Lab. SDK page: https://jra-van.jp/dlb/sdv/sdk.html
- JV-Data 仕様書 PDF: https://jra-van.jp/dlb/sdv/sdk/JV-Data4901.pdf
- JV-Link system overview: https://jra-van.jp/dlb/sdv/about.html
- MacOS JV-Link FAQ: https://jra-van.jp/dlb/sdv/faq.html

## Scope

In scope:

- Define minimal RA and SE fixed-width layouts for the canonical fields Horse Lab needs.
- Parse synthetic CP932 JV-Data fixture lines using byte offsets, not Python character offsets.
- Convert an `RA` record into a `Race`.
- Convert an `SE` record into an `Entry`.
- Convert an `SE` record into a `Result` only when official result fields are populated.
- Keep unknown or currently unmapped JRA-VAN codes in schema metadata where useful.
- Add tests that prove Japanese text and fixed-width byte offsets are handled correctly.

Out of scope:

- Calling JV-Link.
- Downloading or committing real JRA-VAN data.
- Full RA/SE field coverage.
- Odds (`O1` and related records).
- Payout records.
- Master records such as horse, jockey, trainer, breeder, or owner masters.
- Web scraping.
- CSV or Parquet export.

## Proposed File Structure

```text
src/horse_lab/data/jravan/
  __init__.py
  layouts.py
  mappers.py
  raw.py

tests/data/
  test_jravan_mappers.py
  test_jravan_raw.py
```

Existing `raw.py` remains responsible for low-level record envelopes and fixed-width byte extraction. New `layouts.py` should hold named field layouts. New `mappers.py` should hold canonical schema conversion.

## Minimal Field Contract

The implementation should not try to mirror the full JV-Data specification yet. It should define a small versioned layout profile that is enough to build canonical objects and can be replaced with official full layouts later.

### RA Minimal Layout

The synthetic RA fixture must include:

- `record_type`
- `data_kubun`
- `race_date`
- `venue_code`
- `kaiji`
- `nichiji`
- `race_number`
- `race_name`
- `surface_code`
- `distance_m`
- `direction_code`
- `track_condition_code`
- `weather_code`
- `grade_code`
- `start_time`
- `field_size`

Mapping behavior:

- `race_id` is built from `race_date`, `venue_code`, `kaiji`, `nichiji`, and `race_number`.
- `race_date` uses `YYYYMMDD`.
- `start_time` uses `HHMM`; the mapper combines it with `race_date`.
- `surface_code`, `direction_code`, `track_condition_code`, `weather_code`, `grade_code`, and `venue_code` are converted through small local code maps.
- Unknown codes map to `UNKNOWN` where the canonical enum supports it, or to metadata when no canonical enum exists.
- `Race.metadata` keeps raw JRA-VAN keys such as `venue_code`, `kaiji`, `nichiji`, and `data_kubun`.

### SE Minimal Layout

The synthetic SE fixture must include:

- `record_type`
- `data_kubun`
- `race_date`
- `venue_code`
- `kaiji`
- `nichiji`
- `race_number`
- `horse_number`
- `gate_number`
- `horse_id`
- `horse_name`
- `sex_code`
- `age`
- `trainer_id`
- `jockey_id`
- `carried_weight`
- `body_weight`
- `body_weight_diff_sign`
- `body_weight_diff`
- `finish_position`
- `is_disqualified`
- `is_dead_heat`
- `final_time_seconds`
- `prize_jpy`

Mapping behavior:

- `race_id` is built with the same rule as RA.
- `runner_id` is `race_id` plus a two-digit horse number suffix, matching existing sample-data style.
- `horse_id`, `jockey_id`, and `trainer_id` map to their canonical NewType wrappers when populated.
- `carried_weight` is represented as kilograms with one decimal place when the raw value uses deci-kilograms.
- Body-weight difference applies the sign field. A blank difference maps to `None`; a raw zero difference maps to `0`; a `-` sign negates the parsed value; `+`, blank, or `0` signs leave it non-negative.
- `Entry.metadata` keeps `horse_name`, `sex_code`, and `data_kubun`.
- `Result` is returned only when `finish_position` is populated. If result fields are blank, the mapper returns `None`.

## API Design

`layouts.py` should expose named field lists and helper parsing:

```python
JRAVAN_MINIMAL_RA_FIELDS: tuple[FixedWidthField, ...]
JRAVAN_MINIMAL_SE_FIELDS: tuple[FixedWidthField, ...]
parse_minimal_ra_fields(record: JvDataRecord) -> dict[str, str]
parse_minimal_se_fields(record: JvDataRecord) -> dict[str, str]
```

`mappers.py` should expose:

```python
build_jravan_race_id(fields: Mapping[str, str]) -> RaceId
build_jravan_runner_id(fields: Mapping[str, str]) -> RunnerId
map_ra_record_to_race(record: JvDataRecord) -> Race
map_se_record_to_entry(record: JvDataRecord) -> Entry
map_se_record_to_result(record: JvDataRecord) -> Result | None
```

The mappers should validate record type. Passing an `SE` record to the RA mapper, or an `RA` record to the SE mapper, should raise `ValueError` with a clear message.

## Data Quality And Error Handling

The mappers should fail fast when required canonical fields are missing or malformed:

- invalid dates;
- invalid race numbers;
- invalid distance;
- invalid horse numbers;
- invalid finish position values;
- unsupported record type for the mapper.

Unknown JRA-VAN categorical codes should not fail the ingestion slice unless a canonical schema cannot represent them. Unknown surface, direction, and track-condition codes should map to canonical `UNKNOWN` enum values and preserve raw codes in metadata.

## Testing

Tests should use synthetic CP932 fixture builders, not real JRA-VAN data.

Required tests:

- RA fixture maps to a `Race` with expected `race_id`, race date, venue, race number, name, surface, distance, direction, track condition, weather, grade, start time, and field size.
- SE fixture maps to an `Entry` with expected `runner_id`, `race_id`, horse number, gate number, IDs, carried weight, body weight, body-weight diff, age, and metadata.
- SE fixture with populated result fields maps to a `Result`.
- SE fixture with blank result fields returns `None` for result.
- Mapper rejects wrong record types.
- Fixed-width fixture builder includes Japanese text so tests prove byte offsets are still correct with CP932 multibyte characters.

The full suite must continue to run without network access, JV-Link, Windows, or paid credentials.

## Future Extensions

After this slice:

1. Add staging CSV export so RA/SE mapped objects can be written into the existing CSV repository format.
2. Add Windows JV-Link worker contract and manifest files.
3. Add odds record mapping for point-in-time `OddsQuote`.
4. Replace minimal synthetic layouts with complete official layout definitions when full-field ingestion becomes necessary.
