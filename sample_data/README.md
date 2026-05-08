# Sample Data

These CSV files are tiny deterministic fixtures for the historical replay tests.

They are not real JRA records. The field names mirror the domain schemas so a future JRA-VAN adapter can map official records into the same repository contracts.

The fixture intentionally includes:

- two races on May 8, 2026;
- four runners;
- one future odds quote after replay `as_of`;
- two feature rows for the same runner so latest-at-or-before logic is tested;
- same-race win/loss settlement behavior for the backtest simulator.
