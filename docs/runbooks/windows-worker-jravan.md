# Windows worker JRA-VAN runbook

## Realtime historical backfill: 0B41 / 0B42

Use `tools/windows/Invoke-JvLinkRtHistoricalBackfillToS3.ps1` when backfilling realtime race-key based data for the past year. The script splits race keys into weekly or monthly chunks, writes a temporary key file per chunk, runs `Invoke-JvLinkRtRaceListToS3.ps1`, and uploads raw outputs to S3.

Recommended first pass:

```powershell
powershell.exe -ExecutionPolicy Bypass -File C:\horse-lab\scripts\Invoke-JvLinkRtHistoricalBackfillToS3.ps1 `
  -RaceKeyCsv C:\horse-lab\inputs\race_keys_20250509_20260509.csv `
  -RaceKeyColumn race_key `
  -StartDate 20250509 `
  -EndDate 20260509 `
  -ChunkType Weekly `
  -DataSpecs 0B41,0B42 `
  -Bucket horse-lab-jravan-244306245597-apne1 `
  -S3Prefix raw/jravan/realtime-backfill `
  -KeepBatchLogs `
  -SummaryOnly
```

For a stable retry after weekly smoke runs, use `-ChunkType Monthly` to reduce scheduler overhead. Keep weekly chunks when JV-Link is unstable, race key count is high, or partial retry cost matters.

Inputs can be either:

- Plain `-RaceKeyFile`, one race key per line. The key must start with `YYYYMMDD`.
- `-RaceKeyCsv`, with `race_key` / `raceKey` / `RaceKey` / `key` / `Key`. A date column is optional if the key starts with `YYYYMMDD`.

The realtime wrapper now uses `aws s3 sync` by default through `Invoke-S3RawUpload.ps1 -UseSync`. Pass `-UseFileByFileUpload` only when debugging AWS CLI sync behavior.

## 0B30 periodic accumulation

Prefer EventBridge Scheduler + SSM + EC2 start/stop over Windows Task Scheduler for production-like operation.

Reasons:

- The worker can stay stopped except during collection windows, which lowers EC2 cost and avoids Windows drift.
- SSM Run Command gives central logs, exit status, IAM-controlled execution, and a cleaner retry path than an interactive Windows scheduled task.
- EventBridge can model multiple collection windows around race days and can start the instance before the first run, execute the PowerShell command, then stop it after a grace period.
- Task Scheduler is acceptable for local smoke tests or an always-on worker, but it hides failures on the instance and requires more manual monitoring.

Operational shape:

1. EventBridge Scheduler starts the Windows EC2 instance before the target collection window.
2. SSM Run Command invokes the realtime wrapper for `0B30` with the current race key file.
3. Raw files, logs, stdout/stderr, and manifest are uploaded under `s3://<bucket>/raw/jravan/realtime/<run_id>/`.
4. A second EventBridge/SSM step stops the instance after upload, with CloudWatch alarm/notification on non-zero command status.

Example SSM command payload:

```powershell
powershell.exe -ExecutionPolicy Bypass -File C:\horse-lab\scripts\Invoke-JvLinkRtRaceListToS3.ps1 `
  -RaceKeyFile C:\horse-lab\inputs\today_race_keys.txt `
  -DataSpecs 0B30 `
  -Bucket horse-lab-jravan-244306245597-apne1 `
  -S3Prefix raw/jravan/realtime `
  -MaxReadIterations 5000
```

Use a deterministic `RunId` from the schedule window when replayability matters, for example `0B30_yyyyMMdd_HHmm`.
