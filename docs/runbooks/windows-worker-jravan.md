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

## Automated daily fetch wrapper

Use `tools/windows/Invoke-JvLinkAutomatedFetchToS3.ps1` as the scheduled command when one job should fetch both accumulated `RACE` and realtime odds. It composes the existing S3-first scripts and writes a local `automated_fetch_manifest.json`.

Example:

```powershell
powershell.exe -ExecutionPolicy Bypass -File C:\horse-lab\scripts\Invoke-JvLinkAutomatedFetchToS3.ps1 `
  -StartDate 20260511 `
  -EndDate 20260511 `
  -DailyDataSpecs RACE `
  -RealtimeDataSpecs 0B30,0B41,0B42 `
  -RaceKeyFile C:\horse-lab\inputs\today_race_keys.txt `
  -Bucket horse-lab-jravan-244306245597-apne1 `
  -RunId auto_20260511_pm
```

Recommended EventBridge/SSM shape:

1. Start the Windows instance 10-15 minutes before collection.
2. Run the automated wrapper with a deterministic `RunId`.
3. Upload outputs to S3 through the wrapper scripts.
4. Stop the instance after the command finishes or after a fixed grace period.

For non-race days or settlement-only runs, pass `-SkipRealtime` and fetch `RACE` only. For intraday odds accumulation windows, pass `-SkipDaily` and run only `0B30/0B41/0B42` with the current race-key file.

## Windows Task Scheduler smoke operation

Use `tools/windows/Register-JvLinkAutomatedFetchTask.ps1` when the Windows worker is intentionally kept running and a local daily scheduled task is enough. Production-like operation should still prefer EventBridge Scheduler + SSM + EC2 start/stop, but Task Scheduler is useful for smoke tests and short operational trials.

Example:

```powershell
powershell.exe -ExecutionPolicy Bypass -File C:\horse-lab\scripts\Register-JvLinkAutomatedFetchTask.ps1 `
  -TaskName HorseLab-JvLink-AutomatedFetch `
  -ScriptPath C:\horse-lab\scripts\Invoke-JvLinkAutomatedFetchToS3.ps1 `
  -RaceKeyFile C:\horse-lab\inputs\today_race_keys.txt `
  -Bucket horse-lab-jravan-244306245597-apne1 `
  -DailyDataSpecs RACE `
  -RealtimeDataSpecs 0B30,0B41,0B42 `
  -At 21:30 `
  -WhatIf
```

Remove `-WhatIf` to register or replace the task. The wrapper uploads to S3 first and leaves local files only when the lower-level fetch scripts are called with keep-local options. For EC2 cost control, stop the instance after each collection window when the worker is not dumping data.
