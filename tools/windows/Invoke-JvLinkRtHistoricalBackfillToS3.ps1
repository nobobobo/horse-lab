param(
    [string]$RtRaceListToS3ScriptPath = "C:\horse-lab\scripts\Invoke-JvLinkRtRaceListToS3.ps1",

    [string]$RaceKeyFile = "",

    [string]$RaceKeyCsv = "",

    [string]$RaceKeyColumn = "",

    [string]$RaceDateColumn = "",

    [string[]]$DataSpecs = @("0B41", "0B42"),

    [string]$StartDate = "",

    [string]$EndDate = "",

    [ValidateSet("Weekly", "Monthly")]
    [string]$ChunkType = "Weekly",

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string]$S3Prefix = "raw/jravan/realtime-backfill",

    [string]$RunIdPrefix = "",

    [string]$WorkRoot = "C:\horse-lab\work\jravan-rt-backfill",

    [int]$MaxReadIterations = 5000,

    [switch]$UseFileByFileUpload,

    [switch]$StopOnError,

    [switch]$KeepLocal,

    [switch]$KeepBatchLogs,

    [switch]$IncludeFailureLogTail,

    [switch]$SummaryOnly
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Normalize-DateOnly {
    param(
        [string]$Value,
        [string]$Name
    )

    $trimmed = $Value.Trim()
    if ($trimmed -match '^\d{8}$') {
        return [datetime]::ParseExact($trimmed, "yyyyMMdd", $null)
    }
    if ($trimmed -match '^\d{4}-\d{2}-\d{2}$') {
        return [datetime]::ParseExact($trimmed, "yyyy-MM-dd", $null)
    }
    throw "${Name} must be YYYYMMDD or YYYY-MM-DD, got '${Value}'"
}

function Get-RaceDateFromKey {
    param([string]$RaceKey)

    $trimmed = $RaceKey.Trim()
    if ($trimmed.Length -lt 8 -or $trimmed.Substring(0, 8) -notmatch '^\d{8}$') {
        throw "Race key must start with YYYYMMDD when no race-date column is available: ${RaceKey}"
    }
    return [datetime]::ParseExact($trimmed.Substring(0, 8), "yyyyMMdd", $null)
}

function Get-FirstExistingPropertyName {
    param(
        [object]$Row,
        [string[]]$Names
    )

    $propertyNames = @($Row.PSObject.Properties | ForEach-Object { $_.Name })
    foreach ($name in $Names) {
        if ($propertyNames -contains $name) {
            return $name
        }
    }
    return ""
}

function Read-RaceKeyRows {
    $rows = New-Object System.Collections.Generic.List[object]

    if (-not [string]::IsNullOrWhiteSpace($RaceKeyFile)) {
        Get-Content -LiteralPath $RaceKeyFile |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object {
                $key = $_.Trim()
                $rows.Add([pscustomobject]@{
                    raceKey = $key
                    raceDate = Get-RaceDateFromKey -RaceKey $key
                }) | Out-Null
            }
    }

    if (-not [string]::IsNullOrWhiteSpace($RaceKeyCsv)) {
        $csvRows = @(Import-Csv -LiteralPath $RaceKeyCsv)
        foreach ($row in $csvRows) {
            $keyColumn = $RaceKeyColumn
            if ([string]::IsNullOrWhiteSpace($keyColumn)) {
                $keyColumn = Get-FirstExistingPropertyName -Row $row -Names @("race_key", "raceKey", "RaceKey", "key", "Key")
            }
            if ([string]::IsNullOrWhiteSpace($keyColumn)) {
                throw "Race key column was not found. Pass -RaceKeyColumn."
            }

            $key = [string]$row.PSObject.Properties[$keyColumn].Value
            if ([string]::IsNullOrWhiteSpace($key)) {
                continue
            }

            $dateColumn = $RaceDateColumn
            if ([string]::IsNullOrWhiteSpace($dateColumn)) {
                $dateColumn = Get-FirstExistingPropertyName -Row $row -Names @("race_date", "raceDate", "RaceDate", "date", "Date")
            }

            $raceDate = if ([string]::IsNullOrWhiteSpace($dateColumn)) {
                Get-RaceDateFromKey -RaceKey $key
            }
            else {
                Normalize-DateOnly -Value ([string]$row.PSObject.Properties[$dateColumn].Value) -Name $dateColumn
            }

            $rows.Add([pscustomobject]@{
                raceKey = $key.Trim()
                raceDate = $raceDate
            }) | Out-Null
        }
    }

    return $rows
}

function Get-LogTail {
    param(
        [string]$Path,
        [int]$Count = 30
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    return ((Get-Content -LiteralPath $Path -Tail $Count -ErrorAction SilentlyContinue) -join [Environment]::NewLine)
}

function Get-LastExitCodeOrDefault {
    param([int]$Default)

    $lastExitCodeVariable = Get-Variable -Name LASTEXITCODE -ErrorAction SilentlyContinue
    if (($null -eq $lastExitCodeVariable) -or ($null -eq $lastExitCodeVariable.Value)) {
        return $Default
    }
    return [int]$lastExitCodeVariable.Value
}

if (-not (Test-Path -LiteralPath $RtRaceListToS3ScriptPath)) {
    throw "Realtime race-list-to-S3 script was not found: $RtRaceListToS3ScriptPath"
}

if ([string]::IsNullOrWhiteSpace($RaceKeyFile) -and [string]::IsNullOrWhiteSpace($RaceKeyCsv)) {
    throw "Pass -RaceKeyFile, -RaceKeyCsv, or both."
}

$allRows = @(Read-RaceKeyRows)
if ($allRows.Count -eq 0) {
    throw "No race keys were loaded."
}

$sortedRows = @($allRows | Sort-Object raceDate, raceKey)
$minDate = $sortedRows[0].raceDate
$maxDate = $sortedRows[$sortedRows.Count - 1].raceDate
$start = if ([string]::IsNullOrWhiteSpace($StartDate)) { [datetime]$minDate } else { Normalize-DateOnly -Value $StartDate -Name "StartDate" }
$end = if ([string]::IsNullOrWhiteSpace($EndDate)) { [datetime]$maxDate } else { Normalize-DateOnly -Value $EndDate -Name "EndDate" }
if ($start -gt $end) {
    throw "StartDate must be on or before EndDate"
}

$filteredRows = @(
    $allRows |
        Where-Object { $_.raceDate.Date -ge $start.Date -and $_.raceDate.Date -le $end.Date } |
        Sort-Object raceDate, raceKey -Unique
)
if ($filteredRows.Count -eq 0) {
    throw "No race keys matched the requested date range."
}

if ([string]::IsNullOrWhiteSpace($RunIdPrefix)) {
    $safeSpecs = ($DataSpecs -join "_") -replace '[^0-9A-Za-z_-]', '_'
    $RunIdPrefix = "rtbackfill_${safeSpecs}_$($start.ToString('yyyyMMdd'))_$($end.ToString('yyyyMMdd'))"
}

$batchRunId = $RunIdPrefix -replace '[^0-9A-Za-z_-]', '_'
$logRoot = Join-Path (Join-Path $WorkRoot "batch_logs") $batchRunId
$keyRoot = Join-Path (Join-Path $WorkRoot "batch_keys") $batchRunId
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
New-Item -ItemType Directory -Force -Path $keyRoot | Out-Null

$startedAt = Get-Date
$results = New-Object System.Collections.Generic.List[object]
$failedCount = 0
$completedCount = 0

:chunkLoop for ($chunkStart = $start.Date; $chunkStart -le $end.Date;) {
    if ($ChunkType -eq "Monthly") {
        $monthStart = New-Object DateTime -ArgumentList $chunkStart.Year, $chunkStart.Month, 1
        $monthEnd = $monthStart.AddMonths(1).AddDays(-1)
        $chunkEnd = if ($monthEnd -lt $end.Date) { $monthEnd } else { $end.Date }
        $nextStart = $chunkEnd.AddDays(1)
    }
    else {
        $weekEnd = $chunkStart.AddDays(6)
        $chunkEnd = if ($weekEnd -lt $end.Date) { $weekEnd } else { $end.Date }
        $nextStart = $chunkEnd.AddDays(1)
    }

    $chunkKeys = @(
        $filteredRows |
            Where-Object { $_.raceDate.Date -ge $chunkStart -and $_.raceDate.Date -le $chunkEnd } |
            ForEach-Object { $_.raceKey } |
            Sort-Object -Unique
    )

    if ($chunkKeys.Count -eq 0) {
        $chunkStart = $nextStart
        continue
    }

    $chunkName = "$($chunkStart.ToString('yyyyMMdd'))_$($chunkEnd.ToString('yyyyMMdd'))"
    $runId = "${batchRunId}_${chunkName}"
    $keyFilePath = Join-Path $keyRoot "${runId}.keys.txt"
    $logPath = Join-Path $logRoot "${runId}.log"
    Set-Content -LiteralPath $keyFilePath -Value $chunkKeys -Encoding ASCII

    $childParams = @{
        RaceKeyFile = $keyFilePath
        DataSpecs = $DataSpecs
        Bucket = $Bucket
        S3Prefix = $S3Prefix
        WorkRoot = $WorkRoot
        RunId = $runId
        MaxReadIterations = $MaxReadIterations
    }
    if ($KeepLocal) {
        $childParams.KeepLocal = $true
    }
    if ($UseFileByFileUpload) {
        $childParams.UseFileByFileUpload = $true
    }

    $runStartedAt = Get-Date
    $status = "completed"
    $exitCode = 0
    $errorMessage = ""

    try {
        & $RtRaceListToS3ScriptPath @childParams *> $logPath
        $exitCode = Get-LastExitCodeOrDefault -Default 0
        if ($exitCode -ne 0) {
            throw "Realtime race-list-to-S3 exited with code $exitCode"
        }
        $completedCount += 1
    }
    catch {
        $status = "failed"
        $exitCode = Get-LastExitCodeOrDefault -Default 1
        if ($exitCode -eq 0) {
            $exitCode = 1
        }
        $errorMessage = $_.Exception.Message
        $failedCount += 1
    }

    $runFinishedAt = Get-Date
    $results.Add([pscustomobject]@{
        chunkStartDate = $chunkStart.ToString("yyyyMMdd")
        chunkEndDate = $chunkEnd.ToString("yyyyMMdd")
        raceKeyCount = $chunkKeys.Count
        dataSpecs = $DataSpecs
        runId = $runId
        status = $status
        exitCode = $exitCode
        error = $errorMessage
        s3Prefix = ($S3Prefix.Trim("/") + "/" + $runId).Trim("/")
        startedAt = $runStartedAt.ToUniversalTime().ToString("o")
        finishedAt = $runFinishedAt.ToUniversalTime().ToString("o")
        keyFilePath = if ($KeepBatchLogs) { $keyFilePath } else { "" }
        logPath = if ($KeepBatchLogs) { $logPath } else { "" }
        logTail = if (($status -ne "completed") -and $IncludeFailureLogTail) { Get-LogTail -Path $logPath } else { "" }
    }) | Out-Null

    if (-not $KeepBatchLogs) {
        if (Test-Path -LiteralPath $keyFilePath) {
            Remove-Item -LiteralPath $keyFilePath -Force
        }
        if (Test-Path -LiteralPath $logPath) {
            Remove-Item -LiteralPath $logPath -Force
        }
    }

    if (($status -ne "completed") -and $StopOnError) {
        break chunkLoop
    }

    $chunkStart = $nextStart
}

if (-not $KeepBatchLogs) {
    foreach ($path in @($keyRoot, (Join-Path $WorkRoot "batch_keys"), $logRoot, (Join-Path $WorkRoot "batch_logs"))) {
        if (Test-Path -LiteralPath $path) {
            $remaining = Get-ChildItem -LiteralPath $path -Force -ErrorAction SilentlyContinue
            if (-not $remaining) {
                Remove-Item -LiteralPath $path -Force
            }
        }
    }
}

$finishedAt = Get-Date
$summary = [ordered]@{
    batchRunId = $batchRunId
    dataSpecs = $DataSpecs
    startDate = $start.ToString("yyyyMMdd")
    endDate = $end.ToString("yyyyMMdd")
    chunkType = $ChunkType
    raceKeyCount = $filteredRows.Count
    attemptedChunks = $results.Count
    completedChunks = $completedCount
    failedChunks = $failedCount
    bucket = $Bucket
    s3Prefix = $S3Prefix.Trim("/")
    uploadMode = if ($UseFileByFileUpload) { "file" } else { "sync" }
    keepLocal = [bool]$KeepLocal
    keepBatchLogs = [bool]$KeepBatchLogs
    summaryOnly = [bool]$SummaryOnly
    startedAt = $startedAt.ToUniversalTime().ToString("o")
    finishedAt = $finishedAt.ToUniversalTime().ToString("o")
}

if ($SummaryOnly) {
    $summary.completedRunIds = @($results | Where-Object { $_.status -eq "completed" } | ForEach-Object { $_.runId })
    $summary.failed = @(
        $results |
            Where-Object { $_.status -ne "completed" } |
            ForEach-Object {
                [pscustomobject]@{
                    chunkStartDate = $_.chunkStartDate
                    chunkEndDate = $_.chunkEndDate
                    runId = $_.runId
                    error = $_.error
                }
            }
    )
}
else {
    $summary.results = $results
}

[pscustomobject]$summary | ConvertTo-Json -Depth 8

if (($failedCount -gt 0) -and $StopOnError) {
    throw "Realtime historical backfill stopped after $failedCount failed chunk(s)"
}
