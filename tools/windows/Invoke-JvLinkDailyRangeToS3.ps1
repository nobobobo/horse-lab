param(
    [string]$DumpToS3ScriptPath = "C:\horse-lab\scripts\Invoke-JvLinkDumpToS3.ps1",

    [string[]]$DataSpecs = @("RACE"),

    [Parameter(Mandatory = $true)]
    [string]$StartDate,

    [Parameter(Mandatory = $true)]
    [string]$EndDate,

    [int]$Option = 1,

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string]$S3Prefix = "raw/jravan/daily",

    [string]$RunIdPrefix = "",

    [string]$WorkRoot = "C:\horse-lab\work\jravan-daily",

    [int]$MaxReadIterations = 1000000,

    [int]$BufferSize = 110000,

    [int]$LogEveryChunks = 100,

    [int]$FlushEveryChunks = 1000,

    [int]$DownloadWaitTimeoutSeconds = 900,

    [int]$DownloadPollSeconds = 2,

    [string]$FailedS3Prefix = "raw/jravan/failed",

    [switch]$StopOnError,

    [switch]$KeepFailedLocal,

    [switch]$KeepLocal,

    [switch]$KeepBatchLogs,

    [switch]$IncludeLogTail,

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

function Get-LogTail {
    param(
        [string]$Path,
        [int]$Count = 20
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return ""
    }
    return ((Get-Content -LiteralPath $Path -Tail $Count -ErrorAction SilentlyContinue) -join [Environment]::NewLine)
}

function Remove-DirectoryIfEmpty {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $remaining = Get-ChildItem -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if (-not $remaining) {
        Remove-Item -LiteralPath $Path -Force
    }
}

function Get-LastExitCodeOrDefault {
    param(
        [int]$Default
    )

    $lastExitCodeVariable = Get-Variable -Name LASTEXITCODE -ErrorAction SilentlyContinue
    if (($null -eq $lastExitCodeVariable) -or ($null -eq $lastExitCodeVariable.Value)) {
        return $Default
    }
    return [int]$lastExitCodeVariable.Value
}

if (-not (Test-Path -LiteralPath $DumpToS3ScriptPath)) {
    throw "Dump-to-S3 script was not found: $DumpToS3ScriptPath"
}

$start = Normalize-DateOnly -Value $StartDate -Name "StartDate"
$end = Normalize-DateOnly -Value $EndDate -Name "EndDate"
if ($start -gt $end) {
    throw "StartDate must be on or before EndDate"
}

if ([string]::IsNullOrWhiteSpace($RunIdPrefix)) {
    $safeSpecs = ($DataSpecs -join "_") -replace '[^0-9A-Za-z_-]', '_'
    $RunIdPrefix = "daily_${safeSpecs}_$($start.ToString('yyyyMMdd'))_$($end.ToString('yyyyMMdd'))"
}

$batchRunId = $RunIdPrefix -replace '[^0-9A-Za-z_-]', '_'
$logRoot = Join-Path (Join-Path $WorkRoot "batch_logs") $batchRunId
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$startedAt = Get-Date
$results = New-Object System.Collections.Generic.List[object]
$failedCount = 0
$completedCount = 0

:dayLoop for ($day = $start; $day -le $end; $day = $day.AddDays(1)) {
    $dateString = $day.ToString("yyyyMMdd")
    $fromDateRange = "${dateString}000000-${dateString}999999"

    foreach ($dataSpec in $DataSpecs) {
        $safeDataSpec = $dataSpec -replace '[^0-9A-Za-z_-]', '_'
        $runId = if ($DataSpecs.Count -eq 1) {
            "${batchRunId}_${dateString}"
        }
        else {
            "${batchRunId}_${safeDataSpec}_${dateString}"
        }
        $logPath = Join-Path $logRoot "${runId}.log"

        $childParams = @{
            DataSpec = $dataSpec
            FromDate = $fromDateRange
            Option = $Option
            Bucket = $Bucket
            S3Prefix = $S3Prefix
            WorkRoot = $WorkRoot
            RunId = $runId
            MaxReadIterations = $MaxReadIterations
            BufferSize = $BufferSize
            LogEveryChunks = $LogEveryChunks
            FlushEveryChunks = $FlushEveryChunks
            DownloadWaitTimeoutSeconds = $DownloadWaitTimeoutSeconds
            DownloadPollSeconds = $DownloadPollSeconds
            FailedS3Prefix = $FailedS3Prefix
        }
        if ($KeepLocal) {
            $childParams.KeepLocal = $true
        }
        if ($KeepFailedLocal) {
            $childParams.KeepFailedLocal = $true
        }

        $runStartedAt = Get-Date
        $status = "completed"
        $exitCode = 0
        $errorMessage = ""

        try {
            & $DumpToS3ScriptPath @childParams *> $logPath
            $exitCode = Get-LastExitCodeOrDefault -Default 0
            if ($exitCode -ne 0) {
                throw "Dump-to-S3 exited with code $exitCode"
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
        $shouldIncludeTail = $IncludeLogTail -or (($status -ne "completed") -and $IncludeFailureLogTail)
        $logTail = if ($shouldIncludeTail) { Get-LogTail -Path $logPath -Count 30 } else { "" }

        $results.Add([pscustomobject]@{
            dataSpec = $dataSpec
            date = $dateString
            fromDate = $fromDateRange
            option = $Option
            runId = $runId
            status = $status
            exitCode = $exitCode
            error = $errorMessage
            s3Prefix = ($S3Prefix.Trim("/") + "/" + $runId).Trim("/")
            failedS3Prefix = ($FailedS3Prefix.Trim("/") + "/" + $runId).Trim("/")
            startedAt = $runStartedAt.ToUniversalTime().ToString("o")
            finishedAt = $runFinishedAt.ToUniversalTime().ToString("o")
            logPath = if ($KeepBatchLogs) { $logPath } else { "" }
            logTail = $logTail
        }) | Out-Null

        if (-not $KeepBatchLogs -and (Test-Path -LiteralPath $logPath)) {
            Remove-Item -LiteralPath $logPath -Force
        }

        if (($status -ne "completed") -and $StopOnError) {
            break dayLoop
        }
    }
}

if (-not $KeepBatchLogs) {
    Remove-DirectoryIfEmpty -Path $logRoot
    Remove-DirectoryIfEmpty -Path (Join-Path $WorkRoot "batch_logs")
}

$finishedAt = Get-Date
$totalDays = [int](($end.Date - $start.Date).TotalDays + 1)
$completedResults = @($results | Where-Object { $_.status -eq "completed" })
$failedResults = @($results | Where-Object { $_.status -ne "completed" })

$summary = [ordered]@{
    batchRunId = $batchRunId
    dataSpecs = $DataSpecs
    startDate = $start.ToString("yyyyMMdd")
    endDate = $end.ToString("yyyyMMdd")
    totalDays = $totalDays
    attemptedRuns = $results.Count
    completedRuns = $completedCount
    failedRuns = $failedCount
    option = $Option
    bucket = $Bucket
    s3Prefix = $S3Prefix.Trim("/")
    failedS3Prefix = $FailedS3Prefix.Trim("/")
    keepLocal = [bool]$KeepLocal
    keepFailedLocal = [bool]$KeepFailedLocal
    keepBatchLogs = [bool]$KeepBatchLogs
    summaryOnly = [bool]$SummaryOnly
    startedAt = $startedAt.ToUniversalTime().ToString("o")
    finishedAt = $finishedAt.ToUniversalTime().ToString("o")
}

if ($SummaryOnly) {
    $summary.completedRunIds = @($completedResults | ForEach-Object { $_.runId })
    $summary.failed = @(
        $failedResults | ForEach-Object {
            [pscustomobject]@{
                date = $_.date
                runId = $_.runId
                failedS3Prefix = $_.failedS3Prefix
            }
        }
    )
}
else {
    $summary.results = $results
}

[pscustomobject]$summary | ConvertTo-Json -Depth 8

if (($failedCount -gt 0) -and $StopOnError) {
    throw "Daily range dump stopped after $failedCount failed run(s)"
}
