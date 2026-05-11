param(
    [string]$DailyRangeScriptPath = "C:\horse-lab\scripts\Invoke-JvLinkDailyRangeToS3.ps1",

    [string]$RealtimeScriptPath = "C:\horse-lab\scripts\Invoke-JvLinkRtRaceListToS3.ps1",

    [string]$StartDate = "",

    [string]$EndDate = "",

    [string[]]$DailyDataSpecs = @("RACE"),

    [string[]]$RealtimeDataSpecs = @("0B30", "0B41", "0B42"),

    [string]$RaceKeyFile = "",

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string]$DailyS3Prefix = "raw/jravan/daily",

    [string]$RealtimeS3Prefix = "raw/jravan/realtime",

    [string]$RunId = "",

    [string]$WorkRoot = "C:\horse-lab\work\automated-fetch",

    [int]$DailyMaxReadIterations = 1000000,

    [int]$RealtimeMaxReadIterations = 5000,

    [switch]$SkipDaily,

    [switch]$SkipRealtime,

    [switch]$KeepLocal,

    [switch]$KeepBatchLogs
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

function Invoke-CapturedCommand {
    param(
        [scriptblock]$Command
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = (& $Command 2>&1 | Out-String)
        [pscustomobject]@{
            exitCode = $LASTEXITCODE
            output = $output
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

if ([string]::IsNullOrWhiteSpace($StartDate)) {
    $StartDate = (Get-Date).ToString("yyyyMMdd")
}
if ([string]::IsNullOrWhiteSpace($EndDate)) {
    $EndDate = $StartDate
}
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "auto_" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
}

$start = Normalize-DateOnly -Value $StartDate -Name "StartDate"
$end = Normalize-DateOnly -Value $EndDate -Name "EndDate"
if ($start -gt $end) {
    throw "StartDate must be on or before EndDate"
}

$safeRunId = $RunId -replace '[^0-9A-Za-z_-]', '_'
$runWorkRoot = Join-Path $WorkRoot $safeRunId
New-Item -ItemType Directory -Force -Path $runWorkRoot | Out-Null

$startedAt = Get-Date
$dailyResult = $null
$realtimeResult = $null
$status = "completed"
$errorMessage = ""

try {
    if (-not $SkipDaily) {
        if (-not (Test-Path -LiteralPath $DailyRangeScriptPath)) {
            throw "Daily range script was not found: $DailyRangeScriptPath"
        }
        $dailyParams = @{
            DataSpecs = $DailyDataSpecs
            StartDate = $start.ToString("yyyyMMdd")
            EndDate = $end.ToString("yyyyMMdd")
            Bucket = $Bucket
            S3Prefix = $DailyS3Prefix
            RunIdPrefix = "${safeRunId}_daily"
            WorkRoot = (Join-Path $runWorkRoot "daily")
            MaxReadIterations = $DailyMaxReadIterations
            SummaryOnly = $true
        }
        if ($KeepLocal) {
            $dailyParams.KeepLocal = $true
        }
        if ($KeepBatchLogs) {
            $dailyParams.KeepBatchLogs = $true
        }
        $dailyResult = Invoke-CapturedCommand -Command {
            & $DailyRangeScriptPath @dailyParams
        }
        if ($dailyResult.exitCode -ne 0) {
            throw "Daily fetch failed with exit code $($dailyResult.exitCode)"
        }
    }

    if (-not $SkipRealtime) {
        if (-not (Test-Path -LiteralPath $RealtimeScriptPath)) {
            throw "Realtime script was not found: $RealtimeScriptPath"
        }
        if ([string]::IsNullOrWhiteSpace($RaceKeyFile)) {
            throw "RaceKeyFile is required unless -SkipRealtime is supplied"
        }
        if (-not (Test-Path -LiteralPath $RaceKeyFile)) {
            throw "RaceKeyFile was not found: $RaceKeyFile"
        }
        $realtimeParams = @{
            RaceKeyFile = $RaceKeyFile
            DataSpecs = $RealtimeDataSpecs
            Bucket = $Bucket
            S3Prefix = $RealtimeS3Prefix
            WorkRoot = (Join-Path $runWorkRoot "realtime")
            RunId = "${safeRunId}_realtime"
            MaxReadIterations = $RealtimeMaxReadIterations
        }
        if ($KeepLocal) {
            $realtimeParams.KeepLocal = $true
        }
        $realtimeResult = Invoke-CapturedCommand -Command {
            & $RealtimeScriptPath @realtimeParams
        }
        if ($realtimeResult.exitCode -ne 0) {
            throw "Realtime fetch failed with exit code $($realtimeResult.exitCode)"
        }
    }
}
catch {
    $status = "failed"
    $errorMessage = $_.Exception.Message
}

$finishedAt = Get-Date
$manifest = [ordered]@{
    runId = $safeRunId
    status = $status
    startDate = $start.ToString("yyyyMMdd")
    endDate = $end.ToString("yyyyMMdd")
    bucket = $Bucket
    dailyS3Prefix = $DailyS3Prefix.Trim("/")
    realtimeS3Prefix = $RealtimeS3Prefix.Trim("/")
    dailyDataSpecs = $DailyDataSpecs
    realtimeDataSpecs = $RealtimeDataSpecs
    raceKeyFile = $RaceKeyFile
    keepLocal = [bool]$KeepLocal
    startedAt = $startedAt.ToUniversalTime().ToString("o")
    finishedAt = $finishedAt.ToUniversalTime().ToString("o")
    daily = if ($null -eq $dailyResult) { $null } else {
        [ordered]@{
            exitCode = $dailyResult.exitCode
            output = $dailyResult.output
        }
    }
    realtime = if ($null -eq $realtimeResult) { $null } else {
        [ordered]@{
            exitCode = $realtimeResult.exitCode
            output = $realtimeResult.output
        }
    }
}

if ($status -ne "completed") {
    $manifest["error"] = $errorMessage
}

$manifestPath = Join-Path $runWorkRoot "automated_fetch_manifest.json"
$manifest["localManifestPath"] = $manifestPath
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 12

if ($status -ne "completed") {
    throw $errorMessage
}
