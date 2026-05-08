param(
    [string]$RunnerPath = "C:\horse-lab\scripts\JvLinkDump.exe",

    [string]$UploadScriptPath = "C:\horse-lab\scripts\Invoke-S3RawUpload.ps1",

    [Parameter(Mandatory = $true)]
    [string]$DataSpec,

    [Parameter(Mandatory = $true)]
    [string]$FromDate,

    [int]$Option = 1,

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string]$S3Prefix = "raw/jravan",

    [string]$WorkRoot = "C:\horse-lab\work\jravan",

    [string]$RunId = "",

    [int]$MaxReadIterations = 1000000,

    [int]$DownloadWaitTimeoutSeconds = 900,

    [int]$DownloadPollSeconds = 2,

    [switch]$KeepLocal
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $safeDataSpec = $DataSpec -replace '[^0-9A-Za-z_-]', '_'
    $RunId = "${safeDataSpec}_${FromDate}_${timestamp}"
}

$outputRoot = Join-Path $WorkRoot $RunId
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$safeDataSpec = $DataSpec -replace '[^0-9A-Za-z_-]', '_'
$outputPath = Join-Path $outputRoot "${safeDataSpec}_${FromDate}.txt"
$logPath = Join-Path $outputRoot "${safeDataSpec}_${FromDate}.log"

$dumpArgs = @(
    "--data-spec", $DataSpec,
    "--from-date", $FromDate,
    "--option", [string]$Option,
    "--output", $outputPath,
    "--log", $logPath,
    "--max-read-iterations", [string]$MaxReadIterations,
    "--download-wait-timeout-seconds", [string]$DownloadWaitTimeoutSeconds,
    "--download-poll-seconds", [string]$DownloadPollSeconds
)

$startedAt = Get-Date
$dumpOutput = (& $RunnerPath @dumpArgs 2>&1 | Out-String)
$dumpExitCode = $LASTEXITCODE
$finishedDumpAt = Get-Date

if ($dumpExitCode -ne 0) {
    [pscustomobject]@{
        runId = $RunId
        stage = "dump"
        exitCode = $dumpExitCode
        outputRoot = $outputRoot
        stdout = $dumpOutput
    } | ConvertTo-Json -Depth 6
    throw "JvLinkDump failed with exit code $dumpExitCode; local files kept at $outputRoot"
}

$deleteAfterUpload = -not $KeepLocal
$uploadPrefix = ($S3Prefix.Trim("/") + "/" + $RunId).Trim("/")
if ($deleteAfterUpload) {
    $uploadOutput = (& $UploadScriptPath `
        -LocalRoot $outputRoot `
        -Bucket $Bucket `
        -Prefix $uploadPrefix `
        -DeleteAfterUpload 2>&1 | Out-String)
}
else {
    $uploadOutput = (& $UploadScriptPath `
        -LocalRoot $outputRoot `
        -Bucket $Bucket `
        -Prefix $uploadPrefix 2>&1 | Out-String)
}
$uploadExitCode = $LASTEXITCODE
$finishedUploadAt = Get-Date

if ($uploadExitCode -ne 0) {
    [pscustomobject]@{
        runId = $RunId
        stage = "upload"
        exitCode = $uploadExitCode
        outputRoot = $outputRoot
        dumpOutput = $dumpOutput
        uploadOutput = $uploadOutput
    } | ConvertTo-Json -Depth 6
    throw "S3 upload failed with exit code $uploadExitCode; local files kept at $outputRoot"
}

if ($deleteAfterUpload -and (Test-Path -LiteralPath $outputRoot)) {
    $remainingFiles = Get-ChildItem -LiteralPath $outputRoot -Recurse -File -ErrorAction SilentlyContinue
    if (-not $remainingFiles) {
        Remove-Item -LiteralPath $outputRoot -Force
    }
}

[pscustomobject]@{
    runId = $RunId
    dataSpec = $DataSpec
    fromDate = $FromDate
    option = $Option
    s3Prefix = $uploadPrefix
    keepLocal = [bool]$KeepLocal
    startedAt = $startedAt.ToUniversalTime().ToString("o")
    finishedDumpAt = $finishedDumpAt.ToUniversalTime().ToString("o")
    finishedUploadAt = $finishedUploadAt.ToUniversalTime().ToString("o")
    dumpOutput = $dumpOutput
    uploadOutput = $uploadOutput
} | ConvertTo-Json -Depth 8
