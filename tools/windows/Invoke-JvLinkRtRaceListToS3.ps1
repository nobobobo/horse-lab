param(
    [string]$RaceListScriptPath = "C:\horse-lab\scripts\Invoke-JvLinkRtRaceList.ps1",

    [string]$UploadScriptPath = "C:\horse-lab\scripts\Invoke-S3RawUpload.ps1",

    [string[]]$RaceKeys = @(),

    [string]$RaceKeyFile = "",

    [string[]]$DataSpecs = @("0B31", "0B41"),

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string]$S3Prefix = "raw/jravan/realtime",

    [string]$WorkRoot = "C:\horse-lab\work\jravan-rt",

    [string]$RunId = "",

    [int]$MaxReadIterations = 5000,

    [switch]$KeepLocal
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
}

$outputRoot = Join-Path $WorkRoot $RunId
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$raceParams = @{
    OutputRoot = $outputRoot
    MaxReadIterations = $MaxReadIterations
}

if ($RaceKeys.Count -gt 0) {
    $raceParams.RaceKeys = $RaceKeys
}

if (-not [string]::IsNullOrWhiteSpace($RaceKeyFile)) {
    $raceParams.RaceKeyFile = $RaceKeyFile
}

if ($DataSpecs.Count -gt 0) {
    $raceParams.DataSpecs = $DataSpecs
}

$startedAt = Get-Date
$raceOutput = (& $RaceListScriptPath @raceParams 2>&1 | Out-String)
$raceExitCode = $LASTEXITCODE
$finishedRaceAt = Get-Date

if ($raceExitCode -ne 0) {
    [pscustomobject]@{
        runId = $RunId
        stage = "realtime_dump"
        exitCode = $raceExitCode
        outputRoot = $outputRoot
        stdout = $raceOutput
    } | ConvertTo-Json -Depth 6
    throw "Realtime race-list dump failed with exit code $raceExitCode; local files kept at $outputRoot"
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
        raceOutput = $raceOutput
        uploadOutput = $uploadOutput
    } | ConvertTo-Json -Depth 6
    throw "S3 upload failed with exit code $uploadExitCode; local files kept at $outputRoot"
}

if ($deleteAfterUpload -and (Test-Path -LiteralPath $outputRoot)) {
    $remainingFiles = Get-ChildItem -LiteralPath $outputRoot -Recurse -File -ErrorAction SilentlyContinue
    if (-not $remainingFiles) {
        Remove-Item -LiteralPath $outputRoot -Force -Recurse
    }
}

[pscustomobject]@{
    runId = $RunId
    raceKeys = $RaceKeys
    raceKeyFile = $RaceKeyFile
    dataSpecs = $DataSpecs
    s3Prefix = $uploadPrefix
    keepLocal = [bool]$KeepLocal
    startedAt = $startedAt.ToUniversalTime().ToString("o")
    finishedRaceAt = $finishedRaceAt.ToUniversalTime().ToString("o")
    finishedUploadAt = $finishedUploadAt.ToUniversalTime().ToString("o")
    raceOutput = $raceOutput
    uploadOutput = $uploadOutput
} | ConvertTo-Json -Depth 8
