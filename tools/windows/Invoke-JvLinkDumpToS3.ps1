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

    [int]$BufferSize = 110000,

    [int]$LogEveryChunks = 100,

    [int]$FlushEveryChunks = 1000,

    [int]$DownloadWaitTimeoutSeconds = 900,

    [int]$DownloadPollSeconds = 2,

    [string]$FailedS3Prefix = "raw/jravan/failed",

    # Failed dump partials are uploaded to FailedS3Prefix and deleted by
    # default. Use this only when actively debugging the Windows worker.
    [switch]$KeepFailedLocal,

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

function Invoke-NativeCapture {
    param(
        [scriptblock]$Command
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell can promote native stderr to NativeCommandError
        # when ErrorActionPreference is Stop. Capture stdout/stderr first so
        # dump failures still flow through the failed-artifact upload branch.
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

function Invoke-RawUpload {
    param(
        [string]$Prefix,
        [bool]$DeleteAfterUpload
    )

    if ($DeleteAfterUpload) {
        $upload = Invoke-NativeCapture -Command {
            & $UploadScriptPath `
                -LocalRoot $outputRoot `
                -Bucket $Bucket `
                -Prefix $Prefix `
                -DeleteAfterUpload
        }
    }
    else {
        $upload = Invoke-NativeCapture -Command {
            & $UploadScriptPath `
                -LocalRoot $outputRoot `
                -Bucket $Bucket `
                -Prefix $Prefix
        }
    }
    [pscustomobject]@{
        exitCode = $upload.exitCode
        output = $upload.output
    }
}

function Remove-OutputRootIfEmpty {
    if (-not (Test-Path -LiteralPath $outputRoot)) {
        return
    }

    $remainingFiles = Get-ChildItem -LiteralPath $outputRoot -Recurse -File -ErrorAction SilentlyContinue
    if (-not $remainingFiles) {
        Remove-Item -LiteralPath $outputRoot -Force
    }
}

$dumpArgs = @(
    "--data-spec", $DataSpec,
    "--from-date", $FromDate,
    "--option", [string]$Option,
    "--output", $outputPath,
    "--log", $logPath,
    "--max-read-iterations", [string]$MaxReadIterations,
    "--buffer-size", [string]$BufferSize,
    "--log-every-chunks", [string]$LogEveryChunks,
    "--flush-every-chunks", [string]$FlushEveryChunks,
    "--download-wait-timeout-seconds", [string]$DownloadWaitTimeoutSeconds,
    "--download-poll-seconds", [string]$DownloadPollSeconds
)

$startedAt = Get-Date
$dump = Invoke-NativeCapture -Command { & $RunnerPath @dumpArgs }
$dumpOutput = $dump.output
$dumpExitCode = $dump.exitCode
$finishedDumpAt = Get-Date

if ($dumpExitCode -ne 0) {
    $failedUploadPrefix = ($FailedS3Prefix.Trim("/") + "/" + $RunId).Trim("/")
    $deleteFailedAfterUpload = -not ($KeepLocal -or $KeepFailedLocal)
    $failedUploadOutput = ""
    $failedUploadExitCode = $null
    $failedUploadError = ""

    try {
        $failedUpload = Invoke-RawUpload -Prefix $failedUploadPrefix -DeleteAfterUpload $deleteFailedAfterUpload
        $failedUploadOutput = $failedUpload.output
        $failedUploadExitCode = $failedUpload.exitCode
        if ($failedUploadExitCode -ne 0) {
            throw "S3 failed-artifact upload exited with code $failedUploadExitCode"
        }

        if ($deleteFailedAfterUpload) {
            Remove-OutputRootIfEmpty
        }
    }
    catch {
        $failedUploadError = $_.Exception.Message
        [pscustomobject]@{
            runId = $RunId
            stage = "dump"
            exitCode = $dumpExitCode
            outputRoot = $outputRoot
            failedS3Prefix = $failedUploadPrefix
            failedUploadExitCode = $failedUploadExitCode
            failedUploadOutput = $failedUploadOutput
            failedUploadError = $failedUploadError
            stdout = $dumpOutput
        } | ConvertTo-Json -Depth 8
        throw "JvLinkDump failed with exit code $dumpExitCode; failed-artifact upload also failed, so local files were kept at $outputRoot"
    }

    [pscustomobject]@{
        runId = $RunId
        stage = "dump"
        exitCode = $dumpExitCode
        outputRoot = $outputRoot
        failedS3Prefix = $failedUploadPrefix
        failedArtifactsDeletedLocal = $deleteFailedAfterUpload
        failedUploadExitCode = $failedUploadExitCode
        failedUploadOutput = $failedUploadOutput
        stdout = $dumpOutput
    } | ConvertTo-Json -Depth 8
    throw "JvLinkDump failed with exit code $dumpExitCode; partial artifacts uploaded to s3://$Bucket/$failedUploadPrefix"
}

$deleteAfterUpload = -not $KeepLocal
$uploadPrefix = ($S3Prefix.Trim("/") + "/" + $RunId).Trim("/")
$upload = Invoke-RawUpload -Prefix $uploadPrefix -DeleteAfterUpload $deleteAfterUpload
$uploadOutput = $upload.output
$uploadExitCode = $upload.exitCode
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

if ($deleteAfterUpload) {
    Remove-OutputRootIfEmpty
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
