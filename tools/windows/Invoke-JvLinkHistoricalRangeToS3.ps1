param(
    [string]$DumpToS3ScriptPath = "C:\horse-lab\scripts\Invoke-JvLinkDumpToS3.ps1",

    [string[]]$DataSpecs = @("RACE"),

    [Parameter(Mandatory = $true)]
    [string]$FromDate,

    [string]$ToDate = "",

    [int]$Option = 1,

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string]$S3Prefix = "raw/jravan/historical",

    [string]$RunId = "",

    [int]$MaxReadIterations = 1000000,

    [switch]$KeepLocal
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Normalize-JvDate {
    param(
        [string]$Value,
        [string]$Name
    )

    $trimmed = $Value.Trim()
    if ($trimmed -match '^\d{8}$') {
        return "${trimmed}000000"
    }
    if ($trimmed -match '^\d{14}$') {
        return $trimmed
    }
    throw "${Name} must be YYYYMMDD or YYYYMMDDHHMMSS, got '${Value}'"
}

if (-not (Test-Path -LiteralPath $DumpToS3ScriptPath)) {
    throw "Dump-to-S3 script was not found: $DumpToS3ScriptPath"
}

$normalizedFromDate = Normalize-JvDate -Value $FromDate -Name "FromDate"
$normalizedToDate = if ([string]::IsNullOrWhiteSpace($ToDate)) {
    (Get-Date).ToUniversalTime().ToString("yyyyMMdd")
}
else {
    (Normalize-JvDate -Value $ToDate -Name "ToDate").Substring(0, 8)
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $safeSpecs = ($DataSpecs -join "_") -replace '[^0-9A-Za-z_-]', '_'
    $RunId = "historical_${safeSpecs}_${($normalizedFromDate.Substring(0, 8))}_${normalizedToDate}"
}

$startedAt = Get-Date
$results = foreach ($dataSpec in $DataSpecs) {
    $safeDataSpec = $dataSpec -replace '[^0-9A-Za-z_-]', '_'
    $dataSpecRunId = if ($DataSpecs.Count -eq 1) { $RunId } else { "${RunId}_${safeDataSpec}" }

    $arguments = @(
        "-DataSpec", $dataSpec,
        "-FromDate", $normalizedFromDate,
        "-Option", [string]$Option,
        "-Bucket", $Bucket,
        "-S3Prefix", $S3Prefix,
        "-RunId", $dataSpecRunId,
        "-MaxReadIterations", [string]$MaxReadIterations
    )
    if ($KeepLocal) {
        $arguments += "-KeepLocal"
    }

    $specStartedAt = Get-Date
    $output = (& $DumpToS3ScriptPath @arguments 2>&1 | Out-String)
    $exitCode = $LASTEXITCODE
    $specFinishedAt = Get-Date

    [pscustomobject]@{
        dataSpec = $dataSpec
        runId = $dataSpecRunId
        exitCode = $exitCode
        startedAt = $specStartedAt.ToUniversalTime().ToString("o")
        finishedAt = $specFinishedAt.ToUniversalTime().ToString("o")
        output = $output
    }

    if ($exitCode -ne 0) {
        throw "Historical dump failed for DataSpec=$dataSpec with exit code $exitCode"
    }
}
$finishedAt = Get-Date

[pscustomobject]@{
    runId = $RunId
    dataSpecs = $DataSpecs
    fromDate = $normalizedFromDate
    toDate = $normalizedToDate
    option = $Option
    bucket = $Bucket
    s3Prefix = $S3Prefix
    keepLocal = [bool]$KeepLocal
    startedAt = $startedAt.ToUniversalTime().ToString("o")
    finishedAt = $finishedAt.ToUniversalTime().ToString("o")
    results = $results
} | ConvertTo-Json -Depth 8
