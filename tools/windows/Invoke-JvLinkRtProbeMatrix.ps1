param(
    [string]$ExePath = "C:\horse-lab\scripts\JvLinkRtDump.exe",

    [string]$RaceKey = "2026051005020611",

    [string]$OutputRoot = "C:\horse-lab\data\raw\jravan\rt_probe",

    [int]$MaxReadIterations = 1000,

    [switch]$IncludeJvRead
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$probes = @(
    @{ Name = "o1_0B31_jvgets"; DataSpec = "0B31"; Reader = "jvgets" },
    @{ Name = "all_odds_0B30_jvgets"; DataSpec = "0B30"; Reader = "jvgets" },
    @{ Name = "timeseries_o1_0B41_jvgets"; DataSpec = "0B41"; Reader = "jvgets" }
)

if ($IncludeJvRead) {
    $probes += @{ Name = "o1_0B31_jvread"; DataSpec = "0B31"; Reader = "jvread" }
}

$summary = foreach ($probe in $probes) {
    $name = $probe.Name
    $outputPath = Join-Path $OutputRoot "$name.txt"
    $logPath = Join-Path $OutputRoot "$name.log"
    $stdoutPath = Join-Path $OutputRoot "$name.stdout.txt"
    $stderrPath = Join-Path $OutputRoot "$name.stderr.txt"

    $arguments = @(
        "--data-spec", $probe.DataSpec,
        "--key", $RaceKey,
        "--reader", $probe.Reader,
        "--output", $outputPath,
        "--log", $logPath,
        "--max-read-iterations", $MaxReadIterations
    )

    $startedAt = Get-Date
    $stdout = ""
    $stderr = ""
    try {
        $stdout = (& $ExePath @arguments 2>&1 | Out-String)
        $exitCode = $LASTEXITCODE
    }
    catch {
        $stderr = $_.Exception.ToString()
        $exitCode = 1
    }
    $finishedAt = Get-Date
    Set-Content -LiteralPath $stdoutPath -Value $stdout -Encoding UTF8
    Set-Content -LiteralPath $stderrPath -Value $stderr -Encoding UTF8

    $recordCounts = @{}
    if (Test-Path -LiteralPath $outputPath) {
        Get-Content -LiteralPath $outputPath -Encoding Default |
            Where-Object { $_.Length -ge 2 } |
            ForEach-Object {
                $recordType = $_.Substring(0, 2)
                if (-not $recordCounts.ContainsKey($recordType)) {
                    $recordCounts[$recordType] = 0
                }
                $recordCounts[$recordType] += 1
            }
    }

    [pscustomobject]@{
        name = $name
        dataSpec = $probe.DataSpec
        raceKey = $RaceKey
        reader = $probe.Reader
        exitCode = $exitCode
        startedAt = $startedAt.ToString("o")
        finishedAt = $finishedAt.ToString("o")
        outputPath = $outputPath
        outputLength = if (Test-Path -LiteralPath $outputPath) { (Get-Item -LiteralPath $outputPath).Length } else { -1 }
        recordCounts = $recordCounts
        stdout = $stdout
        stderr = $stderr
        logTail = if (Test-Path -LiteralPath $logPath) { (Get-Content -LiteralPath $logPath -Tail 20) -join [Environment]::NewLine } else { "" }
    }
}

$summaryPath = Join-Path $OutputRoot "rt_probe_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Output $summaryPath
