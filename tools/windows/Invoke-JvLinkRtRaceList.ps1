param(
    [string]$ExePath = "C:\horse-lab\scripts\JvLinkRtDump.exe",

    [string[]]$RaceKeys = @(),

    [string]$RaceKeyFile = "",

    [string[]]$DataSpecs = @("0B31", "0B41"),

    [string]$OutputRoot = "C:\horse-lab\data\raw\jravan\rt_races",

    [int]$MaxReadIterations = 5000
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$keys = New-Object System.Collections.Generic.List[string]
foreach ($key in $RaceKeys) {
    if (-not [string]::IsNullOrWhiteSpace($key)) {
        $keys.Add($key.Trim())
    }
}

if (-not [string]::IsNullOrWhiteSpace($RaceKeyFile)) {
    Get-Content -LiteralPath $RaceKeyFile |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $keys.Add($_.Trim()) }
}

if ($keys.Count -eq 0) {
    throw "At least one race key is required via -RaceKeys or -RaceKeyFile"
}

function Get-SafeDataSpecName {
    param([string]$DataSpec)
    return $DataSpec -replace '[^0-9A-Za-z_-]', '_'
}

$summary = foreach ($raceKey in ($keys | Select-Object -Unique)) {
    foreach ($dataSpec in $DataSpecs) {
        $safeDataSpec = Get-SafeDataSpecName $dataSpec
        $outputDir = Join-Path $OutputRoot $raceKey
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

        $outputPath = Join-Path $outputDir "${safeDataSpec}_jvgets.txt"
        $logPath = Join-Path $outputDir "${safeDataSpec}_jvgets.log"
        $stdoutPath = Join-Path $outputDir "${safeDataSpec}_jvgets.stdout.txt"
        $stderrPath = Join-Path $outputDir "${safeDataSpec}_jvgets.stderr.txt"

        $arguments = @(
            "--data-spec", $dataSpec,
            "--key", $raceKey,
            "--reader", "jvgets",
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
            raceKey = $raceKey
            dataSpec = $dataSpec
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
}

$summaryPath = Join-Path $OutputRoot "rt_race_list_summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Output $summaryPath
