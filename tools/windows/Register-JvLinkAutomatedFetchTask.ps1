param(
    [string]$TaskName = "HorseLab-JvLink-AutomatedFetch",

    [string]$ScriptPath = "C:\horse-lab\scripts\Invoke-JvLinkAutomatedFetchToS3.ps1",

    [string]$RaceKeyFile = "C:\horse-lab\inputs\today_race_keys.txt",

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string[]]$DailyDataSpecs = @("RACE"),

    [string[]]$RealtimeDataSpecs = @("0B30", "0B41", "0B42"),

    [string]$DailyS3Prefix = "raw/jravan/daily",

    [string]$RealtimeS3Prefix = "raw/jravan/realtime",

    [string]$At = "21:30",

    [string]$User = "$env:USERNAME",

    [switch]$RunWhetherLoggedOn,

    [switch]$SkipRealtime,

    [switch]$SkipDaily,

    [switch]$WhatIf
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Join-ArgumentList {
    param([string[]]$Values)
    return ($Values -join ",")
}

function Quote-ForArgument {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Automated fetch script was not found: $ScriptPath"
}

$parsedAt = [datetime]::ParseExact($At, "HH:mm", $null)
$trigger = New-ScheduledTaskTrigger -Daily -At $parsedAt

$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-ForArgument $ScriptPath),
    "-DailyDataSpecs", (Join-ArgumentList $DailyDataSpecs),
    "-RealtimeDataSpecs", (Join-ArgumentList $RealtimeDataSpecs),
    "-Bucket", (Quote-ForArgument $Bucket),
    "-DailyS3Prefix", (Quote-ForArgument $DailyS3Prefix),
    "-RealtimeS3Prefix", (Quote-ForArgument $RealtimeS3Prefix)
)

if (-not $SkipRealtime) {
    $arguments += @("-RaceKeyFile", (Quote-ForArgument $RaceKeyFile))
}
if ($SkipRealtime) {
    $arguments += "-SkipRealtime"
}
if ($SkipDaily) {
    $arguments += "-SkipDaily"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($arguments -join " ")
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 15)

$principalLogonType = if ($RunWhetherLoggedOn) {
    "S4U"
}
else {
    "Interactive"
}
$principal = New-ScheduledTaskPrincipal `
    -UserId $User `
    -LogonType $principalLogonType `
    -RunLevel Highest

$task = New-ScheduledTask `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Horse Lab JV-Link automated S3-first daily/realtime fetch"

if ($WhatIf) {
    [pscustomobject]@{
        taskName = $TaskName
        user = $User
        at = $At
        scriptPath = $ScriptPath
        action = "powershell.exe $($arguments -join ' ')"
        runWhetherLoggedOn = [bool]$RunWhetherLoggedOn
        skipDaily = [bool]$SkipDaily
        skipRealtime = [bool]$SkipRealtime
    } | ConvertTo-Json -Depth 6
    exit 0
}

Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName
