param(
    [Parameter(Mandatory = $true)]
    [string]$DataSpec,

    [Parameter(Mandatory = $true)]
    [string]$FromDate,

    [int]$Option = 1,

    [string]$SoftwareId = "UNKNOWN",

    [string]$OutputPath = "C:\horse-lab\data\raw\jravan\jvdata.txt",

    [string]$LogPath = "C:\horse-lab\data\raw\jravan\jvlink_dump.log",

    [int]$MaxReadIterations = 1000000
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Invoke-JvLinkDump {
    param(
        [string]$DataSpec,
        [string]$FromDate,
        [int]$Option,
        [string]$SoftwareId,
        [string]$OutputPath,
        [string]$LogPath,
        [int]$MaxReadIterations
    )

    $outputDirectory = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    $logDirectory = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

    $encoding = [Text.Encoding]::GetEncoding(932)
    $writer = New-Object IO.StreamWriter($OutputPath, $false, $encoding)
    $jv = $null

    function Write-WorkerLog {
        param([string]$Message)
        $timestamp = [DateTime]::UtcNow.ToString("o")
        Add-Content -Path $LogPath -Value "$timestamp $Message" -Encoding UTF8
    }

    try {
        Set-Content -Path $LogPath -Value "" -Encoding UTF8
        Write-WorkerLog "starting DataSpec=$DataSpec FromDate=$FromDate Option=$Option"
        $jv = New-Object -ComObject "JVDTLab.JVLink"
        Write-WorkerLog "created COM object"

        $initCode = $jv.JVInit($SoftwareId)
        Write-WorkerLog "JVInit returned $initCode"
        if ($initCode -ne 0) {
            throw "JVInit failed with return code $initCode"
        }

        $readCount = 0
        $downloadCount = 0
        $lastFileTimestamp = ""
        $openCode = $jv.JVOpen(
            $DataSpec,
            $FromDate,
            $Option,
            [ref]$readCount,
            [ref]$downloadCount,
            [ref]$lastFileTimestamp
        )
        Write-WorkerLog "JVOpen returned $openCode readCount=$readCount downloadCount=$downloadCount lastFileTimestamp=$lastFileTimestamp"
        if ($openCode -ne 0) {
            throw "JVOpen failed with return code $openCode"
        }

        $recordChunks = 0
        $fileMarkers = 0
        $reachedEnd = $false
        $bufferSize = 110000
        for ($i = 0; $i -lt $MaxReadIterations; $i++) {
            # JVGets expects a VARIANT* buffer. Keep the byte array boxed as
            # object so PowerShell COM interop does not pass a typed byte[] ref.
            [object]$buffer = New-Object byte[] $bufferSize
            $bufferName = ""
            $readCode = $jv.JVGets([ref]$buffer, $bufferSize, [ref]$bufferName)

            if ($readCode -gt 0) {
                $recordText = $encoding.GetString([byte[]]$buffer, 0, $readCode).TrimEnd("`r", "`n")
                $writer.WriteLine($recordText)
                $recordChunks += 1
                if (($recordChunks % 100) -eq 0) {
                    Write-WorkerLog "JVGets chunks=$recordChunks"
                }
                continue
            }

            if ($readCode -eq -1) {
                $fileMarkers += 1
                Write-WorkerLog "JVGets file marker $fileMarkers name=$bufferName"
                continue
            }

            if ($readCode -eq 0) {
                $reachedEnd = $true
                Write-WorkerLog "JVGets EOF chunks=$recordChunks fileMarkers=$fileMarkers"
                break
            }

            throw "JVGets failed with return code $readCode"
        }

        if (-not $reachedEnd) {
            Write-WorkerLog "JVGets max iterations reached chunks=$recordChunks fileMarkers=$fileMarkers"
        }

        [pscustomobject]@{
            outputPath = $OutputPath
            dataSpec = $DataSpec
            fromDate = $FromDate
            option = $Option
            readCount = $readCount
            downloadCount = $downloadCount
            lastFileTimestamp = $lastFileTimestamp
            recordChunks = $recordChunks
            fileMarkers = $fileMarkers
        } | ConvertTo-Json -Depth 3
    }
    catch {
        Write-WorkerLog "ERROR $($_.Exception.Message)"
        throw
    }
    finally {
        if ($writer -ne $null) {
            $writer.Dispose()
        }
        if ($jv -ne $null) {
            try {
                $null = $jv.JVClose()
                Write-WorkerLog "JVClose completed"
            }
            catch {
            }
        }
    }
}

if (-not [Environment]::Is64BitProcess) {
    Invoke-JvLinkDump @PSBoundParameters
    return
}

$powerShell32 = Join-Path $env:WINDIR "SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
$scriptPath = $PSCommandPath
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $scriptPath,
    "-DataSpec", $DataSpec,
    "-FromDate", $FromDate,
    "-Option", $Option,
    "-SoftwareId", $SoftwareId,
    "-OutputPath", $OutputPath,
    "-LogPath", $LogPath,
    "-MaxReadIterations", $MaxReadIterations
)

& $powerShell32 @arguments
exit $LASTEXITCODE
