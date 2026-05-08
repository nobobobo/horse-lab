param(
    [string]$InstallerPath = "C:\Windows\Temp\AWSCLIV2.msi"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$aws = Get-Command aws -ErrorAction SilentlyContinue
if ($aws) {
    & $aws.Source --version
    return
}

Invoke-WebRequest `
    -Uri "https://awscli.amazonaws.com/AWSCLIV2.msi" `
    -OutFile $InstallerPath `
    -UseBasicParsing

$process = Start-Process `
    -FilePath "msiexec.exe" `
    -ArgumentList @("/i", $InstallerPath, "/qn") `
    -Wait `
    -PassThru

if ($process.ExitCode -ne 0) {
    throw "AWS CLI installer failed with exit code $($process.ExitCode)"
}

$awsPath = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
& $awsPath --version
