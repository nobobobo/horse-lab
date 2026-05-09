param(
    [Parameter(Mandatory = $true)]
    [string]$LocalRoot,

    [string]$Bucket = "horse-lab-jravan-244306245597-apne1",

    [string]$Prefix = "raw/jravan",

    [string[]]$IncludePatterns = @("*.txt", "*.log", "*.json"),

    [string]$AwsPath = "C:\Program Files\Amazon\AWSCLIV2\aws.exe",

    [switch]$UseSync,

    [switch]$DeleteAfterUpload,

    [string]$ManifestPath = "",

    [switch]$NoUploadManifest
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $AwsPath)) {
    $aws = Get-Command aws -ErrorAction SilentlyContinue
    if (-not $aws) {
        throw "AWS CLI was not found. Run Install-AwsCli.ps1 first."
    }
    $AwsPath = $aws.Source
}

$root = Resolve-Path -LiteralPath $LocalRoot
$normalizedPrefix = $Prefix.Trim("/")
$uploaded = New-Object System.Collections.Generic.List[object]
$failed = New-Object System.Collections.Generic.List[object]

$files = foreach ($pattern in $IncludePatterns) {
    Get-ChildItem -LiteralPath $root -Recurse -File -Filter $pattern
}

$uniqueFiles = $files | Sort-Object FullName -Unique

if ($UseSync) {
    $syncUri = "s3://$Bucket/$normalizedPrefix"
    $syncArgs = @(
        "s3", "sync",
        $root.Path,
        $syncUri,
        "--only-show-errors",
        "--sse", "AES256",
        "--exclude", "*"
    )
    foreach ($pattern in $IncludePatterns) {
        $syncArgs += @("--include", $pattern)
    }

    & $AwsPath @syncArgs
    $syncExitCode = $LASTEXITCODE

    foreach ($file in $uniqueFiles) {
        $relativePath = $file.FullName.Substring($root.Path.Length).TrimStart("\", "/")
        $s3Key = ($normalizedPrefix + "/" + ($relativePath -replace "\\", "/")).TrimStart("/")
        $s3Uri = "s3://$Bucket/$s3Key"

        if ($syncExitCode -eq 0) {
            $record = [pscustomobject]@{
                localPath = $file.FullName
                s3Uri = $s3Uri
                bytes = $file.Length
                uploadedAt = (Get-Date).ToUniversalTime().ToString("o")
                deletedLocal = $false
            }

            if ($DeleteAfterUpload) {
                Remove-Item -LiteralPath $file.FullName -Force
                $record.deletedLocal = $true
            }

            $uploaded.Add($record) | Out-Null
        }
        else {
            $failed.Add([pscustomobject]@{
                localPath = $file.FullName
                s3Uri = $s3Uri
                exitCode = $syncExitCode
            }) | Out-Null
        }
    }
}
else {
    foreach ($file in $uniqueFiles) {
        $relativePath = $file.FullName.Substring($root.Path.Length).TrimStart("\", "/")
        $s3Key = ($normalizedPrefix + "/" + ($relativePath -replace "\\", "/")).TrimStart("/")
        $s3Uri = "s3://$Bucket/$s3Key"

        & $AwsPath s3 cp $file.FullName $s3Uri --only-show-errors --sse AES256
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            $record = [pscustomobject]@{
                localPath = $file.FullName
                s3Uri = $s3Uri
                bytes = $file.Length
                uploadedAt = (Get-Date).ToUniversalTime().ToString("o")
                deletedLocal = $false
            }

            if ($DeleteAfterUpload) {
                Remove-Item -LiteralPath $file.FullName -Force
                $record.deletedLocal = $true
            }

            $uploaded.Add($record) | Out-Null
        }
        else {
            $failed.Add([pscustomobject]@{
                localPath = $file.FullName
                s3Uri = $s3Uri
                exitCode = $exitCode
            }) | Out-Null
        }
    }
}

if ($UseSync -and $DeleteAfterUpload -and (Test-Path -LiteralPath $root.Path)) {
    $emptyDirectories = Get-ChildItem -LiteralPath $root.Path -Recurse -Directory |
        Sort-Object FullName -Descending
    foreach ($directory in $emptyDirectories) {
        $remaining = Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction SilentlyContinue
        if (-not $remaining) {
            Remove-Item -LiteralPath $directory.FullName -Force
        }
    }
}

$summary = [pscustomobject]@{
    localRoot = $root.Path
    bucket = $Bucket
    prefix = $normalizedPrefix
    includePatterns = $IncludePatterns
    uploadMode = if ($UseSync) { "sync" } else { "file" }
    deleteAfterUpload = [bool]$DeleteAfterUpload
    uploadedCount = $uploaded.Count
    failedCount = $failed.Count
    uploadedBytes = ($uploaded | Measure-Object -Property bytes -Sum).Sum
    uploaded = $uploaded
    failed = $failed
}

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    $ManifestPath = Join-Path $root.Path "s3_upload_manifest.json"
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

if (-not $NoUploadManifest) {
    $manifestItem = Get-Item -LiteralPath $ManifestPath
    $manifestRelativePath = $manifestItem.FullName.Substring($root.Path.Length).TrimStart("\", "/")
    if ([string]::IsNullOrWhiteSpace($manifestRelativePath)) {
        $manifestRelativePath = $manifestItem.Name
    }
    $manifestKey = ($normalizedPrefix + "/" + ($manifestRelativePath -replace "\\", "/")).TrimStart("/")
    $manifestUri = "s3://$Bucket/$manifestKey"
    & $AwsPath s3 cp $manifestItem.FullName $manifestUri --only-show-errors --sse AES256
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload manifest to $manifestUri"
    }

    if ($DeleteAfterUpload) {
        Remove-Item -LiteralPath $manifestItem.FullName -Force
    }
}

$summary | ConvertTo-Json -Depth 8

if ($failed.Count -gt 0) {
    throw "One or more files failed to upload to S3"
}
