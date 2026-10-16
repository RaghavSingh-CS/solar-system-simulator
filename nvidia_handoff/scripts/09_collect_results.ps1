[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunDirectory)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '09_collect_results'
$run = Resolve-HandoffRunDirectory $RunDirectory
$paths = Get-HandoffPaths
$resultsRoot = [IO.Path]::GetFullPath($paths.DefaultResultsRoot)
$runId = Split-Path -Leaf $run
$zip = Join-Path $resultsRoot "$runId.zip"
$temporaryRoot = Join-Path $resultsRoot ('.collecting-' + $runId + '-' + [Guid]::NewGuid().ToString('N'))
$temporaryZip = Join-Path $resultsRoot ('.collecting-' + $runId + '-' + [Guid]::NewGuid().ToString('N') + '.zip')
$maximumFiles = 512
$maximumFileBytes = 32MB
$maximumTotalBytes = 128MB

try {
    Assert-HandoffStageFresh $run $stage
    if (Test-Path -LiteralPath $zip) {
        throw "Result archive already exists and will not be overwritten: $zip"
    }
    Assert-HandoffNoReparsePoints $run $resultsRoot
    $allowedDirectories = @('logs', 'stages', 'diagnostics', 'numerics')
    foreach ($directory in @(Get-ChildItem -Force -LiteralPath $run -Directory -Recurse)) {
        Assert-HandoffNoReparsePoints $directory.FullName $run
        $relativeDirectory = $directory.FullName.Substring($run.Length).TrimStart('\', '/').Replace('\', '/')
        if ($allowedDirectories -notcontains $relativeDirectory) {
            throw "Unrecognized result directory cannot be collected: $relativeDirectory"
        }
    }
    $sourceFiles = @()
    $totalBytes = [int64]0
    foreach ($file in @(Get-ChildItem -Force -LiteralPath $run -File -Recurse | Sort-Object FullName)) {
        Assert-HandoffNoReparsePoints $file.FullName $run
        $relative = $file.FullName.Substring($run.Length).TrimStart('\', '/').Replace('\', '/')
        $allowed = $relative -in @('.handoff-run.json', 'preflight.json', 'executables.json', 'demo_observations.txt') -or
            $relative -match '^(logs|diagnostics)/[^/]+\.(log|txt|json|csv)$' -or
            $relative -match '^numerics/[^/]+\.(txt|json|csv)$' -or
            $relative -match '^stages/[^/]+\.json$'
        if (-not $allowed) {
            throw "Unrecognized result file cannot be collected: $relative"
        }
        if ($file.Length -gt $maximumFileBytes) {
            throw "Result file exceeds the $maximumFileBytes-byte collection limit: $relative"
        }
        if ($file.Length -gt ([int64]::MaxValue - $totalBytes)) {
            throw 'Result byte accounting overflowed Int64.'
        }
        $totalBytes += $file.Length
        $sourceFiles += $file
    }
    if ($sourceFiles.Count -gt $maximumFiles) {
        throw "Result file count exceeds the collection limit of $maximumFiles."
    }
    if ($totalBytes -gt $maximumTotalBytes) {
        throw "Result bytes exceed the $maximumTotalBytes-byte collection limit."
    }

    $repository = Get-RepositoryRecord
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    $copyRoot = Join-Path $temporaryRoot $runId
    New-Item -ItemType Directory -Path $copyRoot | Out-Null
    foreach ($file in $sourceFiles) {
        $relative = $file.FullName.Substring($run.Length).TrimStart('\', '/')
        $destination = Join-Path $copyRoot $relative
        $parent = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $parent)) {
            New-Item -ItemType Directory -Path $parent | Out-Null
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destination
    }
    foreach ($name in @('logs', 'stages')) {
        $directory = Join-Path $copyRoot $name
        if (-not (Test-Path -LiteralPath $directory)) {
            New-Item -ItemType Directory -Path $directory | Out-Null
        }
    }
    $collectionLog = Join-Path $copyRoot "logs/$stage.log"
    "stage=$stage`nstarted_utc=$([DateTime]::UtcNow.ToString('o'))`nvalidated_files=$($sourceFiles.Count)`nvalidated_bytes=$totalBytes" |
        Set-Content -LiteralPath $collectionLog -Encoding utf8
    Write-HandoffStageStatus $copyRoot $stage passed 0 'Bounded allowlisted result archive created.'

    $files = @()
    foreach ($file in @(Get-ChildItem -Force -LiteralPath $copyRoot -File -Recurse | Sort-Object FullName)) {
        if ($file.Name -eq 'collection_manifest.json') { continue }
        $relative = $file.FullName.Substring($copyRoot.Length).TrimStart('\', '/')
        $files += [ordered]@{
            path = $relative.Replace('\', '/')
            bytes = $file.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
        }
    }
    $binaryHashes = @()
    $executableManifest = Join-Path $copyRoot 'executables.json'
    if (Test-Path -LiteralPath $executableManifest) {
        $inventory = Get-Content -Raw $executableManifest | ConvertFrom-Json
        foreach ($entry in $inventory.executables) {
            $verified = Get-HandoffStagedExecutable $run ([string]$entry.build_configuration) ([string]$entry.cmake_target)
            $file = Get-Item -LiteralPath $verified
            $binaryHashes += [ordered]@{
                cmake_target = $entry.cmake_target
                build_configuration = $entry.build_configuration
                path = $file.FullName.Substring($paths.HandoffRoot.Length).TrimStart('\', '/').Replace('\', '/')
                bytes = $file.Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
            }
        }
    }
    $manifest = [ordered]@{
        schema_version = 1
        collected_utc = [DateTime]::UtcNow.ToString('o')
        repository = $repository
        run_id = $runId
        files = $files
        staged_executable_hashes = $binaryHashes
        limits = [ordered]@{
            maximum_files = $maximumFiles
            maximum_file_bytes = $maximumFileBytes
            maximum_total_bytes = $maximumTotalBytes
        }
        allowlist = @('root handoff records', 'logs', 'stage status', 'diagnostics', 'numerics')
    }
    Write-JsonFile $manifest (Join-Path $copyRoot 'collection_manifest.json')
    Compress-Archive -LiteralPath $copyRoot -DestinationPath $temporaryZip -CompressionLevel Optimal
    Move-Item -LiteralPath $temporaryZip -Destination $zip
    Write-Host "Result archive: $zip"
    Write-Host "Archive SHA-256: $((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant())"
} catch {
    throw
} finally {
    if (Test-Path -LiteralPath $temporaryZip) {
        $resolvedZip = [IO.Path]::GetFullPath($temporaryZip)
        if (-not (Test-HandoffPathWithin $resolvedZip $resultsRoot)) {
            throw "Refusing to remove temporary archive outside results: $resolvedZip"
        }
        Remove-Item -LiteralPath $resolvedZip
    }
    if (Test-Path -LiteralPath $temporaryRoot) {
        $resolvedRoot = [IO.Path]::GetFullPath($temporaryRoot)
        if (-not (Test-HandoffPathWithin $resolvedRoot $resultsRoot)) {
            throw "Refusing to remove temporary collection directory outside results: $resolvedRoot"
        }
        Remove-Item -LiteralPath $resolvedRoot -Recurse
    }
}
