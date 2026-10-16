Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-HandoffPaths {
    $handoffRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $repositoryRoot = (Resolve-Path (Join-Path $handoffRoot '..')).Path
    [pscustomobject]@{
        RepositoryRoot = $repositoryRoot
        HandoffRoot = $handoffRoot
        BinRoot = Join-Path $handoffRoot 'bin'
        DefaultResultsRoot = Join-Path $handoffRoot 'results'
        Configuration = Get-Content -Raw (Join-Path $handoffRoot 'config/handoff.json') | ConvertFrom-Json
        ExpectedManifest = Get-Content -Raw (Join-Path $handoffRoot 'MANIFEST.json') | ConvertFrom-Json
    }
}

function Test-HandoffPathWithin {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary
    )
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $root = [IO.Path]::GetFullPath($Boundary).TrimEnd('\', '/')
    return $full.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or
        $full.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-HandoffNoReparsePoints {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary
    )
    $full = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetFullPath($Boundary).TrimEnd('\', '/')
    if (-not (Test-HandoffPathWithin $full $root)) {
        throw "Path escapes the NVIDIA handoff boundary: $full"
    }
    $probe = if (Test-Path -LiteralPath $full) {
        Get-Item -Force -LiteralPath $full
    } else {
        Get-Item -Force -LiteralPath (Split-Path -Parent $full)
    }
    while ($probe) {
        if (($probe.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse points are not permitted in NVIDIA handoff paths: $($probe.FullName)"
        }
        if ($probe.FullName.TrimEnd('\', '/').Equals($root, [StringComparison]::OrdinalIgnoreCase)) {
            return
        }
        $probe = if ($probe -is [IO.DirectoryInfo]) { $probe.Parent } else { $probe.Directory }
    }
    throw "Unable to prove path containment beneath $root"
}

function Resolve-HandoffRunDirectory {
    param([string]$RunDirectory)
    $paths = Get-HandoffPaths
    if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
        $timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss.fffZ')
        $shortSha = (& git -C $paths.RepositoryRoot rev-parse --short=12 HEAD).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Unable to identify the source commit.' }
        $RunDirectory = Join-Path $paths.DefaultResultsRoot "run-$timestamp-$shortSha"
    }
    $resultsRoot = [IO.Path]::GetFullPath($paths.DefaultResultsRoot)
    $full = [IO.Path]::GetFullPath($RunDirectory)
    if ($full.TrimEnd('\', '/').Equals($resultsRoot.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-HandoffPathWithin $full $resultsRoot)) {
        throw "Run directories must be direct children of the NVIDIA handoff results directory: $resultsRoot"
    }
    $relative = $full.Substring($resultsRoot.TrimEnd('\', '/').Length).TrimStart('\', '/')
    if ($relative.Contains('\') -or $relative.Contains('/')) {
        throw "Run directories must be direct children of the NVIDIA handoff results directory: $resultsRoot"
    }
    Assert-HandoffNoReparsePoints $full $resultsRoot
    $created = -not (Test-Path -LiteralPath $full)
    if ($created) {
        New-Item -ItemType Directory -Path $full | Out-Null
    }
    Assert-HandoffNoReparsePoints $full $resultsRoot
    $marker = Join-Path $full '.handoff-run.json'
    if ($created) {
        Write-JsonFile -Path $marker -Value ([ordered]@{
            schema_version = 1
            kind = 'solar-system-simulator-nvidia-handoff-run'
            run_id = Split-Path -Leaf $full
            created_utc = [DateTime]::UtcNow.ToString('o')
        })
    } elseif (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw "Existing directory is not a marked NVIDIA handoff run: $full"
    } else {
        $record = Get-Content -Raw -LiteralPath $marker | ConvertFrom-Json
        if ($record.schema_version -ne 1 -or
            $record.kind -ne 'solar-system-simulator-nvidia-handoff-run' -or
            $record.run_id -ne (Split-Path -Leaf $full)) {
            throw "Invalid NVIDIA handoff run marker: $marker"
        }
    }
    foreach ($name in @('logs', 'stages')) {
        $child = Join-Path $full $name
        if ($created) {
            New-Item -ItemType Directory -Path $child | Out-Null
        } elseif (-not (Test-Path -LiteralPath $child -PathType Container)) {
            throw "Marked NVIDIA handoff run is incomplete; missing directory: $child"
        }
    }
    return $full
}

function Write-JsonFile {
    param([Parameter(Mandatory)]$Value, [Parameter(Mandatory)][string]$Path)
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Write-HandoffStageStatus {
    param(
        [Parameter(Mandatory)][string]$RunDirectory,
        [Parameter(Mandatory)][string]$Stage,
        [Parameter(Mandatory)][ValidateSet('passed', 'failed', 'not_run')][string]$Status,
        [int]$ExitCode = 0,
        [string]$Message = ''
    )
    $path = Join-Path $RunDirectory "stages/$Stage.json"
    if (Test-Path -LiteralPath $path) {
        throw "Stage status already exists and will not be overwritten: $path"
    }
    Write-JsonFile -Path $path -Value ([ordered]@{
        schema_version = 1
        stage = $Stage
        status = $Status
        exit_code = $ExitCode
        message = $Message
        recorded_utc = [DateTime]::UtcNow.ToString('o')
    })
}

function Assert-HandoffStageFresh {
    param(
        [Parameter(Mandatory)][string]$RunDirectory,
        [Parameter(Mandatory)][string]$Stage
    )
    $log = Join-Path $RunDirectory "logs/$Stage.log"
    $status = Join-Path $RunDirectory "stages/$Stage.json"
    if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $status)) {
        throw "Stage '$Stage' already has immutable results; use a fresh run directory."
    }
}

function Assert-HandoffStagePassed {
    param([Parameter(Mandatory)][string]$RunDirectory, [Parameter(Mandatory)][string]$Stage)
    $path = Join-Path $RunDirectory "stages/$Stage.json"
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required stage '$Stage' has not run in $RunDirectory"
    }
    $record = Get-Content -Raw $path | ConvertFrom-Json
    if ($record.status -ne 'passed') {
        throw "Required stage '$Stage' did not pass (status: $($record.status))."
    }
}

function New-HandoffStageLog {
    param([Parameter(Mandatory)][string]$RunDirectory, [Parameter(Mandatory)][string]$Stage)
    Assert-HandoffStageFresh $RunDirectory $Stage
    $path = Join-Path $RunDirectory "logs/$Stage.log"
    "stage=$Stage`nstarted_utc=$([DateTime]::UtcNow.ToString('o'))" |
        Set-Content -LiteralPath $path -Encoding utf8
    return $path
}

function Get-HandoffStagedExecutable {
    param(
        [Parameter(Mandatory)][string]$RunDirectory,
        [Parameter(Mandatory)][string]$Configuration,
        [Parameter(Mandatory)][string]$Target
    )
    $paths = Get-HandoffPaths
    $manifestPath = Join-Path $RunDirectory 'executables.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Generated executable manifest is missing: $manifestPath"
    }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $records = @($manifest.executables | Where-Object {
        $_.build_configuration -eq $Configuration -and $_.cmake_target -eq $Target
    })
    if ($records.Count -ne 1) {
        throw "Expected exactly one staged executable for ${Configuration}/${Target}; found $($records.Count)."
    }
    $record = $records[0]
    $runId = Split-Path -Leaf $RunDirectory
    $expectedRoot = Join-Path (Join-Path $paths.BinRoot $runId) $Configuration
    $staged = [IO.Path]::GetFullPath([string]$record.staged_location)
    if (-not (Test-HandoffPathWithin $staged $expectedRoot)) {
        throw "Staged executable path escapes its expected directory: $staged"
    }
    Assert-HandoffNoReparsePoints $staged $paths.BinRoot
    if (-not (Test-Path -LiteralPath $staged -PathType Leaf)) {
        throw "Staged executable is missing: $staged"
    }
    $expectedName = "$Target.exe"
    if ((Split-Path -Leaf $staged) -ne $expectedName) {
        throw "Staged executable name mismatch: expected $expectedName, found $(Split-Path -Leaf $staged)"
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $staged).Hash.ToLowerInvariant()
    if ($actualHash -ne ([string]$record.sha256).ToLowerInvariant()) {
        throw "Staged executable SHA-256 mismatch: ${Configuration}/${Target}"
    }
    $source = [IO.Path]::GetFullPath([string]$record.source_build_location)
    $expectedBuildRoot = Join-Path $paths.RepositoryRoot "build/$Configuration"
    if (-not (Test-HandoffPathWithin $source $expectedBuildRoot) -or
        -not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Manifest source executable is missing or outside the expected build: $source"
    }
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
    if ($sourceHash -ne $actualHash) {
        throw "Build/staging executable identity mismatch: ${Configuration}/${Target}"
    }
    return $staged
}

function Assert-HandoffStagedExecutables {
    param(
        [Parameter(Mandatory)][string]$RunDirectory,
        [Parameter(Mandatory)][string]$Configuration,
        [Parameter(Mandatory)][string[]]$Targets
    )
    foreach ($target in $Targets) {
        Get-HandoffStagedExecutable $RunDirectory $Configuration $target | Out-Null
    }
}

function Format-CommandLine {
    param([string]$FilePath, [string[]]$ArgumentList)
    $parts = @($FilePath) + @($ArgumentList | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    })
    return ($parts -join ' ')
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory)][string]$LogPath,
        [string]$WorkingDirectory = '',
        [switch]$AllowFailure
    )
    $commandLine = Format-CommandLine $FilePath $ArgumentList
    "`ncommand=$commandLine`ncommand_started_utc=$([DateTime]::UtcNow.ToString('o'))" |
        Add-Content -LiteralPath $LogPath -Encoding utf8
    Write-Host "> $commandLine"
    $previous = (Get-Location).Path
    try {
        if ($WorkingDirectory) { Set-Location -LiteralPath $WorkingDirectory }
        & $FilePath @ArgumentList 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            $line | Add-Content -LiteralPath $LogPath -Encoding utf8
        }
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) { $exitCode = 0 }
    } finally {
        Set-Location -LiteralPath $previous
    }
    "command_exit_code=$exitCode`ncommand_completed_utc=$([DateTime]::UtcNow.ToString('o'))" |
        Add-Content -LiteralPath $LogPath -Encoding utf8
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "Command failed with exit code ${exitCode}: $commandLine"
    }
    return [int]$exitCode
}

function Invoke-CtestNoSkips {
    param(
        [Parameter(Mandatory)][string]$RunDirectory,
        [Parameter(Mandatory)][string]$Stage,
        [Parameter(Mandatory)][string]$RegularExpression,
        [Parameter(Mandatory)][string[]]$ExpectedTargets
    )
    $paths = Get-HandoffPaths
    Assert-HandoffStagedExecutables $RunDirectory 'cuda-release' $ExpectedTargets
    $log = New-HandoffStageLog $RunDirectory $Stage
    Invoke-LoggedCommand -FilePath 'ctest' -ArgumentList @(
        '--test-dir', (Join-Path $paths.RepositoryRoot 'build/cuda-release'),
        '-C', 'Release', '--output-on-failure', '--no-tests=error', '-R', $RegularExpression
    ) -LogPath $log -WorkingDirectory $paths.RepositoryRoot | Out-Null
    $text = Get-Content -Raw $log
    if ($text -match '(?im)\*\*\*Skipped|\bNot Run\b|tests? skipped') {
        throw "CUDA validation contained a skipped or not-run test; inspect $log"
    }
    return $log
}

function Get-BuildExecutable {
    param([Parameter(Mandatory)][string]$BuildDirectory, [Parameter(Mandatory)][string]$Target)
    foreach ($candidate in @(
        (Join-Path $BuildDirectory "$Target.exe"),
        (Join-Path $BuildDirectory "Release/$Target.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "Expected executable was not built: $Target ($BuildDirectory)"
}

function Get-RepositoryRecord {
    $paths = Get-HandoffPaths
    $head = (& git -C $paths.RepositoryRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'git rev-parse failed.' }
    $branch = (& git -C $paths.RepositoryRoot rev-parse --abbrev-ref HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'git branch discovery failed.' }
    $status = @(& git -C $paths.RepositoryRoot status --porcelain=v1)
    if ($LASTEXITCODE -ne 0) { throw 'git status failed.' }
    [ordered]@{
        head = $head
        branch = $branch
        clean = ($status.Count -eq 0)
        status = @($status)
    }
}
