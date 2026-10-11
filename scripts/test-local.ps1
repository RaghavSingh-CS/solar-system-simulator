[CmdletBinding()]
param(
    [ValidateSet('CPU', 'OpenMP', 'CUDA')]
    [string]$Configuration = 'CPU',

    [string[]]$ConfigureArguments = @()
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$preset = switch ($Configuration) {
    'CPU'    { 'cpu-release' }
    'OpenMP' { 'openmp-release' }
    'CUDA'   { 'cuda-release' }
}

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$Program $($Arguments -join ' ')' failed with exit code $LASTEXITCODE."
    }
}

$bundledCMake = Join-Path $repositoryRoot '.tools/cmake-4.4.3-windows-x86_64/bin/cmake.exe'
$bundledCTest = Join-Path $repositoryRoot '.tools/cmake-4.4.3-windows-x86_64/bin/ctest.exe'
if ((Test-Path -LiteralPath $bundledCMake -PathType Leaf) -and
    (Test-Path -LiteralPath $bundledCTest -PathType Leaf)) {
    $cmakeProgram = $bundledCMake
    $ctestProgram = $bundledCTest
} else {
    $cmakeCommand = Get-Command cmake -ErrorAction Stop
    $ctestCommand = Get-Command ctest -ErrorAction Stop
    $cmakeProgram = $cmakeCommand.Source
    $ctestProgram = $ctestCommand.Source
}

Push-Location $repositoryRoot
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$previousTmpDir = $env:TMPDIR
try {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        Invoke-Checked $python.Source @('-B', 'scripts/check_repository_layout.py')
    } else {
        Write-Warning 'Python was not found; the CTest layout check will be unavailable.'
    }

    $dependencyManifestPath = Join-Path $repositoryRoot 'tools/dependency_cache_manifest.json'
    $dependencyManifest = Get-Content -Raw -LiteralPath $dependencyManifestPath | ConvertFrom-Json
    $dependencyRoot = Join-Path $repositoryRoot $dependencyManifest.storage_root
    $dependencyPresence = @(
        $dependencyManifest.dependencies | ForEach-Object {
            Test-Path -LiteralPath (Join-Path $dependencyRoot $_.cache_directory) -PathType Container
        }
    )
    $dependencyOverrides = @()
    if ($dependencyPresence -contains $true) {
        if ($dependencyPresence -contains $false) {
            throw "Persistent dependency cache under '$dependencyRoot' is incomplete. Restore all manifest entries or remove the incomplete cache."
        }
        if (-not $python) {
            throw 'Python is required to verify the installed persistent dependency cache.'
        }
        Invoke-Checked $python.Source @('-B', 'scripts/verify_dependency_cache.py', '--require-present')
        $dependencyOverrides = @(
            $dependencyManifest.dependencies | ForEach-Object {
                $path = (Join-Path $dependencyRoot $_.cache_directory).Replace('\', '/')
                "-D$($_.cmake_override)=$path"
            }
        )
    }

    Invoke-Checked $cmakeProgram (@('--preset', $preset) + $dependencyOverrides + $ConfigureArguments)
    Invoke-Checked $cmakeProgram @('--build', '--preset', $preset)
    $testTemp = Join-Path $repositoryRoot "build/$preset/tmp"
    New-Item -ItemType Directory -Path $testTemp -Force | Out-Null
    $env:TEMP = $testTemp
    $env:TMP = $testTemp
    $env:TMPDIR = $testTemp
    Invoke-Checked $ctestProgram @('--preset', $preset)
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    $env:TMPDIR = $previousTmpDir
    Pop-Location
}
