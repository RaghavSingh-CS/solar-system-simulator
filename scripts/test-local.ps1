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
    Invoke-Checked $cmakeProgram (@('--preset', $preset) + $ConfigureArguments)
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
