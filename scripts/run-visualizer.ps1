[CmdletBinding()]
param(
    [ValidateSet('CPU', 'OpenMP', 'CUDA')]
    [string]$Backend = 'CPU',

    [string[]]$ApplicationArguments = @(),

    [switch]$PrintCommand
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$selection = switch ($Backend) {
    'CPU'    { @{ Directory = 'cpu-release'; Backend = 'sequential' } }
    'OpenMP' { @{ Directory = 'openmp-release'; Backend = 'openmp' } }
    'CUDA'   { @{ Directory = 'cuda-release'; Backend = 'cuda' } }
}
$executable = Join-Path $repositoryRoot "build/$($selection.Directory)/nbody_visualizer.exe"

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    $preset = $selection.Directory
    throw "Viewer not found at '$executable'. Build it first with '.\scripts\test-local.ps1 -Configuration $Backend'. No backend substitution was attempted."
}

if ($PrintCommand) {
    [pscustomobject]@{
        Executable = $executable
        Backend = $selection.Backend
        Arguments = @('--backend', $selection.Backend) + $ApplicationArguments
    }
    return
}

Push-Location $repositoryRoot
try {
    & $executable '--backend' $selection.Backend @ApplicationArguments
    if ($LASTEXITCODE -ne 0) {
        throw "The $Backend viewer exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
