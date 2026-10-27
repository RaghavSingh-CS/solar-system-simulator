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

if ($Backend -eq 'CUDA') {
    $cachePath = Join-Path $repositoryRoot 'build/cuda-release/CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf) -or
        (Get-Content -Raw -LiteralPath $cachePath) -notmatch '(?m)^NBODY_ENABLE_OPENMP:[^=]+=ON\r?$') {
        throw "CUDA viewer cache lacks OpenMP. Rebuild with '.\scripts\test-local.ps1 -Configuration CUDA' or 'cmake --preset cuda-release -DNBODY_ENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=89', then build. The source-custodied preset itself is retained unchanged."
    }
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
