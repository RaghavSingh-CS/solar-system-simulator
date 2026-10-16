[CmdletBinding()]
param(
    [string]$RunDirectory,
    [int]$Device = 0,
    [switch]$LaunchViewer,
    [ValidateSet('solar', 'catalog', 'synthetic')][string]$ViewerScenario = 'solar'
)

. (Join-Path $PSScriptRoot 'common.ps1')
$run = Resolve-HandoffRunDirectory $RunDirectory
$failed = $false
$failure = $null

try {
    & (Join-Path $PSScriptRoot '00_preflight.ps1') -RunDirectory $run -Device $Device
    & (Join-Path $PSScriptRoot '01_build.ps1') -RunDirectory $run
    & (Join-Path $PSScriptRoot '02_cpu_baseline.ps1') -RunDirectory $run
    & (Join-Path $PSScriptRoot '03_cuda_direct.ps1') -RunDirectory $run
    & (Join-Path $PSScriptRoot '04_cuda_barnes_hut.ps1') -RunDirectory $run
    & (Join-Path $PSScriptRoot '05_cuda_numerics.ps1') -RunDirectory $run
    & (Join-Path $PSScriptRoot '06_cuda_resources.ps1') -RunDirectory $run

    if ($LaunchViewer) {
        & (Join-Path $PSScriptRoot '08_viewer.ps1') `
            -RunDirectory $run -Scenario $ViewerScenario -Device $Device
    }
} catch {
    $failed = $true
    $failure = $_
    Write-Warning "Validation stopped: $($_.Exception.Message)"
} finally {
    try {
        & (Join-Path $PSScriptRoot '09_collect_results.ps1') -RunDirectory $run
    } catch {
        Write-Warning "Result collection also failed: $($_.Exception.Message)"
        if (-not $failed) {
            $failed = $true
            $failure = $_
        }
    }
}

Write-Host "Run directory: $run"
if ($failed) { throw $failure }
Write-Host 'All requested NVIDIA handoff stages passed.'
