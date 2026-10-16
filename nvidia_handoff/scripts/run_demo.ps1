[CmdletBinding()]
param(
    [string]$RunDirectory,
    [int]$Device = 0,
    [ValidateSet('solar', 'catalog')][string]$Scenario = 'solar'
)

. (Join-Path $PSScriptRoot 'common.ps1')
$run = Resolve-HandoffRunDirectory $RunDirectory
$failed = $false
$failure = $null

try {
    & (Join-Path $PSScriptRoot '00_preflight.ps1') -RunDirectory $run -Device $Device
    & (Join-Path $PSScriptRoot '01_build.ps1') -RunDirectory $run -CudaOnly
    & (Join-Path $PSScriptRoot '03_cuda_direct.ps1') -RunDirectory $run
    & (Join-Path $PSScriptRoot '04_cuda_barnes_hut.ps1') -RunDirectory $run
    & (Join-Path $PSScriptRoot '08_viewer.ps1') `
        -RunDirectory $run -Scenario $Scenario -Device $Device
} catch {
    $failed = $true
    $failure = $_
    Write-Warning "Demonstration preparation stopped: $($_.Exception.Message)"
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
Write-Host 'Five-minute CUDA demonstration path completed.'
