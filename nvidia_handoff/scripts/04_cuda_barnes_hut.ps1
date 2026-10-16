[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunDirectory)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '04_cuda_barnes_hut'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
try {
    Assert-HandoffStagePassed $run '03_cuda_direct'
    $regex = '^(nbody_barnes_hut_cuda|nbody_barnes_hut_cuda_integration)$'
    Invoke-CtestNoSkips $run $stage $regex @(
        'nbody_barnes_hut_cuda_tests',
        'nbody_barnes_hut_cuda_integration_tests'
    ) | Out-Null
    Write-HandoffStageStatus $run $stage passed 0 'CUDA Barnes-Hut force and Velocity Verlet tests executed without skips.'
} catch {
    $log = Join-Path $run "logs/$stage.log"
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
