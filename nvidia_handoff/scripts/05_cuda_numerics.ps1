[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunDirectory)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '05_cuda_numerics'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
try {
    Assert-HandoffStagePassed $run '04_cuda_barnes_hut'
    $regex = '^(nbody_cuda_validation|nbody_barnes_hut_cuda|nbody_barnes_hut_cuda_integration|nbody_counterfactual_cuda|nbody_visualization_cuda)$'
    Invoke-CtestNoSkips $run $stage $regex @(
        'nbody_cuda_tests',
        'nbody_barnes_hut_cuda_tests',
        'nbody_barnes_hut_cuda_integration_tests',
        'nbody_counterfactual_cuda_tests',
        'nbody_visualization_cuda_tests'
    ) | Out-Null
    Write-HandoffStageStatus $run $stage passed 0 'Combined CUDA numerical agreement, rollback, repeatability, and metadata tests passed without skips.'
} catch {
    $log = Join-Path $run "logs/$stage.log"
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
