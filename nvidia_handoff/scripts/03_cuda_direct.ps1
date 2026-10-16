[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunDirectory)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '03_cuda_direct'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
try {
    Assert-HandoffStagePassed $run '01_build'
    $regex = '^(nbody_cuda_validation|nbody_visualization_cuda|nbody_counterfactual_cuda)$'
    Invoke-CtestNoSkips $run $stage $regex @(
        'nbody_cuda_tests',
        'nbody_visualization_cuda_tests',
        'nbody_counterfactual_cuda_tests'
    ) | Out-Null
    Write-HandoffStageStatus $run $stage passed 0 'CUDA Direct, viewer-controller, and paired counterfactual tests executed without skips.'
} catch {
    $log = Join-Path $run "logs/$stage.log"
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
