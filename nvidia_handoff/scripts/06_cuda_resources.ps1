[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunDirectory)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '06_cuda_resources'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
$log = New-HandoffStageLog $run $stage
$paths = Get-HandoffPaths

try {
    Assert-HandoffStagePassed $run '05_cuda_numerics'
    foreach ($target in @(
        'nbody_barnes_hut_cuda_tests',
        'nbody_barnes_hut_cuda_integration_tests',
        'nbody_counterfactual_cuda_tests'
    )) {
        $executable = Get-HandoffStagedExecutable $run 'cuda-release' $target
        Invoke-LoggedCommand $executable @() $log $paths.RepositoryRoot | Out-Null
    }
    Write-HandoffStageStatus $run $stage passed 0 'Resource limits, failure rollback, host-side injection, and repeated executor lifetime tests passed.'
} catch {
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
