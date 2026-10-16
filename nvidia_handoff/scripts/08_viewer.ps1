[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RunDirectory,
    [ValidateSet('solar', 'catalog', 'synthetic')][string]$Scenario = 'solar',
    [int]$Device = 0,
    [int]$Width = 1920,
    [int]$Height = 1080
)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '08_viewer'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
$log = New-HandoffStageLog $run $stage
$paths = Get-HandoffPaths

try {
    Assert-HandoffStagePassed $run '04_cuda_barnes_hut'
    $viewer = Get-HandoffStagedExecutable $run 'cuda-release' 'nbody_visualizer'
    Write-Warning 'The viewer supports CUDA Direct only. CUDA Barnes-Hut viewer execution is intentionally unavailable.'
    Write-Warning 'This launcher is a manual visual inspection step; close the window to complete it.'
    Invoke-LoggedCommand $viewer @(
        '--scenario', $Scenario, '--backend', 'cuda', '--solver', 'direct',
        '--device', "$Device", '--width', "$Width", '--height', "$Height"
    ) $log $paths.RepositoryRoot | Out-Null
    Write-HandoffStageStatus $run $stage passed 0 'CUDA Direct viewer launched and exited normally; operator visual observations remain manual.'
} catch {
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
