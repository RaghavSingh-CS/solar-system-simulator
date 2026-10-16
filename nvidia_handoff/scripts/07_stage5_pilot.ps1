[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RunDirectory,
    [string]$DeploymentPacket,
    [int]$Device = 0,
    [switch]$ConfirmPilot
)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '07_stage5_pilot'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
$log = New-HandoffStageLog $run $stage

try {
    throw 'Automated Stage 5 execution is disabled because the authoritative deployment packet and Stage 5 v2 runner commands are inconsistent. Use the separate Stage 5 runbook after explicit review.'
} catch {
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
