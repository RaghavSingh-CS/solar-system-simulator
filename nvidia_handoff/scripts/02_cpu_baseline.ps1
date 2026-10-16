[CmdletBinding()]
param([Parameter(Mandatory)][string]$RunDirectory)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '02_cpu_baseline'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
$log = New-HandoffStageLog $run $stage
$paths = Get-HandoffPaths

try {
    Assert-HandoffStagePassed $run '01_build'
    $manifest = Get-Content -Raw (Join-Path $run 'executables.json') | ConvertFrom-Json
    foreach ($required in @('cpu-release', 'openmp-release')) {
        if (@($manifest.configured_builds) -notcontains $required) {
            throw "CPU/OpenMP baseline requires a full build; missing configuration: $required"
        }
    }
    foreach ($preset in @('cpu-release', 'openmp-release')) {
        Invoke-LoggedCommand 'ctest' @('--preset', $preset, '--output-on-failure', '--no-tests=error') `
            $log $paths.RepositoryRoot | Out-Null
    }
    Write-HandoffStageStatus $run $stage passed 0 'Canonical CPU and OpenMP Release CTest baselines passed.'
} catch {
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
