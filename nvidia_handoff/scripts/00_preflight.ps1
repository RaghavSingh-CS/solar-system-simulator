[CmdletBinding()]
param(
    [string]$RunDirectory,
    [int]$Device = 0
)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '00_preflight'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
$log = New-HandoffStageLog $run $stage
$paths = Get-HandoffPaths

try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw 'The NVIDIA handoff supports native Windows only.'
    }
    $toolNames = @('git', 'python', 'cmake', 'ctest', 'ninja', 'cl', 'dumpbin', 'nvcc', 'nvidia-smi')
    $tools = [ordered]@{}
    foreach ($name in $toolNames) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $command) { throw "Required tool is unavailable on PATH: $name" }
        $tools[$name] = $command.Source
    }
    foreach ($name in @('cmake', 'ctest', 'ninja')) {
        $lower = $tools[$name].ToLowerInvariant().Replace('/', '\')
        if ($lower.Contains('\msys') -or $lower.Contains('\mingw')) {
            throw "$name resolves to an unsupported MSYS/MinGW path: $($tools[$name])"
        }
    }

    Invoke-LoggedCommand $tools.git @('--version') $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.python @('--version') $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.cmake @('--version') $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.ctest @('--version') $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.ninja @('--version') $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.cl @() $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.dumpbin @('/?') $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.nvcc @('--version') $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand $tools.nvcc @('--list-gpu-arch') $log $paths.RepositoryRoot | Out-Null

    $architectureOutput = & $tools.nvcc --list-gpu-arch 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $architectureOutput -notmatch '(?m)^compute_89\s*$') {
        throw 'nvcc does not report native compute_89 support; CUDA 11.8 or newer is required.'
    }

    Invoke-LoggedCommand $tools['nvidia-smi'] @() $log $paths.RepositoryRoot | Out-Null
    $queryArgs = @('--query-gpu=index,name,memory.total,driver_version,compute_cap', '--format=csv,noheader,nounits')
    Invoke-LoggedCommand $tools['nvidia-smi'] $queryArgs $log $paths.RepositoryRoot | Out-Null
    $rows = @(& $tools['nvidia-smi'] @queryArgs 2>&1)
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi device query failed.' }
    $selected = $null
    foreach ($row in $rows) {
        $fields = "$row".Split(',') | ForEach-Object { $_.Trim() }
        if ($fields.Count -ge 5 -and $fields[0] -eq "$Device") {
            $selected = $fields
            break
        }
    }
    if (-not $selected) { throw "CUDA device index $Device was not reported by nvidia-smi." }
    if ($selected[1] -notlike "*$($paths.Configuration.expected_gpu_name_contains)*") {
        throw "Selected GPU is not the required RTX 4050: $($selected[1])"
    }
    if ($selected[4] -ne $paths.Configuration.required_compute_capability) {
        throw "Selected GPU compute capability is $($selected[4]); expected $($paths.Configuration.required_compute_capability)."
    }
    if ([string]::IsNullOrWhiteSpace($env:VCToolsInstallDir) -or
        [string]::IsNullOrWhiteSpace($env:WindowsSdkDir)) {
        throw 'Visual C++ tools or Windows SDK environment is missing; use an x64 VS2022 Native Tools shell.'
    }

    $record = Get-RepositoryRecord
    if (-not $record.clean) { throw 'The source checkout must be clean before NVIDIA validation.' }
    if ($record.branch -ne $paths.ExpectedManifest.required_source_branch) {
        throw "Expected branch '$($paths.ExpectedManifest.required_source_branch)', found '$($record.branch)'."
    }

    $provenance = [ordered]@{
        schema_version = 1
        captured_utc = [DateTime]::UtcNow.ToString('o')
        repository = $record
        device = [ordered]@{
            index = [int]$selected[0]
            name = $selected[1]
            memory_total_mib = [int64]$selected[2]
            driver_version = $selected[3]
            compute_capability = $selected[4]
        }
        tools = [ordered]@{
            paths = $tools
            git_version = (& $tools.git --version 2>&1 | Out-String).Trim()
            python_version = (& $tools.python --version 2>&1 | Out-String).Trim()
            cmake_version = (& $tools.cmake --version 2>&1 | Out-String).Trim()
            ninja_version = (& $tools.ninja --version 2>&1 | Out-String).Trim()
            nvcc_version = (& $tools.nvcc --version 2>&1 | Out-String).Trim()
            cuda_path = $env:CUDA_PATH
            vc_tools_install_dir = $env:VCToolsInstallDir
            windows_sdk_dir = $env:WindowsSdkDir
        }
        requested_cuda_architecture = $paths.Configuration.cuda_architecture
        validation_status = 'preflight only; CUDA correctness not yet established'
    }
    Write-JsonFile $provenance (Join-Path $run 'preflight.json')
    Write-HandoffStageStatus $run $stage passed 0 'RTX 4050 hardware and native Windows toolchain preflight passed.'
    Write-Host "Preflight passed. Run directory: $run"
} catch {
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
