[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RunDirectory,
    [switch]$CudaOnly
)

. (Join-Path $PSScriptRoot 'common.ps1')
$stage = '01_build'
$run = Resolve-HandoffRunDirectory $RunDirectory
Assert-HandoffStageFresh $run $stage
$log = New-HandoffStageLog $run $stage
$paths = Get-HandoffPaths

try {
    Assert-HandoffStagePassed $run '00_preflight'
    $source = Get-RepositoryRecord
    $preflight = Get-Content -Raw (Join-Path $run 'preflight.json') | ConvertFrom-Json
    if (-not $source.clean -or $source.head -ne $preflight.repository.head) {
        throw 'Source identity or cleanliness changed after preflight.'
    }

    $buildConfigurations = @('cuda-release')
    if (-not $CudaOnly) {
        $buildConfigurations = @('cpu-release', 'openmp-release', 'cuda-release')
        foreach ($preset in @('cpu-release', 'openmp-release')) {
            Invoke-LoggedCommand 'cmake' @('--preset', $preset) $log $paths.RepositoryRoot | Out-Null
            Invoke-LoggedCommand 'cmake' @('--build', '--preset', $preset, '--parallel') $log $paths.RepositoryRoot | Out-Null
        }
    }
    Invoke-LoggedCommand 'cmake' @(
        '--preset', 'cuda-release',
        '-DCMAKE_CUDA_ARCHITECTURES=89',
        '-DNBODY_ENABLE_OPENMP=ON'
    ) $log $paths.RepositoryRoot | Out-Null
    Invoke-LoggedCommand 'cmake' @('--build', '--preset', 'cuda-release', '--parallel') $log $paths.RepositoryRoot | Out-Null

    $cudaCache = Join-Path $paths.RepositoryRoot 'build/cuda-release/CMakeCache.txt'
    if (-not (Test-Path -LiteralPath $cudaCache)) { throw 'CUDA CMakeCache.txt is missing.' }
    $cacheText = Get-Content -Raw $cudaCache
    $requiredCacheValues = [ordered]@{
        CMAKE_CUDA_ARCHITECTURES = '89'
        NBODY_ENABLE_CUDA = 'ON'
        NBODY_ENABLE_OPENMP = 'ON'
        CMAKE_BUILD_TYPE = 'Release'
    }
    foreach ($required in $requiredCacheValues.GetEnumerator()) {
        $pattern = "(?m)^$([regex]::Escape([string]$required.Key)):[^=]+=$([regex]::Escape([string]$required.Value))\r?$"
        if ($cacheText -notmatch $pattern) {
            throw "CUDA CMake cache does not contain required value: $($required.Key)=$($required.Value)"
        }
    }
    $cudaCompiler = [regex]::Match($cacheText, '(?m)^CMAKE_CUDA_COMPILER:[^=]+=(.+)\r?$').Groups[1].Value
    $hostCompiler = [regex]::Match($cacheText, '(?m)^CMAKE_CXX_COMPILER:[^=]+=(.+)\r?$').Groups[1].Value
    if (-not $cudaCompiler -or -not $hostCompiler) {
        throw 'CUDA or C++ compiler identity is missing from CMakeCache.txt.'
    }
    if ((Split-Path -Leaf $cudaCompiler) -notmatch '^nvcc(\.exe)?$' -or
        (Split-Path -Leaf $hostCompiler) -notmatch '^cl(\.exe)?$') {
        throw "Unexpected CUDA/host compiler tuple: CUDA=$cudaCompiler CXX=$hostCompiler"
    }

    $runId = Split-Path -Leaf $run
    $stagingRoot = Join-Path $paths.BinRoot $runId
    if (Test-Path -LiteralPath $stagingRoot) {
        throw "Executable staging directory already exists and will not be overwritten: $stagingRoot"
    }
    New-Item -ItemType Directory -Path $stagingRoot | Out-Null
    $entries = @()
    foreach ($target in @($paths.ExpectedManifest.targets | Where-Object {
        $buildConfigurations -contains [string]$_.configuration
    })) {
        $configuration = [string]$target.configuration
        $buildDirectory = Join-Path $paths.RepositoryRoot "build/$configuration"
        $sourcePath = Get-BuildExecutable $buildDirectory ([string]$target.target)
        $destinationDirectory = Join-Path $stagingRoot $configuration
        if (-not (Test-Path -LiteralPath $destinationDirectory)) {
            New-Item -ItemType Directory -Path $destinationDirectory | Out-Null
        }
        $destination = Join-Path $destinationDirectory (Split-Path -Leaf $sourcePath)
        Copy-Item -LiteralPath $sourcePath -Destination $destination
        $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash.ToLowerInvariant()
        $stagedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
        if ($sourceHash -ne $stagedHash) {
            throw "Executable staging changed bytes: $configuration/$($target.target)"
        }
        $entries += [ordered]@{
            file_name = Split-Path -Leaf $destination
            cmake_target = [string]$target.target
            role = [string]$target.role
            source_build_location = $sourcePath
            staged_location = $destination
            sha256 = $stagedHash
            bytes = (Get-Item -LiteralPath $destination).Length
            build_configuration = $configuration
            cuda_architecture = $(if ($configuration -eq 'cuda-release') { '89' } else { $null })
            gpu_validation_status = $(if ($configuration -eq 'cuda-release') { 'unverified until genuine NVIDIA execution' } else { 'not applicable' })
        }
        if ($configuration -eq 'cuda-release') {
            Invoke-LoggedCommand 'dumpbin' @('/dependents', $destination) $log $paths.RepositoryRoot | Out-Null
        }
    }

    $runtimeFiles = @()
    foreach ($configuration in $buildConfigurations) {
        $buildDirectory = Join-Path $paths.RepositoryRoot "build/$configuration"
        foreach ($dll in @(Get-ChildItem -LiteralPath $buildDirectory -Filter '*.dll' -File -ErrorAction SilentlyContinue)) {
            $destinationDirectory = Join-Path $stagingRoot $configuration
            $destination = Join-Path $destinationDirectory $dll.Name
            Copy-Item -LiteralPath $dll.FullName -Destination $destination
            $runtimeFiles += [ordered]@{
                file_name = $dll.Name
                source_build_location = $dll.FullName
                staged_location = $destination
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
            }
        }
    }

    $manifest = [ordered]@{
        schema_version = 1
        generated_utc = [DateTime]::UtcNow.ToString('o')
        source_commit = $source.head
        source_clean = $source.clean
        build_configuration = 'Release'
        configured_builds = $buildConfigurations
        cmake_cuda_architecture = '89'
        cuda_compiler = $cudaCompiler
        cxx_compiler = $hostCompiler
        executable_count = $entries.Count
        executables = $entries
        adjacent_runtime_files = $runtimeFiles
        note = 'CUDA binaries are compiled but remain GPU-unverified until subsequent stages pass.'
    }
    $manifestPath = Join-Path $stagingRoot 'executables.json'
    Write-JsonFile $manifest $manifestPath
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $run 'executables.json')
    Write-HandoffStageStatus $run $stage passed 0 "Release builds completed ($($buildConfigurations -join ', ')); executables staged under $stagingRoot"
    Write-Host "Build and staging passed: $manifestPath"
} catch {
    $_ | Out-String | Add-Content -LiteralPath $log -Encoding utf8
    Write-HandoffStageStatus $run $stage failed 1 $_.Exception.Message
    throw
}
