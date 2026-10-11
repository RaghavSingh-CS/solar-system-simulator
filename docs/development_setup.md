# Development setup

## Canonical Windows builds

Prerequisites: CMake with preset support, Ninja, a C++17 compiler, Python for
the repository-layout and data tests, and Git. Ordinary CPU and OpenMP builds
must use the tracked presets:

```powershell
.\scripts\test-local.ps1 -Configuration CPU
.\scripts\test-local.ps1 -Configuration OpenMP
```

These use `build/cpu-release/` and `build/openmp-release/`. CUDA is optional:

```powershell
.\scripts\test-local.ps1 -Configuration CUDA
```

The CUDA preset requires a compatible NVIDIA toolkit; runtime validation also
requires an NVIDIA device. MPI remains outside the ordinary preset set and
requires a separately approved `build/scratch/<purpose>/` configuration. Exact
historical Stage 3--5 and MPI commands remain evidence records, not templates
for new root-level build directories.

## Optional MPI build

Install Microsoft MPI's runtime and SDK, then configure through an MPI-aware compiler wrapper:

```powershell
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' -S . -B build/scratch/mpi-release -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/mpicxx.exe `
  -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/ninja.exe `
  -DNBODY_ENABLE_MPI=ON -DNBODY_ENABLE_OPENMP=ON
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' --build build/scratch/mpi-release
ctest --test-dir build/scratch/mpi-release --output-on-failure
```

The test matrix launches one, two, and four MPI ranks plus a two-rank/two-thread hybrid case. Keep ranks multiplied by OpenMP threads within the available logical CPU count for local diagnostics. See `docs/mpi_and_large_n.md` for topology and evidence rules.

## Current environment note

At Stage 2, MSYS2 UCRT64 GCC 15.1.0, `libgomp`, and `mingw32-make` were present. GCC defines `_OPENMP=201511` (OpenMP 4.5). CMake was not installed or on `PATH`. Installing `mingw-w64-ucrt-x86_64-cmake 4.4.3-3` via pacman produced a dependency entry-point failure (`0xC0000139`) in this partially updated MSYS2 environment; a full MSYS2 upgrade was deliberately avoided.

The safe fallback was Kitware's official portable CMake 4.4.3 ZIP, downloaded into ignored `.tools/`, verified against the official `cmake-4.4.3-SHA-256.txt`, and invoked by repository-relative path. No system-wide PATH was changed. The canonical equivalent commands (when CMake and Ninja are on `PATH`) are:

```powershell
cmake --preset openmp-release
cmake --build --preset openmp-release
ctest --preset openmp-release
```

The CPU preset guarantees `NBODY_ENABLE_OPENMP=OFF`. Requesting the OpenMP
backend from that binary returns a clear diagnostic rather than silently
falling back.

The validated run used Ninja 1.13.2 and the portable executable at `.tools/cmake-4.4.3-windows-x86_64/bin/cmake.exe`; substitute that path for `cmake` and the adjacent `ctest.exe` for `ctest` in the commands above when reproducing this exact environment.

## Google Colab CUDA workflow

1. Install Ninja if absent: `apt-get update && apt-get install -y ninja-build`.
2. Clone the exact Git commit, or upload a hashed `git archive` and declare both its commit and archive SHA-256.
3. Record `nvidia-smi`, `nvcc --version`, OS image, compiler, and CMake versions.
4. Configure with CUDA enabled and an architecture supported by the assigned Colab GPU.
5. Run deterministic cross-backend correctness fixtures before timing.
6. Store raw JSON/CSV results with the commit hash, provenance mode, and hardware metadata.
7. Treat Colab timings as Colab-only; do not infer the RTX 4050 crossover from them.

The optional CUDA code from preparation commit `ae6117b` was compiled and validated on a Tesla T4. The accepted CTest log, raw timings, statistical summary, environment metadata, and independent verification are retained under `data/benchmarks/stage4_cuda/`. The local Windows machine still has no CUDA compiler or NVIDIA device.
