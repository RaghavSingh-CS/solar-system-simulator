# Development setup

## Windows CPU build

Prerequisites: CMake 3.20+, a C++17 compiler (Visual Studio 2022 Build Tools recommended), and Git.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA and graphics are not required for the default build. `NBODY_ENABLE_OPENMP` and `NBODY_ENABLE_CUDA` are implemented and remain off by default. `NBODY_ENABLE_MPI` and `NBODY_BUILD_VISUALIZATION` remain unsupported and off. CUDA-enabled configuration requires a real NVIDIA toolkit and is documented in `docs/colab_cuda_workflow.md`.

## Current environment note

At Stage 2, MSYS2 UCRT64 GCC 15.1.0, `libgomp`, and `mingw32-make` were present. GCC defines `_OPENMP=201511` (OpenMP 4.5). CMake was not installed or on `PATH`. Installing `mingw-w64-ucrt-x86_64-cmake 4.4.3-3` via pacman produced a dependency entry-point failure (`0xC0000139`) in this partially updated MSYS2 environment; a full MSYS2 upgrade was deliberately avoided.

The safe fallback was Kitware's official portable CMake 4.4.3 ZIP, downloaded into ignored `.tools/`, verified against the official `cmake-4.4.3-SHA-256.txt`, and invoked by absolute path. No system-wide PATH was changed. General equivalent commands (when CMake and Ninja are on `PATH`) are:

```powershell
cmake -S . -B build-openmp -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/c++.exe `
  -DNBODY_ENABLE_OPENMP=ON
cmake --build build-openmp --config Release
ctest --test-dir build-openmp -C Release --output-on-failure
```

For a guaranteed CPU-only build, set `-DNBODY_ENABLE_OPENMP=OFF`. Requesting the OpenMP backend from that benchmark binary returns a clear diagnostic rather than silently falling back.

The validated run used Ninja 1.13.2 and the portable executable at `.tools/cmake-4.4.3-windows-x86_64/bin/cmake.exe`; substitute that path for `cmake` and the adjacent `ctest.exe` for `ctest` in the commands above when reproducing this exact environment.

## Google Colab CUDA workflow

1. Clone or upload the exact Git commit.
2. Record `nvidia-smi`, `nvcc --version`, OS image, compiler, and CMake versions.
3. Configure with CUDA enabled and an architecture supported by the assigned Colab GPU.
4. Run deterministic cross-backend correctness fixtures before timing.
5. Store raw JSON/CSV results with the commit hash and hardware metadata.
6. Treat Colab timings as Colab-only; do not infer the RTX 4050 crossover from them.

The optional CUDA code is prepared but has not been compiled or executed on NVIDIA hardware. Use the exact preparation commit and retain the CTest log, raw timings, statistical summary, and environment metadata. Stage 4 remains incomplete until that evidence is returned and reviewed.
