# Development setup

## Windows CPU build

Prerequisites: CMake 3.20+, a C++17 compiler (Visual Studio 2022 Build Tools recommended), and Git.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

CUDA and graphics are not required for the default build. Current options are reserved as `NBODY_ENABLE_OPENMP`, `NBODY_ENABLE_CUDA`, `NBODY_ENABLE_MPI`, and `NBODY_BUILD_VISUALIZATION`; unsupported optional stages remain off.

## Current environment note

At foundation creation, `c++` (MSYS2 GCC 15.1.0) was available but `cmake`, `cl`, and `clang++` were not on `PATH`. The stage was therefore compiled and tested directly with `c++`; exact commands are in `docs/MASTER_LEDGER.md`.

## Google Colab CUDA plan

1. Clone or upload the exact Git commit.
2. Record `nvidia-smi`, `nvcc --version`, OS image, compiler, and CMake versions.
3. Configure with CUDA enabled and an architecture supported by the assigned Colab GPU.
4. Run deterministic cross-backend correctness fixtures before timing.
5. Store raw JSON/CSV results with the commit hash and hardware metadata.
6. Treat Colab timings as Colab-only; do not infer the RTX 4050 crossover from them.

The CUDA option is intentionally not implemented in the foundation commit.
