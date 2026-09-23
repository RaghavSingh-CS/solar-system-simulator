# Colab CUDA workflow

Stage 4 must run from the repository implementation on a real NVIDIA GPU. Colab assignments vary, so every run needs a unique environment identifier and retained metadata. Do not combine its GPU timings with the Stage 3 Windows CPU timings to claim a workstation crossover.

## 1. Start and identify the runtime

Select a GPU runtime in Colab. In a shell cell, clone the repository and check out the exact preparation commit reported in the Stage 4 handoff:

```bash
git clone https://github.com/RaghavSingh-CS/solar-system-simulator.git
cd solar-system-simulator
git checkout ae6117b
git status --short
nvidia-smi
nvcc --version
```

The status must be clean. Do not substitute another commit without recording it.

## 2. Run the pilot

Choose a unique identifier containing the environment and date, then run:

```bash
export STAGE4_COMMIT="$(git rev-parse HEAD)"
export STAGE4_ENVIRONMENT_ID="colab-$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' /' '__')-$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/run_colab_stage4.sh pilot
```

This configures a Release build for the assigned compute capability, builds the repository CUDA targets, runs CPU and CUDA CTest validation, and writes the pilot log/raw/summary/environment files under `data/benchmarks/stage4_cuda/pilot/`.

Stop immediately if configuration, compilation, CTest, correctness status, repeatability checksum, or CUDA error checking fails. Preserve the output; do not loosen tolerances to continue.

Archive and return the pilot before starting the full matrix:

```bash
tar -czf stage4-cuda-pilot.tar.gz data/benchmarks/stage4_cuda/pilot
sha256sum stage4-cuda-pilot.tar.gz
```

Return `stage4-cuda-pilot.tar.gz` and its SHA-256 for inspection. Do not run `full` until the pilot evidence has been accepted.

## 3. Select the full upper bound

After the pilot has been accepted, inspect its runtime and variability. If 4096 bodies is practical, retain the default full matrix. Otherwise record the reason and set a reduced, explicit matrix that still preserves 64–1024:

```bash
export STAGE4_BODIES="64,96,128,160,192,256,384,512,768,1024,1536,2048,3072,4096"
bash scripts/run_colab_stage4.sh full
```

The full run writes separately under `data/benchmarks/stage4_cuda/run/`. Do not overwrite or merge the pilot.

## 4. Return the evidence

Archive both directories without editing their contents:

```bash
tar -czf stage4-cuda-results.tar.gz data/benchmarks/stage4_cuda/pilot data/benchmarks/stage4_cuda/run
sha256sum stage4-cuda-results.tar.gz
```

Return `stage4-cuda-results.tar.gz` and its SHA-256 to the project workspace. Stage 4 remains incomplete until the real-GPU CTest log, raw rows, summaries, environment metadata, and numerical differences are inspected and committed.
