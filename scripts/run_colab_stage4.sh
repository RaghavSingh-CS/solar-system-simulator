#!/usr/bin/env bash
set -euo pipefail

mode="${1:-pilot}"
expected_commit="${STAGE4_COMMIT:?Set STAGE4_COMMIT to the preparation commit SHA}"
environment_id="${STAGE4_ENVIRONMENT_ID:?Set STAGE4_ENVIRONMENT_ID, e.g. colab-t4-2026-09-23-a}"

test "$(git rev-parse HEAD)" = "$expected_commit"
test -z "$(git status --porcelain)"
nvidia-smi
nvcc --version

compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$compute_capability" \
  -DNBODY_ENABLE_CUDA=ON
cmake --build build-cuda --config Release

if [[ "$mode" == "pilot" ]]; then
  output_dir="data/benchmarks/stage4_cuda/pilot"
  mkdir -p "$output_dir"
  ctest --test-dir build-cuda -C Release --output-on-failure | tee "$output_dir/cuda_ctest.txt"
  python3 -B scripts/run_cuda_benchmarks.py \
    --executable build-cuda/nbody_cuda_benchmark \
    --output-dir "$output_dir" \
    --bodies 64,256,1024,2048,4096 \
    --steps 10 --warmups 1 --repetitions 3 \
    --environment-id "$environment_id" \
    --experiment-id stage4-cuda-pilot
elif [[ "$mode" == "full" ]]; then
  bodies="${STAGE4_BODIES:-64,96,128,160,192,256,384,512,768,1024,1536,2048,3072,4096}"
  output_dir="data/benchmarks/stage4_cuda/run"
  mkdir -p "$output_dir"
  ctest --test-dir build-cuda -C Release --output-on-failure | tee "$output_dir/cuda_ctest.txt"
  python3 -B scripts/run_cuda_benchmarks.py \
    --executable build-cuda/nbody_cuda_benchmark \
    --output-dir "$output_dir" \
    --bodies "$bodies" \
    --steps 100 --warmups 1 --repetitions 11 \
    --environment-id "$environment_id" \
    --experiment-id stage4-cuda-controlled
else
  echo "usage: $0 pilot|full" >&2
  exit 2
fi
