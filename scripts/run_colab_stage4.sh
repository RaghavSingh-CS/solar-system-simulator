#!/usr/bin/env bash
set -euo pipefail

mode="${1:-pilot}"
expected_commit="${STAGE4_COMMIT:?Set STAGE4_COMMIT to the preparation commit SHA}"
environment_id="${STAGE4_ENVIRONMENT_ID:?Set STAGE4_ENVIRONMENT_ID, e.g. colab-t4-2026-09-23-a}"

missing_tools=()
for tool in nvidia-smi nvcc cmake ctest ninja python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    missing_tools+=("$tool")
  fi
done
if (( ${#missing_tools[@]} > 0 )); then
  printf 'missing required tool(s): %s\n' "${missing_tools[*]}" >&2
  if [[ " ${missing_tools[*]} " == *" ninja "* ]]; then
    echo "Google Colab repair: apt-get update && apt-get install -y ninja-build" >&2
  fi
  exit 1
fi

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  actual_commit="$(git rev-parse HEAD)"
  test "$actual_commit" = "$expected_commit" || {
    echo "Git HEAD $actual_commit does not match STAGE4_COMMIT $expected_commit" >&2
    exit 1
  }
  git diff --quiet && git diff --cached --quiet || {
    echo "Git checkout has tracked changes" >&2
    exit 1
  }
  unexpected_untracked=""
  while IFS= read -r path; do
    case "$path" in
      data/benchmarks/stage4_cuda/pilot/*|data/benchmarks/stage4_cuda/run/*) ;;
      *) unexpected_untracked+="$path"$'\n' ;;
    esac
  done < <(git ls-files --others --exclude-standard)
  if [[ -n "$unexpected_untracked" ]]; then
    printf 'Git checkout has unexpected untracked files:\n%s' "$unexpected_untracked" >&2
    exit 1
  fi
  provenance_mode="git-verified"
  source_archive_sha256="${STAGE4_SOURCE_ARCHIVE_SHA256:-}"
else
  provenance_mode="archive-declared"
  source_archive_sha256="${STAGE4_SOURCE_ARCHIVE_SHA256:?Archive mode requires STAGE4_SOURCE_ARCHIVE_SHA256}"
  echo "No Git metadata found; provenance is archive-declared, not Git-verified." >&2
fi

provenance_args=(
  --source-commit "$expected_commit"
  --provenance-mode "$provenance_mode"
)
if [[ -n "$source_archive_sha256" ]]; then
  provenance_args+=(--source-archive-sha256 "$source_archive_sha256")
fi

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
    --experiment-id stage4-cuda-pilot \
    "${provenance_args[@]}"
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
    --experiment-id stage4-cuda-controlled \
    "${provenance_args[@]}"
else
  echo "usage: $0 pilot|full" >&2
  exit 2
fi
