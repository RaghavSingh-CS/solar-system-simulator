# Stage 4 CUDA benchmark data

Stage 4 GPU results have **not** been generated yet. This directory is reserved for real NVIDIA-hardware evidence produced by `scripts/run_cuda_benchmarks.py`; do not add fabricated or estimated rows.

Expected run outputs:

- `raw_timings.csv`: one row per measured repetition, with execution order, workload, CUDA device metadata, correctness status, device-compute timing, transfer-inclusive timing, allocation, host staging, diagnostics, and checksum;
- `summary.csv`: median, mean, sample standard deviation, min/max, IQR, coefficient of variation, and transfer penalty by body count and timing regime;
- `environment.json`: GPU/driver/toolkit, host, Git revision, complete matrix, and exact timing definitions.

Pilot and final results must use separate subdirectories. Stage 3 Windows CPU data must not be merged with Colab GPU data. A CPU/GPU speedup is permitted only if its CPU baseline was collected on the same Colab VM and is labeled environment-specific.
