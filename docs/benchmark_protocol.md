# Benchmark protocol

## Principle

Crossover is a property of a particular workload, implementation, build, and machine. Only backends timed on the same machine in the same experimental block may be compared.

## Workload families

- **Physical:** verified Solar System snapshot, small N, used for correctness and visualization—not scalability claims.
- **Synthetic:** deterministic seeded distributions over increasing N, used for scalability. These are never described as a Solar System.

## Measurement

- Release build; record compiler and flags.
- One untimed warm-up per configuration, then at least 10 randomized-order repetitions (increase until confidence intervals stabilize).
- Steady-clock wall time around initialization + required transfers + integration + final diagnostics. Exclude CSV/rendering; report them separately.
- Record median, interquartile range, arithmetic mean, standard deviation, and bootstrap 95% confidence interval.
- Pin OpenMP thread counts in the matrix and record affinity/environment.
- Synchronize CUDA before stopping the timer; retain transfer costs unless a separately labeled kernel-only experiment is reported.
- Check thermal/power state where possible and preserve raw observations.

## Matrix

Vary at least N, steps, dt, backend, thread count, and precision. Use a pilot grid to identify transition regions, then sample densely around them. Use identical input states and output cadence across backends.

## Selector evaluation

Split workload configurations into calibration and held-out sets before fitting. Baselines: sequential-only, OpenMP-only, CUDA-only, size-only threshold, and offline fastest-valid oracle. Report validity rate, backend-choice accuracy, selector overhead, regret distribution, and failure cases. Never train and evaluate on the same timing records.

## Accuracy gate

Validity is evaluated before performance ranking. A fast run that violates a declared bound is not a candidate. Thresholds, horizon, and diagnostic formulae are frozen before final measurements.
