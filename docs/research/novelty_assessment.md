# Novelty assessment

## Verdict

The original phrase “workload-aware backend selection” is **not defensible as algorithmic novelty**. Qilin and StarPU established performance-model-based heterogeneous mapping, and Sapporo2 directly studied where CPU or GPU N-body kernels win as workload and precision change.

The project should therefore be framed as a **reproducible comparative experimental study with a modest application-specific policy**, not as a new scheduling algorithm.

## Narrow proposed contribution

On one fully specified Windows/RTX 4050 machine, calibrate sequential, OpenMP, and CUDA implementations of the same O(N^2) force law and timestep scheme. Given `(N, step count, precision mode, accuracy budget)`, select the fastest configuration whose validation run satisfies predeclared bounds on relative energy and angular-momentum drift. Compare this policy with always-sequential, always-OpenMP, always-CUDA, a size-only threshold, and an offline oracle.

The contribution, if supported, is the **measurement protocol and evidence** that adding a scientific-validity gate changes or stabilizes backend choice for this implementation. It is not a claim that backend selection or crossover discovery is new.

## Established prior work

- Direct gravitational force evaluation maps well to GPUs at sufficiently large workloads.
- Small jobs can favor CPUs because launch, transfer, and synchronization overheads dominate.
- Heterogeneous runtimes can learn performance models and map work dynamically.
- N-body frameworks already expose interchangeable solvers and hybrid MPI/OpenMP/GPU execution.
- Precision, timestep choice, reduction order, and integrator design affect conservation and sometimes total runtime.

## Hypotheses to test

- **H1:** no single backend minimizes end-to-end runtime over the full tested `(N, steps)` grid on the target machine.
- **H2:** a calibration-derived selector chooses the fastest valid backend within 10% of an offline oracle for at least 90% of held-out configurations.
- **H3:** enforcing a conservation bound changes at least some choices relative to a size-only performance selector.
- **H4:** calibration is hardware-specific; a threshold learned on Colab does not transfer reliably to the RTX 4050.

These are unverified hypotheses. The percentage targets are preregistered evaluation criteria, not expected results.

## Claims explicitly not made

- No universal CPU/GPU crossover value.
- No claim that Velocity Verlet is the best Solar System integrator.
- No claim that different hardware can be compared to infer a crossover.
- No publication promise and no claim that the focused review proves an unoccupied research gap.
- No claim that identical mathematical formulas produce bitwise-identical reductions across backends.

## Go/no-go rule for a later paper

Proceed from a faculty project to a paper only if the same-machine experiments are repeatable, the selector is evaluated on held-out workloads, all compared outputs meet the same declared physics criteria, and the accuracy gate adds information beyond N alone. Otherwise report a useful negative result and retain the work as a comparative implementation study.
