# Novelty assessment

## Verdict

The original phrase “workload-aware backend selection” is **not defensible as algorithmic novelty**. Qilin and StarPU established performance-model-based heterogeneous mapping, and Sapporo2 directly studied where CPU or GPU N-body kernels win as workload and precision change.

The project should therefore be framed as a **reproducible comparative experimental study with a modest application-specific policy**, not as a new scheduling algorithm.

## Narrow proposed contribution

On one fully specified Windows/RTX 4050 machine, calibrate Sequential,
topology-derived OpenMP, and CUDA implementations of the same O(N^2) force law
and timestep scheme. For the fixed Stage 5 v2 workload, map only N to the
configuration with the smallest calibration-predicted `selector_steady_state`
median among configurations satisfying predeclared relative energy and
angular-momentum-drift bounds. Evaluate the frozen mapping against the
held-out empirical oracle without refitting.

The contribution, if supported, is the **measurement protocol, custody, and
same-machine evidence** for this narrow N-dependent mapping. Step count,
precision, accuracy budget, scenario, seed, and cold-start behavior are future
extensions rather than demonstrated selector inputs. It is not a claim that
backend selection or crossover discovery is new.

## Established prior work

- Direct gravitational force evaluation maps well to GPUs at sufficiently large workloads.
- Small jobs can favor CPUs because launch, transfer, and synchronization overheads dominate.
- Heterogeneous runtimes can learn performance models and map work dynamically.
- N-body frameworks already expose interchangeable solvers and hybrid MPI/OpenMP/GPU execution.
- Precision, timestep choice, reduction order, and integrator design affect conservation and sometimes total runtime.

## Hypotheses to test

- **H1:** no single configuration minimizes `selector_steady_state` over the tested N grid on the target machine.
- **H2:** report empirical-fastest agreement and multiplicative regret on all six held-out N values without converting this small set into a universal success claim.
- **H3:** enforcing a conservation bound changes at least some choices relative to a size-only performance selector.
- **H4:** calibration is hardware-specific; a threshold learned on Colab does not transfer reliably to the RTX 4050.

These are unverified hypotheses. Stage 5 v2 reports the six held-out outcomes
directly and does not introduce a percentage target that the small held-out set
cannot resolve cleanly.

## Claims explicitly not made

- No universal CPU/GPU crossover value.
- No claim that Velocity Verlet is the best Solar System integrator.
- No claim that different hardware can be compared to infer a crossover.
- No publication promise and no claim that the focused review proves an unoccupied research gap.
- No claim that identical mathematical formulas produce bitwise-identical reductions across backends.

## Go/no-go rule for a later paper

Proceed from a faculty project to a paper only if the same-machine experiments are repeatable, the selector is evaluated on held-out workloads, all compared outputs meet the same declared physics criteria, and the accuracy gate adds information beyond N alone. Otherwise report a useful negative result and retain the work as a comparative implementation study.
