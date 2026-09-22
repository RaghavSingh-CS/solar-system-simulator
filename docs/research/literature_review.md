# Focused literature review

## Scope and method

This reconnaissance was performed on 2026-09-22 before choosing the project methodology. It targets direct gravitational N-body computation, CPU/GPU and distributed parallelism, heterogeneous scheduling, and conservation-aware integration. The eight sources below are primary publications selected for direct relevance. Reported results belong to the cited hardware and workloads; none is treated as a transferable crossover threshold.

## 1. CUDA direct summation

**Robert G. Belleman, Jeroen Bedorf, and Simon F. Portegies Zwart (2008), “High performance direct gravitational N-body simulations on graphics processing units II: An implementation in CUDA,” _New Astronomy_ 13(2), 103–112.** [DOI 10.1016/j.newast.2007.07.004](https://doi.org/10.1016/j.newast.2007.07.004)

- **Hardware/workload:** NVIDIA GeForce 8800 GTX, a host CPU, and GRAPE-6Af; direct integrations plus a Barnes–Hut code, with particle count varied.
- **Algorithm/protocol:** CUDA force evaluation; fourth-order Hermite predictor-corrector with block timesteps in two codes, and second-order leapfrog in the tree code; integration remained on the host. Runtime and energy error were compared with CPU/GRAPE implementations.
- **Metrics/findings:** GPU beat GRAPE-6Af above roughly 512 particles when softening was accepted; at large N (about 100,000 and above) it was about 100 times faster than the host CPU. GPU total-energy conservation was better than one part in 10^6, but about an order of magnitude worse than GRAPE under the stated criterion.
- **Limitations:** 2007-era hardware and precision, unlike current CUDA devices; crossover depended on softening and timestep. It did not automatically select a backend.
- **Relation:** establishes both a workload-dependent crossover and an accuracy/performance coupling in CUDA N-body work. Our project must measure its own RTX 4050 crossover and must include transfers and integration in end-to-end timing.

## 2. MPI direct N-body parallelization

**Simon Portegies Zwart, Stephen McMillan, Derek Groen, Alessia Gualandris, Michael Sipior, and Willem Vermin (2008), “A parallel gravitational N-body kernel,” _New Astronomy_ 13(5), 285–295.** [DOI 10.1016/j.newast.2007.11.002](https://doi.org/10.1016/j.newast.2007.11.002)

- **Hardware/workload:** four parallel systems (high-speed interconnect, NUMA, and geographically distributed grid configurations); Plummer models from 1,024 to 65,536 equal-mass stars; 1–128 processors; one eighth of an N-body time unit.
- **Algorithm/protocol:** MPI j-parallelization of the kira direct integrator. Each process retained the full state, computed partial forces for a source subset, then used a global reduction. Timings were compared against an analytic performance model.
- **Metrics/findings:** good scaling in the tested range; the three-site grid became inefficient above about 60 processors because communication dominated.
- **Limitations:** distributed-memory stellar-cluster workload and older systems; replicated state limits scaling; no GPU selection or accuracy-constrained policy.
- **Relation:** supports MPI as an optional stretch backend and shows why it must not be critical path for a one-node demonstration.

## 3. Hybrid OpenMP/MPI/GPU N-body code

**Roberto Capuzzo-Dolcetta, Mario Spera, and Davide Punzo (2013), “A fully parallel, high precision, N-body code running on hybrid computing platforms,” _Journal of Computational Physics_ 236, 580–593.** [DOI 10.1016/j.jcp.2012.11.013](https://doi.org/10.1016/j.jcp.2012.11.013)

- **Hardware/workload:** CINECA IBM iDataPlex DX360M3 Linux InfiniBand cluster, up to 256 NVIDIA GPUs; Plummer-like stellar systems from 32,000 to 8 million particles.
- **Algorithm/protocol:** HiGPUs uses direct force summation, a sixth-order Hermite scheme, block timesteps, OpenMP and MPI on CPUs, and CUDA or OpenCL on GPUs. The paper reports both performance and accuracy experiments.
- **Metrics/findings:** demonstrated integration of up to eight million bodies for several crossing times and strong use of hybrid parallel resources.
- **Limitations:** cluster-scale, collisional stellar-dynamics focus; high-order block-step design is too large for this mini-project and does not isolate an automatic CPU/OpenMP/CUDA selector.
- **Relation:** proves that the proposed backend set and accuracy measurements are established engineering territory. Our contribution must be the controlled selector evaluation, not the existence of these backends.

## 4. N-body CPU/GPU crossover and precision

**Jeroen Bedorf, Evghenii Gaburov, and Simon Portegies Zwart (2015), “Sapporo2: a versatile direct N-body library,” _Computational Astrophysics and Cosmology_ 2, article 8.** [DOI 10.1186/s40668-015-0012-z](https://doi.org/10.1186/s40668-015-0012-z)

- **Hardware/workload:** NVIDIA GTX 680 and K20m, AMD HD 7970, and an OpenMP/SSE2 CPU implementation; force-kernel grids over source/sink counts; Plummer spheres from 16,000 to 512,000 bodies for full block-step runs.
- **Algorithm/protocol:** CUDA/OpenCL direct-force library supporting leapfrog and fourth-/sixth-order Hermite integration, multiple precisions, and multiple GPUs. Timings included transfers. Full runs recorded execution time, energy error, active-particle count, and 1-vs-4-GPU speedup.
- **Metrics/findings:** the CPU could be preferable around small workloads where transfer costs matter (the paper highlights 1,024 particles and below as the relevant regime). Multiple GPUs became useful for sufficiently large work. On K20m sixth-order runs, native double precision was both faster overall and about two orders of magnitude more accurate than double-single because lower precision forced smaller timesteps; reported energy errors were about 10^-12 versus 10^-10.
- **Limitations:** crossover is hardware-, kernel-, precision-, and active-set-specific; the CPU kernel covered only one integrator; it measured/tuned rather than presenting our proposed pass/fail selector.
- **Relation:** this is the strongest prior-art constraint. “GPU crossover for N-body” is not novel. A narrower study may test whether a reproducible calibrated policy can choose among sequential/OpenMP/CUDA while rejecting configurations that violate a conservation bound.

## 5. Adaptive CPU/GPU mapping

**Chi-Keung Luk, Sunpyo Hong, and Hyesoon Kim (2009), “Qilin: exploiting parallelism on heterogeneous multiprocessors with adaptive mapping,” _42nd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO-42)_, 45–55.** [DOI 10.1145/1669112.1669121](https://doi.org/10.1145/1669112.1669121)

- **Hardware/workload:** two-socket, eight-core 2.4 GHz Intel Core 2 system plus NVIDIA 8800 GTX; eight kernels from finance, image processing, dense linear algebra, bioinformatics, and vision.
- **Algorithm/protocol:** generated CPU and GPU variants, profiled training partitions, fitted simple execution-time curves, and selected a CPU/GPU work split for new problem sizes. Wall time included GPU transfers. Manual 10%-granularity mappings and CPU-only/GPU-only baselines were used.
- **Metrics/findings:** adaptive mapping averaged 41% faster than CPU-only and 25% faster than GPU-only, reaching 94% of manually found performance; it adapted to problem-size and hardware/compiler changes.
- **Limitations:** not an N-body study, used training amortized over repeated runs, and optimized time rather than numerical validity.
- **Relation:** automatic workload-aware backend mapping is established. Our potentially distinct variable is a scientific-validity constraint, evaluated for one application on one machine rather than claimed as a general scheduling algorithm.

## 6. Heterogeneous task scheduling runtime

**Cedric Augonnet, Samuel Thibault, Raymond Namyst, and Pierre-Andre Wacrenier (2011), “StarPU: a unified platform for task scheduling on heterogeneous multicore architectures,” _Concurrency and Computation: Practice and Experience_ 23(2), 187–198.** [DOI 10.1002/cpe.1631](https://doi.org/10.1002/cpe.1631)

- **Hardware/workload:** multicore CPUs with NVIDIA GPUs; dense numerical kernels and LU/Cholesky-style factorizations, including matrices larger than accelerator memory.
- **Algorithm/protocol:** application tasks expose CPU/GPU implementations; StarPU manages data movement and uses calibrated per-worker performance models to schedule tasks. Dynamic strategies were compared with static approaches and optimized MAGMA routines.
- **Metrics/findings:** performance-model scheduling reduced load imbalance and could achieve superlinear aggregate efficiency by assigning different kernel types to the processor where each was most efficient.
- **Limitations:** general runtime rather than N-body; task scheduling does not decide whether a numerical method satisfies a physics-specific error budget.
- **Relation:** reinforces that performance-aware heterogeneous scheduling is not new. It suggests keeping our policy small and application-specific instead of claiming a new runtime system.

## 7. Long-term conservation in planetary integration

**Hanno Rein and Daniel Tamayo (2015), “WHFast: a fast and unbiased implementation of a symplectic Wisdom–Holman integrator for long-term gravitational simulations,” _Monthly Notices of the Royal Astronomical Society_ 452(1), 376–388.** [DOI 10.1093/mnras/stv1257](https://doi.org/10.1093/mnras/stv1257)

- **Hardware/workload:** CPU implementation; two-body eccentricity/timestep sweeps and an outer-Solar-System test using the Sun and four giant planets.
- **Algorithm/protocol:** improved Wisdom–Holman mapping, stable Jacobi-coordinate transforms, Kepler solver, and optional symplectic correctors. Compared final/bounded relative energy error and time per step with other integrators; long-run tests used 1.5-day steps and perturbed replicas.
- **Metrics/findings:** reported 1.5–5x speedups, 2–3 orders of magnitude better energy conservation in important regimes, and removal of secular bias so small-step error follows an unbiased random-walk law.
- **Limitations:** specialized for near-Keplerian planetary systems; not a generic direct-summation CPU/GPU backend comparison.
- **Relation:** confirms that energy drift alone must be interpreted over a defined horizon and timestep. Velocity Verlet is acceptable for the first deterministic reference, but a realistic long-horizon Solar System claim may require a Wisdom–Holman-class method or explicit scope limits.

## 8. Interchangeable astrophysical solvers

**Simon Portegies Zwart, Steve McMillan, Arjen van Elteren, Inti Pelupessy, and Nathan de Vries (2013), “Multi-physics simulations using a hierarchical interchangeable software interface,” _Computer Physics Communications_ 184(3), 456–468.** [DOI 10.1016/j.cpc.2012.09.024](https://doi.org/10.1016/j.cpc.2012.09.024)

- **Hardware/workload:** distributed and local solver components in the MUSE/AMUSE framework; demonstrations combine astrophysical modules implemented in multiple languages.
- **Algorithm/protocol:** homogeneous interfaces, automatic unit conversion, separated solver processes, message passing, error recovery, and interchangeable components; framework overhead was measured in representative simulations.
- **Metrics/findings:** reported framework overhead below 1% of wall time and consistent results relative to traditional monolithic workflows.
- **Limitations:** a broad software architecture, not automatic performance/accuracy backend selection; Python-led ecosystem is intentionally heavier than this project needs.
- **Relation:** validates a stable solver interface and independent validation/visualization, but argues against reimplementing a large framework for a one-month project.

## Synthesis

Established prior work already covers CUDA acceleration, MPI/OpenMP/CUDA hybrid codes, empirical CPU/GPU crossover, interchangeable N-body modules, and adaptive heterogeneous mapping. It also shows that precision can change both error and runtime through timestep behavior. What is not established by this focused set is a portable universal threshold or a validated selector for this exact C++ code and target RTX 4050 that minimizes measured end-to-end time **subject to a predeclared conservation-error limit**.

That absence is only a project hypothesis, not proof of a research gap. A broader systematic review would be required before any conference novelty claim.
