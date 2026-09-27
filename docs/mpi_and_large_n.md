# MPI and large-N execution

## Scope

Stage 6D adds an optional MPI implementation without changing the direct
Newtonian force law, double precision, or Velocity Verlet integrator. It also
adds a prospective, runtime-bounded large-N experiment runner. No result from
this stage calibrates the Stage 5 selector, and no local diagnostic is promoted
to controlled benchmark evidence.

## Replicated-state MPI design

`MpiForceBackend` fits the existing `ForceBackend` acceleration contract:

1. Every rank owns the full ordered `SimulationState`.
2. Destination body indices are divided into contiguous ranges. `N % P`
   remainder targets go one each to ranks `0 .. remainder-1`.
3. Each rank evaluates its local targets with the same ascending source order
   and shared direct-force kernel used by Sequential/OpenMP.
4. `MPI_Allreduce` propagates invalid-position status.
5. `MPI_Allgatherv` reconstructs the full ordered acceleration vector on every
   rank.
6. Every rank performs the same Velocity Verlet position/velocity updates.

Both force evaluations in each step therefore use globally consistent
positions. The design costs O(N) memory per rank, O(N^2/P) local work, and O(N)
acceleration communication per force evaluation. It prioritizes a clear,
testable baseline over distributed-state complexity.

With OpenMP enabled, `--threads N` parallelizes only the local target loop with
static scheduling. MPI is initialized with and requires at least
`MPI_THREAD_FUNNELED`; collective calls occur after the OpenMP region on the
initializing thread. This is a clean hybrid baseline, not an assertion that a
particular ranks-by-threads topology is optimal. One MPI rank per GPU is a
possible future architecture only; it is neither implemented nor measured.

## Build, correctness, and direct benchmark

```powershell
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' -S . -B build-stage6d-mpi -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/mpicxx.exe `
  -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/ninja.exe `
  -DNBODY_ENABLE_MPI=ON -DNBODY_ENABLE_OPENMP=ON `
  -DNBODY_ENABLE_CUDA=OFF -DNBODY_BUILD_VISUALIZATION=OFF
& '.\.tools\cmake-4.4.3-windows-x86_64\bin\cmake.exe' --build build-stage6d-mpi
ctest --test-dir build-stage6d-mpi --output-on-failure
```

CTest compares one, two, and four ranks with Sequential, including an N=11
non-divisible partition, repeatability, finite values, and the established
energy `5e-6` / angular momentum `5e-10` gates. It also runs two ranks by two
OpenMP threads when hybrid support is compiled.

```powershell
mpiexec -n 4 ./build-stage6d-mpi/nbody_mpi_benchmark.exe `
  --bodies 512 --steps 50 --dt 10 --threads 1 `
  --warmups 1 --repetitions 5 --seed 24301
```

The primary elapsed time is the maximum across ranks. Output also separates
cumulative local force-compute and MPI collective-communication time, reports
the provided MPI thread level, topology, build identity, conservation, final
checksum, and correctness. Keep ranks multiplied by threads no greater than
available logical CPUs unless oversubscription is itself the declared study.

## Stage 6D single-machine diagnostic

The following one-session Windows measurements used GCC 15.1.0, Microsoft MPI
10.1.12498.52, N=512, 10 steps, dt=10 s, one warm-up, three repetitions, and
seed 24301. They are **SINGLE-MACHINE LOCAL DIAGNOSTICS -- NOT CONTROLLED
BENCHMARK EVIDENCE**:

| Configuration | Median elapsed (s) | Per step (s) | Local compute (s) | Communication (s) |
|---|---:|---:|---:|---:|
| MPI 1 rank x 1 thread | 0.0489955 | 0.00489955 | 0.0485291 | 0.0002430 |
| MPI 2 ranks x 1 thread | 0.0236225 | 0.00236225 | 0.0223961 | 0.0011655 |
| MPI 4 ranks x 1 thread | 0.0229480 | 0.00229480 | 0.0204617 | 0.0056052 |
| MPI 2 ranks x 2 threads | 0.0184575 | 0.00184575 | 0.0166969 | 0.0017773 |

All four passed correctness with checksum `-382567846011.50037`, relative
energy drift `1.557002540516534e-14`, and angular drift
`1.4733993293434963e-15`. Same-session informal Sequential and OpenMP-4 medians
were 0.0397994 s and 0.0145921 s. The measurements show the expected local MPI
scaling trend, communication growth, and that this tiny single-node workload
does not beat OpenMP-4. They establish no multi-machine scaling claim.

## Prospective bounded large-N protocol

`scripts/run_large_n_pilot.py` implements protocol `stage6d-large-n-v1.0` for
Sequential, OpenMP, MPI, and MPI+OpenMP. It uses a fixed geometric size schedule
selected before execution, explicit steps/dt/warm-ups/repetitions/seed, and a
per-configuration process timeout. A timeout is retained as
`infeasible_within_budget`; configurations are never silently dropped. Error,
correctness-failure, completed, and timeout rows all remain in the CSV with raw
stdout/stderr logs.

The runner records source revision, command/build paths, OS, CPU and available
topology, RAM, compiler, CMake, Ninja, OpenMP environment, Microsoft MPI, GPU
and driver when visible, and CUDA compiler metadata when present. It rejects a
local ranks-by-threads request that exceeds the reported logical CPU count.
Results include elapsed/per-step time, MPI local compute and communication,
conservation, checksum, correctness, and topology.

Use `--dry-run` to inspect every prospective command before execution and write
results only to a new or empty output directory. A pilot becomes benchmark
evidence only when its environment, affinity/topology policy, runtime budget,
complete schedule, raw logs, and verification are retained and reviewed. To
claim multi-machine scaling, rerun the frozen protocol on a declared cluster
with host/network/interconnect metadata, fixed rank placement, synchronized
software, and repeated measurements; these local results cannot support that
claim.

## Large-N boundary

The solver remains exact direct O(N^2). Viewer support through N=100,000 is a
batched rendering and inspection feature, not a claim that direct integration
is practical at that size. The viewer's exact safety thresholds are documented
in `docs/visualization.md`. No Barnes--Hut tree, fast multipole method, domain
decomposition, softening, adaptive timestep, distributed state, or multi-GPU
path is hidden behind the large-N selector.
