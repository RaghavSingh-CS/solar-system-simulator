# CUDA backend

The optional CUDA backend implements the same direct O(N²), double-precision Newtonian interaction as the CPU reference. It is enabled only with `NBODY_ENABLE_CUDA=ON`; default CPU and OpenMP configurations do not enable the CUDA language or require a toolkit.

`CudaForceBackend` implements the shared `ForceBackend` contract. It transfers masses and positions for each acceleration request, launches one destination-body thread, accumulates source bodies in ascending index order, synchronizes explicitly, and returns accelerations to the host. This path exists for architectural compatibility and correctness comparison.

`CudaVelocityVerlet` is the intended benchmark path. It allocates reusable device and pinned-host buffers once, keeps state resident during a run, and executes old acceleration, position update, new acceleration, and velocity update kernels for every step.

CUDA events record H2D, complete device computation, D2H, and transfer-inclusive intervals on one stream. Allocation/context setup, host structure staging, and host wall time are reported separately. CUDA API calls, kernel launches, event synchronization, and stream synchronization are checked. Kernels flag coincident or invalid separations, and returned states are scanned for non-finite values.

`CudaVelocityVerlet` destruction frees stream-referenced device and pinned-host
storage only after successful synchronization of its owning stream. If device
selection or synchronization fails, quiescence is unknown, so that executor's
allocations are deliberately quarantined until CUDA context/process teardown.
Each such executor can add another retained allocation set; repeated fatal CUDA
failures can therefore accumulate device and pinned-host memory. No automatic
device-wide reset is performed because other viewer/experiment executors may
share the primary context. After recording diagnostics for a genuine
unrecoverable CUDA error, stop GPU experimentation and restart the application
process. Recovery from a fatal device error is not guaranteed.

The existing direct-force implementation uses `double` throughout and does not
use approximate math, mixed precision, tree methods, or CPU/GPU decomposition.
CUDA and CPU may still differ slightly because compiler contraction/FMA and
platform execution differ; validation uses predeclared mixed tolerances instead
of bit identity.

## Barnes--Hut Stage 3A (source-reviewed, GPU unverified)

`CudaBarnesHutForceBackend` is a separate force-only evaluator. It is not used
by `CudaVelocityVerlet`, the viewer, the counterfactual engine, or any automatic
backend selector. The validated deterministic CPU builder produces a
pointer-free `BarnesHutPackedTree` with fixed-width node, child, body-order, and
inverse-rank indices. Every force call transfers that immutable snapshot plus
mass/position arrays to a dedicated CUDA stream; GPU tree construction is not
implemented. CPU construction and host-to-device transfer can dominate and are
an explicit performance risk, especially at 50K+ bodies.

The kernel assigns one target to each logical CUDA work item. It uses double
precision, the unchanged unsoftened Newtonian law, the same monopoles and
strict `side / distance < theta` acceptance rule, inverse-rank target
containment, explicit leaf self-index exclusion, compensated per-target
accumulation, and reverse-pushed children so traversal order matches the CPU
depth-first order. A fixed 1024-entry (`4096` byte) compiled local stack bounds
each work item; the configured logical stack limit can be lower and exhaustion
is an explicit failure. A global 64-bit compare/exchange counter enforces the
configured traversal-work ceiling without overshoot. Singular geometry,
nonrepresentable geometry/acceleration, malformed packed data, stack exhaustion,
and work exhaustion are reported through a bounded device failure record. No
failure falls back to CPU or direct force, and partial acceleration output is
never returned.

Failure reporting uses two atomic words. One thread atomically claims the
record, writes the code and target/node metadata, executes a device fence, and
then atomically publishes the completed record. Other device threads inspect
only the atomic publication word; host code reads metadata only after stream
synchronization. This avoids mixing atomic writes with ordinary concurrent
reads and prevents a published code from preceding its metadata.

Persistent device buffers and pinned host staging buffers are reusable and
separately budgeted with checked size arithmetic. The reported persistent
device bytes exclude CUDA allocator/context overhead and compiler-managed
per-thread local stack backing. The executor owns a dedicated stream. If device
selection or stream synchronization fails, the executor becomes terminal and
stream-referenced device/pinned allocations are quarantined rather than freed
without proven quiescence. No implicit `cudaDeviceReset()` is performed; after
a genuine fatal CUDA error, record diagnostics and restart the process before
more GPU experimentation.

Capacity growth uses a checked live-release path: every device and pinned-host
free is checked, successfully freed pointers are cleared individually, and a
failed free makes the executor terminal without discarding the failed or
not-yet-released handles or resetting their accounting. Replacement allocation
does not proceed. Destructor cleanup remains noexcept; after proven quiescence
it attempts all remaining releases, while uncertain quiescence retains the
complete allocation set until context/process teardown. A test-only host fault
injection exercises this terminal transition without claiming a device fault.

Independent adversarial source review passed for committed implementation
`5e1864e`. CUDA compilation and execution of this Stage 3A path remain unverified on the
AMD-only development host. Registered GPU tests use strict enumeration: only a
successful zero-device result returns skip code 77; enumeration, allocation,
kernel, copy, or synchronization failures fail the test. NVIDIA validation must
characterize mixed absolute/relative, RMS, maximum-relative, and percentile
errors against independent Sequential Barnes--Hut and direct-force references.
No GPU equivalence, throughput, crossover, or 50K--100K claim is made.

Preparation commit `ae6117b` was compiled and validated on a Tesla T4; controlled evidence is under `data/benchmarks/stage4_cuda/`. See `docs/colab_cuda_workflow.md` for the repaired reproducible workflow. The local Windows host still lacks CUDA hardware/tooling.
