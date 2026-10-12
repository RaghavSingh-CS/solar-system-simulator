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

The implementation uses `double` throughout and does not use approximate math, mixed precision, tree methods, or CPU/GPU decomposition. CUDA and CPU may still differ slightly because compiler contraction/FMA and platform execution differ; validation uses predeclared mixed tolerances instead of bit identity.

Preparation commit `ae6117b` was compiled and validated on a Tesla T4; controlled evidence is under `data/benchmarks/stage4_cuda/`. See `docs/colab_cuda_workflow.md` for the repaired reproducible workflow. The local Windows host still lacks CUDA hardware/tooling.
