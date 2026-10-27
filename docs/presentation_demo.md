# October 29 presentation correction — RTX verification candidate

Status: uncommitted; local CPU/OpenMP tests and window inspection only. The
operator reports that the previous CUDA viewer launched on the RTX 4050, but
this correction has not been compiled or run there. No new GPU timing or
correctness result is claimed. The existing untracked `interview_handoff/`
directory is outside this correction and is preserved.

## Observed defects and focused corrections

| Observation | Repository cause / status | Correction or constraint |
|---|---|---|
| OpenMP grey in CUDA viewer | The ordinary CUDA preset explicitly sets OpenMP OFF; model and experiment capabilities follow their compile definitions. The special handoff build previously overrode this, but ordinary builds did not. Exact teammate cache remains to be checked. | Ordinary CUDA build script enables OpenMP. CUDA launcher rejects an OpenMP-disabled cache with a rebuild instruction. The source-custodied preset is preserved byte-for-byte. |
| Removal only produces numbers | Existing experiment leaves live state untouched and overlays only the selected/significant bodies; a one-day catalog horizon often has subpixel separation at full-system scale. | Completed results can display either paired baseline or body-removed endpoint at identical time and origin. Removed body is hidden only in the counterfactual endpoint. Results use stable-ID mapping and the real generation gate. Live state is preserved; Return to live baseline and Reset are explicit. |
| Barnes--Hut seems unavailable | CUDA/Barnes--Hut is intentionally rejected by the GUI. Large synthetic admission also requires pause/manual steps at 4096/8192. | Demo exposes CPU solver choice, actual committed path, theta and executed node/worker counts. CUDA/Barnes--Hut remains unavailable. Existing math and bounds are unchanged. |
| CUDA jitter | Verified source path blocks the GL/event thread on every CUDA batch; catalog speeds allowed 10–120 steps per batch. Live GPU and experiment GPU could also run together. Actual RTX frame-time cause remains unmeasured. | GUI CUDA batches capped at four unchanged physical steps. Experiment submission pauses live advancement. FPS and measured batch milliseconds/steps per second are shown separately. No invented interpolated states or guaranteed smoothness claim. |
| Sequential work despite CUDA selected | Paired experiments previously defaulted independently to Sequential; unavailable selections were rewritten to Sequential in the UI. CPU diagnostics and explicit CPU comparison also execute CPU code. No live CUDA integration fallback was found. | Experiment initialization/removal preparation uses the selected live path; later edits are explicit independent settings. Availability errors never change that selection to Sequential. Last committed live path and immutable experiment path are separately displayed. |
| Catalog lacks visible orbits | Catalog had no initial reference guides and sampled history only every 1800 simulated seconds. Early arcs at dt=60 s are short. | History samples completed advancement boundaries at >=60 s. Bounded 192-segment initial two-body osculating ellipses are clearly labelled presentation guides; they do not seed or advance physics. Local guides use the actual parent; full-system guides are heliocentric. Missing-parent, degenerate and unbound conics have no closed guide. |
| Poor zoom | Universal 2–120 render-unit clamp and comparatively strong wheel response. | Proportional wheel zoom with finite input checks, 0.15–160 render-unit bounds and matching clipping planes. Preset reframing restores navigation. |
| Restrictive rotation | Pitch was clamped to +/-85 degrees with fixed world-up. No pan. | Wrapped full pitch/yaw with analytic camera-up avoids pole singularities; right drag pans relative to camera distance. Existing initial poses and legacy physical data are preserved. |
| Excessive sidebar scrolling | All simulation, catalog composition, body properties, experiment controls and diagnostics were stacked. Saved window scrolling could hide the title/status. | Demo, Comparison and Advanced pages; persistent header and independently scrolling page content. Primary scene/backend/solver/pause/step/reset/speed/view/body/removal controls are on Demo. |
| Additional 5000 objects | No admitted 5060-body astronomical composition exists; Direct >1024 remains rejected for synthetic scenes. | Deferred. Existing separate deterministic synthetic presets remain available. No decorative objects are presented as gravitating catalog members. |

## Actual execution paths

| Selection / action | Owner / thread | Executed code |
|---|---|---|
| Live Sequential + Direct | SimulationController / UI thread | Existing CPU Velocity Verlet + Sequential force backend |
| Live OpenMP + Direct | SimulationController / UI thread + OpenMP team | Existing CPU Velocity Verlet + OpenMpForceBackend |
| Live Sequential/OpenMP + Barnes--Hut | SimulationController / UI thread + optional OpenMP team | Existing Velocity Verlet + selected Barnes--Hut force backend; node and worker statistics published after success |
| Live CUDA + Direct | Persistent controller CUDA executor / UI thread, GPU stream | CudaVelocityVerlet::run; synchronize/read back completed batch, then publish host state; no CPU integration fallback |
| Live CUDA + Barnes--Hut | Unavailable | Explicit rejection; CPU selection required |
| Removal experiment | Independent worker thread, selected explicit backend | Two owned branches using existing Direct or CPU Barnes--Hut integrators. CUDA supports Direct only. Live scene pauses. |
| Conservation/guards/labels/rendering | CPU / UI or experiment worker | Auxiliary diagnostics and drawing, not CUDA force/integration throughput |
| Advanced Compare CPU backends | CPU / UI thread | Explicit independent Sequential/OpenMP reference diagnostic, not a CUDA measurement |
| Completed paired display | UI thread / OpenGL | Read-only immutable paired result; no new integration or live deletion |

The ACTIVE EXECUTION record is created only after successful advancement. It is
cleared on backend/solver selection, reset and scenario replacement. Failed
batches preserve the last successful record and display a paused/error state.
Experiment metadata comes from its immutable executed descriptor, not current
live or next-run controls. Failed, stale and mismatched publications do not
display counterfactual endpoints. Removing the requested local origin makes the
comparison unavailable in that frame; choose Full System or a surviving parent.

## Orbit guide interpretation

Catalog guides use initial parent-relative r and v, mu=G*(body mass+parent mass),
h=r cross v, eccentricity vector=(v cross h)/mu-r/|r|, and
p=|h|^2/mu. For bound nondegenerate ellipses, the drawn radius is
p/(1+e*cos(anomaly)). This is a presentation-only instantaneous two-body conic,
not a new force solver, an integrated trail, or a Horizons prediction. The
catalog initial conditions, GM, reference frame and dt=60 s remain unchanged.
The timestep is still provisional. Measured separations are never magnified;
short-horizon planetary responses may be indistinguishable at full-system scale.

## RTX 4050 build and focused validation

Apply the reviewed candidate files to the teammate's checkout first. These
changes are intentionally uncommitted until RTX verification; the clean-source
handoff preflight cannot certify an uncommitted candidate. Use an x64 VS2022
Native Tools PowerShell and record the source diff and toolchain versions.

```powershell
cmake --preset cuda-release -DNBODY_ENABLE_OPENMP=ON -DCMAKE_CUDA_ARCHITECTURES=89
Select-String -Path build/cuda-release/CMakeCache.txt -Pattern 'NBODY_ENABLE_OPENMP:.*=ON','NBODY_ENABLE_CUDA:.*=ON','CMAKE_CUDA_ARCHITECTURES:.*=89'
cmake --build --preset cuda-release --parallel 4
ctest --preset cuda-release --output-on-failure --no-tests=error -R '^(nbody_openmp_validation|nbody_cuda_validation|nbody_visualization_model|nbody_visualization_cuda|nbody_counterfactual_worker|nbody_counterfactual_cuda)$'
.\scripts\run-visualizer.ps1 -Backend CUDA -ApplicationArguments '--scenario','catalog','--device','0','--width','1920','--height','1080'
```

All six named tests must exist; CUDA skips/NOT RUN are not passes on the RTX.
Do not continue after a compiler, driver, allocation, synchronization or test
failure. No device reset or CPU fallback is performed. Record driver, nvcc/MSVC,
cache values, CTest results and actual ACTIVE EXECUTION status. Inspect both
1920x1080 and 1280x720, full/inner and moon-system views, full orbit/pan/zoom,
pause/step/reset, backend switching, and completed/cancelled/failed comparison.
Measure displayed FPS separately from the last physics batch milliseconds and
steps/s; capture whether capping GPU batches reduces the reported jitter.

## Practical five-minute presentation

1. Before class: build, run the focused tests, rehearse on the RTX, retain logs.
2. 0:00–1:00: show the catalog's 60 gravitating real bodies, sourced epoch,
   clearly labelled guide/history distinction, CUDA Direct active execution,
   and separate display versus simulation timing.
3. 1:00–2:00: Full/Inner/Earth–Moon views; left-drag full orbit, wheel zoom,
   right-drag pan, Reframe, pause and Step.
4. 2:00–3:00: explicitly select OpenMP then Barnes--Hut. Advance; show real node
   count and workers. Explain why CUDA Barnes--Hut GUI is unavailable.
5. 3:00–4:30: choose Earth or Jupiter, Remove / compare. Confirm the
   experiment backend/solver, run the bounded comparison, toggle Baseline
   endpoint / Removed-body endpoint. The removed body disappears from the
   altered branch; solid baseline and dashed pink paths share time and frame.
   Show actual numeric separation without claiming ejection or dramatic
   short-term planetary disruption. For tiny differences use a local view.
6. 4:30–5:00: Return to live baseline / Reset. State provisional timestep,
   approximate Barnes--Hut forces, and the deferred extra 5000-body request.

Stage 5, large-N scaling, CUDA kernel development and evidence acquisition are
outside this correction. No commit or push is authorized before RTX verification.

## Validation and exact file inventory

Local CPU/OpenMP Release builds passed. Focused CPU viewer/worker tests passed
2/2; focused OpenMP viewer/worker/backend tests passed 3/3. The extended host
handoff suite passed. Full serial CPU CTest passed 20/21 and OpenMP passed 24/25;
both fail only the existing layout check for six CSV files in the pre-existing
untracked `interview_handoff/` directory. That directory and the checker remain
unchanged. L1/L2 strict evidence verifiers, dependency verification and whitespace
checks pass. The preset remains exactly hash-custodied. An initial preset edit
was reverted after the verifier exposed the source-byte constraint.

The real AMD OpenGL windows were inspected locally for guides, fixed status
header and CPU Barnes--Hut execution. These observations and headless tests do
not establish RTX visual stability or human approval. nvcc is unavailable here;
the new CUDA/combined-OpenMP test assertions remain uncompiled and unexecuted.

Changed tracked files:

```text
README.md
docs/MASTER_LEDGER.md
docs/visualization.md
include/nbody/visualization/camera.hpp
include/nbody/visualization/controller.hpp
include/nbody/visualization/counterfactual_worker.hpp
include/nbody/visualization/presentation.hpp
nvidia_handoff/DEMO.md
scripts/run-visualizer.ps1
scripts/test-local.ps1
tests/test_counterfactual_worker.cpp
tests/test_nvidia_handoff.py
tests/test_visualization.cpp
tests/test_visualization_cuda.cpp
visualization/camera.cpp
visualization/controller.cpp
visualization/counterfactual_worker.cpp
visualization/main.cpp
visualization/presentation.cpp
visualization/scenario.cpp
```

New file: `docs/presentation_demo.md` (this report). No file under the existing
untracked `interview_handoff/` has been edited, staged or removed.
