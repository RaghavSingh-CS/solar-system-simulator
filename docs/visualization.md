# Final visualization system (Stage 6D)

## Scope and status

Stage 6D completes the feature scope of the scientifically functional desktop
application. It retains the Stage 6C presentation system and adds camera-aware
Solar body sizing, fixed inner-system callouts, a resizable polished panel with
complete hover help, and safely bounded synthetic displays through 100,000
bodies. The physics library remains independent of graphics and UI dependencies.

Earlier stages were manually inspected and their findings drove these passes.
Stage 6D automated and real-context validation is recorded below, but automation
cannot judge final composition and interaction feel. Status therefore remains
**IMPLEMENTED / USER VISUAL VALIDATION PENDING**.

## Solar scenario provenance and limits

The default nine-body scenario is a deterministic approximation at J2000.0
(Julian date 2451545.0 TDB), in mean ecliptic/equinox J2000 coordinates:

- Mercury through Neptune use Table 1 elements and rates from JPL Solar System
  Dynamics, [Approximate Positions of the Planets](https://ssd.jpl.nasa.gov/planets/approx_pos.html),
  whose stated interval is 1800--2050. The implementation follows the page's
  Kepler-equation and ecliptic-coordinate formulas.
- Initial velocities are deterministic centered derivatives of that position
  model at plus/minus 3600 seconds, not sampled ephemeris velocities.
- The Sun begins at the heliocentric origin; a mass-weighted center-of-mass
  position and velocity shift produces the simulator's barycentric state.
- Earth uses JPL's Earth--Moon-barycenter orbit but is assigned Earth's mass;
  the Moon is omitted. This is an explicit approximation.
- Masses and volumetric mean radii come from NASA NSSDCA's
  [planetary fact sheets](https://nssdc.gsfc.nasa.gov/planetary/factsheet/).

This is suitable for an educational demonstration, regression testing, and
bounded numerical experiments. It is not a JPL Horizons state vector, an
observational ephemeris, or evidence for long-horizon planetary accuracy. The
committed one-Mercury-orbit check (352 six-hour steps) observed relative energy
drift `6.46017e-09` and angular-momentum-vector drift `1.61093e-16` on the
validation machine; those values validate this build and test case only.

## Horizons-derived 60-body sandbox

The viewer also exposes a distinct `jpl-ssd-j2000-60-v1` scenario. It never
mixes the legacy mean-element state with catalog records. All included bodies
come from the committed catalog's geometric SSB-centered J2000 ICRF Cartesian
state at JD 2451545.0 TDB, with SI position/velocity, sourced positive radius,
and mass derived from sourced GM using the project's unchanged gravitational
constant. The dataset version and provenance contract are visible in the
technical panel.

The default extended composition contains all 60 records in deterministic
parent-before-child order. The catalog panel can create a physical subset by
stable ID; this explicitly rebuilds the solver state and reports missing major
gravitational sources or catalog parents. A separate visibility mask controls
individual and class-level rendering. Hiding a body never removes its force.
Metadata also records the future curated counterfactual-removal eligibility,
but this increment provides no removal experiment control.

Effective render visibility is the intersection of three independent policies:
membership in the active subsystem view, the body-class filter, and the
individual body's visibility preference. Changing a class or individual
preference never changes active-view membership, so re-enabling a class cannot
reveal bodies outside the current subsystem. Returning to Overview restores
full view membership while retaining the user's class and individual choices.
None of these presentation policies changes solver composition.

The extended scenario uses a provisional fixed timestep of 60 seconds. A
one-day three-level check found worst parent-relative final-position differences
of `2.661430167e-2` for 300 versus 150 seconds and `6.676367043e-3` for 150
versus 75 seconds, so the earlier 300-second proposal was rejected. The
selected 60-second policy differed from 30 seconds by `1.069240913e-3`
(`138269.1228 m` worst absolute parent-relative difference); Metis was the
worst normalized case at every refinement. During the one-day 60-second run,
the closest center separation was Jupiter--Metis at `1.831596045` times their
combined sourced radii, so no physical-radius contact occurred in that bounded
check. This is not a general close-encounter safety result. The 300-second
run's relative global energy drift was `5.050393981e-11`; that small global
drift does not override the fast-moon phase error. The 60-second policy remains
explicitly unvalidated for long horizons and against later Horizons states.
Point-mass dynamics omit oblateness, tides, relativity, non-gravitational comet
forces, adaptive steps, and close-encounter regularization.

## Dependency acquisition

Visualization is guarded by `NBODY_BUILD_VISUALIZATION` and is `OFF` by
default. A headless configure neither fetches nor compiles graphics code.

When visualization is enabled, CMake FetchContent obtains exact commits:

- GLFW 3.4: `a74efa0d5628b74adc0426af4c5710e287fa7c2c`;
- GLM 1.0.1: `0af55ccecd98d4e5a8d1fad7de25ba429d60e863`;
- Dear ImGui 1.92.9b: `f1cc2ae15e53a861a874c3034aae6798fde194ab`.

The GLAD 2.0.6 loader was reproducibly generated from commit
`658f48e72aee3c6582e80b05ac0f8787a64fe6bb` for OpenGL 3.3 core and
`GL_KHR_debug`, then committed in `third_party/glad`. This keeps clean builds
independent of Python/Jinja2 and avoids copied runtime DLLs. Initial dependency
fetching requires Git/network access; subsequent configuration can reuse the
FetchContent checkout/cache.

## Build and launch

From the repository root:

```powershell
cmake -S . -B build-visual -G Ninja -DCMAKE_BUILD_TYPE=Release -DNBODY_BUILD_VISUALIZATION=ON -DNBODY_ENABLE_OPENMP=ON -DNBODY_ENABLE_CUDA=OFF
cmake --build build-visual
ctest --test-dir build-visual --output-on-failure
./build-visual/nbody_visualizer.exe --backend sequential
```

CUDA remains opt-in. On a system with a usable CUDA toolkit and NVIDIA device,
configure with `-DNBODY_ENABLE_CUDA=ON` and select it with
`--backend cuda --device 0` or the enabled CUDA control. The viewer never
silently substitutes a CPU backend.

Options are `--scenario solar|catalog|synthetic`, `--backend sequential|openmp|cuda`,
`--threads N`, `--device N`, `--bodies N`, `--seed N`, `--width N`, and
`--height N`. Body count and seed apply to the synthetic scenario. CUDA is
accepted only when support was compiled and the requested device is available;
otherwise it issues a clear diagnostic and never silently falls back. MPI is
exercised by its separate benchmark executable and Auto is not exposed because
Stage 5 calibration is not complete.

## Loop and separation

Each active frame advances an integer number of Velocity Verlet steps at the
scenario's fixed physical timestep, updates diagnostics and bounded physical-
space trail history, then creates a read-only render snapshot. Wall-clock frame
duration never becomes a physics timestep. Pausing stops physics advancement
but leaves rendering and interaction active.

Sequential and OpenMP retain the CPU integrator path. CUDA uses one persistent
`CudaVelocityVerlet` allocation per active body count and performs one complete
upload/integration/download operation per requested frame batch. Host
`SimulationState` remains authoritative at completed advancement boundaries;
there is no CUDA/OpenGL interoperability or automatic backend selection.

The legacy Solar scenario uses a six-hour timestep. Named presets map to 1, 4, 16, or
64 steps per render: six hours, one day, four days, or sixteen days of simulated
time. Trails are sampled every simulated day and retained in a bounded deque.
The synthetic fixture uses its established deterministic generator and a
10-second timestep.

Positions, trails, orbit guides, labels, rings, picking, and markers share one
linear render transform. Overview uses the SSB origin. Subsystem views first
subtract a moving local origin in binary64 and only then convert the offset to
render floats; their view scale is fixed when the preset is selected. Physics
coordinates remain absolute SSB-centered SI values. Parent-relative trails
subtract the matching historical parent position for every sample instead of
subtracting the current parent from past child points. Display radii are
presentation only: the Sun uses a stable camera-dependent 20--36 px screen-
space radius and other bodies use 3--13 px bounds. No position distortion is
used and no display value mutates mass, position, velocity, force, or
diagnostics.

## Stage 6C presentation policy

### UI and numerical health

The 1080p-oriented panel uses scaled default ImGui fonts (no external font or
runtime asset), increased spacing, restrained teal/gold accents, clear sections,
and concise primary values. Position/velocity tuples, maximum drift, fixed
timestep, steps per render, trail capacity, provenance, and synthetic seed live
under `Advanced / Technical Details`.

The primary health view compares current relative energy drift with `5e-6` and
angular-momentum-vector drift with `5e-10`. Values at or below the unchanged
bounds show `PASS`; exceeded/non-finite values show an obvious warning. Maximum
values remain visible in advanced details. Viewer FPS is labeled display-only
and **NOT BENCHMARK EVIDENCE**.

### Camera and scale

The legacy deterministic compositions coexist with manual orbit/dolly, and the
catalog adds Earth-Moon, Galilean/Jovian, Saturnian, Uranian, Neptune-Triton,
Pluto-Charon, and asteroid-belt profiles. All profile membership, focus, and
local origins resolve through stable IDs rather than catalog indices.

The legacy views are:

- `Overview`: complete outer system at the established linear scale;
- `Inner System`: Sun through Mars readable in one frame;
- `Earth`, `Jupiter`, and `Saturn`: tracked body-centered presentation views.

`C` returns to Overview. Presets change only camera target, orientation, and
distance. Focus tracking follows a moving body; manual controls remain active
afterward. This camera-first policy addresses the Solar System's physical
dynamic range without altering coordinates.

### Orbit context and trails

Subtle reference guides are sampled over one approximate revolution from the documented JPL
mean-element model used for initialization. They are static presentation
geometry in the initial barycentric frame, not scripted trajectories. Actual
body positions always come from N-body integration. Separately, bounded
simulation-history trails use physical samples, body-associated colors, and an
old-to-new alpha fade. Reset and scenario switching clear/reseed history.

### Labels, selection, and scene identity

Solar labels are projected, prioritized (selected body first, then Sun), tried
at several offsets, rejected when they overlap, and connected with a subtle
leader when displaced. Labels outside/behind the view remain suppressed. The
selected body also receives a camera-facing gold marker. `Compute Experiment`
uses a selected-only label policy by default, avoiding 32-body text clutter.

The renderer remains asset-free and lightweight:

- Sun: enlarged warm emissive procedural surface;
- Mercury: restrained grey mottling;
- Venus: warm cream cloud-band variation;
- Earth: ocean blue with procedural green land and subtle cloud suggestion;
- Mars: red/orange identity;
- Jupiter: enlarged warm banding;
- Saturn: pale banding plus a tilted translucent striped ring mesh;
- Uranus: pale cyan; Neptune: deeper blue.

A fixed-seed 720-point camera-consistent star sphere replaces empty pure black.
Stars, materials, rings, guides, markers, radii, and labels are rendering only.
They require no texture pack, network, or runtime file discovery.

## Stage 6C.1 corrective pass

### Mercury verification and render overlap

The actual nine-body state was integrated for ten years at the recommended
six-hour timestep. Minimum Sun--Mercury center separation was `4.60084e10 m`
(`0.307547 AU`); the Sun-plus-Mercury physical contact distance is
`6.9814e8 m`. Mercury therefore never physically intersects the Sun in this
validation interval. The misleading pass-through was caused by exaggerated
render radii. A stable render-only maximum Sun radius of `0.052` now remains
smaller than the validated rendered perihelion clearance, while physical radii,
positions, forces, integration, and diagnostics remain unchanged.

### Stable labels

Decluttering now accepts the prior frame's relative placement first. If its
actual text rectangle remains inside the viewport and collision-free, that
anchor is retained; only then are alternate anchors tried. A seven-pixel
separation margin, priority order (selected, Sun, others), subtle leaders, and
low-priority suppression handle dense inner-system layouts. This hysteresis
removes anchor hopping caused by tiny frame-to-frame projected motion. The
selected label is processed first and is never suppressed by another label.

### Barycentric motion

`Barycenter & real Sun motion` defaults on in Solar mode. It draws a gold
diamond at the barycentric origin and a distinct fading trail built from the
Sun's actual integrated physical positions stored by `TrailHistory`. Because
the displacement is small beside planetary orbits, the trail has an explicit
`Sun-motion visual amplification` control (`10x` default, `1x` physical render
scale). Amplification is applied only after physical-to-render conversion, never
to state, stored trails, diagnostics, or forces. No helical/time-axis motion or
external frame translation was added.

### Parallel-computing demonstration

Solar mode states that its physical `N=9` scene is a functional backend-
equivalence demonstration, not a meaningful speedup workload. `Compute
Experiment` now reconstructs deterministic workloads at `N=32, 64, 128, 256,
512, 1024`. Changes reset state, time, diagnostics, trails, selection, and any
live comparison result.

Synthetic workloads above 64 bodies use one dynamic OpenGL point-sprite draw,
with selected-only labeling and the existing selection marker. Solar rendering
keeps its sphere/material/ring quality. This prevents hundreds of sphere draw
calls from obscuring the compute demonstration.

`Compare CPU Backends` runs only on explicit request. It performs one untimed
initialization step per backend, then times identical isolated initial-state
copies, the same fixed `dt`, and a predeclared N-based step count bounded to
8--256 steps. It reports raw Sequential/OpenMP milliseconds and the honest
Sequential/OpenMP ratio, including ratios below one. The visible simulation is
not advanced. Final states must be finite and pass the established mixed CPU
tolerances (`1e-6 m` absolute or `1e-13` scaled position; `1e-9 m/s` absolute
or `1e-13` scaled velocity) before a ratio is shown as valid. Results invalidate
when N or comparison thread count changes. The panel labels every result
**LIVE DIAGNOSTIC -- NOT BENCHMARK EVIDENCE**.

## Controls and diagnostics

The renderer reuses one sphere mesh, one Saturn ring mesh, one static star
buffer, and one dynamic line buffer. The panel provides scenario switching,
play/pause/reset, human-oriented fixed-batch speed presets, explicit backend
availability, selected-body physical data, camera presets/focus, numerical
health, guides/trails/labels, and visual radius. Switching or resetting
reconstructs state and dependent histories consistently.

- Click near a visible body to select it; labels outside or behind the view are
  suppressed.
- Use `Focus Selected` for any body or the stable-ID astronomical camera
  profiles for repeatable live-demo and screenshot composition.
- Left-drag, `A`/`D`, or arrow keys orbit; mouse wheel or `W`/`S` dollies.
- `Space` pauses/resumes, `R` resets exactly, `C` returns to overview, and
  `Esc` exits. ImGui capture prevents camera shortcuts while editing the UI.

Energy drift is `abs(E-E0)/max(abs(E),abs(E0),1)`. Angular drift is
`norm(L-L0)/max(norm(L0),1)`. The panel shows current and maximum values since
scenario initialization/reset through the existing diagnostics API. FPS and
frame time are labeled **NOT BENCHMARK EVIDENCE**; controlled Stage 3/4/5
protocols remain the only performance evidence.

### Counterfactual experiment panel

Astronomical scenarios expose a separate `Counterfactual Experiment` panel.
The panel captures an immutable current or initial state, permits exactly one
curated non-Sun physical removal, and runs an independent paired Sequential or
OpenMP experiment on a persistent worker. The live controller is never used as
an experiment branch and remains responsive. Scenario replacement requests
cancellation, window shutdown joins the worker, and restart uses a new worker
generation. A separate application-owned scenario generation advances on every
replacement, even an identical stable-ID composition. Progress, results, and
overlays require both generations plus the full stable-ID composition to
match, so late completion cannot revive stale work after switching away and
back. Pending, running, cancelled, failed, and completed states are distinct;
an invalidated prior-scenario publication is labeled stale and suppressed
without exposing its progress, diagnostics, or overlay as current.

Controls are labeled for the next run. Every publication instead carries an
immutable executed-run descriptor containing snapshot source/time/provenance
and frame, scenario/dataset identity, removal, backend, timestep, horizon and
accepted steps, cadence, monitored/reference IDs, thresholds, persistence, and
checked work/memory estimates. Cancelled and failed runs retain the same
descriptor; displayed historical values never come from subsequently edited
controls.

The panel shows checked steps, retained samples, direct-force work, and a
conservative memory estimate before launch. Horizon, cadence, monitored-body,
history, and engine work limits remain authoritative. Progress is published
only at synchronized paired batch boundaries. Terminal publication contains
the bounded result, per-branch conservation diagnostics, warnings and dropped-
warning count, or a separate bounded worker failure message.

Completed overlays use retained samples only. Baseline paths are solid and
counterfactual paths dashed; both use the same inertial origin or the matching
historical baseline-parent origin. If that surviving origin is not monitored,
bounded baseline-only reference positions are retained at the same sample
times, separately from metric bodies and included in memory admission. Removed,
missing, unretained, or incomplete origins suppress the local overlay with an
explicit explanation; there is no live-origin fallback. The viewer does not
independently recenter the two branches. At most the selected body plus a
bounded set of significant bodies is drawn, and each path must also pass the
active-profile, class, and individual visibility masks. Displayed separation,
its fixed normalization scale, the rule requiring both thresholds, and first
and persistent sampled crossing times are counterfactual metrics, not an escape
or stability claim.
Instantaneous removal remains hypothetical, and the provisional catalog
60-second timestep, point-mass model, linear swept-step warning, and one-day
engine ceiling do not establish ephemeris accuracy.

## Validation boundary and deferred work

Headless tests cover exact Solar construction, finite sampled reference paths,
center-of-mass removal, one-Mercury-orbit conservation, scenario switching/reset,
diagnostics reset semantics, bounded trails, render immutability, deterministic
camera framing, numerical-health boundary mapping, label collision avoidance,
selected-label priority, scenario-specific label policy, and manual backend
behavior. Release CTest results are recorded in the Stage 6C.1 ledger entry.

The fresh Stage 6C.1 matrix passed CPU-only `2/2`, OpenMP `3/3`, and
visualization/OpenMP `4/4`. Solar/Sequential, Solar/OpenMP-4, Compute
Experiment/Sequential at N=32 and N=256, and Compute Experiment/OpenMP-4 at
N=512 and N=1024 each initialized a real AMD Radeon OpenGL 3.3 core context and
entered the application loop. The CUDA request exited `1` with its intended
unavailable diagnostic. This proves runtime initialization, not subjective
visual quality or interaction behavior.

The fresh N=512/OpenMP-4 live diagnostic used 32 fixed steps and observed
Sequential `100.13 ms`, OpenMP `33.4279 ms`, ratio `2.99541x`, correctness PASS.
This is **NOT BENCHMARK EVIDENCE**; it is one uncontrolled presentation-layer
observation and establishes no crossover or Auto-selection threshold.

Pending user visual checks are theme/readability at 1920x1080 and laptop sizes,
all five preset compositions, inner-label separation, trail/guide distinction,
planet material identity, Saturn ring depth/tilt, starfield restraint, selection
marker, barycenter/Sun-trail appearance, resize behavior, N switching, the live
comparison interaction, and live backend switching. CUDA compilation and runtime
validation still require an NVIDIA/CUDA host. Auto selection, high-resolution
texture assets, advanced atmosphere/post-processing, instanced high-detail
large-N meshes, final RTX capture, and ephemeris-grade Horizons input remain
deferred.

## Final user visual-validation checklist

Launch at 1920x1080 where available:

```powershell
./build-stage6c1-visual/nbody_visualizer.exe --scenario solar --backend sequential --width 1920 --height 1080
```

1. Confirm the panel is readable, intentionally styled, unclipped, and does not
   dominate the scene; expand/collapse `Advanced / Technical Details`.
2. Exercise `Overview`, `Inner System`, `Earth`, `Jupiter`, and `Saturn`; verify
   deterministic useful framing, body tracking, then manual orbit/dolly.
3. In Inner System, confirm labels no longer form a text pile, selected label is
   retained, displaced labels have subtle leaders, and hidden bodies stay hidden.
4. Verify Sun emissive identity, Earth ocean/land/cloud cue, Mars color, Jupiter
   bands, Saturn bands and tilted translucent rings, Uranus cyan, Neptune blue,
   and restrained Mercury/Venus differentiation.
5. Toggle reference orbit guides and simulation-history trails independently;
   verify guides exist immediately and history grows/fades with simulated time.
6. In `Inner System`, watch Mercury through perihelion and confirm the capped
   render-only Sun disc never misleadingly covers it.
7. Toggle `Barycenter & real Sun motion`; verify the gold barycenter diamond and
   warm fading Sun path, compare `1x` with the explicit default `10x` visual
   amplification, and confirm there is no artificial helix.
8. Select several bodies by click and combo; verify the gold marker and label,
   then use `Focus Selected`.
9. Confirm energy/angular values, unchanged bounds, green PASS state, maximum
   values in advanced details, and conspicuous non-benchmark FPS labeling.
10. Switch Sequential/OpenMP and thread count; verify active state is obvious,
   CUDA/Auto remain intentionally disabled, and motion/reset stay coherent.
11. Switch to `Compute Experiment`; select N=32, 256, 512, and 1024 and verify
   each workload resets coherently, stays responsive, uses selected-only labels,
   and makes no astronomical claim.
12. At N=512, set four comparison threads and click `COMPARE CPU BACKENDS`;
   verify both raw times, the honest ratio, correctness PASS, and the prominent
   `LIVE DIAGNOSTIC - NOT BENCHMARK EVIDENCE` label. Change N and thread count
   separately and confirm the stale result disappears each time.
13. Exercise `Space`, `R`, `C`, resize to a laptop-like window, and return to
    Solar mode; confirm ImGui capture prevents accidental camera input.
14. Confirm the starfield is subtle/static, trails do not overwhelm the scene,
    Saturn ring depth looks reasonable, and no visual effect resembles an
    unsupported physics claim.

For the N=512 Compute Experiment check:

```powershell
./build-stage6c1-visual/nbody_visualizer.exe --scenario synthetic --backend openmp --threads 4 --bodies 512 --seed 24301 --width 1920 --height 1080
```

## Stage 6D completion pass

### Solar readability and stable association

The renderer derives apparent size from the active camera and scene viewport,
then applies bounded display radii without changing a body's projected center.
The Sun/Mercury clearance constraint is evaluated in screen space and works in
both the 18-pixel inner-system regression and a deliberately compressed 4-pixel
overview regression. Sun, Mercury, Venus, Earth, and Mars use deterministic
screen slots with thin leader lines to their live projected anchors. The slots
remain stable under small body movement, stay inside the scene viewport, retain
the selected state, and never overlap each other. Outer bodies continue to use
priority-based dynamic decluttering.

### Panel, typography, and help

The panel is resizable during the session. Its width is clamped to 380--680 px
and never exceeds 55% of the current display; camera aspect, projection,
picking, and labels always use the remaining scene rectangle. Property tables
wrap long values and show full physical units with fixed or scientific notation
chosen by magnitude. Sections use user-facing names: `Simulation`, `Compute
Backend`, `Body Inspector`, `Numerical Accuracy`, and `Visualization & View`.
A platform system sans-serif font is preferred (Segoe UI/Arial, DejaVu Sans, or
SFNS); Dear ImGui's built-in font is the no-asset fallback. Every primary
control has delayed hover help. CUDA, Auto-selection development history, and
stage diary wording are not displayed as inactive pseudo-features.

### Large-N viewer safety policy

Supported structured sizes are `32, 64, 128, 256, 512, 1024, 2048, 4096,
8192, 16384, 32768, 65536, 100000`. The viewer enforces these boundaries:

| N | Viewer behavior |
|---:|---|
| 32--1024 | Continuous advancement, bounded trails, conservation diagnostics, and live Sequential/OpenMP comparison are available. |
| 2048 | Starts paused; one explicit manual direct-force step is available with Sequential or OpenMP. |
| 4096--8192 | Starts paused; one explicit manual step requires OpenMP. |
| 16384--100000 | Display and selection inspection only; the viewer never invokes the O(N^2) solver. |

Above 1024 the application does not allocate per-body trail deques or execute
O(N^2) live conservation diagnostics. Above 64 the renderer reuses one point
buffer and one `GL_POINTS` draw. Synthetic labels and inspection remain
selected-body-only, and the selector stores an integer index rather than a
100,000-entry name list. These limits provide practical 100,000-body display;
they do not claim a scalable force algorithm. Barnes--Hut, FMM, and GPU
instancing are not implemented.

### Stage 6D automated and runtime validation

- Fresh GCC 15.1 Release CPU-only, OpenMP, visualization/OpenMP, and
  MPI+OpenMP configurations passed CTest `3/3`, `4/4`, `5/5`, and `8/8`.
- Headless presentation tests cover camera-aware Sun/Mercury clearance, stable
  five-body callouts, selected retention, viewport containment, panel width,
  unit formatting, every tooltip, controller manual stepping, all supported
  sizes, deterministic 100,000-body construction, complete render snapshots,
  and safety-policy boundaries.
- Solar/Sequential, Solar/OpenMP-4, synthetic Sequential N=32 and N=256,
  synthetic OpenMP-4 N=512, N=1024, and N=2048, and synthetic display-only
  N=100,000 each created an AMD Radeon OpenGL 3.3 core context, initialized the
  application, and remained in the event loop until deliberate interruption.
  This proves runtime construction and batched rendering initialization, not
  subjective appearance or force performance. A CUDA request exited `1` with
  the intended explicit unavailable diagnostic and no silent fallback.

Native-app automation was unavailable on the validation host. Real contexts can
be launched and observed alive, but resize handles, presets, hover tooltips,
selection, and visual composition cannot be clicked or judged programmatically.
The final status is therefore **IMPLEMENTED / USER VISUAL VALIDATION PENDING**.

### Stage 6D.2 stable Sun sizing

Stage 6D.1 incorrectly capped the Sun from the instantaneous projected
Sun--Mercury separation, so Mercury's orbit visibly changed the Sun radius.
Stage 6D.2 removes all planet positions from the Sun sizing policy. Its target
pixel radius is the Sun's render-only base radius projected using vertical
viewport height, fixed 45-degree field of view, and `OrbitCamera::distance()`,
then clamped to 20--36 pixels. The target is converted back at the Sun's actual
camera depth solely so projection produces that invariant pixel radius.

At fixed camera, viewport, and body-size presentation settings, orbital phase
therefore cannot change the Sun size. The focused 1920x1080 test reports
Overview `20 px`, Inner System `36 px`, and `27.9539 px` unchanged across 352
six-hour steps at an unclamped fixed camera distance. Zoom response is smooth
and bounded. Exaggerated Sun/Mercury glyphs may temporarily overlap at distant
scales; correct centers, normal depth rendering, and the existing stable callout
are less misleading than breathing or displaced orbits.

### Stage 6D.3 inner-orbit shape verification

No physics, guide, trail, or projection defect was found. The JPL approximate
J2000 elements used by the initializer are:

| Body | a (AU) | e | i (deg) | ascending node Ω (deg) | perihelion longitude ϖ (deg) | mean longitude L (deg) | derived mean anomaly M (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mercury | 0.38709927 | 0.20563593 | 7.00497902 | 48.33076593 | 77.45779628 | 252.25032350 | 174.79252722 |
| Venus | 0.72333566 | 0.00677672 | 3.39467605 | 76.67984255 | 131.60246718 | 181.97909950 | 50.37663232 |
| Earth/EMB approximation | 1.00000261 | 0.01671123 | -0.00001531 | 0 | 102.93768193 | 100.46457166 | 357.52688973 |
| Mars | 1.52371034 | 0.09339410 | 1.84969142 | 49.55953891 | -23.94362959 | -4.55343205 | 19.39019754 |

The implementation solves `E - e sin(E) = M`, constructs orbital-plane
coordinates `a(cos(E)-e)` and `a sqrt(1-e^2) sin(E)`, applies perihelion/node/
inclination rotations, derives velocity by a centered one-hour difference of
the same fitted model, and finally subtracts one common center-of-mass position
and velocity. Initial Cartesian osculating eccentricities were `0.205628851`,
`0.006788612`, `0.016723353`, and `0.093428916`, respectively, confirming that
the conversion retains the source shape.

A deterministic 700-day (2800 six-hour step) nine-body integration measured:

| Body | min r (m / AU) | max r (m / AU) | radial range (m / AU) | max/min | e from extrema | source e | absolute Δe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mercury | 4.600927371e10 / 0.307552999 | 6.981784139e10 / 0.466703443 | 2.380856768e10 / 0.159150445 | 1.517473234 | 0.205552626 | 0.205635930 | 8.33e-5 |
| Venus | 1.074731247e11 / 0.718413465 | 1.089527934e11 / 0.728304440 | 1.479668765e9 / 0.009890975 | 1.013767803 | 0.006836837 | 0.006776720 | 6.01e-5 |
| Earth | 1.470978866e11 / 0.983288638 | 1.521060978e11 / 1.016766462 | 5.008211195e9 / 0.033477824 | 1.034046792 | 0.016738451 | 0.016711230 | 2.72e-5 |
| Mars | 2.066432935e11 / 1.381325099 | 2.492708609e11 / 1.666272786 | 4.262756733e10 / 0.284947688 | 1.206285753 | 0.093499109 | 0.093394100 | 1.05e-4 |

The separate ten-year regression reproduced Mercury minimum separation as
`4.600842350e10 m` (`0.307547315 AU`). A 4097-sample guide check measured:

| Body | guide min (m / AU) | guide max (m / AU) | guide e from extrema |
|---|---:|---:|---:|
| Mercury | 4.600100898e10 / 0.307497752 | 6.981744364e10 / 0.466700785 | 0.205635925 |
| Venus | 1.074761958e11 / 0.718433995 | 1.089427713e11 / 0.728237446 | 0.006776557 |
| Earth | 1.470982908e11 / 0.983291340 | 1.520982031e11 / 1.016713690 | 0.016711133 |
| Mars | 2.066549460e11 / 1.381402991 | 2.492326029e11 / 1.666017048 | 0.093395086 |

`TrailHistory` stores the integrated barycentric Cartesian positions verbatim.
Planet and Sun samples receive the same linear render transform before upload as
`GL_LINE_STRIP` vertices; no normalization, fixed-radius reconstruction, guide
substitution, or planet-trail amplification occurs. A daily one-Mercury-orbit
regression preserves its greater-than-1.45 heliocentric radius ratio through
render preparation. The optional amplification applies only to the separately
constructed Sun-motion trail.

The trajectories can nevertheless look nearly circular because eccentricity is
not the visual axis flattening. The intrinsic `b/a = sqrt(1-e^2)` values are
Mercury `0.978629`, Venus `0.999977`, Earth `0.999860`, and Mars `0.995629`.
Mercury's shape is clearer from the Sun's focus offset and its 1.517 radial ratio
than from a dramatically oval outline. In the existing 1920x1080 Inner System
preset, the projected Mercury guide has covariance axis ratio `0.291186`,
apparent major/minor radii `55.1233/16.0512 px`, and a `7.00421 px` Sun-to-guide-
centroid offset. Thin low-alpha guides and partial fading history trails can make
that focus relationship easy to miss. No camera change or eccentricity
exaggeration was introduced; this remains an approximate mean-element demo, not
an ephemeris-grade claim.

### Final user visual-validation checklist

Launch a Solar view at 1920x1080 and a laptop-sized window, plus a 100,000-body
display-only view:

```powershell
./build-stage6d-visual/nbody_visualizer.exe --scenario solar --backend openmp --threads 4 --width 1920 --height 1080
./build-stage6d-visual/nbody_visualizer.exe --scenario synthetic --backend sequential --bodies 100000 --seed 24301 --width 1280 --height 800
```

1. Resize the window and panel; verify readable wrapping and that the scene never
   appears underneath the panel.
2. Exercise all five Solar presets and manual orbit/dolly. Confirm Sun through
   Mars have stable, non-overlapping callouts with clear leaders in Inner System
   and compressed Overview compositions.
3. Watch Mercury near the Sun while zooming. Confirm centers/orbits never move,
   the discs remain distinct, and the Sun label preserves identity when its disc
   is reduced for clearance.
4. Hover every primary control and confirm concise help appears; inspect full
   units and long numeric values at both window sizes.
5. Confirm selection/focus, trails, guides, barycenter, 1x/10x real Sun motion,
   conservation status, reset, pause, and live Sequential/OpenMP switching.
6. Visit N=1024, 2048, 8192, and 100000. Confirm warnings and paused/manual or
   display-only behavior match the table, selected-only inspection remains
   responsive, and no large workload silently begins direct-force integration.
7. Confirm the 100,000-body point cloud and restrained starfield are legible and
   that no UI text presents FPS or live comparison as benchmark evidence.
