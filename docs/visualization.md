# Stage 6C final presentation pass

## Scope and status

Stage 6C is the final major visual/UX pass over the scientifically functional
Stage 6B application. It adds a deliberate dark-scientific theme and information
hierarchy, deterministic camera compositions, reference orbit context, fading
history trails, screen-space label decluttering, procedural planet materials,
Saturn rings, a deterministic starfield, and an explicit selection marker. The
physics library remains independent of graphics and UI dependencies.

Stage 6B was manually inspected and its findings drove this pass. Stage 6C's
automated and real-context validation is recorded below, but automation cannot
judge final composition and interaction feel. Status therefore remains
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

Options are `--scenario solar|synthetic`, `--backend sequential|openmp|cuda`,
`--threads N`, `--bodies N`, `--seed N`, `--width N`, and `--height N`. Body
count and seed apply to the synthetic scenario. CUDA is accepted only to issue
a clear unavailable diagnostic; it never silently falls back. Auto is shown
disabled because Stage 5 calibration is not complete.

## Loop and separation

Each active frame advances an integer number of Velocity Verlet steps at the
scenario's fixed physical timestep, updates diagnostics and bounded physical-
space trail history, then creates a read-only render snapshot. Wall-clock frame
duration never becomes a physics timestep. Pausing stops physics advancement
but leaves rendering and interaction active.

The Solar scenario uses a six-hour timestep. Named presets map to 1, 4, 16, or
64 steps per render: six hours, one day, four days, or sixteen days of simulated
time. Trails are sampled every simulated day and retained in a bounded deque.
The synthetic fixture uses its established deterministic generator and a
10-second timestep.

Positions, trails, orbit guides, labels, rings, and markers share one fixed
linear render transform derived from the initial state. Body radii are
display-only: the default Solar multiplier is 80, with a minimum render radius
of 0.035 and small per-body identity factors. Changing these values never
mutates mass, position, velocity, or force calculations. No nonlinear position
distortion is used.

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

Five deterministic Solar compositions coexist with manual orbit/dolly:

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

## Controls and diagnostics

The renderer reuses one sphere mesh, one Saturn ring mesh, one static star
buffer, and one dynamic line buffer. The panel provides scenario switching,
play/pause/reset, human-oriented fixed-batch speed presets, explicit backend
availability, selected-body physical data, camera presets/focus, numerical
health, guides/trails/labels, and visual radius. Switching or resetting
reconstructs state and dependent histories consistently.

- Click near a visible body to select it; labels outside or behind the view are
  suppressed.
- Use `Focus Selected` for any body or the five deterministic Solar camera
  presets for repeatable live-demo and screenshot composition.
- Left-drag, `A`/`D`, or arrow keys orbit; mouse wheel or `W`/`S` dollies.
- `Space` pauses/resumes, `R` resets exactly, `C` returns to overview, and
  `Esc` exits. ImGui capture prevents camera shortcuts while editing the UI.

Energy drift is `abs(E-E0)/max(abs(E),abs(E0),1)`. Angular drift is
`norm(L-L0)/max(norm(L0),1)`. The panel shows current and maximum values since
scenario initialization/reset through the existing diagnostics API. FPS and
frame time are labeled **NOT BENCHMARK EVIDENCE**; controlled Stage 3/4/5
protocols remain the only performance evidence.

## Validation boundary and deferred work

Headless tests cover exact Solar construction, finite sampled reference paths,
center-of-mass removal, one-Mercury-orbit conservation, scenario switching/reset,
diagnostics reset semantics, bounded trails, render immutability, deterministic
camera framing, numerical-health boundary mapping, label collision avoidance,
selected-label priority, scenario-specific label policy, and manual backend
behavior. Release CTest results are recorded in the Stage 6C ledger entry.

The fresh Stage 6C matrix passed CPU-only `2/2`, OpenMP `3/3`, and
visualization/OpenMP `4/4`. Solar/Sequential, Solar/OpenMP-4, and Compute
Experiment/Sequential each initialized a real AMD OpenGL 3.3 core context and
entered the application loop; the CUDA request exited `1` with its intended
diagnostic. This proves runtime initialization, not subjective visual quality.

Pending user visual checks are theme/readability at 1920x1080 and laptop sizes,
all five preset compositions, inner-label separation, trail/guide distinction,
planet material identity, Saturn ring depth/tilt, starfield restraint, selection
marker, resize behavior, and live Sequential/OpenMP switching. CUDA viewer
integration, Auto selection, high-resolution texture assets, advanced atmosphere/
post-processing, large-N point/instanced rendering, final RTX capture, and
ephemeris-grade Horizons input are intentionally deferred beyond Stage 6C.

## Final user visual-validation checklist

Launch at 1920x1080 where available:

```powershell
./build-stage6c-visual/nbody_visualizer.exe --scenario solar --backend sequential --width 1920 --height 1080
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
6. Select several bodies by click and combo; verify the gold marker and label,
   then use `Focus Selected`.
7. Confirm energy/angular values, unchanged bounds, green PASS state, maximum
   values in advanced details, and conspicuous non-benchmark FPS labeling.
8. Switch Sequential/OpenMP and thread count; verify active state is obvious,
   CUDA/Auto remain intentionally disabled, and motion/reset stay coherent.
9. Switch to `Compute Experiment`; verify N/seed/workload language, selected-only
   labeling, useful overview, numerical health, and no astronomical claim.
10. Exercise `Space`, `R`, `C`, resize to a laptop-like window, and return to
    Solar mode; confirm ImGui capture prevents accidental camera input.
11. Confirm the starfield is subtle/static, trails do not overwhelm the scene,
    Saturn ring depth looks reasonable, and no visual effect resembles an
    unsupported physics claim.
