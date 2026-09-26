# Stage 6B Solar System visualization

## Scope and status

Stage 6B turns the optional Stage 6A OpenGL shell into an explorable desktop
application. It adds an approximate Solar System scenario, an ImGui control and
diagnostics panel, physical-space orbital trails, body labels, click selection,
smooth focus/overview navigation, and explicit visual-size controls. The physics
library remains independent of graphics and UI dependencies.

Release CPU-only, OpenMP, and visualization builds pass their complete CTest
suites. Real OpenGL 3.3 contexts were smoke-launched on the development machine
for Solar/Sequential, Solar/OpenMP, and Synthetic/Sequential modes. Automated
tests cannot judge the native window's composition and interaction feel, so the
status is **IMPLEMENTED / USER VISUAL VALIDATION PENDING**.

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
independent of Python/Jinja2 and avoids copied runtime DLLs. Initial GLFW/GLM
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

Positions, trails, and labels share one fixed render transform derived from the
initial state. Body radii are display-only: the default Solar multiplier is 80,
with a minimum render radius of 0.035. Changing the multiplier never mutates
mass, position, velocity, or force calculations. The Sun is emissive; planet
shading is a simple Sun-centered diffuse cue.

## Controls and diagnostics

The renderer reuses one sphere mesh for all bodies and one dynamic line buffer
for all trails. The ImGui panel provides scenario switching, play/pause/reset,
named speed presets, Sequential/OpenMP choice and thread count, selected-body
physical data, focus actions, conservation drift, trails/labels, trail length,
and visual radius. Switching or resetting reconstructs state and dependent
histories consistently.

- Click near a visible body to select it; labels outside or behind the view are
  suppressed.
- Use panel buttons to focus the selected body, focus Sun/body 0, or return to
  overview. Focus tracking follows a moving body smoothly.
- Left-drag, `A`/`D`, or arrow keys orbit; mouse wheel or `W`/`S` dollies.
- `Space` pauses/resumes, `R` resets exactly, `C` returns to overview, and
  `Esc` exits. ImGui capture prevents camera shortcuts while editing the UI.

Energy drift is `abs(E-E0)/max(abs(E),abs(E0),1)`. Angular drift is
`norm(L-L0)/max(norm(L0),1)`. The panel shows current and maximum values since
scenario initialization/reset through the existing diagnostics API. FPS and
frame time are labeled **NOT BENCHMARK EVIDENCE**; controlled Stage 3/4/5
protocols remain the only performance evidence.

## Validation boundary and deferred work

Headless tests cover exact Solar construction, finiteness, center-of-mass
removal, one-Mercury-orbit conservation, scenario switching/reset, diagnostics
reset semantics, bounded trails, render immutability, camera focus/framing, and
manual backend behavior. Release CTest passed 2/2 CPU-only, 3/3 OpenMP, and 4/4
visualization configurations.

Pending user visual checks are readability at overview and focused scales,
label overlap, trail appearance, selection feel, resize behavior, and live
Sequential/OpenMP switching. CUDA viewer integration, Auto selection, large-N
point/instanced rendering, textures/effects, presentation capture, and
ephemeris-grade Horizons input are outside Stage 6B.
