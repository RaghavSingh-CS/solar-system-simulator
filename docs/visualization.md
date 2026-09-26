# Stage 6A interactive visualization

## Scope and status

Stage 6A supplies the minimal optional desktop application layer: a real
OpenGL window, 3D body rendering, fixed-step simulation advancement, an orbit
camera, pause/reset, and explicit Sequential/OpenMP backend selection. It is
an architecture foundation rather than the final presentation UI.

The application has opened real OpenGL 3.3 core contexts on the development
machine with both Sequential and OpenMP backends. Automated model tests pass.
Because the automation environment cannot reliably inspect or drive the native
window, appearance and interactive controls remain **user visual validation
pending**.

## Dependency acquisition

Visualization is guarded by `NBODY_BUILD_VISUALIZATION` and is `OFF` by
default. A headless configure neither fetches nor compiles graphics code.

When visualization is enabled, CMake FetchContent obtains exact commits:

- GLFW 3.4: `a74efa0d5628b74adc0426af4c5710e287fa7c2c`;
- GLM 1.0.1: `0af55ccecd98d4e5a8d1fad7de25ba429d60e863`.

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

Useful options are `--backend sequential|openmp|cuda`, `--threads N`,
`--scenario synthetic32|analytic`, `--bodies N`, `--seed N`, `--dt SECONDS`,
`--steps-per-frame N`, `--width N`, and `--height N`. `--backend cuda` is
accepted only to issue an explicit Stage 6A unavailable diagnostic; it never
falls back silently. Auto is not an option.

## Loop and separation

Each non-minimized frame performs these phases in order:

1. poll events and apply camera/controller input;
2. advance `steps-per-frame` Velocity Verlet steps using the explicit physical
   `dt` and selected existing `ForceBackend`;
3. update a reusable read-only render snapshot from the physical state;
4. draw the snapshot and swap buffers.

Pause skips only phase 2. Frame elapsed time is not used as an integration
timestep, and display FPS is not reported as simulation performance.

The default is the existing deterministic seed-24301 synthetic fixture with
32 bodies, `dt=10 s`, and 256 steps per rendered frame. It is labeled as a
synthetic fixture, not the Solar System. Reset copies the exact initial state.

Physics remains double-precision SI data. `RenderSpaceTransform` derives a
fixed position scale from the initial state and maps positions into a compact
render extent. It derives modest mass-based display radii and colors only for
the snapshot. Neither scaled positions nor display radii flow back into the
simulation.

## Rendering and controls

`BodyRenderer` creates one 16-by-24 smooth sphere mesh and reuses its VAO/VBO/
index buffer for every body. Shaders compile once. The renderer uses perspective
projection, resize-correct aspect, depth testing, back-face culling, basic
directional lighting, distinguishable colors, and a dark background.

- `Space`: pause/resume.
- `R`: reset the scenario exactly.
- `1`: select Sequential.
- `2`: select OpenMP with the configured `--threads` count; unavailable builds
  log an actionable input error and keep running.
- Left mouse drag, `A`/`D`, and arrow keys: orbit.
- Mouse wheel or `W`/`S`: dolly.
- `C`: reset the camera; `Esc`: exit.

The title identifies the fixture, active backend/thread count, paused/running
state, simulated time, and fixed steps per rendered frame.

## Validation boundary and deferred work

Headless tests cover controller advancement/pause/resume/reset, manual backend
selection, render transformation immutability/finite output, and camera matrix
and reset behavior. Graphical screenshot tests are deliberately absent.

Stage 6B/6C may add presentation controls, scalable point/instanced rendering,
trails, selection, diagnostics, textures/effects, UI typography, and a
provenance-backed Solar System scenario. Final RTX 4050 CUDA viewer integration
and presentation capture are also deferred. None of this viewer's diagnostic
display is Stage 3/4/5 benchmark evidence.
