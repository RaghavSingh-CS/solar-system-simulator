# Repository build and artifact policy

These rules are binding for automated agents and developers working in this
repository.

- All ordinary CMake builds must use the tracked canonical presets and the
  `build/` directory.
- Never create a new root-level `build-*` directory for tests, reviews,
  experiments, or debugging.
- Reuse the canonical `build/cpu-release`, `build/openmp-release`, or
  `build/cuda-release` tree when its configuration matches the task.
- Temporary configurations must live under `build/scratch/<purpose>/`.
- Remove disposable scratch builds safely after validation, but never remove a
  directory until its resolved path is confirmed to be inside `build/scratch/`.
- Never store scientific results, provenance records, or irreplaceable evidence
  in `build/` or another disposable scratch path.
- Do not alter, replace, or reinterpret frozen Stage 5 environments,
  protocols, scripts, evidence, or exact historical rerun instructions.
- Do not create root-level logs, binaries, test results, validation output, or
  generated dependency caches.
- Any exception to this policy requires an explicit justification and user
  approval before files or directories are created.
- Run `python -B scripts/check_repository_layout.py` before proposing a commit
  that changes build, test, or repository-organization behavior.

The existing root-level `build-*` directories are temporary grandfathered
artifacts recorded in `tools/repository_layout_baseline.json`. Their presence
does not authorize creating more. Never delete or move them without the
separate evidence review and cleanup approval described in
`docs/repository_layout.md`.
