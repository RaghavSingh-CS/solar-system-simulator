# Generated executables

`01_build.ps1` creates configuration subdirectories beneath a unique run ID and
writes `executables.json` with hashes and build provenance. The short demo
build stages CUDA Release only; the full workflow also stages CPU and OpenMP.
Every executable is re-hashed against its manifest and matching build output
before use. Generated binaries and manifests are intentionally ignored by Git.

Do not copy binaries between machines. CUDA-enabled binaries must be compiled
on the NVIDIA validation host.
