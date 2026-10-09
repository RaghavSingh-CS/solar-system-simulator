# Solar System 60 v1 snapshot

This directory is the immutable source snapshot and generated representation
for catalog `jpl-ssd-j2000-60-v1`, dataset version `1.0.0`. It is separate from
the legacy nine-body approximate Solar System scenario, which remains
unchanged.

## Coordinate contract

- epoch: `JD 2451545.0 TDB` (`2000-Jan-01 12:00:00 TDB`)
- origin: Solar System barycenter (`@0`)
- axes: J2000 ICRF/equatorial
- vectors: geometric Cartesian state, no aberration correction
- archived Horizons units: `KM-S`
- normalized units: metres, metres per second, cubic metres per second squared,
  kilograms, and metres

All 60 state-vector responses must satisfy that contract before admission.
Mass is generated from the sourced body-specific GM using the project's
existing gravitational constant. Planet body GMs—not planet-system GMs—are
used because represented satellites contribute their own mass.

## Custody layout

- `roster.json`: ordered roster, stable identities, requested source mappings,
  limitations, and deferred expansion queue.
- `raw/horizons/`: exact, unmodified Horizons batch responses for 60 bodies.
- `raw/sbdb/`: exact, unmodified Small-Body Database API responses used for 13
  small-body physical parameters.
- `raw/sources/`: exact JPL GM kernel and planetary/satellite physical-parameter
  pages.
- `snapshot_manifest.json`: complete request parameters, source URLs, UTC
  retrieval times, response sizes, media types, and SHA-256 hashes.
- `normalized/catalog.json`: deterministic SI-normalized catalog.

Snapshot manifest SHA-256:
`82b271f3fe7470172693c9ab4810a664153ab59f94f293498000b19504749229`.

## Reproduction

Acquisition is the only network-facing step. For a new versioned snapshot,
run from the repository root:

```powershell
python -B tools/astronomy/acquire_catalog_snapshot.py `
  --roster data/astronomy/solar_system_60_v1/roster.json `
  --output data/astronomy/solar_system_60_v1
```

Do not overwrite this version to refresh upstream data; create a new dataset
version and review every source/solution change.

Generation is offline and first verifies every archived response hash:

```powershell
python -B tools/astronomy/generate_catalog_dataset.py `
  --roster data/astronomy/solar_system_60_v1/roster.json `
  --snapshot data/astronomy/solar_system_60_v1/snapshot_manifest.json `
  --normalized-output data/astronomy/solar_system_60_v1/normalized/catalog.json `
  --cpp-output src/astronomy/generated/solar_system_60_v1.generated.inc
```

The focused custody/reconstruction test reruns the generator in a temporary
directory and requires byte-identical normalized JSON and C++ output.

## Scientific scope

Admission establishes source identity, custody, a common state-vector contract,
and internally consistent unit conversion. It does not establish ephemeris-grade
long-duration accuracy. The simulator remains a Newtonian point-mass,
fixed-timestep Velocity Verlet model without perturbation-model parity with JPL
ephemerides, relativistic terms, figure effects, nongravitational comet forces,
adaptive stepping, uncertainty propagation, or close-encounter regularization.
