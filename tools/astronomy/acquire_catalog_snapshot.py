#!/usr/bin/env python3
"""Acquire the immutable raw-source snapshot for the 60-body catalog.

The script is intentionally network-facing. It writes response bytes exactly as
received and records request parameters, retrieval time, URL, and SHA-256. The
offline generator is a separate script and never performs network access.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request


HORIZONS_PARAMETERS = {
    "format": "text",
    "OBJ_DATA": "YES",
    "MAKE_EPHEM": "YES",
    "EPHEM_TYPE": "VECTORS",
    "CENTER": "'@0'",
    "TLIST": "'2451545.0'",
    "TLIST_TYPE": "JD",
    "TIME_TYPE": "TDB",
    "REF_SYSTEM": "ICRF",
    "REF_PLANE": "FRAME",
    "VEC_TABLE": "2",
    "VEC_CORR": "NONE",
    "OUT_UNITS": "KM-S",
    "CSV_FORMAT": "YES",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def request_bytes(url: str, attempts: int = 3) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "solar-system-simulator-catalog-snapshot/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as error:  # pragma: no cover - exercised only on network failure
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"failed to retrieve {url}: {last_error}")


def safe_name(stable_id: str) -> str:
    return stable_id.replace(":", "_").replace("/", "_")


def archive_response(
    root: pathlib.Path,
    relative_path: pathlib.Path,
    url: str,
    parameters: dict[str, str] | None,
    media_type: str,
) -> dict[str, object]:
    data = request_bytes(url)
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "path": relative_path.as_posix(),
        "source_url": url,
        "request_parameters": parameters or {},
        "retrieved_utc": utc_now(),
        "media_type": media_type,
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def validate_horizons_identity(data: bytes, body: dict[str, object]) -> None:
    text = data.decode("utf-8", errors="strict")
    identity = str(body["horizons_identity"])
    solution = str(body["expected_solution"])
    target_match = re.search(r"(?m)^Target body name:\s*(.+?)\s+\{source:\s*([^}]+)\}", text)
    if target_match is None:
        raise RuntimeError(f"{body['stable_id']}: Horizons target identity is missing")
    if identity not in target_match.group(1):
        raise RuntimeError(
            f"{body['stable_id']}: target mismatch: {target_match.group(1)!r}"
        )
    if target_match.group(2).strip() != solution:
        raise RuntimeError(
            f"{body['stable_id']}: solution mismatch: "
            f"expected {solution!r}, received {target_match.group(2).strip()!r}"
        )
    required = (
        "Center body name: Solar System Barycenter (0)",
        "Output units    : KM-S",
        "Output type     : GEOMETRIC cartesian states",
        "Reference frame : ICRF",
        "Start time      : A.D. 2000-Jan-01 12:00:00.0000 TDB",
        "$$SOE",
        "$$EOE",
    )
    for marker in required:
        if marker not in text:
            raise RuntimeError(f"{body['stable_id']}: missing Horizons marker {marker!r}")


def build_query_url(endpoint: str, parameters: dict[str, str]) -> str:
    return endpoint + "?" + urllib.parse.urlencode(parameters)


def acquire(roster_path: pathlib.Path, output_root: pathlib.Path) -> None:
    roster_bytes = roster_path.read_bytes()
    roster = json.loads(roster_bytes)
    bodies = roster["bodies"]
    if len(bodies) != 60:
        raise RuntimeError(f"frozen roster must contain exactly 60 bodies, found {len(bodies)}")
    if [body["order"] for body in bodies] != list(range(60)):
        raise RuntimeError("frozen roster order must be contiguous from zero")

    snapshot_path = output_root / "snapshot_manifest.json"
    if snapshot_path.exists():
        raise RuntimeError(
            f"snapshot already exists at {snapshot_path}; remove it explicitly before reacquiring"
        )
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    endpoints = roster["source_endpoints"]
    sources: dict[str, dict[str, object]] = {}
    static_sources = (
        ("gm_kernel", "sources/gm_Horizons.pck", "application/octet-stream"),
        ("planetary_physical_parameters", "sources/planetary_physical_parameters.html", "text/html"),
        ("satellite_physical_parameters", "sources/satellite_physical_parameters.html", "text/html"),
    )
    for key, relative, media_type in static_sources:
        sources[key] = archive_response(
            raw_root,
            pathlib.Path(relative),
            endpoints[key],
            None,
            media_type,
        )

    state_records: dict[str, dict[str, object]] = {}
    sbdb_records: dict[str, dict[str, object]] = {}
    for body in bodies:
        parameters = dict(HORIZONS_PARAMETERS)
        parameters["COMMAND"] = f"'{body['horizons_command']}'"
        url = build_query_url(endpoints["horizons"], parameters)
        relative = pathlib.Path("horizons") / (
            f"{body['order']:02d}_{safe_name(body['stable_id'])}.txt"
        )
        record = archive_response(raw_root, relative, url, parameters, "text/plain")
        validate_horizons_identity((raw_root / relative).read_bytes(), body)
        state_records[body["stable_id"]] = record

        if "sbdb_query" in body:
            sbdb_parameters = {"sstr": str(body["sbdb_query"]), "phys-par": "1"}
            sbdb_url = build_query_url(endpoints["sbdb"], sbdb_parameters)
            sbdb_relative = pathlib.Path("sbdb") / (
                f"{body['order']:02d}_{safe_name(body['stable_id'])}.json"
            )
            sbdb_records[body["stable_id"]] = archive_response(
                raw_root,
                sbdb_relative,
                sbdb_url,
                sbdb_parameters,
                "application/json",
            )

    manifest = {
        "schema_version": 1,
        "dataset_id": roster["dataset_id"],
        "dataset_version": roster["dataset_version"],
        "roster_path": roster_path.relative_to(output_root.parents[2]).as_posix(),
        "roster_sha256": sha256_bytes(roster_bytes),
        "acquisition_completed_utc": utc_now(),
        "sources": sources,
        "state_records": state_records,
        "sbdb_records": sbdb_records,
    }
    snapshot_path.write_bytes(canonical_json_bytes(manifest))
    print(f"archived {len(state_records)} Horizons states and {len(sbdb_records)} SBDB records")
    print(f"snapshot manifest: {snapshot_path}")
    print(f"snapshot manifest SHA-256: {sha256_bytes(snapshot_path.read_bytes())}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        acquire(args.roster.resolve(), args.output.resolve())
        return 0
    except Exception as error:
        print(f"catalog snapshot acquisition failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
