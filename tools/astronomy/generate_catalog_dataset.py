#!/usr/bin/env python3
"""Reconstruct normalized and typed catalog artifacts from archived responses only."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import io
import json
import math
import pathlib
import re
import sys


CLASS_CPP = {
    "star": "AstronomicalBodyClass::star",
    "planet": "AstronomicalBodyClass::planet",
    "dwarf_planet": "AstronomicalBodyClass::dwarf_planet",
    "natural_satellite": "AstronomicalBodyClass::natural_satellite",
    "asteroid": "AstronomicalBodyClass::asteroid",
    "comet": "AstronomicalBodyClass::comet",
    "trans_neptunian_object": "AstronomicalBodyClass::trans_neptunian_object",
}


class HtmlRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def read_verified(raw_root: pathlib.Path, record: dict[str, object]) -> bytes:
    data = (raw_root / str(record["path"])).read_bytes()
    actual = sha256_bytes(data)
    if actual != record["sha256"]:
        raise RuntimeError(
            f"raw response checksum mismatch for {record['path']}: "
            f"expected {record['sha256']}, found {actual}"
        )
    return data


def first_number(text: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", text)
    if match is None:
        raise RuntimeError(f"no numeric value in {text!r}")
    value = float(match.group(0))
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite numeric value in {text!r}")
    return value


def table_rows(data: bytes) -> list[list[str]]:
    parser = HtmlRows()
    parser.feed(data.decode("utf-8", errors="strict"))
    return parser.rows


def table_mean_radius_km(rows: list[list[str]], name: str) -> float:
    matches = [row for row in rows if row and row[0] == name]
    if len(matches) != 1 or len(matches[0]) < 3:
        raise RuntimeError(f"expected one physical-parameter row for {name}, found {len(matches)}")
    radius = first_number(matches[0][2])
    if radius <= 0.0:
        raise RuntimeError(f"{name}: radius is not positive")
    return radius


def parse_state(body: dict[str, object], data: bytes) -> tuple[list[float], str]:
    text = data.decode("utf-8", errors="strict")
    target = re.search(r"(?m)^Target body name:\s*(.+?)\s+\{source:\s*([^}]+)\}", text)
    if target is None:
        raise RuntimeError(f"{body['stable_id']}: missing target identity")
    if str(body["horizons_identity"]) not in target.group(1):
        raise RuntimeError(f"{body['stable_id']}: target identity changed")
    solution = target.group(2).strip()
    if solution != body["expected_solution"]:
        raise RuntimeError(
            f"{body['stable_id']}: expected solution {body['expected_solution']!r}, "
            f"found {solution!r}"
        )
    required = (
        "Center body name: Solar System Barycenter (0)",
        "Output units    : KM-S",
        "Output type     : GEOMETRIC cartesian states",
        "Reference frame : ICRF",
        "Start time      : A.D. 2000-Jan-01 12:00:00.0000 TDB",
    )
    for marker in required:
        if marker not in text:
            raise RuntimeError(f"{body['stable_id']}: missing state marker {marker!r}")
    match = re.search(r"\$\$SOE\s*\n([^\n]+)\n\$\$EOE", text)
    if match is None:
        raise RuntimeError(f"{body['stable_id']}: expected exactly one state-vector row")
    row = next(csv.reader(io.StringIO(match.group(1)), skipinitialspace=True))
    if len(row) < 8:
        raise RuntimeError(f"{body['stable_id']}: malformed state-vector row")
    if abs(float(row[0]) - 2451545.0) > 1.0e-9 or "2000-Jan-01 12:00:00" not in row[1]:
        raise RuntimeError(f"{body['stable_id']}: state epoch is not JD 2451545.0 TDB")
    values = [float(value) for value in row[2:8]]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{body['stable_id']}: state contains a non-finite value")
    return values, solution


def parse_gm_kernel(data: bytes) -> dict[str, float]:
    text = data.decode("ascii", errors="strict")
    result: dict[str, float] = {}
    for body_id, value in re.findall(
        r"(?m)^\s*BODY(\d+)_GM\s*=\s*\(\s*([-+0-9.Ee]+)", text
    ):
        result[body_id] = float(value)
    if not result:
        raise RuntimeError("JPL GM kernel did not contain any BODY*_GM values")
    return result


def parse_sbdb(body: dict[str, object], data: bytes) -> dict[str, object]:
    payload = json.loads(data)
    obj = payload.get("object", {})
    orbit = payload.get("orbit", {})
    if str(obj.get("spkid")) != str(body["naif_id"]):
        raise RuntimeError(
            f"{body['stable_id']}: SBDB SPK identity mismatch: {obj.get('spkid')!r}"
        )
    if str(orbit.get("orbit_id")) != str(body["expected_sbdb_orbit"]):
        raise RuntimeError(
            f"{body['stable_id']}: SBDB orbit solution mismatch: "
            f"expected {body['expected_sbdb_orbit']!r}, found {orbit.get('orbit_id')!r}"
        )
    parameters = {entry.get("name"): entry for entry in payload.get("phys_par", [])}
    return {"object": obj, "orbit": orbit, "parameters": parameters}


def provenance(record: dict[str, object], source_name: str, record_id: str) -> dict[str, str]:
    return {
        "source_name": source_name,
        "source_uri": str(record["source_url"]),
        "record_id": record_id,
        "retrieved_utc": str(record["retrieved_utc"]),
        "sha256": str(record["sha256"]),
    }


def limitations(body: dict[str, object]) -> str:
    body_class = body["class"]
    if body_class == "natural_satellite":
        base = (
            "JPL state is an initial condition only; the project point-mass Newtonian "
            "model omits parent oblateness, tides, resonances, and ephemeris-fit dynamics."
        )
    elif body_class in ("asteroid", "comet"):
        base = (
            "The small-body orbital solution is mutable and its covariance is not propagated; "
            "the project uses Newtonian point masses and a fixed timestep."
        )
    else:
        base = (
            "JPL state is an initial condition only; the project omits relativistic, multipole, "
            "tidal, and ephemeris-fit dynamics, so long-term ephemeris agreement is not implied."
        )
    specific = body.get("specific_limitation")
    return base + (" " + str(specific) if specific else "")


def cpp_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def cpp_number(value: float) -> str:
    return format(value, ".17g")


def cpp_provenance(value: dict[str, str]) -> str:
    return "{" + ", ".join(
        cpp_string(value[key])
        for key in ("source_name", "source_uri", "record_id", "retrieved_utc", "sha256")
    ) + "}"


def generate_cpp(dataset: dict[str, object], snapshot_sha256: str) -> bytes:
    lines = [
        "// Generated by tools/astronomy/generate_catalog_dataset.py.",
        "// Do not edit; regenerate from the archived raw responses.",
        f"// Dataset: {dataset['dataset_id']} {dataset['dataset_version']}",
        f"// Roster SHA-256: {dataset['roster_sha256']}",
        f"// Snapshot manifest SHA-256: {snapshot_sha256}",
        "",
        "CatalogDefinition generated_solar_system_60_v1_definition() {",
        "    const CoordinateContract contract{",
        '        "J2000.0 (JD 2451545.0 TDB)",',
        "        2451545.0,",
        "        EphemerisTimeScale::tdb,",
        "        EphemerisReferenceFrame::icrf,",
        "        EphemerisOrigin::solar_system_barycenter,",
        "        StateVectorKind::geometric,",
        "        PositionUnit::metre,",
        "        VelocityUnit::metre_per_second,",
        "    };",
        "    CatalogDefinition definition;",
        f"    definition.catalog_id = {cpp_string(str(dataset['dataset_id']))};",
        "    definition.coordinate_contract = contract;",
        "    definition.bodies = {",
    ]
    for body in dataset["bodies"]:
        identifiers = ", ".join(
            "{" + cpp_string(item["authority"]) + ", " + cpp_string(item["value"]) + "}"
            for item in body["identifiers"]
        )
        parent = (
            "std::nullopt"
            if body["parent"] is None
            else "std::optional<std::string>{" + cpp_string(body["parent"]) + "}"
        )
        pos = ", ".join(cpp_number(value) for value in body["position_m"])
        vel = ", ".join(cpp_number(value) for value in body["velocity_m_s"])
        prov = body["provenance"]
        lines.extend(
            [
                "        {",
                f"            {body['order']}, {cpp_string(body['stable_id'])}, {cpp_string(body['name'])},",
                f"            {CLASS_CPP[body['class']]}, {parent},",
                f"            {{{identifiers}}}, contract,",
                f"            {cpp_number(body['gravitational_parameter_m3_s2'])},",
                f"            {cpp_number(body['mean_radius_m'])},",
                f"            {{{pos}}}, {{{vel}}},",
                "            {",
                f"                {cpp_provenance(prov['state_vector'])},",
                f"                {cpp_provenance(prov['gravitational_parameter'])},",
                f"                {cpp_provenance(prov['mean_radius'])},",
                f"                {cpp_string(prov['model_limitations'])},",
                "            },",
                "        },",
            ]
        )
    lines.extend(["    };", "    return definition;", "}", ""])
    return "\n".join(lines).encode("utf-8")


def generate(
    roster_path: pathlib.Path,
    snapshot_path: pathlib.Path,
    normalized_output: pathlib.Path,
    cpp_output: pathlib.Path,
) -> None:
    roster_bytes = roster_path.read_bytes()
    snapshot_bytes = snapshot_path.read_bytes()
    roster = json.loads(roster_bytes)
    snapshot = json.loads(snapshot_bytes)
    if snapshot["roster_sha256"] != sha256_bytes(roster_bytes):
        raise RuntimeError("snapshot manifest does not match the frozen roster")
    if roster["dataset_id"] != snapshot["dataset_id"] or roster["dataset_version"] != snapshot["dataset_version"]:
        raise RuntimeError("snapshot dataset identity does not match the frozen roster")

    output_root = snapshot_path.parent
    raw_root = output_root / "raw"
    source_bytes = {
        key: read_verified(raw_root, record) for key, record in snapshot["sources"].items()
    }
    gm_kernel = parse_gm_kernel(source_bytes["gm_kernel"])
    planet_rows = table_rows(source_bytes["planetary_physical_parameters"])
    satellite_rows = table_rows(source_bytes["satellite_physical_parameters"])

    bodies_output: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for body in roster["bodies"]:
        stable_id = body["stable_id"]
        if stable_id in seen_ids:
            raise RuntimeError(f"duplicate stable ID {stable_id}")
        seen_ids.add(stable_id)
        state_record = snapshot["state_records"][stable_id]
        state_values, solution = parse_state(body, read_verified(raw_root, state_record))

        sbdb = None
        sbdb_record = None
        if "sbdb_query" in body:
            sbdb_record = snapshot["sbdb_records"][stable_id]
            sbdb = parse_sbdb(body, read_verified(raw_root, sbdb_record))

        if body["gm_source"] == "gm_kernel":
            kernel_id = str(body.get("gm_kernel_id", body["naif_id"]))
            if kernel_id not in gm_kernel:
                raise RuntimeError(f"{stable_id}: GM kernel has no BODY{kernel_id}_GM")
            gm_km3_s2 = gm_kernel[kernel_id]
            gm_record = snapshot["sources"]["gm_kernel"]
            gm_provenance = provenance(gm_record, "JPL Horizons GM kernel", f"BODY{kernel_id}_GM")
        elif body["gm_source"] == "sbdb":
            assert sbdb is not None and sbdb_record is not None
            gm_parameter = sbdb["parameters"].get("GM")
            if gm_parameter is None or gm_parameter.get("units") != "km^3/s^2":
                raise RuntimeError(f"{stable_id}: SBDB lacks GM in km^3/s^2")
            gm_km3_s2 = first_number(str(gm_parameter.get("value", "")))
            gm_provenance = provenance(
                sbdb_record,
                "NASA/JPL Small-Body Database API",
                f"SBDB:{body['sbdb_query']}:orbit-{body['expected_sbdb_orbit']}:GM",
            )
        else:
            raise RuntimeError(f"{stable_id}: unsupported GM source {body['gm_source']}")
        if gm_km3_s2 <= 0.0:
            raise RuntimeError(f"{stable_id}: sourced GM is not positive")

        radius_source = body["radius_source"]
        if radius_source == "horizons":
            text = read_verified(raw_root, state_record).decode("utf-8")
            match = re.search(r"Vol\. mean radius, km\s*=\s*([-+0-9.Ee]+)", text)
            if match is None:
                raise RuntimeError(f"{stable_id}: Horizons response lacks volume mean radius")
            radius_km = float(match.group(1))
            radius_provenance = provenance(
                state_record, "NASA/JPL Horizons API", f"{solution}:volume-mean-radius"
            )
        elif radius_source == "planet_table":
            radius_km = table_mean_radius_km(planet_rows, body["name"])
            radius_record = snapshot["sources"]["planetary_physical_parameters"]
            radius_provenance = provenance(
                radius_record, "JPL Planetary Physical Parameters", f"{body['name']}:mean-radius"
            )
        elif radius_source == "satellite_table":
            radius_km = table_mean_radius_km(satellite_rows, body["name"])
            radius_record = snapshot["sources"]["satellite_physical_parameters"]
            radius_provenance = provenance(
                radius_record, "JPL Planetary Satellite Physical Parameters", f"{body['name']}:mean-radius"
            )
        elif radius_source == "sbdb":
            assert sbdb is not None and sbdb_record is not None
            diameter = sbdb["parameters"].get("diameter")
            if diameter is None or diameter.get("units") != "km":
                raise RuntimeError(f"{stable_id}: SBDB lacks diameter in km")
            radius_km = first_number(str(diameter.get("value", ""))) / 2.0
            radius_provenance = provenance(
                sbdb_record,
                "NASA/JPL Small-Body Database API",
                f"SBDB:{body['sbdb_query']}:diameter-to-characteristic-radius",
            )
        else:
            raise RuntimeError(f"{stable_id}: unsupported radius source {radius_source}")
        if radius_km <= 0.0:
            raise RuntimeError(f"{stable_id}: sourced radius is not positive")

        identifiers = [
            {"authority": "Horizons", "value": str(body["horizons_command"])},
            {"authority": "NAIF/SPK", "value": str(body["naif_id"])},
        ]
        if "sbdb_query" in body:
            identifiers.append({"authority": "SBDB", "value": str(body["sbdb_query"])})

        bodies_output.append(
            {
                "order": body["order"],
                "stable_id": stable_id,
                "name": body["name"],
                "class": body["class"],
                "parent": body["parent"],
                "identifiers": identifiers,
                "ephemeris_solution": solution,
                "gravitational_parameter_m3_s2": gm_km3_s2 * 1.0e9,
                "mean_radius_m": radius_km * 1.0e3,
                "position_m": [value * 1.0e3 for value in state_values[:3]],
                "velocity_m_s": [value * 1.0e3 for value in state_values[3:]],
                "provenance": {
                    "state_vector": provenance(
                        state_record,
                        "NASA/JPL Horizons API",
                        f"{body['horizons_command']}:{solution}:JD2451545.0",
                    ),
                    "gravitational_parameter": gm_provenance,
                    "mean_radius": radius_provenance,
                    "model_limitations": limitations(body),
                },
            }
        )

    if len(bodies_output) != 60 or [body["order"] for body in bodies_output] != list(range(60)):
        raise RuntimeError("normalized catalog is not the exact ordered 60-body roster")
    dataset = {
        "schema_version": 1,
        "dataset_id": roster["dataset_id"],
        "dataset_version": roster["dataset_version"],
        "roster_sha256": sha256_bytes(roster_bytes),
        "snapshot_manifest_sha256": sha256_bytes(snapshot_bytes),
        "coordinate_contract": roster["coordinate_contract"],
        "bodies": bodies_output,
        "deferred_catalog_expansion": roster["deferred_catalog_expansion"],
    }
    normalized_bytes = canonical_json_bytes(dataset)
    cpp_bytes = generate_cpp(dataset, sha256_bytes(snapshot_bytes))
    normalized_output.parent.mkdir(parents=True, exist_ok=True)
    cpp_output.parent.mkdir(parents=True, exist_ok=True)
    normalized_output.write_bytes(normalized_bytes)
    cpp_output.write_bytes(cpp_bytes)
    print(f"generated {len(bodies_output)} admitted body definitions")
    print(f"normalized SHA-256: {sha256_bytes(normalized_bytes)}")
    print(f"generated C++ SHA-256: {sha256_bytes(cpp_bytes)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=pathlib.Path, required=True)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--normalized-output", type=pathlib.Path, required=True)
    parser.add_argument("--cpp-output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        generate(
            args.roster.resolve(),
            args.snapshot.resolve(),
            args.normalized_output.resolve(),
            args.cpp_output.resolve(),
        )
        return 0
    except Exception as error:
        print(f"catalog generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
