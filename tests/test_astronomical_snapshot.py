#!/usr/bin/env python3
"""Offline custody and deterministic-reconstruction tests for the catalog snapshot."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "astronomy" / "solar_system_60_v1"
ROSTER = DATA / "roster.json"
SNAPSHOT = DATA / "snapshot_manifest.json"
NORMALIZED = DATA / "normalized" / "catalog.json"
CPP = ROOT / "src" / "astronomy" / "generated" / "solar_system_60_v1.generated.inc"
GENERATOR = ROOT / "tools" / "astronomy" / "generate_catalog_dataset.py"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AstronomicalSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.roster = json.loads(ROSTER.read_text(encoding="utf-8"))
        cls.snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        cls.normalized = json.loads(NORMALIZED.read_text(encoding="utf-8"))

    def test_roster_identity_and_deferred_queue(self) -> None:
        self.assertEqual(60, len(self.roster["bodies"]))
        self.assertEqual(list(range(60)), [body["order"] for body in self.roster["bodies"]])
        self.assertEqual(60, len({body["stable_id"] for body in self.roster["bodies"]}))
        deferred = {body["name"] for body in self.roster["deferred_catalog_expansion"]}
        self.assertTrue({"Eris", "Haumea", "Quaoar", "Orcus", "Gonggong"} <= deferred)

    def test_raw_response_checksums_and_request_contract(self) -> None:
        raw = DATA / "raw"
        records = list(self.snapshot["sources"].values())
        records += list(self.snapshot["state_records"].values())
        records += list(self.snapshot["sbdb_records"].values())
        self.assertEqual(76, len(records))
        for record in records:
            path = raw / record["path"]
            self.assertTrue(path.is_file(), record["path"])
            self.assertEqual(record["sha256"], sha256(path), record["path"])
            self.assertRegex(record["retrieved_utc"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        for record in self.snapshot["state_records"].values():
            parameters = record["request_parameters"]
            self.assertEqual("'@0'", parameters["CENTER"])
            self.assertEqual("'2451545.0'", parameters["TLIST"])
            self.assertEqual("JD", parameters["TLIST_TYPE"])
            self.assertEqual("TDB", parameters["TIME_TYPE"])
            self.assertEqual("ICRF", parameters["REF_SYSTEM"])
            self.assertEqual("FRAME", parameters["REF_PLANE"])
            self.assertEqual("NONE", parameters["VEC_CORR"])
            self.assertEqual("KM-S", parameters["OUT_UNITS"])

    def test_normalized_identity_units_and_provenance(self) -> None:
        self.assertEqual(self.snapshot["roster_sha256"], sha256(ROSTER))
        self.assertEqual(sha256(SNAPSHOT), self.normalized["snapshot_manifest_sha256"])
        self.assertEqual(60, len(self.normalized["bodies"]))
        for body in self.normalized["bodies"]:
            self.assertTrue(body["ephemeris_solution"])
            self.assertGreater(body["gravitational_parameter_m3_s2"], 0.0)
            self.assertGreater(body["mean_radius_m"], 0.0)
            self.assertEqual(3, len(body["position_m"]))
            self.assertEqual(3, len(body["velocity_m_s"]))
            for key in ("state_vector", "gravitational_parameter", "mean_radius"):
                source = body["provenance"][key]
                self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(source["record_id"])
                self.assertTrue(source["source_uri"].startswith("https://"))

        sun = self.normalized["bodies"][0]
        raw_state = (DATA / "raw" / self.snapshot["state_records"]["sol:sun"]["path"]).read_text(
            encoding="utf-8"
        )
        state_line = raw_state.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip()
        raw_x_km = float(state_line.split(",")[2])
        raw_vx_km_s = float(state_line.split(",")[5])
        self.assertEqual(raw_x_km * 1000.0, sun["position_m"][0])
        self.assertEqual(raw_vx_km_s * 1000.0, sun["velocity_m_s"][0])
        self.assertEqual(695700000.0, sun["mean_radius_m"])

    def test_offline_generation_is_deterministic_and_reconstructs_committed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as first_dir, tempfile.TemporaryDirectory(
            dir=ROOT
        ) as second_dir:
            outputs = []
            for directory in (first_dir, second_dir):
                normalized = pathlib.Path(directory) / "catalog.json"
                cpp = pathlib.Path(directory) / "catalog.inc"
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(GENERATOR),
                        "--roster",
                        str(ROSTER),
                        "--snapshot",
                        str(SNAPSHOT),
                        "--normalized-output",
                        str(normalized),
                        "--cpp-output",
                        str(cpp),
                    ],
                    check=True,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                outputs.append((normalized.read_bytes(), cpp.read_bytes()))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(NORMALIZED.read_bytes(), outputs[0][0])
            self.assertEqual(CPP.read_bytes(), outputs[0][1])


if __name__ == "__main__":
    unittest.main()
