"""Fast tests for the I2EM teacher request/result contract."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "research_pilots"
    / "scattering"
    / "surfaces"
    / "teacher_contract.py"
)
SPEC = importlib.util.spec_from_file_location("teacher_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def request_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "request_id": ["anchor"],
            "scenario": ["baseline"],
            "correlation_model": ["exponential"],
            "frequency_ghz": [1.26],
            "incidence_angle_deg": [40.0],
            "rms_height_m": [0.01],
            "correlation_length_m": [0.10],
            "dielectric_real": [15.0],
            "dielectric_loss_positive": [0.5],
        }
    )


class TeacherContractTests(unittest.TestCase):
    def test_valid_anchor(self) -> None:
        result = MODULE.validate_teacher_requests(request_frame())
        self.assertTrue(bool(result.loc[0, "i2em_request_valid"]))

    def test_reference_slope_limit(self) -> None:
        frame = request_frame()
        frame.loc[0, "rms_height_m"] = 0.03
        with self.assertRaises(ValueError):
            MODULE.validate_teacher_requests(frame)

    def test_duplicate_identifier(self) -> None:
        frame = pd.concat([request_frame(), request_frame()], ignore_index=True)
        with self.assertRaises(ValueError):
            MODULE.validate_teacher_requests(frame)

    def test_result_ids_must_match(self) -> None:
        results = pd.DataFrame(
            {"request_id": ["wrong"], "i2em_hh_db": [-12.0], "i2em_vv_db": [-10.0]}
        )
        with self.assertRaises(ValueError):
            MODULE.validate_teacher_results(request_frame(), results)

    def test_complete_results_merge(self) -> None:
        results = pd.DataFrame(
            {"request_id": ["anchor"], "i2em_hh_db": [-12.0], "i2em_vv_db": [-10.0]}
        )
        merged = MODULE.validate_teacher_results(request_frame(), results)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(float(merged.loc[0, "i2em_hh_db"]), -12.0)


if __name__ == "__main__":
    unittest.main()
