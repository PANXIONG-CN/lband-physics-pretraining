"""Fast unit tests for the selective physics-transfer experiment."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("evaluate_selective_physics_transfer.py")
SPEC = importlib.util.spec_from_file_location("selective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SelectivePhysicsTransferTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "spm_k_rms_height": [0.05, 0.15, 0.29],
                "spm_rms_slope_proxy": [0.02, 0.08, 0.20],
                "vegetation_water_content_map_kg_m2": [0.1, 1.0, 3.0],
                "vegetation_water_content_in_situ_kg_m2": [0.2, 1.1, 2.9],
                "vwc_map_source": ["direct_satellite", "interpolated", "extrapolated"],
            }
        )

    def test_gate_is_bounded_and_decreases_with_challenge(self) -> None:
        gate = MODULE.physics_confidence(self.frame(), alpha=0.8)
        self.assertTrue(np.all((gate.physics_confidence >= 0) & (gate.physics_confidence <= 1)))
        self.assertGreater(gate.physics_confidence.iloc[0], gate.physics_confidence.iloc[1])
        self.assertGreater(gate.physics_confidence.iloc[1], gate.physics_confidence.iloc[2])

    def test_alpha_zero_removes_only_vegetation_attenuation(self) -> None:
        gate = MODULE.physics_confidence(self.frame(), alpha=0.0)
        np.testing.assert_allclose(gate.gate_vegetation, 1.0)
        np.testing.assert_allclose(
            gate.physics_confidence,
            gate.gate_roughness * gate.gate_provenance,
        )

    def test_zero_weight_returns_observations(self) -> None:
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        physics = observed + 10.0
        result = MODULE.selective_blended_target(observed, physics, [0.2, 0.8], 0.0)
        np.testing.assert_allclose(result, observed)

    def test_unit_gate_matches_fixed_blend(self) -> None:
        observed = np.array([[1.0, 2.0], [3.0, 4.0]])
        physics = observed + 2.0
        selective = MODULE.selective_blended_target(observed, physics, [1.0, 1.0], 0.2)
        fixed = MODULE.blended_physics_target(observed, physics, 0.2)
        np.testing.assert_allclose(selective, fixed)

    def test_candidate_grid_deduplicates_zero_weight(self) -> None:
        pairs = MODULE.candidate_pairs([0.0, 0.1, 0.2], [0.0, 0.8, 1.6])
        self.assertEqual(pairs.count((0.0, 0.0)), 1)
        self.assertEqual(len(pairs), 7)

    def test_metric_schema_matches_report(self) -> None:
        frame = pd.DataFrame(
            {
                "sigma0_hh_db": [-10.0, -9.0, -8.0],
                "sigma0_vv_db": [-8.0, -7.0, -6.0],
            }
        )
        for method in MODULE.METHODS:
            frame[f"hh_{method}"] = frame["sigma0_hh_db"] + 0.1
            frame[f"vv_{method}"] = frame["sigma0_vv_db"] - 0.1
        metrics = MODULE.evaluate_metrics(frame)
        self.assertIn("r_squared_skill", metrics.columns)


if __name__ == "__main__":
    unittest.main()
