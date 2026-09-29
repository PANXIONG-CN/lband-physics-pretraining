"""Unit tests for evaluate_synthetic_pretraining_domain.py."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_synthetic_pretraining_domain.py")
SPEC = importlib.util.spec_from_file_location("synthetic_domain", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SyntheticDomainTests(unittest.TestCase):
    def test_sample_size_parser(self):
        self.assertEqual(MODULE.parse_sample_sizes("3000,500,1000,500"), [500, 1000, 3000])
        with self.assertRaises(ValueError):
            MODULE.parse_sample_sizes("50")

    def test_zero_error_metrics(self):
        values = np.array([1.0, 2.0, 3.0])
        metrics = MODULE.error_metrics(values, values)
        self.assertEqual(metrics["rmse_db"], 0.0)
        self.assertAlmostEqual(metrics["r_squared"], 1.0)

    def test_special_domains_obey_definitions(self):
        frequency_hz = 1.26e9
        boundary, _, boundary_meta = MODULE.sample_special_domain(
            100, frequency_hz, 40.0, 7, "near_validity_boundary"
        )
        validity = MODULE.spm_validity(
            boundary[:, 2] / 100.0, boundary[:, 3] / 100.0, frequency_hz
        )
        margin = np.maximum(
            validity["k_rms_height"] / 0.3,
            validity["rms_slope_proxy"] / 0.21,
        )
        self.assertTrue(np.all(validity["valid"]))
        self.assertTrue(np.all(margin >= 0.8))
        self.assertEqual(boundary_meta["sample_count"], 100)

        extension, _, _ = MODULE.sample_special_domain(
            100, frequency_hz, 40.0, 8, "range_extension"
        )
        outside = np.zeros(len(extension), dtype=bool)
        for index, name in enumerate(MODULE.FEATURE_NAMES):
            low, high = MODULE.TRAINING_BOX[name]
            outside |= (extension[:, index] < low) | (extension[:, index] > high)
        self.assertTrue(np.all(outside))

    def test_common_and_difference_rows(self):
        reference = np.array([[1.0, 3.0], [2.0, 5.0]])
        prediction = reference + np.array([[1.0, -1.0], [1.0, -1.0]])
        rows = MODULE.metric_rows(reference, prediction, "test", 100, 1)
        by_component = {row["component"]: row for row in rows}
        self.assertAlmostEqual(by_component["common_HH_VV"]["rmse_db"], 0.0)
        self.assertAlmostEqual(by_component["difference_VV_minus_HH"]["rmse_db"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
