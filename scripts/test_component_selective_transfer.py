"""Fast tests for component-selective transfer."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("evaluate_component_selective_transfer.py")
SPEC = importlib.util.spec_from_file_location("component_selective", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ComponentSelectiveTransferTests(unittest.TestCase):
    def test_component_round_trip_is_exact(self) -> None:
        channels = np.array([[-12.0, -9.0], [-18.5, -14.2], [-7.1, -7.0]])
        recovered = MODULE.from_components(MODULE.to_components(channels))
        np.testing.assert_allclose(recovered, channels, atol=1e-12)

    def test_component_definition(self) -> None:
        values = MODULE.to_components(np.array([[-12.0, -8.0]]))
        np.testing.assert_allclose(values, [[-10.0, 4.0]])

    def test_zero_weight_returns_observed(self) -> None:
        observed = np.array([[-10.0, 2.0], [-12.0, 4.0]])
        physics = observed + 5.0
        for mode in ["both", "differential_only"]:
            target = MODULE.component_constraint_target(observed, physics, 0.0, mode)
            np.testing.assert_allclose(target, observed)

    def test_differential_only_preserves_common(self) -> None:
        observed = np.array([[-10.0, 2.0], [-12.0, 4.0]])
        physics = np.array([[-20.0, 8.0], [-18.0, 10.0]])
        target = MODULE.component_constraint_target(
            observed, physics, 0.25, "differential_only"
        )
        np.testing.assert_allclose(target[:, 0], observed[:, 0])
        np.testing.assert_allclose(target[:, 1], [3.5, 5.5])

    def test_both_mode_matches_existing_blend(self) -> None:
        observed = np.array([[-10.0, 2.0], [-12.0, 4.0]])
        physics = observed + 2.0
        target = MODULE.component_constraint_target(observed, physics, 0.2, "both")
        expected = MODULE.blended_physics_target(observed, physics, 0.2)
        np.testing.assert_allclose(target, expected)

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.to_components(np.ones((3, 1)))

    def test_component_pretraining_labels_are_explicit(self) -> None:
        metadata = {
            "pretrain_epochs": 10,
            "synthetic_training_rmse_db": {"HH": 0.2, "VV": 0.03},
        }
        result = MODULE.relabel_component_pretraining_metadata(metadata)
        self.assertEqual(
            result["synthetic_training_rmse_db"],
            {"common": 0.2, "differential": 0.03},
        )


if __name__ == "__main__":
    unittest.main()
