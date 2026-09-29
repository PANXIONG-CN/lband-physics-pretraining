"""Fast tests for the fully decoupled physics-head experiment."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("evaluate_decoupled_physics_heads.py")
SPEC = importlib.util.spec_from_file_location("decoupled", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DecoupledPhysicsHeadTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        channels = np.array([[-12.0, -9.0], [-18.5, -14.2], [-7.1, -7.0]])
        components = MODULE.to_components(channels)
        recovered = MODULE.from_components(components[:, 0], components[:, 1])
        np.testing.assert_allclose(recovered, channels, atol=1e-12)

    def test_error_identity(self) -> None:
        observed = np.array([[-12.0, -9.0], [-18.5, -14.2]])
        predicted = np.array([[-11.0, -10.0], [-17.1, -13.4]])
        MODULE.verify_error_identity(observed, predicted)
        channel_error = predicted - observed
        component_error = MODULE.to_components(predicted) - MODULE.to_components(observed)
        np.testing.assert_allclose(
            np.mean(channel_error**2, axis=1),
            component_error[:, 0] ** 2 + component_error[:, 1] ** 2 / 4,
        )

    def test_zero_constraint_returns_observation(self) -> None:
        observed = np.array([1.0, 2.0, 3.0])
        physics = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            MODULE.constrained_differential_target(observed, physics, 0.0),
            observed,
        )

    def test_full_constraint_returns_calibrated_physics(self) -> None:
        observed = np.array([1.0, 2.0, 3.0])
        physics = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            MODULE.constrained_differential_target(observed, physics, 1.0),
            physics,
        )

    def test_common_is_independent_of_differential(self) -> None:
        common = np.array([-10.0, -12.0])
        first = MODULE.to_components(MODULE.from_components(common, [1.0, 2.0]))[:, 0]
        second = MODULE.to_components(
            MODULE.from_components(common, [4.1234567890123, -3.9876543210987])
        )[:, 0]
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=1e-12)

    def test_invalid_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.constrained_differential_target([1.0], [2.0], 1.1)

    def test_prediction_shrinkage_endpoints(self) -> None:
        predicted = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(
            MODULE.shrink_prediction_to_mean(predicted, 2.0, 0.0),
            np.full(3, 2.0),
        )
        np.testing.assert_allclose(
            MODULE.shrink_prediction_to_mean(predicted, 2.0, 1.0),
            predicted,
        )

    def test_joint_channel_metric(self) -> None:
        frame = MODULE.pd.DataFrame(
            {
                "sigma0_hh_db": [-10.0, -12.0],
                "sigma0_vv_db": [-8.0, -9.0],
            }
        )
        for method in MODULE.METHODS:
            frame[f"hh_{method}"] = frame["sigma0_hh_db"] + 1.0
            frame[f"vv_{method}"] = frame["sigma0_vv_db"] - 2.0
        metrics = MODULE.joint_channel_metrics(frame).set_index("method")
        self.assertAlmostEqual(
            metrics.loc["training_mean", "mean_channel_rmse_db"], 1.5
        )
        self.assertAlmostEqual(
            metrics.loc["training_mean", "pooled_channel_rmse_db"],
            np.sqrt(2.5),
        )

    def test_joint_bootstrap_identical_predictions(self) -> None:
        observed = np.array([[-10.0, -8.0], [-12.0, -9.0]])
        predicted = observed + np.array([[1.0, -2.0], [1.0, -2.0]])
        result = MODULE.grouped_bootstrap_joint_delta(
            observed,
            predicted,
            predicted,
            np.array(["A", "B"]),
            iterations=50,
            seed=42,
        )
        self.assertEqual(result["mean_channel_rmse_delta_db"], 0.0)
        self.assertEqual(result["ci_2_5_percent_db"], 0.0)
        self.assertEqual(result["ci_97_5_percent_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
