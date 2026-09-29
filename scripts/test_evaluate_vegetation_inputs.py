"""Focused tests for the matched vegetation-input ablation."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("evaluate_vegetation_inputs.py")
SPEC = importlib.util.spec_from_file_location("evaluate_vegetation_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_cohort() -> pd.DataFrame:
    rows = []
    for field in range(10):
        for day in range(3):
            vwc = 0.1 * field + 0.02 * day
            rows.append(
                {
                    "acquisition_date": f"2012-06-{day + 1:02d}",
                    "field_id": str(field),
                    "soil_moisture_m3_m3": 0.2 + 0.01 * day,
                    "soil_real_dielectric": 10.0 + day,
                    "pals_rms_height_cm": 0.5 + 0.1 * field,
                    "pals_correlation_length_cm": 8.0 + field,
                    MODULE.VWC_COLUMN: vwc,
                    "vegetation_time_offset_days": float(day),
                    "sigma0_hh_db": -15.0 + 3.0 * vwc,
                    "sigma0_vv_db": -14.0 + 4.0 * vwc,
                    "exponential_spm_hh_raw_db": -16.0,
                    "exponential_spm_vv_raw_db": -15.0,
                    "cohort": "synthetic",
                    "record_id": f"record_{field}_{day}",
                    "vegetation_match_abs_gap_days": float(day),
                }
            )
    return pd.DataFrame(rows)


class VegetationAblationTests(unittest.TestCase):
    def test_field_grouping_never_splits_a_field(self) -> None:
        assigned = MODULE.assign_outer_folds(synthetic_cohort(), n_splits=5)
        self.assertEqual(assigned["outer_fold"].nunique(), 5)
        self.assertTrue(
            (assigned.groupby("field_id")["outer_fold"].nunique() == 1).all()
        )

    def test_duplicate_sample_keys_are_rejected(self) -> None:
        data = synthetic_cohort().drop(
            columns=["cohort", "record_id", "vegetation_match_abs_gap_days"]
        )
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.csv"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate sample keys"):
                MODULE.prepare_cohort(path, "duplicate")

    def test_scaled_ridge_returns_two_finite_targets(self) -> None:
        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(30, 4))
        y_train = np.column_stack([2.0 * x_train[:, 0], -x_train[:, 1]])
        prediction, warned = MODULE.fit_predict_scaled(
            "ridge", x_train, y_train, rng.normal(size=(5, 4)), seed=42
        )
        self.assertEqual(prediction.shape, (5, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertFalse(warned)

    def test_metric_delta_sign_convention(self) -> None:
        observed = np.array([0.0, 1.0, 2.0])
        original = np.array([2.0, 3.0, 4.0])
        augmented = observed.copy()
        delta = MODULE.rmse(observed, augmented) - MODULE.rmse(
            observed, original
        )
        self.assertLess(delta, 0.0)


if __name__ == "__main__":
    unittest.main()
