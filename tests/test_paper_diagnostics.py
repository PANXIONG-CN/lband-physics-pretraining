from pathlib import Path
"""Small regression tests; synthetic cases only, no input files are changed."""
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
import unittest
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from paper_diagnostics_core import (
    FEATURES, OBS, PHYS, VWC, assemble_cohort, association,
    normalize_keys, pair_blocks, parse_bool, residuals, rho,
)


def example():
    base = pd.DataFrame({"field_id": ["1", "1", "2"],
                         "acquisition_date": ["2012-06-20", "2012-06-21", "2012-06-20"],
                         "spm_valid": [True, True, False]})
    for col in FEATURES:
        base[col] = [1., 2., 3.]
    for col, values in zip(OBS + PHYS, [[-10, -9, -8], [-8, -7, -6],
                                        [-15, -14, -13], [-10, -9, -8]]):
        base[col] = np.asarray(values, dtype=float)
    veg = base.copy()
    veg[VWC] = [0.0, 1.0, 2.0]
    veg["vegetation_sample_date"] = ["2012-06-20", "2012-06-24", "2012-06-20"]
    veg["vegetation_time_offset_days"] = [0, 3, 0]
    return base, veg


class DiagnosticTests(unittest.TestCase):
    def test_keyed_join_is_order_independent(self):
        base, veg = example()
        result = assemble_cohort(base, veg.iloc[::-1])
        self.assertEqual(result.in_original_model_cohort.sum(), 2)
        self.assertEqual(result.in_common_cohort.sum(), 2)
        self.assertEqual(result.in_high_quality_cohort.sum(), 1)
        self.assertEqual(result.iloc[0][VWC], 0.0)  # zero is a valid observation

    def test_duplicate_keys_rejected(self):
        base, _ = example()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_keys(pd.concat([base, base.iloc[:1]]), "test")

    def test_conflicting_sources_rejected(self):
        base, veg = example()
        veg.loc[0, OBS[0]] += 0.2
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            assemble_cohort(base, veg)

    def test_inconsistent_date_gap_rejected(self):
        base, veg = example()
        veg.loc[0, "vegetation_time_offset_days"] = 2
        with self.assertRaisesRegex(ValueError, "disagree"):
            assemble_cohort(base, veg)

    def test_missing_vwc_is_not_zero_filled(self):
        base, veg = example()
        veg.loc[0, VWC] = np.nan
        result = assemble_cohort(base, veg)
        self.assertEqual(result.in_common_cohort.sum(), 1)
        self.assertTrue(pd.isna(result.iloc[0][VWC]))

    def test_unmatched_keys_retained_in_audit(self):
        base, veg = example()
        result = assemble_cohort(base, veg.iloc[1:])
        self.assertEqual(len(result), 3)
        self.assertIn("no_vegetation_key_match", result.iloc[0].common_cohort_exclusion_reason)

    def test_boolean_false_is_not_truthy_string(self):
        self.assertEqual(parse_bool(pd.Series(["False", "True", None])).tolist(), [False, True, False])
        with self.assertRaises(ValueError):
            parse_bool(pd.Series(["maybe"]))

    def test_residual_inverse_and_spectrum_cancellation(self):
        base, _ = example()
        out = residuals(base)
        np.testing.assert_allclose(out.common_residual_db - out.differential_residual_db / 2,
                                   out.residual_hh_db)
        np.testing.assert_allclose(out.common_residual_db + out.differential_residual_db / 2,
                                   out.residual_vv_db)
        shifted = base.copy()
        shifted[PHYS] += 6
        np.testing.assert_allclose(residuals(shifted).differential_residual_db, out.differential_residual_db)
        np.testing.assert_allclose(residuals(shifted).common_residual_db, out.common_residual_db - 6)

    def test_constant_within_field_covariate_unavailable(self):
        frame = pd.DataFrame({"field_id": ["a", "a", "b", "b"],
                              "x": [1., 1., 2., 2.], "y": [1., 2., 4., 5.]})
        self.assertEqual(pair_blocks(frame, "x", "y", "within_field"), [])

    def test_within_centering_and_seed_reproducibility(self):
        frame = pd.DataFrame({"field_id": np.repeat(np.arange(8).astype(str), 3),
                              "x": np.tile([1., 2., 3.], 8), "y": np.tile([2., 4., 6.], 8)})
        blocks = pair_blocks(frame, "x", "y", "within_field")
        np.testing.assert_allclose(blocks[0].mean(axis=0), 0)
        self.assertAlmostEqual(rho(np.concatenate(blocks)), 1)
        a = association(frame, "x", "y", "within_field", 100, 7)
        b = association(frame, "x", "y", "within_field", 100, 7)
        self.assertEqual(a, b)
        self.assertEqual(a["n_fields_used"], 8)
        self.assertEqual(a["bootstrap_valid"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
