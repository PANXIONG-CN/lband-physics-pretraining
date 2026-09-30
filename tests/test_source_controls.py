"""Leakage, audit-record, and matching invariants for the ancillary controls."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


structure = load_script("review_source_structure")
matching = load_script("review_match_vegetation")


class SourceControlTests(unittest.TestCase):
    def test_nested_folds_keep_fields_separate_and_cover_each_row(self):
        frame = pd.DataFrame({"field_id": np.repeat(np.arange(10), 3).astype(str),
                              "acquisition_date": list(pd.date_range("2012-06-01", periods=3)) * 10})
        outer, heldout, inner = structure.grouped_assignments(frame)
        self.assertEqual(sorted(heldout.row_id), list(frame.index))
        self.assertTrue(heldout.groupby("field_id").fold.nunique().eq(1).all())
        for fold, (train, test) in enumerate(outer, 1):
            self.assertFalse(set(frame.loc[train, "field_id"]) & set(frame.loc[test, "field_id"]))
            valid = inner.loc[inner.fold.eq(fold)]
            self.assertEqual(sorted(valid.row_id), sorted(train))
            self.assertTrue(valid.groupby("field_id").inner_fold.nunique().eq(1).all())

    def test_ridge_audit_does_not_change_fit_and_records_every_candidate(self):
        rng = np.random.default_rng(21)
        x, y, groups = rng.normal(size=(24, 4)), rng.normal(size=(24, 2)), np.repeat(np.arange(8), 3)
        original, alpha = structure.fit_ridge(x, y, groups)
        records = []
        def capture(fold, rows, predictions, **candidate):
            records.append((fold, rows.copy(), predictions.copy(), candidate["candidate_alpha"]))
        audited, selected = structure.fit_ridge(x, y, groups, capture)
        self.assertEqual(selected, alpha)
        np.testing.assert_array_equal(audited.predict(x), original.predict(x))
        self.assertEqual(len(records), 20)
        for candidate in {r[3] for r in records}:
            rows = np.concatenate([r[1] for r in records if r[3] == candidate])
            self.assertEqual(sorted(rows), list(range(len(x))))

    def test_fixed_common_preserves_differential_and_uses_training_mean(self):
        p = np.array([[1., 4.], [3., -1.], [-2., 5.]])
        train = np.array([[10., 6.], [4., 8.]])
        fixed = structure.fixed_common(p, train)
        np.testing.assert_allclose(fixed[:, 1] - fixed[:, 0], p[:, 1] - p[:, 0])
        np.testing.assert_allclose(fixed.mean(axis=1), train.mean())

    def test_matching_tie_boundary_missing_field_and_invalid_vwc(self):
        source = pd.DataFrame({"field_id": ["1", "2", "3"], "acquisition_date": ["2012-06-03"] * 3})
        vegetation = pd.DataFrame({"field_id": ["1", "1", "2", "2"],
            "sample_date": ["2012-06-01", "2012-06-05", "2012-06-03", "2012-06-06"],
            matching.VWC: [1., 2., -1., 3.]})
        result = matching.match_source(source, vegetation, 2)
        self.assertEqual(result.field_id.tolist(), ["1"])
        self.assertEqual(result[matching.VWC].iloc[0], 1.)
        self.assertEqual(result.vegetation_gap_days.iloc[0], 2)
        self.assertEqual(result.vegetation_sample_date.iloc[0], pd.Timestamp("2012-06-01"))

    def test_matching_rejects_duplicate_keys_and_empty_match(self):
        source = pd.DataFrame({"field_id": ["1"], "acquisition_date": ["2012-06-03"]})
        vegetation = pd.DataFrame({"field_id": ["1"], "sample_date": ["2012-06-01"], matching.VWC: [1.]})
        with self.assertRaises(ValueError):
            matching.match_source(pd.concat([source, source]), vegetation, 2)
        with self.assertRaises(ValueError):
            matching.match_source(source, pd.concat([vegetation, vegetation]), 2)
        with self.assertRaises(RuntimeError):
            matching.match_source(source, vegetation, 1)


if __name__ == "__main__":
    unittest.main()
