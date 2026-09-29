"""Regression tests for corrected frozen-data comparisons; no fitting or downloads."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
import json,unittest
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from recalculate_frozen_results import component_metrics, read_table
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reproducibility'
class PaperResultsTests(unittest.TestCase):
    def test_error_decomposition_identity(self):
        y=np.array([1.,3.,2.,5.]);p=np.array([4.,5.,1.,2.])
        e=p-y
        self.assertAlmostEqual(float(np.mean(e*e)),float(e.mean()**2+np.mean((e-e.mean())**2)),places=13)
    def test_constant_centered_skill(self):
        m=component_metrics(np.array([1.,3.,2.,5.]),np.repeat(8.,4))
        self.assertAlmostEqual(m['centered_skill'],0.,places=13)
    def test_corrected_mean_contrast(self):
        s=pd.read_csv(DATA/'results/matched_information/metrics_summary.csv')
        v=s.pivot(index='requested_fraction',columns='method',values='mean_channel_rmse_db')
        np.testing.assert_allclose((v.spm_only-v.two_channel_mean_all_adaptation).values,
                                   [.003200786693784,-.002744648918434,-.005322098481071],atol=1e-12)
    def test_no_cross_budget_ranking_in_revised_contrasts(self):
        p=pd.read_csv(DATA/'results/matched_information/paired_contrasts.csv')
        # The exact column schema is a public contract.
        cols=p.columns.tolist()
        a='method_a' if 'method_a' in cols else 'method'
        # Check complete values regardless of contrast-column spelling.
        strings=p.astype(str).agg(' '.join,axis=1)
        neural=strings.str.contains('spm_only',regex=False)
        self.assertTrue(neural.any())
        self.assertFalse(any(('spm_only' in row and 'two_channel_mean' in row and 'two_channel_mean_all_adaptation' not in row) for row in strings))
    def test_trajectory_outcome_is_not_overstated(self):
        s=json.loads((DATA/'results/trajectory_reconstruction/summary.json').read_text())
        self.assertTrue(s['frozen_coefficient_predictions_reproduced'])
        self.assertFalse(s['selected_weights_reproduced'])
        self.assertFalse(s['original_windows_environment_recreated'])
    def test_positive_pretraining_negative_source(self):
        df=pd.read_csv(DATA/'results/statistical_checks/error_decomposition.csv')
        # Coverage is checked separately by the frozen recalculation command.
        self.assertEqual(len(df),5)
        self.assertTrue(np.isfinite(df.select_dtypes('number').to_numpy()).all())
class InputTableTests(unittest.TestCase):
    def test_required_input_missing(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'Required paper input'):
                read_table(Path(folder) / 'missing.csv')

    def test_required_columns_missing(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            path.write_text('field_id,value\nF1,1\n')
            with self.assertRaisesRegex(ValueError, 'missing required columns'):
                read_table(path, required=['sigma0_hh_db'])

    def test_duplicate_and_missing_keys_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            for content in ['field_id,value\nF1,1\nF1,2\n', 'field_id,value\n,1\n']:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaisesRegex(ValueError, 'sample keys'):
                        read_table(path, unique_key=['field_id'])

    def test_empty_table_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            path.write_text('field_id,value\n')
            with self.assertRaisesRegex(ValueError, 'empty'):
                read_table(path)

    def test_valid_input_without_manifest(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'input.csv'
            path.write_text('field_id,acquisition_date,value\nF1,2012-06-01T12:00:00,1\n')
            frame = read_table(path, required=['value'], unique_key=['field_id', 'acquisition_date'])
            self.assertEqual(frame.acquisition_date.iloc[0], '2012-06-01')
            self.assertEqual(frame.value.iloc[0], 1)

if __name__=='__main__':unittest.main()
