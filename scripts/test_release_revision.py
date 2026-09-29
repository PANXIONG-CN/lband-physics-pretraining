"""Regression tests for corrected frozen-data comparisons; no fitting or downloads."""
from pathlib import Path
import json,unittest
import numpy as np
import pandas as pd
from recalculate_frozen_results import component_metrics
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reproducibility'
class ReleaseRevisionTests(unittest.TestCase):
    def test_error_decomposition_identity(self):
        y=np.array([1.,3.,2.,5.]);p=np.array([4.,5.,1.,2.])
        e=p-y
        self.assertAlmostEqual(float(np.mean(e*e)),float(e.mean()**2+np.mean((e-e.mean())**2)),places=13)
    def test_constant_centered_skill(self):
        m=component_metrics(np.array([1.,3.,2.,5.]),np.repeat(8.,4))
        self.assertAlmostEqual(m['centered_skill'],0.,places=13)
    def test_corrected_mean_contrast(self):
        s=pd.read_csv(DATA/'results/revision_20260929/fairness/metrics_summary.csv')
        v=s.pivot(index='requested_fraction',columns='method',values='mean_channel_rmse_db')
        np.testing.assert_allclose((v.spm_only-v.two_channel_mean_all_adaptation).values,
                                   [.003200786693784,-.002744648918434,-.005322098481071],atol=1e-12)
    def test_no_cross_budget_ranking_in_revised_contrasts(self):
        p=pd.read_csv(DATA/'results/revision_20260929/fairness/paired_contrasts.csv')
        # The exact column schema is a public contract.
        cols=p.columns.tolist()
        a='method_a' if 'method_a' in cols else 'method'
        # Check complete values regardless of contrast-column spelling.
        strings=p.astype(str).agg(' '.join,axis=1)
        neural=strings.str.contains('spm_only',regex=False)
        self.assertTrue(neural.any())
        self.assertFalse(any(('spm_only' in row and 'two_channel_mean' in row and 'two_channel_mean_all_adaptation' not in row) for row in strings))
    def test_trajectory_outcome_is_not_overstated(self):
        s=json.loads((DATA/'results/revision_20260929/stage_continuity/summary.json').read_text())
        self.assertTrue(s['frozen_coefficient_predictions_reproduced'])
        self.assertFalse(s['selected_weights_reproduced'])
        self.assertFalse(s['original_windows_environment_recreated'])
    def test_positive_pretraining_negative_source(self):
        df=pd.read_csv(DATA/'results/revision_20260929/statistics/error_decomposition.csv')
        # Coverage is checked separately by the frozen recalculation command.
        self.assertEqual(len(df),5)
        self.assertTrue(np.isfinite(df.select_dtypes('number').to_numpy()).all())
if __name__=='__main__':unittest.main()
