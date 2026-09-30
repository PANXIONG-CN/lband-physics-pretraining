"""Regression checks for frozen comparisons and the fixed source Adam control."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]
import json,unittest
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from recalculate_frozen_results import (component_metrics, read_table,
    source_offset_predictions, unified_features, predict_saved_member,
    field_weights, field_response_sensitivity)
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
        self.assertEqual(len(df),8)
        # A constant prediction has undefined correlation, not a failed metric.
        values=df.drop(columns=['correlation']).select_dtypes('number')
        self.assertTrue(np.isfinite(values.to_numpy()).all())
        mean=df.loc[df.method.eq('source_mean')].iloc[0]
        self.assertTrue(pd.isna(mean.correlation))
        self.assertAlmostEqual(mean.variance_ratio,0.,places=12)
        self.assertGreater(df.loc[df.method.eq('pretraining_only_surrogate'),'correlation'].iloc[0],0)
        self.assertLess(df.loc[df.method.eq('source_finetuned_surrogate'),'correlation'].iloc[0],0)
class ResponseSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage=pd.read_csv(DATA/'results/stage_diagnostics/stage_predictions.csv')
        cls.field=cls.stage.field_id.astype(str).to_numpy()
        cls.dates=pd.to_datetime(cls.stage.acquisition_date).dt.strftime('%Y-%m-%d').to_numpy()
        cls.y=(cls.stage.sigma0_vv_db-cls.stage.sigma0_hh_db).to_numpy()
        cls.w,cls.inv,_=field_weights(cls.field,10000,27260910)

    def score(self,p):
        return field_response_sensitivity(self.y,p,self.field,self.dates,self.w,self.inv)

    def test_saved_within_field_and_date_sensitivity(self):
        saved=pd.read_csv(DATA/'results/statistical_checks/exploratory_within_between_field.csv').set_index('method')
        for method,column in [
            ('pretraining_only_surrogate','differential_pretraining_only_surrogate'),
            ('source_finetuned_surrogate','differential_source_finetuned_surrogate'),
            ('source_reset_adam_003','differential_source_reset_adam_003')]:
            got=self.score(self.stage[column].to_numpy())
            for key,value in got.items():
                self.assertAlmostEqual(value,float(saved.loc[method,key]),places=11)
        pre=saved.loc['pretraining_only_surrogate']
        self.assertLess(pre.within_field_skill_ci_low,0)
        self.assertGreater(pre.within_field_skill_ci_high,0)
        self.assertGreater(pre.leave_one_date_min_skill,0)
        self.assertLess(saved.loc['source_reset_adam_003','leave_one_date_max_skill'],0)

    def test_intercept_preserves_all_centered_sensitivity_scores(self):
        p=self.stage.differential_pretraining_only_surrogate.to_numpy()
        a,b=self.score(p),self.score(p-3.8079110886071925)
        for key in a:
            self.assertAlmostEqual(a[key],b[key],places=11)

    def test_new_source_offset_pairwise_comparisons(self):
        pairs=pd.read_csv(DATA/'results/statistical_checks/stage_pairwise_bootstrap_independent.csv')
        for first,delta,low,high in [
            ('source_reset_adam_003',-.5042379919216757,-.730476,-.274752),
            ('source_mean',-.3319688320258454,-.432649,-.231267)]:
            row=pairs.loc[pairs['first'].eq(first)&pairs['second'].eq('pretraining_source_offset')].iloc[0]
            self.assertAlmostEqual(row.delta_rmse_db,delta,places=10)
            self.assertAlmostEqual(row.rmse_ci_low_db,low,places=5)
            self.assertAlmostEqual(row.rmse_ci_high_db,high,places=5)
            self.assertLess(row.rmse_ci_high_db,0)


class SourceOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = pd.read_csv(DATA/'data/source/smapvex12_portable_source.csv')
        cls.zero = pd.read_csv(DATA/'results/cross_domain/zero_shot_predictions.csv')
        cls.weights = DATA/'results/trajectory_reconstruction'
        cls.result = source_offset_predictions(cls.weights, cls.source, unified_features(cls.zero))

    def test_source_only_intercept(self):
        self.assertAlmostEqual(self.result['source_intercept_db'], -3.8079110886071925, places=11)
        d = (self.source.sigma0_vv_db-self.source.sigma0_hh_db).to_numpy()
        residual = self.result['source_predictions_by_member'] + self.result['intercepts_by_member_db'][:,None] - d
        np.testing.assert_allclose(residual.mean(axis=1), 0, atol=1e-12)

    def test_target_labels_cannot_affect_intercept_or_prediction(self):
        changed = self.zero.copy()
        changed['sigma0_hh_db'] += 1000
        changed['sigma0_vv_db'] -= 1000
        other = source_offset_predictions(self.weights, self.source, unified_features(changed))
        np.testing.assert_array_equal(other['target_offset'], self.result['target_offset'])
        np.testing.assert_array_equal(other['intercepts_by_member_db'], self.result['intercepts_by_member_db'])

    def test_constant_shift_preserves_centered_response(self):
        y = self.zero.sigma0_vv_db-self.zero.sigma0_hh_db
        a = component_metrics(y, self.result['target_pretraining'])
        b = component_metrics(y, self.result['target_offset'])
        for key in ['centered_skill','centered_rmse_db','variance_ratio','correlation']:
            self.assertAlmostEqual(a[key], b[key], places=12)

    def test_saved_predictions_match_forward_pass(self):
        stored = self.zero.vv_pretraining_source_offset-self.zero.hh_pretraining_source_offset
        np.testing.assert_allclose(stored, self.result['target_offset'], atol=1e-11, rtol=0)
        self.assertAlmostEqual(component_metrics(self.zero.sigma0_vv_db-self.zero.sigma0_hh_db,
                               stored)['rmse_db'], 2.710210985852595, places=11)

    def test_member_correction_equals_ensemble_correction(self):
        np.testing.assert_allclose(self.result['target_offset'],
            self.result['target_pretraining']+self.result['source_intercept_db'], atol=1e-13, rtol=0)

    def test_pretraining_weights_are_order_independent(self):
        x = unified_features(self.zero)
        p = predict_saved_member(self.weights/'rebuilt_member_1.npz', x)
        q = predict_saved_member(self.weights/'rebuilt_member_1.npz', x[::-1])
        np.testing.assert_allclose(p, q[::-1], atol=1e-13, rtol=0)

    def test_spm_log_ratio_cancels_common_roughness_factor(self):
        from research_pilots.scattering.surfaces.spm import spm_backscatter_db, spm_polarization_factors
        out = spm_backscatter_db(12-0.6j, np.array([.004,.007,.01]),
                                 np.array([.06,.09,.14]), 1.26e9, 40., 'exponential')
        ah, av = spm_polarization_factors(12-0.6j, 40.)
        expected = 20*np.log10(abs(av/ah))
        np.testing.assert_allclose(out['vv_db']-out['hh_db'], expected, atol=1e-12)


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



class SourceAdamResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = DATA/'results/trajectory_reconstruction'
        cls.members = pd.read_csv(cls.weights/'predictions_by_member.csv')
        cls.zero = pd.read_csv(DATA/'results/cross_domain/zero_shot_predictions.csv')
        cls.stage = pd.read_csv(DATA/'results/stage_diagnostics/stage_predictions.csv')
        cls.result = json.loads((cls.weights/'summary.json').read_text())['source_reset_adam_003']

    def test_reset_loads_exact_arrays_and_zero_adam(self):
        from verify_stage_continuity import restore_for_source_reset, MEMBER_SEEDS
        for repeat, seed in enumerate(MEMBER_SEEDS, 1):
            with self.subTest(repeat=repeat):
                path = self.weights/f'rebuilt_member_{repeat}.npz'
                model, scales = restore_for_source_reset(path, seed)
                self.assertEqual(model._optimizer.t, 0)
                self.assertEqual(model._optimizer.learning_rate_init, 0.003)
                self.assertFalse(model.shuffle)
                self.assertEqual(len(model.loss_curve_), 0)
                self.assertTrue(all(np.count_nonzero(v) == 0 for v in model._optimizer.ms+model._optimizer.vs))
                with np.load(path, allow_pickle=False) as saved:
                    for i in range(3):
                        np.testing.assert_array_equal(model.coefs_[i], saved[f'pretraining_coefs_{i}'])
                        np.testing.assert_array_equal(model.intercepts_[i], saved[f'pretraining_intercepts_{i}'])
                    for key, value in scales.items():
                        np.testing.assert_array_equal(value, saved[key])

    def test_reset_start_matches_archived_member_predictions(self):
        from verify_stage_continuity import restore_for_source_reset, predict_reset_model, MEMBER_SEEDS
        for repeat, seed in enumerate(MEMBER_SEEDS, 1):
            model, scales = restore_for_source_reset(self.weights/f'rebuilt_member_{repeat}.npz', seed)
            pred = self.zero[['field_id','acquisition_date']].assign(
                starting=predict_reset_model(model, scales, unified_features(self.zero)))
            joined = self.members[self.members['repeat'].eq(repeat)].merge(pred,
                on=['field_id','acquisition_date'], validate='one_to_one')
            self.assertEqual(len(joined),189)
            np.testing.assert_allclose(joined.starting, joined.d_pretraining, atol=1e-12, rtol=0)

    def test_reset_saved_weights_match_member_predictions(self):
        for repeat in range(1,6):
            pred = predict_saved_member(self.weights/f'rebuilt_member_{repeat}.npz',
                                        unified_features(self.zero), 'source_reset_adam_003')
            lookup = self.zero[['field_id','acquisition_date']].assign(prediction=pred)
            joined = self.members[self.members['repeat'].eq(repeat)].merge(
                lookup, on=['field_id','acquisition_date'], validate='one_to_one')
            np.testing.assert_allclose(joined.prediction, joined.d_source_reset_adam_003, atol=1e-12, rtol=0)

    def test_reset_training_counts_and_settings(self):
        self.assertEqual(self.result['status'], 'COMPLETE')
        self.assertEqual(self.result['training_runs'],5)
        self.assertFalse(self.result['target_labels_used_for_training_or_selection'])
        self.assertEqual(self.result['source_minibatch_sizes'],[200,40])
        for member in self.result['members']:
            self.assertEqual(member['epochs'],120)
            self.assertEqual(member['source_order_seed'],member['seed']+10102)
            self.assertEqual(member['final_state']['optimizer_t'],240)
            self.assertEqual(member['final_state']['samples_seen'],28800)
            self.assertEqual(member['final_state']['optimizer_learning_rate_init'],0.003)
            self.assertEqual(member['archived_source_state']['optimizer_t'],4640)
            self.assertEqual(member['archived_source_state']['optimizer_learning_rate_init'],0.003)

    def test_reset_ensemble_and_common_cohort_agree(self):
        key=['field_id','acquisition_date']
        ensemble=self.members.groupby(key,as_index=False).d_source_reset_adam_003.mean()
        zero=self.zero.merge(ensemble,on=key,validate='one_to_one')
        self.assertEqual(len(zero),189)
        np.testing.assert_allclose(zero.vv_source_reset_adam_003-zero.hh_source_reset_adam_003,
                                   zero.d_source_reset_adam_003,atol=1e-12,rtol=0)
        common=self.stage.merge(ensemble,on=key,validate='one_to_one')
        self.assertEqual((len(common),common.field_id.nunique()),(138,22))
        np.testing.assert_allclose(common.differential_source_reset_adam_003,
                                   common.d_source_reset_adam_003,atol=1e-12,rtol=0)

    def test_reset_metrics_and_pairing(self):
        y=self.stage.sigma0_vv_db-self.stage.sigma0_hh_db
        new=component_metrics(y,self.stage.differential_source_reset_adam_003)
        self.assertAlmostEqual(new['rmse_db'],self.result['shared_138']['rmse_db'],places=12)
        self.assertAlmostEqual(new['rmse_db']**2,new['bias_db']**2+new['centered_rmse_db']**2,places=12)
        for comparison in self.result['paired_contrasts']:
            old=component_metrics(y,self.stage['differential_'+comparison['first_stage']])
            self.assertAlmostEqual(new['centered_skill']-old['centered_skill'],
                comparison['centered_skill_delta_second_minus_first'],places=12)
            self.assertAlmostEqual(new['rmse_db']-old['rmse_db'],
                comparison['rmse_delta_second_minus_first_db'],places=12)
        self.assertEqual(self.result['bootstrap']['seed'],27260910)
        self.assertEqual(self.result['bootstrap']['iterations'],10000)

    def test_reset_branch_skips_teacher_reconstruction(self):
        from unittest.mock import patch
        import verify_stage_continuity as control
        with patch.object(sys,'argv',['verify_stage_continuity.py','--reset-source-adam']), \
             patch.object(control,'run_source_adam_reset',return_value={}) as run, \
             patch.object(control.multi,'load_paired',side_effect=AssertionError('Teacher stage called')):
            control.main()
        run.assert_called_once_with(ROOT.resolve(),1e-8)

    def test_explicit_training_order_with_synthetic_batch(self):
        import copy
        from threadpoolctl import threadpool_limits
        from verify_stage_continuity import restore_for_source_reset
        from research_pilots.scattering.surrogate.physics_pretraining import train_epochs
        model,_=restore_for_source_reset(self.weights/'rebuilt_member_1.npz',20260910)
        expected=copy.deepcopy(model)
        generator=np.random.default_rng(7)
        x=generator.normal(size=(201,4));y=generator.normal(size=201)
        order=np.random.default_rng(20271012).permutation(len(x))
        with threadpool_limits(limits=1):
            train_epochs(model,x,y,epochs=1,seed=20271012)
            expected.partial_fit(x[order],y[order])
        self.assertEqual(model._optimizer.t,2)
        for left,right in zip(model.coefs_+model.intercepts_,expected.coefs_+expected.intercepts_):
            np.testing.assert_array_equal(left,right)


if __name__=='__main__':unittest.main()
