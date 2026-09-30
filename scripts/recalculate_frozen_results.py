#!/usr/bin/env python3
"""Recalculate published metrics and diagnostic decompositions from frozen tables.

This command does not train models, regenerate I2EM outputs, or replay raw data.
It reads the supplied numerical inputs and writes recalculated result tables.
The trajectory reconstruction is handled by verify_stage_continuity.py.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['field_id', 'acquisition_date']
YCOL = ['sigma0_hh_db', 'sigma0_vv_db']
FEATURES = ['soil_moisture_m3_m3', 'soil_real_dielectric',
            'pals_rms_height_cm', 'pals_correlation_length_cm']
STAGES = ['i2em_fixed40_teacher', 'pretraining_only_surrogate',
          'source_finetuned_surrogate', 'risk_shrunk_surrogate',
          'i2em_actual_angle_control', 'pretraining_source_offset',
          'source_reset_adam_003', 'source_mean']



def unified_features(frame: pd.DataFrame) -> np.ndarray:
    """Use the paper's Topp rule on both campaigns without modifying input tables."""
    x = frame[FEATURES].to_numpy(dtype=float, copy=True)
    mv = x[:, 0]
    x[:, 1] = 3.03 + 9.3 * mv + 146.0 * mv**2 - 76.7 * mv**3
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite scattering-model features")
    return x


def predict_saved_member(path: Path, features: np.ndarray,
                         stage: str = 'pretraining') -> np.ndarray:
    """Forward-only evaluation of archived tanh (4,16,8,1) NPZ arrays."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise ValueError("Expected finite N-by-4 input features")
    with np.load(path, allow_pickle=False) as weights:
        scale = weights['x_scale']
        if np.any(scale <= 0):
            raise ValueError("Invalid archived input scale")
        h = (x - weights['x_mean']) / scale
        for layer in range(3):
            h = h @ weights[f'{stage}_coefs_{layer}'] + weights[f'{stage}_intercepts_{layer}']
            if layer < 2:
                h = np.tanh(h)
        prediction = (h * weights['y_scale'] + weights['y_mean']).ravel()
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite archived-network prediction")
    return prediction


def source_offset_predictions(weights_dir: Path, source: pd.DataFrame,
                              target_features: np.ndarray) -> dict:
    """Fit five source-only intercepts; target labels are not an input.

    Averaging the five corrected predictions equals correcting the ensemble
    by its source residual mean. All network parameters and scales stay fixed.
    """
    xs = unified_features(source)
    ds = (source[YCOL[1]] - source[YCOL[0]]).to_numpy(float)
    ps = np.stack([predict_saved_member(weights_dir / f'rebuilt_member_{j}.npz', xs)
                   for j in range(1, 6)])
    pt = np.stack([predict_saved_member(weights_dir / f'rebuilt_member_{j}.npz', target_features)
                   for j in range(1, 6)])
    intercepts = np.mean(ds[None, :] - ps, axis=1)
    return {'source_predictions_by_member': ps,
            'target_pretraining_by_member': pt,
            'intercepts_by_member_db': intercepts,
            'source_intercept_db': float(intercepts.mean()),
            'target_offset_by_member': pt + intercepts[:, None],
            'target_pretraining': pt.mean(axis=0),
            'target_offset': (pt + intercepts[:, None]).mean(axis=0),
            'common_source_mean_db': float(source[YCOL].to_numpy(float).mean()),
            'differential_source_mean_db': float(ds.mean())}


def read_table(path, required=(), unique_key=()):
    """Read a required input and validate its schema without a checksum file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required paper input is missing: {path}')
    frame = pd.read_csv(path)
    missing = sorted(set(required).union(unique_key).difference(frame.columns))
    if missing:
        raise ValueError(f'{path}: missing required columns {missing}')
    if frame.empty:
        raise ValueError(f'{path}: input table is empty')
    if 'acquisition_date' in frame:
        frame['acquisition_date'] = pd.to_datetime(frame.acquisition_date).dt.strftime('%Y-%m-%d')
    if unique_key and (frame[list(unique_key)].isna().any().any()
                       or frame.duplicated(list(unique_key)).any()):
        raise ValueError(f'{path}: missing or duplicate sample keys {list(unique_key)}')
    return frame


def component_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Prediction/reference shape mismatch or nonfinite value')
    yc, pc = y-y.mean(), p-p.mean()
    vy, vp, cov = np.mean(yc**2), np.mean(pc**2), np.mean(yc*pc)
    if vy <= 0:
        raise ValueError('Observed variance is zero')
    return {'bias_db': float(np.mean(p-y)),
            'rmse_db': float(np.sqrt(np.mean((p-y)**2))),
            'centered_rmse_db': float(np.sqrt(np.mean((pc-yc)**2))),
            'centered_skill': float(1-np.mean((pc-yc)**2)/vy),
            'variance_ratio': float(vp/vy),
            'correlation': float(cov/np.sqrt(vy*vp)) if vp > 1e-20 else None,
            'observed_variance': float(vy), 'predicted_variance': float(vp)}


def rmse_avg(y, p):
    return float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2, axis=0)).mean())


def field_weights(field, iterations, seed):
    fields = np.array(sorted(pd.unique(field)))
    draw = np.random.default_rng(seed).integers(0, len(fields), (iterations, len(fields)))
    w = np.zeros((iterations, len(fields)), dtype=np.int32)
    np.add.at(w, (np.arange(iterations)[:, None], draw), 1)
    inv = pd.Categorical(field, categories=fields).codes
    return w, inv, fields


def component_bootstrap(y, p, inv, w):
    y, p = np.asarray(y), np.asarray(p)
    vectors = [np.ones(len(y)), y, p, y*y, p*p, y*p]
    sums = np.array([np.bincount(inv, weights=v, minlength=w.shape[1]) for v in vectors])
    n, sy, sp, sy2, sp2, syp = (w@sums.T).T
    my, mp = sy/n, sp/n
    vy, vp = np.maximum(sy2/n-my*my, 0), np.maximum(sp2/n-mp*mp, 0)
    cov = syp/n-my*mp
    return {'skill': (2*cov-vp)/vy, 'bias': mp-my,
            'rmse': np.sqrt(np.maximum((sp2 + sy2 - 2*syp)/n, 0))}


def field_response_sensitivity(y, p, field, dates, weights, inverse):
    """Score fixed predictions within fields, with equal field weights and by date.

    The field bootstrap retains every observation in each sampled field. Dates,
    model weights and source calibration stay fixed; deleting a date only rescores
    the remaining predictions. No model is trained or selected here.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    field, dates = np.asarray(field), np.asarray(dates)
    if not (y.shape == p.shape == field.shape == dates.shape):
        raise ValueError('Response-sensitivity arrays must have equal length')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite response-sensitivity input')
    frame = pd.DataFrame({'field': field, 'y': y, 'p': p})
    means = frame.groupby('field')[['y', 'p']].transform('mean')
    yw, pw = y-means.y.to_numpy(), p-means.p.to_numpy()
    within = component_metrics(yw, pw)
    between = component_metrics(means.y, means.p)
    total = component_metrics(y, p)
    nf = weights.shape[1]
    obs_ss = np.bincount(inverse, weights=yw**2, minlength=nf)
    error_ss = np.bincount(inverse, weights=(pw-yw)**2, minlength=nf)
    denom = weights @ obs_ss
    if np.any(denom <= 0):
        raise ValueError('A field bootstrap replicate has no within-field variance')
    ci = np.quantile(1-(weights @ error_ss)/denom, [.025, .975])
    # Each field has the same total weight, regardless of its date count.
    sample_weight = 1/frame.groupby('field').y.transform('size').to_numpy(float)
    yc = y-np.average(y, weights=sample_weight)
    pc = p-np.average(p, weights=sample_weight)
    equal_skill = 1-np.average((pc-yc)**2, weights=sample_weight)/np.average(yc**2, weights=sample_weight)
    removed = {}
    for date in sorted(pd.unique(dates)):
        keep = dates != date
        if keep.sum() < 2:
            raise ValueError('Date deletion leaves fewer than two observations')
        removed[str(date)] = component_metrics(y[keep], p[keep])['centered_skill']
    return {'total_skill': total['centered_skill'],
            'within_field_skill': within['centered_skill'],
            'within_field_skill_ci_low': float(ci[0]),
            'within_field_skill_ci_high': float(ci[1]),
            'between_field_skill': between['centered_skill'],
            'within_obs_variance_fraction': within['observed_variance']/total['observed_variance'],
            'within_field_correlation': within['correlation'],
            'between_field_correlation': between['correlation'],
            'equal_field_weight_skill': float(equal_skill),
            'leave_one_date_min_skill': min(removed.values()),
            'leave_one_date_max_skill': max(removed.values()),
            **{'leave_date_'+date+'_skill': value for date, value in removed.items()}}


def shared_conditional_bootstrap(df, predictors, iterations, seed):
    """Shared weights across fixed splits; no refitting or adaptation resampling.

    The resampling population is the common-valid test-field union (22 fields).
    Thus new control intervals need not be bitwise identical to authors' offsets
    intervals if their unpublished implementation used a different field universe
    or random-number draw order. They estimate conditional, NOT unconditional,
    prediction uncertainty.
    """
    w, _, fields = field_weights(df.field_id, iterations, seed)
    out = {m: np.zeros(iterations) for m in predictors}
    counts = np.zeros(iterations)
    for _, idx in df.groupby(['split_repeat', 'requested_fraction']).groups.items():
        idx = np.asarray(list(idx))
        g = df.loc[idx]
        inv = pd.Categorical(g.field_id, categories=fields).codes
        den = w@np.bincount(inv, minlength=len(fields))
        ok = den > 0
        counts += ok
        for method, p in predictors.items():
            e2 = (np.asarray(p)[idx]-g[YCOL].values)**2
            ss = np.column_stack([np.bincount(inv, weights=e2[:, j],
                                             minlength=len(fields)) for j in range(2)])
            weighted = w@ss
            scores = np.zeros(iterations)
            scores[ok] = np.sqrt(weighted[ok]/den[ok, None]).mean(axis=1)
            out[method] += scores
    if (counts == 0).any():
        raise ValueError('Bootstrap replicate has no scored split')
    return {m: v/counts for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.bundle.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError('Choose an output directory outside the input bundle')
    out.mkdir(parents=True, exist_ok=True)

    checked_inputs = set()

    def csv(rel, required=(), unique_key=()):
        frame = read_table(root/rel, required, unique_key)
        checked_inputs.add(rel)
        return frame

    def save(df, name):
        df.to_csv(out/name, index=False)

    files = [p for p in root.rglob('*') if p.is_file()]
    src = csv('data/source/smapvex12_portable_source.csv', FEATURES+YCOL, KEY)
    tgt = csv('data/target/smex02_field_day_model_ready.csv', FEATURES+YCOL, KEY)
    zero = csv('results/cross_domain/zero_shot_predictions.csv', FEATURES+YCOL, KEY)
    split_key = ['split_repeat', 'requested_fraction']
    held = csv('results/cross_domain/heldout_predictions.csv', YCOL, split_key+KEY)
    assignments = csv('results/cross_domain/field_split_assignments.csv', ['role'], split_key+['field_id'])
    common = csv('results/common_cohort/predictions.csv', FEATURES+YCOL, KEY)
    stage = csv('results/stage_diagnostics/stage_predictions.csv', FEATURES+YCOL+['differential_'+m for m in STAGES], KEY)
    offset = csv('results/offset_diagnostics/heldout_common_predictions.csv', YCOL, split_key+KEY)
    for label, frame, count in [('source',src,240),('target',tgt,189),('zero-shot',zero,189),('common',common,138),('stage',stage,138)]:
        if len(frame) != count:
            raise ValueError(f'Expected {count} published {label} samples, found {len(frame)}')
    cohorts = []
    for name, df in [('source', src), ('target', tgt), ('zero', zero),
                     ('common', common), ('stage', stage)]:
        assert not df.duplicated(KEY).any()
        assert np.isfinite(df[FEATURES+YCOL].values).all()
        cohorts.append({'table': name, 'rows': len(df), 'fields': df.field_id.nunique(),
                        'dates': df.acquisition_date.nunique(), 'duplicate_keys': 0})
    save(pd.DataFrame(cohorts), 'cohort_key_checks.csv')

    joinchecks = []
    for name, df in [('zero', zero), ('heldout', held), ('common', common), ('stage', stage)]:
        j = df[KEY+YCOL].merge(tgt[KEY+YCOL], on=KEY, suffixes=('_a', '_b'), validate='many_to_one')
        assert len(j) == len(df)
        err = max(float(np.max(abs(j[c+'_a']-j[c+'_b']))) for c in YCOL)
        assert err < 1e-10
        joinchecks.append({'table': name, 'observation_max_absolute_difference': err})
    save(pd.DataFrame(joinchecks), 'cross_file_observation_checks.csv')

    tr, te = (csv(f'data/teachers/i2em_{split}_requests.csv', FEATURES+['i2em_request_valid'], ['request_id']) for split in ['train','test'])
    assert not set(tr.request_id)&set(te.request_id)
    assert not set(map(tuple, tr[FEATURES].values))&set(map(tuple, te[FEATURES].values))
    assert not tr.duplicated(FEATURES).any() and not te.duplicated(FEATURES).any()
    assert tr.i2em_request_valid.all() and te.i2em_request_valid.all()
    for split, req in [('train', tr), ('test', te)]:
        values = csv(f'data/teachers/i2em_{split}_results.csv', ['i2em_hh_db','i2em_vv_db'], ['request_id'])
        assert set(req.request_id) == set(values.request_id)
        assert values.request_id.is_unique
        assert np.isfinite(values[['i2em_hh_db','i2em_vv_db']].to_numpy(float)).all()

    zero_scores = []
    for c in zero:
        if c.startswith('hh_'):
            m = c[3:]
            zero_scores.append({'method': m, 'mean_hh_vv_rmse_db': rmse_avg(zero[YCOL], zero[[c, 'vv_'+m]])})
    assert np.max(abs(zero[['hh_source_mean', 'vv_source_mean']].values-src[YCOL].mean().values)) < 1e-10
    save(pd.DataFrame(zero_scores), 'zero_shot_metrics_independent.csv')

    full_mu, splitchecks, full_scores = {}, [], []
    for (rep, frac), a in assignments.groupby(['split_repeat', 'requested_fraction']):
        af = set(a.loc[a.role == 'adaptation', 'field_id'])
        tf = set(a.loc[a.role == 'test', 'field_id'])
        assert not af & tf and af|tf == set(tgt.field_id)
        g = held[(held.split_repeat == rep)&(held.requested_fraction == frac)]
        expected = tgt[tgt.field_id.isin(tf)]
        assert not g.duplicated(KEY).any()
        assert set(map(tuple, g[KEY].values)) == set(map(tuple, expected[KEY].values))
        ad = tgt[tgt.field_id.isin(af)]
        mu = ad[YCOL].mean().values
        full_mu[(rep, frac)] = mu
        splitchecks.append({'split_repeat': rep, 'requested_fraction': frac,
                            'adaptation_fields': len(af), 'adaptation_rows': len(ad),
                            'test_fields': len(tf), 'test_rows': len(g), 'overlap_fields': 0})
        full_scores.append({'split_repeat': rep, 'requested_fraction': frac,
                            'method': 'two_channel_mean_full_adapt', 'rmse_avg': rmse_avg(g[YCOL], mu)})
        for c in g:
            if c.startswith('hh_'):
                m = c[3:]
                full_scores.append({'split_repeat': rep, 'requested_fraction': frac,
                                    'method': m, 'rmse_avg': rmse_avg(g[YCOL], g[[c, 'vv_'+m]])})
        assert np.max(abs(g[['hh_spm_only', 'vv_spm_only']].mean(axis=1)-mu.mean())) < 1e-10
    save(pd.DataFrame(splitchecks), 'split_checks.csv')
    save(pd.DataFrame(full_scores), 'full_fewshot_metrics_independent.csv')

    offset_check = source_offset_predictions(root/'results/trajectory_reconstruction',
                                             src, unified_features(zero))
    np.testing.assert_allclose(zero.vv_pretraining_source_offset-zero.hh_pretraining_source_offset,
                               offset_check['target_offset'], atol=1e-11, rtol=0)
    joined_offset = stage[KEY+['differential_pretraining_source_offset']].merge(
        zero[KEY+['hh_pretraining_source_offset','vv_pretraining_source_offset']],
        on=KEY, validate='one_to_one')
    np.testing.assert_allclose(joined_offset.differential_pretraining_source_offset,
                               joined_offset.vv_pretraining_source_offset-joined_offset.hh_pretraining_source_offset,
                               atol=1e-11, rtol=0)
    reset_prediction = np.mean([
        predict_saved_member(root/'results/trajectory_reconstruction'/f'rebuilt_member_{j}.npz',
                             unified_features(zero), stage='source_reset_adam_003')
        for j in range(1, 6)], axis=0)
    np.testing.assert_allclose(zero.vv_source_reset_adam_003-zero.hh_source_reset_adam_003,
                               reset_prediction, atol=1e-11, rtol=0)
    reset_join = stage[KEY+['differential_source_reset_adam_003']].merge(
        zero[KEY].assign(reset_prediction=reset_prediction), on=KEY, validate='one_to_one')
    np.testing.assert_allclose(reset_join.differential_source_reset_adam_003,
                               reset_join.reset_prediction, atol=1e-11, rtol=0)
    sy = stage.sigma0_vv_db-stage.sigma0_hh_db
    sm = pd.DataFrame([{'method': m, **component_metrics(sy, stage['differential_'+m])} for m in STAGES])
    stored = csv('results/stage_diagnostics/stage_metrics.csv').set_index('method')
    cols = ['bias_db', 'rmse_db', 'centered_rmse_db', 'centered_skill', 'variance_ratio']
    maxerr = float((sm.set_index('method')[cols]-stored[cols]).abs().max().max())
    assert maxerr < 1e-10
    save(sm, 'stage_metrics_independent.csv')
    decomposition=sm[['method','bias_db','centered_rmse_db','rmse_db','correlation','variance_ratio']].copy()
    decomposition['bias_squared_db2']=decomposition.bias_db**2
    decomposition['centered_mse_db2']=decomposition.centered_rmse_db**2
    decomposition['mse_db2']=decomposition.rmse_db**2
    assert np.allclose(decomposition.mse_db2,decomposition.bias_squared_db2+decomposition.centered_mse_db2,atol=1e-10)
    save(decomposition,'error_decomposition.csv')
    d=decomposition.set_index('method')
    changes=[]
    for first,second in [*zip(STAGES[:3],STAGES[1:4]),
                         (STAGES[2], STAGES[6]), (STAGES[1], STAGES[6])]:
        changes.append({'first_stage':first,'second_stage':second,
          **{name:float(d.loc[second,name]-d.loc[first,name]) for name in ['bias_squared_db2','centered_mse_db2','mse_db2']}})
    save(pd.DataFrame(changes),'error_decomposition_changes.csv')

    w, inv, _ = field_weights(stage.field_id.astype(str), 10000, 27260910)
    bs = {m: component_bootstrap(sy, stage['differential_'+m], inv, w) for m in STAGES}
    pairs, cis = [], []
    for m in STAGES:
        q = np.quantile(bs[m]['skill'], [.025, .975])
        cis.append({'method': m, 'skill_ci_low': q[0], 'skill_ci_high': q[1]})
    for ia, ib in [(0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 5), (2, 6), (1, 6), (6, 5), (7, 5)]:
        a, b = STAGES[ia], STAGES[ib]
        q = np.quantile(bs[b]['skill']-bs[a]['skill'], [.025, .975])
        qb = np.quantile(bs[b]['bias']-bs[a]['bias'], [.025, .975])
        qr = np.quantile(bs[b]['rmse']-bs[a]['rmse'], [.025, .975])
        pairs.append({'first': a, 'second': b,
                      'delta_skill': sm.set_index('method').loc[b, 'centered_skill']-sm.set_index('method').loc[a, 'centered_skill'],
                      'ci_low': q[0], 'ci_high': q[1], 'bias_ci_low': qb[0], 'bias_ci_high': qb[1],
                      'delta_rmse_db': sm.set_index('method').loc[b,'rmse_db']-sm.set_index('method').loc[a,'rmse_db'],
                      'rmse_ci_low_db': qr[0], 'rmse_ci_high_db': qr[1]})
    save(pd.DataFrame(cis), 'stage_bootstrap_intervals_independent.csv')
    save(pd.DataFrame(pairs), 'stage_pairwise_bootstrap_independent.csv')

    offscores, adcounts, comparisons = [], [], []
    for (rep, frac), g in offset.groupby(['split_repeat', 'requested_fraction']):
        a = assignments[(assignments.split_repeat == rep)&(assignments.requested_fraction == frac)]
        af = set(a.loc[a.role == 'adaptation', 'field_id'])
        ad = common[common.field_id.isin(af)]
        mu_c, mu_f = ad[YCOL].mean().values, full_mu[(rep, frac)]
        assert len(ad) > 0
        assert np.max(abs(g[['hh_two_channel_mean', 'vv_two_channel_mean']].values-mu_c)) < 1e-10
        assert np.max(abs(g[['hh_spm_only', 'vv_spm_only']].mean(axis=1)-mu_f.mean())) < 1e-10
        j = g[KEY].merge(common[KEY+['i2em_hh_db', 'i2em_vv_db']], on=KEY, validate='one_to_one')
        shift = np.mean(ad[YCOL].values-ad[['i2em_hh_db', 'i2em_vv_db']].values, axis=0)
        p = j[['i2em_hh_db', 'i2em_vv_db']].values+shift
        assert np.max(abs(p-g[['hh_i2em_actual_angle_plus_offset', 'vv_i2em_actual_angle_plus_offset']].values)) < 1e-10
        adcounts.append({'split_repeat': rep, 'requested_fraction': frac,
                         'full_adaptation_rows': len(tgt[tgt.field_id.isin(af)]),
                         'common_adaptation_rows': len(ad), 'full_adaptation_fields': len(af),
                         'common_adaptation_fields': ad.field_id.nunique()})
        for c in g:
            if c.startswith('hh_'):
                m = c[3:]
                offscores.append({'split_repeat': rep, 'requested_fraction': frac,
                                  'method': m, 'rmse_avg': rmse_avg(g[YCOL], g[[c, 'vv_'+m]])})
        offscores.append({'split_repeat': rep, 'requested_fraction': frac,
                          'method': 'two_channel_mean_full_adapt', 'rmse_avg': rmse_avg(g[YCOL], mu_f)})
    offdf = pd.DataFrame(offscores)
    means = offdf.groupby(['requested_fraction', 'method']).rmse_avg.mean().unstack()
    save(offdf, 'common_fewshot_metrics_and_equal_information_mean.csv')
    save(means.reset_index(), 'common_comparison_summary.csv')
    save(pd.DataFrame(adcounts), 'common_adaptation_counts.csv')
    for frac in sorted(offset.requested_fraction.unique()):
        g = offset[offset.requested_fraction == frac].reset_index(drop=True)
        p = {c[3:]: g[[c, 'vv_'+c[3:]]].values for c in g if c.startswith('hh_')}
        p['two_channel_mean_full_adapt'] = np.array([full_mu[(r, frac)] for r in g.split_repeat])
        bb = shared_conditional_bootstrap(g, p, 10000, 28260911)
        for a, b in [('i2em_actual_angle_plus_offset', 'two_channel_mean'),
                     ('spm_only', 'two_channel_mean'), ('spm_only', 'two_channel_mean_full_adapt'),
                     ('spm_to_i2em', 'two_channel_mean_full_adapt')]:
            q = np.quantile(bb[a]-bb[b], [.025, .975])
            comparisons.append({'fraction': frac, 'first': a, 'second': b,
                                'delta': means.loc[frac, a]-means.loc[frac, b],
                                'conditional_ci_low': q[0], 'conditional_ci_high': q[1]})
    save(pd.DataFrame(comparisons), 'common_paired_contrasts_independent.csv')

    breakdown, leaveout, coverage, inbox = [], [], [], []
    for m in STAGES:
        sensitivity = field_response_sensitivity(
            sy, stage['differential_'+m], stage.field_id,
            stage.acquisition_date, w, inv)
        breakdown.append({'method': m, **sensitivity})
        for f in sorted(stage.field_id.unique()):
            mask = stage.field_id != f
            leaveout.append({'method': m, 'left_out': f,
                             **component_metrics(sy[mask], stage.loc[mask, 'differential_'+m])})
    for name, df in [('source', src), ('target', tgt), ('common', stage)]:
        # Rounded ranges printed in manuscript Table III, NOT a convex-hull test.
        masks = {'mv': df.soil_moisture_m3_m3.between(.0698, .4859),
                 's': df.pals_rms_height_cm.between(.395, 1.135),
                 'l': df.pals_correlation_length_cm.between(5, 18.25)}
        inside = np.logical_and.reduce(list(masks.values()))
        ks = 2*np.pi*1.26e9/299792458*df.pals_rms_height_cm/100
        slope = df.pals_rms_height_cm/df.pals_correlation_length_cm
        coverage.append({'table': name, 'rows': len(df), 'inside_reported_sim_box': int(inside.sum()),
                         'outside_reported_sim_box': int((~inside).sum()),
                         'spm_valid': int(((ks < .3)&(slope <= .21)).sum()),
                         'i2em_valid': int(((ks < 1)&(slope <= .25)).sum())})
        if name == 'common':
            for m in STAGES:
                inbox.append({'method': m, 'n': int(inside.sum()),
                              **component_metrics(sy[inside], stage.loc[inside, 'differential_'+m])})
    save(pd.DataFrame(breakdown), 'exploratory_within_between_field.csv')
    save(pd.DataFrame(leaveout), 'exploratory_leave_one_field_out.csv')
    save(pd.DataFrame(coverage), 'reported_domain_coverage.csv')
    save(pd.DataFrame(inbox), 'exploratory_training_box_subset.csv')

    simulation = csv('results/sample_efficiency/metrics_by_repeat.csv')
    sim = simulation[simulation.response.isin(['HH', 'VV'])].groupby(
        ['repeat', 'i2em_train_samples', 'method']).rmse_db.mean().reset_index()
    save(sim, 'simulation_average_channel_by_repeat.csv')
    save(sim.groupby(['i2em_train_samples', 'method']).rmse_db.agg(['mean', 'std']).reset_index(),
         'simulation_sample_efficiency_reaggregated.csv')
    rb = csv('results/residual_baselines/metrics_by_repeat.csv')
    save(rb.groupby(['size', 'method']).rmse_mean_hh_vv_db.agg(['mean', 'std']).reset_index(),
         'residual_baselines_reaggregated.csv')

    report = {
        'scope': 'Frozen-table numerical recalculation; trajectory reconstruction and raw-data replay are separate commands',
        'input_file_count': len(files), 'input_file_extensions': dict(Counter(p.suffix for p in files)),
        'input_validation': 'Required tables, schemas, sample keys, cohorts, teacher pairing and field splits; versions are tracked by Git',
        'input_tables_checked': sorted(checked_inputs),
        'author_source_code_files': [str(p.relative_to(root)) for p in files if p.suffix in {'.py', '.m', '.ipynb', '.R', '.jl'}],
        'cohorts': cohorts, 'stage_numeric_max_difference': maxerr,
        'stage_metrics': sm.to_dict('records'), 'stage_bootstrap': pairs,
        'common_comparison': means.reset_index().to_dict('records'),
        'common_new_contrasts': comparisons,
        'new_conditional_bootstrap': {'iterations': 10000, 'seed': 28260911,
            'field_population': 'union of common-valid held-out fields (22)',
            'limitation': 'frozen models, adaptation sets and offsets; no source-training or campaign uncertainty'},
        'exploratory_within_between': breakdown, 'coverage': coverage,
        'response_sensitivity_protocol': {'analysis_role': 'post-hoc fixed-prediction analysis',
            'iterations': 10000, 'seed': 27260910, 'unit': 'field_id',
            'date_deletion': 'one observed date removed at a time; no refitting',
            'equal_field_weights': 'each observation has weight 1 / its field record count'},
        'not_verified_by_this_command': [
                         'Ancillary source-control replay; use audit_source_controls.py for supplied OOF evidence',
                         'Raw NSIDC collocation replay', 'SPM and MATLAB I2EM implementation',
                         'Historical ordering of model/protocol selection']}
    (out/'audit_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'FROZEN_RECALCULATION_COMPLETE',
                      'input_tables_checked': len(checked_inputs), 'source_code_files': report['author_source_code_files'],
                      'common_comparison': report['common_comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
