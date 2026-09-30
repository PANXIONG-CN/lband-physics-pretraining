#!/usr/bin/env python3
"""Reconstruct the archived trajectory or run its five-member reset-Adam control."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import audit_external_domain as audit
import evaluate_decoupled_physics_heads as base
import evaluate_multifidelity_pretraining as multi


def state(model):
    opt=getattr(model,'_optimizer',None)
    return {'learning_rate_init_attribute':float(model.learning_rate_init),
      'optimizer_learning_rate_init':float(opt.learning_rate_init) if opt else None,
      'optimizer_t':int(opt.t) if opt else None,'samples_seen':int(model.t_),
      'batch_size':model.batch_size,'shuffle':bool(model.shuffle)}


RESET_METHOD = 'source_reset_adam_003'
MEMBER_SEEDS = (20260910, 20360910, 20460910, 20560910, 20660910)


def restore_for_source_reset(weight_path: Path, seed: int):
    """Load fixed pretraining arrays and allocate a fresh Adam without an update."""
    from sklearn.utils import check_random_state
    from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
    from research_pilots.scattering.surrogate.physics_pretraining import make_mlp

    with np.load(weight_path, allow_pickle=False) as saved:
        coefs = [saved[f'pretraining_coefs_{i}'].copy() for i in range(3)]
        intercepts = [saved[f'pretraining_intercepts_{i}'].copy() for i in range(3)]
        scales = {k: saved[k].copy() for k in ('x_mean', 'x_scale', 'y_mean', 'y_scale')}
    for value, shape in zip(coefs + intercepts,
                            [(4, 16), (16, 8), (8, 1), (16,), (8,), (1,)]):
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f'Invalid archived float64 network array: {weight_path}')
    if any(not np.isfinite(v).all() for v in scales.values()) or any(
            np.any(scales[k] <= 0) for k in ('x_scale', 'y_scale')):
        raise ValueError(f'Invalid archived scales: {weight_path}')
    model = make_mlp(seed, learning_rate=0.003)
    model._random_state = check_random_state(seed)
    model._initialize(np.zeros((1, 1)), [4, 16, 8, 1], np.dtype('float64'))
    model.n_features_in_ = 4
    model.coefs_, model.intercepts_ = coefs, intercepts
    model._best_coefs = [v.copy() for v in coefs]
    model._best_intercepts = [v.copy() for v in intercepts]
    model._optimizer = AdamOptimizer(
        coefs + intercepts, learning_rate_init=0.003,
        beta_1=model.beta_1, beta_2=model.beta_2, epsilon=model.epsilon)
    return model, scales


def predict_reset_model(model, scales: dict, features: np.ndarray) -> np.ndarray:
    """Return physical-unit differential predictions using the original scales."""
    x = (np.asarray(features, dtype=np.float64) - scales['x_mean']) / scales['x_scale']
    return (model.predict(x).reshape(-1, 1) * scales['y_scale'] + scales['y_mean']).ravel()


def run_source_adam_reset(root: Path, atol: float = 1e-8) -> dict:
    """Train only the five 120-epoch source stages and extend existing artifacts."""
    from research_pilots.scattering.surrogate.physics_pretraining import train_epochs
    from recalculate_frozen_results import (
        KEY, YCOL, component_metrics, component_bootstrap, field_weights, rmse_avg)

    data = root / 'reproducibility'
    result = data / 'results'
    trajectory = result / 'trajectory_reconstruction'
    started = time.monotonic()

    def read(rel):
        frame = pd.read_csv(result / rel, dtype={'field_id': str})
        if 'acquisition_date' in frame:
            frame['acquisition_date'] = pd.to_datetime(frame.acquisition_date).dt.strftime('%Y-%m-%d')
        return frame

    def attach(frame, values, columns, keys=KEY):
        linked = frame.drop(columns=columns, errors='ignore').merge(
            values[list(keys) + list(columns)], on=list(keys), how='left',
            sort=False, validate='one_to_one')
        if len(linked) != len(frame) or linked[list(columns)].isna().any().any():
            raise ValueError('Missing or duplicated field-date predictions')
        return linked

    def save(frame, rel):
        path = result / rel
        if not path.is_file():
            raise FileNotFoundError(f'Expected an existing result file: {path}')
        frame.to_csv(path, index=False)

    def extend_rows(rel, rows, id_column, value):
        frame = read(rel)
        frame = frame.loc[frame[id_column].ne(value)]
        save(pd.concat([frame, pd.DataFrame(rows)], ignore_index=True), rel)

    source = audit.prepare_table(data/'data/source/smapvex12_portable_source.csv', 'SMAPVEX12', 'topp_both')
    target = audit.prepare_table(data/'data/target/smex02_field_day_model_ready.csv', 'SMEX02', 'topp_both')
    source = source.loc[source.finite_model_row].sort_values(KEY).reset_index(drop=True)
    target = target.loc[target.finite_model_row].sort_values(KEY).reset_index(drop=True)
    for frame, count, fields in [(source, 240, 24), (target, 189, 30)]:
        frame['field_id'] = frame.field_id.astype(str)
        frame['acquisition_date'] = frame.acquisition_date.dt.strftime('%Y-%m-%d')
        if len(frame) != count or frame.field_id.nunique() != fields or frame.duplicated(KEY).any():
            raise ValueError('Source/target cohort differs from the planned control')
    xs = source[list(base.FEATURE_NAMES)].to_numpy(np.float64)
    ds = (source[YCOL[1]] - source[YCOL[0]]).to_numpy(np.float64)
    xt = target[list(base.FEATURE_NAMES)].to_numpy(np.float64)
    c_source = float(source[YCOL].to_numpy(float).mean())
    members = read('trajectory_reconstruction/predictions_by_member.csv')
    stage = read('stage_diagnostics/stage_predictions.csv')
    zero = read('cross_domain/zero_shot_predictions.csv')
    if len(stage) != 138 or stage.field_id.nunique() != 22:
        raise ValueError('The original common target cohort is required')
    if len(members) != 5 * len(target) or set(members['repeat']) != set(range(1, 6)):
        raise ValueError('Expected five complete archived members')
    summary_path = trajectory/'summary.json'
    archived = json.loads(summary_path.read_text())
    column = 'd_' + RESET_METHOD
    member_predictions, run_records, trained_arrays = [], [], []

    with threadpool_limits(limits=1):
        for repeat, seed in enumerate(MEMBER_SEEDS, 1):
            path = trajectory/f'rebuilt_member_{repeat}.npz'
            model, scales = restore_for_source_reset(path, seed)
            before = state(model)
            moment_zero = all(np.count_nonzero(v) == 0 for v in model._optimizer.ms + model._optimizer.vs)
            with np.load(path, allow_pickle=False) as saved:
                unchanged = all(np.array_equal(model.coefs_[i], saved[f'pretraining_coefs_{i}'])
                                and np.array_equal(model.intercepts_[i], saved[f'pretraining_intercepts_{i}'])
                                for i in range(3))
            if not unchanged or not moment_zero or before['optimizer_t'] != 0:
                raise ValueError('Reset initialization did not match the fixed starting state')
            start_predictions = target[KEY].assign(d_start=predict_reset_model(model, scales, xt))
            reference = members.loc[members['repeat'].eq(repeat)]
            check = reference.merge(start_predictions, on=KEY, validate='one_to_one')
            if len(check) != len(target):
                raise ValueError('Incomplete starting-prediction comparison')
            start_error = float(np.max(np.abs(check.d_start - check.d_pretraining)))
            if start_error > atol:
                raise ValueError(f'Member {repeat}: starting predictions differ by {start_error} dB')
            scaled_x = (xs - scales['x_mean']) / scales['x_scale']
            scaled_d = ((ds[:, None] - scales['y_mean']) / scales['y_scale']).ravel()
            train_epochs(model, scaled_x, scaled_d, epochs=120, seed=seed + 10102)
            after = state(model)
            if after['optimizer_t'] != 240 or after['optimizer_learning_rate_init'] != 0.003 or len(model.loss_curve_) != 120:
                raise ValueError('Reset source stage did not finish the specified updates')
            pred = predict_reset_model(model, scales, xt)
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite reset-Adam predictions')
            member_predictions.append(target[KEY].assign(repeat=repeat, **{column: pred}))
            arrays = {}
            for kind, values in [('coefs', model.coefs_), ('intercepts', model.intercepts_)]:
                arrays.update({f'{RESET_METHOD}_{kind}_{i}': v.copy() for i, v in enumerate(values)})
            trained_arrays.append((path, arrays))
            old_member = next(m for m in archived['members'] if m['repeat'] == repeat)
            run_records.append({
                'repeat': repeat, 'seed': seed, 'source_order_seed': seed + 10102,
                'epochs': len(model.loss_curve_), 'starting_arrays_equal': unchanged,
                'starting_prediction_max_abs_difference_db': start_error,
                'moments_initially_zero': moment_zero, 'initial_state': before,
                'final_state': after, 'archived_source_state': old_member['source'],
                'source_training_rmse_db': component_metrics(ds, predict_reset_model(model, scales, xs))['rmse_db']})
            print(f'Reset-Adam member {repeat}/5: 120 epochs, 240 updates; starting difference {start_error:.3g} dB', flush=True)

    new_members = pd.concat(member_predictions, ignore_index=True)
    members = attach(members, new_members, [column], ['repeat'] + KEY)
    ensemble = new_members.groupby(KEY, as_index=False)[column].mean()
    new_stage = attach(stage, ensemble, [column])
    new_stage['differential_' + RESET_METHOD] = new_stage.pop(column)
    new_zero = attach(zero, ensemble, [column])
    new_zero['hh_' + RESET_METHOD] = c_source - new_zero[column]/2
    new_zero['vv_' + RESET_METHOD] = c_source + new_zero[column]/2
    new_zero = new_zero.drop(columns=column)
    y = (stage[YCOL[1]] - stage[YCOL[0]]).to_numpy(float)
    p = new_stage['differential_' + RESET_METHOD].to_numpy(float)
    methods = {RESET_METHOD: p,
               'source_finetuned_surrogate': stage.differential_source_finetuned_surrogate.to_numpy(float),
               'pretraining_only_surrogate': stage.differential_pretraining_only_surrogate.to_numpy(float)}
    points = {m: component_metrics(y, pred) for m, pred in methods.items()}
    w, inv, fields = field_weights(stage.field_id.astype(str), 10000, 27260910)
    counts = w @ np.bincount(inv, minlength=len(fields))
    boot = {}
    for method, pred in methods.items():
        boot[method] = component_bootstrap(y, pred, inv, w)
        boot[method]['rmse'] = np.sqrt((w @ np.bincount(inv, weights=(pred-y)**2,
                                                      minlength=len(fields))) / counts)
    def interval(values):
        return [float(v) for v in np.quantile(values, [0.025, 0.975])]
    bs = boot[RESET_METHOD]
    metric_row = {'method': RESET_METHOD, 'n': len(y), **points[RESET_METHOD],
                  'centered_skill_probability_above_zero': float(np.mean(bs['skill'] > 0))}
    for metric, score in [('centered_skill', 'skill'), ('bias_db', 'bias'), ('rmse_db', 'rmse')]:
        metric_row[metric + '_ci_low'], metric_row[metric + '_ci_high'] = interval(bs[score])
    paired = []
    for first in ['source_finetuned_surrogate', 'pretraining_only_surrogate']:
        row = {'comparison_type': 'source_adam_reset_control', 'first_stage': first,
               'second_stage': RESET_METHOD}
        for point_name, score, delta_name, low_name, high_name in [
                ('centered_skill', 'skill', 'centered_skill_delta_second_minus_first', 'centered_skill_delta_ci_low', 'centered_skill_delta_ci_high'),
                ('bias_db', 'bias', 'bias_delta_second_minus_first_db', 'bias_delta_ci_low_db', 'bias_delta_ci_high_db'),
                ('rmse_db', 'rmse', 'rmse_delta_second_minus_first_db', 'rmse_delta_ci_low_db', 'rmse_delta_ci_high_db')]:
            row[delta_name] = points[RESET_METHOD][point_name] - points[first][point_name]
            row[low_name], row[high_name] = interval(bs[score] - boot[first][score])
        paired.append(row)
    member_metrics = []
    for repeat in range(1, 6):
        current = members.loc[members['repeat'].eq(repeat)]
        for cohort, truth in [('shared_138', stage), ('full_189', target)]:
            joined = truth[KEY + YCOL].merge(current, on=KEY, how='left', validate='one_to_one')
            yd = (joined[YCOL[1]] - joined[YCOL[0]]).to_numpy(float)
            for method, col in [('pretraining_only_surrogate', 'd_pretraining'),
                                ('source_finetuned_surrogate', 'd_source'), (RESET_METHOD, column)]:
                values = joined[col].to_numpy(float)
                metrics = component_metrics(yd, values)
                np.testing.assert_allclose(metrics['rmse_db']**2,
                    metrics['bias_db']**2 + metrics['centered_rmse_db']**2, atol=1e-12, rtol=0)
                member_metrics.append({'repeat': repeat, 'seed': MEMBER_SEEDS[repeat-1],
                    'cohort': cohort, 'method': method, 'n': len(joined), **metrics,
                    'mean_hh_vv_rmse_db': rmse_avg(joined[YCOL],
                        np.column_stack([c_source-values/2, c_source+values/2]))})
    full_y = (new_zero[YCOL[1]]-new_zero[YCOL[0]]).to_numpy(float)
    full_p = (new_zero['vv_' + RESET_METHOD]-new_zero['hh_' + RESET_METHOD]).to_numpy(float)
    full = {'rows': len(zero), 'fields': int(zero.field_id.nunique()), **component_metrics(full_y, full_p),
            'mean_hh_vv_rmse_db': rmse_avg(zero[YCOL], new_zero[['hh_' + RESET_METHOD, 'vv_' + RESET_METHOD]])}
    record = {
        'status': 'COMPLETE', 'analysis_role': 'post-hoc source-stage optimizer-state control',
        'training_performed': True, 'training_runs': 5, 'epochs_per_member': 120,
        'source_rows': len(source), 'source_fields': int(source.field_id.nunique()),
        'architecture': [4, 16, 8, 1], 'activation': 'tanh', 'alpha': 0.001,
        'optimizer': {'name': 'Adam', 'learning_rate_init': 0.003, 'beta_1': 0.9,
                      'beta_2': 0.999, 'epsilon': 1e-8, 'reset': 'first/second moments and update clock'},
        'batch_size': 'auto', 'source_minibatch_sizes': [200, 40], 'shuffle': False,
        'endpoint': 'epoch 120', 'feature_rule': 'Topp on both campaigns; NPZ standardization unchanged',
        'common_source_mean_db': c_source, 'target_labels_used_for_training_or_selection': False,
        'original_primary_comparison': 'unchanged',
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        **{p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl']},
                        'training_threads': 1, 'dtype': 'float64'},
        'members': run_records, 'member_metrics': member_metrics,
        'shared_138': metric_row, 'full_189': full, 'paired_contrasts': paired,
        'bootstrap': {'unit': 'field_id', 'iterations': 10000, 'seed': 27260910,
                      'field_order': 'string sorted', 'fields': len(fields), 'models_fixed': True},
        'elapsed_seconds': time.monotonic() - started}

    # Extend existing artifacts only, retaining all historical columns and arrays.
    for path, arrays in trained_arrays:
        with np.load(path, allow_pickle=False) as saved:
            original = {k: saved[k].copy() for k in saved.files}
        original.update(arrays)
        np.savez_compressed(path, **original)
    save(members, 'trajectory_reconstruction/predictions_by_member.csv')
    save(attach(read('trajectory_reconstruction/ensemble_comparison.csv'), ensemble, [column]),
         'trajectory_reconstruction/ensemble_comparison.csv')
    save(new_stage, 'stage_diagnostics/stage_predictions.csv')
    save(new_zero, 'cross_domain/zero_shot_predictions.csv')
    extend_rows('stage_diagnostics/stage_metrics.csv', [metric_row], 'method', RESET_METHOD)
    extend_rows('stage_diagnostics/stage_pairwise_deltas.csv', paired, 'comparison_type', 'source_adam_reset_control')
    channel_mse = np.mean((new_zero[['hh_' + RESET_METHOD, 'vv_' + RESET_METHOD]].to_numpy(float)
                          - zero[YCOL].to_numpy(float))**2, axis=0)
    extend_rows('cross_domain/zero_shot_joint_channel_metrics.csv', [{
        'method': RESET_METHOD, 'mean_hh_vv_rmse_db': float(np.sqrt(channel_mse).mean()),
        'pooled_hh_vv_rmse_db': float(np.sqrt(channel_mse.mean())),
        'hh_rmse_db': float(np.sqrt(channel_mse[0])), 'vv_rmse_db': float(np.sqrt(channel_mse[1]))}],
        'method', RESET_METHOD)
    for rel in ['trajectory_reconstruction/summary.json', 'stage_diagnostics/manifest.json',
                'cross_domain/zero_shot_summary.json', 'paper_summaries/phase3_summary.json']:
        path = result / rel
        content = json.loads(path.read_text())
        content[RESET_METHOD] = record
        # The saved networks gained keys; keep the existing offset-input hashes current.
        if 'source_offset_inputs' in content:
            for key in content['source_offset_inputs']:
                if key.endswith('.npz'):
                    content['source_offset_inputs'][key] = hashlib.sha256((root/key).read_bytes()).hexdigest()
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'shared_138': metric_row, 'full_189': full, 'paired_contrasts': paired}, indent=2), flush=True)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=ROOT)
    ap.add_argument('--output',type=Path,help='New output directory for full trajectory reconstruction')
    ap.add_argument('--reset-source-adam', action='store_true',
                    help='Train the fixed five-member source reset control and extend existing result files')
    ap.add_argument('--atol',type=float,default=1e-8)
    a=ap.parse_args();root=a.project_root.resolve()
    if a.reset_source_adam:
        if a.output is not None:
            ap.error('--reset-source-adam updates the existing reproducibility files; omit --output')
        run_source_adam_reset(root, a.atol)
        return
    if a.output is None:
        ap.error('--output is required for full trajectory reconstruction')
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False); data=root/'reproducibility';start=time.monotonic()
    source=audit.prepare_table(data/'data/source/smapvex12_portable_source.csv','SMAPVEX12','topp_both')
    source=source.loc[source.finite_model_row].sort_values(['field_id','acquisition_date']).reset_index(drop=True)
    target=audit.prepare_table(data/'data/target/smex02_field_day_model_ready.csv','SMEX02','topp_both')
    target=target.loc[target.finite_model_row].sort_values(['field_id','acquisition_date']).reset_index(drop=True)
    xs=source[list(base.FEATURE_NAMES)].to_numpy(float)
    ds=base.to_components(source[base.OBSERVED_COLUMNS].to_numpy(float))[:,1]
    xt=target[list(base.FEATURE_NAMES)].to_numpy(float)
    spm=pd.read_csv(data/'data/teachers/spm_pretraining.csv')
    i2em=multi.load_paired(data/'data/teachers/i2em_train_requests.csv',data/'data/teachers/i2em_train_results.csv')
    xl=multi.features_from(spm);dl=base.to_components(spm[['spm_hh_db','spm_vv_db']].to_numpy(float))[:,1]
    xh=multi.features_from(i2em);dh=base.to_components(i2em[['i2em_hh_db','i2em_vv_db']].to_numpy(float))[:,1]
    member_frames=[];metadata=[];tuning=[]
    with threadpool_limits(limits=1):
      for rep in range(5):
        seed=20260910+100000*rep
        low=base.pretrain_single_output(xl,dl,200,seed)
        seq=base.fine_tune_single(low,xh,dh,100,seed+2)
        bundle=base.SingleOutputBundle(seq,low.feature_scaler,low.target_scaler,{})
        pre=base.predict_single(bundle,seq,xt)
        src=base.fine_tune_single(bundle,xs,ds,120,seed+10102)
        pred=base.predict_single(bundle,src,xt)
        weight,rec=base.tune_differential_shrinkage(bundle,xs,ds,source.field_id.astype(str).to_numpy(),np.arange(len(source)),[0.,.1,.25,.5,.75,1.],4,120,seed+500000)
        risk=base.shrink_prediction_to_mean(pred,float(ds.mean()),weight)
        f=target[['field_id','acquisition_date']].copy();f['repeat']=rep+1
        f['d_pretraining']=pre;f['d_source']=pred;f['d_risk_reselected']=risk
        frozen_weight=[.1,.1,0.,0.,0.][rep]
        f['d_risk']=base.shrink_prediction_to_mean(pred,float(ds.mean()),frozen_weight)
        member_frames.append(f)
        metadata.append({'repeat':rep+1,'seed':seed,'selected_weight':weight,'spm':state(low.model),'i2em':state(seq),'source':state(src),'scalers_fixed':bool(np.array_equal(low.feature_scaler.mean_,bundle.feature_scaler.mean_))})
        for r in rec:r.update(repeat=rep+1,selected_weight=weight)
        tuning.extend(rec)
        weights={}
        for stage,model in [('spm',low.model),('pretraining',seq),('source',src)]:
          for i,v in enumerate(model.coefs_):weights[f'{stage}_coefs_{i}']=v
          for i,v in enumerate(model.intercepts_):weights[f'{stage}_intercepts_{i}']=v
        weights.update(x_mean=low.feature_scaler.mean_,x_scale=low.feature_scaler.scale_,y_mean=low.target_scaler.mean_,y_scale=low.target_scaler.scale_)
        np.savez_compressed(out/f'rebuilt_member_{rep+1}.npz',**weights)
        print(f'Member {rep+1}/5 complete; lambda={weight}; elapsed={time.monotonic()-start:.1f}s',flush=True)
    members=pd.concat(member_frames,ignore_index=True)
    members.acquisition_date=pd.to_datetime(members.acquisition_date).dt.strftime('%Y-%m-%d')
    members.to_csv(out/'predictions_by_member.csv',index=False)
    ens=members.groupby(['field_id','acquisition_date'],as_index=False)[['d_pretraining','d_source','d_risk','d_risk_reselected']].mean()
    frozen=pd.read_csv(data/'results/cross_domain/zero_shot_predictions.csv',dtype={'field_id':str})
    stage=pd.read_csv(data/'results/stage_diagnostics/stage_predictions.csv',dtype={'field_id':str})
    joined=ens.merge(frozen,on=['field_id','acquisition_date'],validate='one_to_one');checks=[]
    for m,prefix in [('d_source','spm_to_i2em'),('d_risk','risk_spm_to_i2em'),('d_risk_reselected','risk_spm_to_i2em')]:
      diff=joined[m].to_numpy()-(joined[f'vv_{prefix}']-joined[f'hh_{prefix}']).to_numpy()
      checks.append({'comparison':m,'rows':len(diff),'max_abs_difference_db':float(abs(diff).max()),'rmse_difference_db':float(np.sqrt(np.mean(diff**2))),'pass_atol':bool(np.all(abs(diff)<=a.atol))})
    prejoin=ens.merge(stage,on=['field_id','acquisition_date'],validate='one_to_one')
    diff=prejoin.d_pretraining-prejoin.differential_pretraining_only
    checks.append({'comparison':'d_pretraining','rows':len(diff),'max_abs_difference_db':float(abs(diff).max()),'rmse_difference_db':float(np.sqrt(np.mean(diff**2))),'pass_atol':bool(np.all(abs(diff)<=a.atol))})
    pd.DataFrame(checks).to_csv(out/'prediction_equivalence.csv',index=False)
    joined.to_csv(out/'ensemble_comparison.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'source_only_shrinkage_tuning.csv',index=False)
    used=[data/'data/source/smapvex12_portable_source.csv',data/'data/target/smex02_field_day_model_ready.csv',data/'data/teachers/spm_pretraining.csv',data/'data/teachers/i2em_train_requests.csv',data/'data/teachers/i2em_train_results.csv',root/'scripts/evaluate_decoupled_physics_heads.py',root/'src/research_pilots/scattering/surrogate/physics_pretraining.py']
    summary={'status':'COMPLETE','role':'post-hoc trajectory reconstruction; no target selection','risk_comparisons':'d_risk uses archived coefficients [0.1,0.1,0,0,0]; d_risk_reselected reports independently reselected coefficients without replacing archived results','atol_db':a.atol,'checks':checks,'frozen_coefficient_predictions_reproduced':all(c['pass_atol'] for c in checks if c['comparison']!='d_risk_reselected'),'selected_weights_reproduced':[x['selected_weight'] for x in metadata]==[.1,.1,0.,0.,0.],'original_windows_environment_recreated':False,'environment':{'python':platform.python_version(),'platform':platform.platform(),**{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','threadpoolctl']}},'elapsed_seconds':time.monotonic()-start,'members':metadata,'inputs':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in used}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['checks','selected_weights_reproduced','elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
