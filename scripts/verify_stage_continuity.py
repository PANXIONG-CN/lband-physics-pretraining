#!/usr/bin/env python3
"""One fixed five-seed trajectory check using the uploaded training functions.
No target labels are used for fitting or selection. No hyperparameter search
is added. The source-only shrinkage selection is the historical four-fold rule.
"""
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=ROOT)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--atol',type=float,default=1e-8)
    a=ap.parse_args();root=a.project_root.resolve();out=a.output.resolve()
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
    stage=pd.read_csv(data/'results/advisor_final/stage_retention/stage_predictions.csv',dtype={'field_id':str})
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
