"""Paired post-hoc input-contract sensitivity; frozen research code is read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo', default='D:/research-pilots')
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    repo=Path(args.repo); out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    sys.path.insert(0,str(repo/'scripts')); sys.path.insert(0,str(repo/'src'))
    import evaluate_smex02_preregistered_few_shot as ev
    root=repo/'outputs/scattering/rough_ground'
    manifest=json.loads((root/'smex02_prospective_freeze_20260911_v1/manifest.json').read_text())
    files=manifest['files']; checks={k:digest(v['path'])==v['sha256'] for k,v in files.items()}
    if not all(checks.values()): raise AssertionError(checks)
    tables={}
    audit={}
    for role,key in [('SMAPVEX12','source_table'),('SMEX02','external_table')]:
        raw=ev.audit.prepare_table(Path(files[key]['path']),role,'measured')
        unified=ev.audit.prepare_table(Path(files[key]['path']),role,'topp_both')
        if not np.array_equal(raw.finite_model_row,unified.finite_model_row):
            raise AssertionError('Policies change sample membership; stop rather than change frozen cohort')
        tables[role]={}
        for name,frame in [('original',raw),('unified',unified)]:
            tables[role][name]=frame.loc[frame.finite_model_row].sort_values(['field_id','acquisition_date']).reset_index(drop=True)
        delta=tables[role]['original'].soil_real_dielectric-tables[role]['unified'].soil_real_dielectric
        audit[role]={'rows':len(delta),'fields':int(tables[role]['original'].field_id.nunique()),'max_abs_epsilon_change':float(np.max(abs(delta))),'mean_abs_epsilon_change':float(np.mean(abs(delta)))}
    spm=pd.read_csv(files['spm_teacher_samples']['path'])
    i2em=ev.multi.load_paired(Path(files['i2em_requests']['path']),Path(files['i2em_results']['path']))
    methods=ev.METHODS+['two_channel_means']
    seed=20260910; fractions=[.05,.10,.20]; records=[]; tuning={}; assignments=[]
    out.mkdir(parents=True)
    protocol={'status':'RUNNING','classification':'post-hoc sensitivity, not prospective validation','frozen_hash_checks':checks,'input_audit':audit,'arms':{'original':'Retain original table dielectric: source probe-derived, target moisture-derived','unified':'Recompute both real-domain dielectric inputs using the existing frozen cubic'},'teachers':'Unchanged in both arms; this tests real-input provenance, not teacher regeneration','budget':{'members':5,'splits':20,'spm_epochs':200,'i2em_epochs':100,'source_epochs':120,'adapt_epochs':30,'inner_folds':4,'seed':seed},'code_hashes':{str(p):digest(p) for p in (repo/'scripts').glob('*.py')}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for arm in ['unified','original']:
        print('Starting '+arm,flush=True)
        source=tables['SMAPVEX12'][arm]; target=tables['SMEX02'][arm]
        members,tune=ev.source_ensemble(source,spm,i2em,5,[0,.1,.25,.5,.75,1],200,100,120,4,seed)
        tuning[arm]=tune.to_dict('records')
        fields=np.sort(target.field_id.astype(str).unique())
        for repeat in range(1,21):
            shuffled=np.random.default_rng(seed+repeat*100000).permutation(fields)
            for fraction in fractions:
                count=int(np.ceil(fraction*len(fields))); selected=set(shuffled[:count])
                mask=target.field_id.astype(str).isin(selected)
                adaptation=target.loc[mask].reset_index(drop=True); test=target.loc[~mask].reset_index(drop=True)
                predicted=ev.ensemble([ev.adapt_member(m,adaptation,test,30,seed+repeat*1000000+count*10000+int(m['repeat'])*100) for m in members])
                predicted['two_channel_means']=np.tile(adaptation[ev.base.OBSERVED_COLUMNS].mean().to_numpy(),(len(test),1))
                y=test[ev.base.OBSERVED_COLUMNS].to_numpy(float)
                for i,row in enumerate(test.itertuples()):
                    records.append({'arm':arm,'split':repeat,'fraction':fraction,'field':str(row.field_id),'date':str(row.acquisition_date),'observed':y[i].tolist(),'predicted':{m:predicted[m][i].tolist() for m in methods}})
                assignments.append({'arm':arm,'split':repeat,'fraction':fraction,'adaptation_fields':sorted(selected)})
            print(f'{arm}: split {repeat}/20',flush=True)
        (out/f'{arm}_predictions.json').write_text(json.dumps([r for r in records if r['arm']==arm]),encoding='utf-8')
    # Paired point estimates; centering is diagnostic only and never changes predictions.
    scores=[]
    for arm in ['unified','original']:
        for fraction in fractions:
            for method in methods:
                vals=[]
                for repeat in range(1,21):
                    rr=[r for r in records if r['arm']==arm and r['fraction']==fraction and r['split']==repeat]
                    y=np.array([r['observed'] for r in rr]); p=np.array([r['predicted'][method] for r in rr])
                    yc=ev.base.to_components(y); pc=ev.base.to_components(p)
                    ec=(pc-pc.mean(0))-(yc-yc.mean(0))
                    vals.append([np.sqrt(np.mean((p-y)**2,axis=0)).mean(),*np.sqrt(np.mean(ec**2,axis=0)),*(1-np.mean(ec**2,axis=0)/np.var(yc,axis=0)),*(np.var(pc,axis=0)/np.var(yc,axis=0))])
                names=['rmse_mean_hh_vv_db','centered_rmse_common','centered_rmse_differential','centered_skill_common','centered_skill_differential','variance_ratio_common','variance_ratio_differential']
                scores.append(dict(arm=arm,fraction=fraction,method=method,**dict(zip(names,np.mean(vals,axis=0).tolist()))))
    # Resample each field once per bootstrap replicate, sharing its weight across ALL splits and both arms.
    fields=sorted({r['field'] for r in records}); fi={f:i for i,f in enumerate(fields)}
    weights=np.random.default_rng(seed+7000000).multinomial(len(fields),np.ones(len(fields))/len(fields),4000)
    intervals=[]
    for fraction in fractions:
        values={}
        for arm in ['unified','original']:
            rr=[r for r in records if r['arm']==arm and r['fraction']==fraction]
            counts=np.zeros((20,len(fields))); errors={m:np.zeros((20,len(fields),2)) for m in methods}
            for r in rr:
                s=r['split']-1; f=fi[r['field']]; counts[s,f]+=1
                for m in methods: errors[m][s,f]+=(np.array(r['predicted'][m])-r['observed'])**2
            denom=weights@counts.T
            if (denom==0).any(): raise AssertionError('Empty bootstrap test split')
            for m in methods:
                values[arm,m]=np.sqrt(np.einsum('bf,sfc->bsc',weights,errors[m])/denom[:,:,None]).mean(axis=(1,2))
        comparisons=[('original',m,'unified',m) for m in methods]+[(a,'spm_to_i2em',a,m) for a in ['unified','original'] for m in ['spm_only','two_channel_means']]
        for a,m,b,n in comparisons:
            d=values[a,m]-values[b,n]
            intervals.append({'fraction':fraction,'comparison':f'{a}/{m} minus {b}/{n}','ci95':np.quantile(d,[.025,.975]).tolist(),'interpretation':'conditional shared-field bootstrap; not across new campaigns'})
    # Confirm same assignments and exact reproduction of frozen baseline point estimates.
    aa=[{k:v for k,v in x.items() if k!='arm'} for x in assignments if x['arm']=='unified']
    bb=[{k:v for k,v in x.items() if k!='arm'} for x in assignments if x['arm']=='original']
    assert aa==bb
    frozen=json.loads((root/'smex02_preregistered_few_shot_20260911_v1/summary.json').read_text())
    differences=[]
    for row in frozen['mean_hh_vv_rmse_db']:
        now=next(s for s in scores if s['arm']=='unified' and s['fraction']==row['requested_fraction'] and s['method']==row['method'])
        differences.append(abs(now['rmse_mean_hh_vv_db']-row['rmse_db']))
    protocol.update(status='COMPLETE',same_assignments=True,unified_frozen_max_rmse_difference=max(differences))
    for name,obj in [('protocol',protocol),('scores',scores),('paired_intervals',intervals),('source_tuning',tuning),('assignments',assignments)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm,style in [('unified','-'),('original','--')]:
        for m in ['spm_only','spm_to_i2em','two_channel_means']:
            rr=[s for s in scores if s['arm']==arm and s['method']==m]
            for ax,key in zip(axes,['rmse_mean_hh_vv_db','centered_skill_differential']):
                ax.plot([6.67,10,20],[s[key] for s in rr],style,marker='o',label=arm+' / '+m)
    axes[0].set_ylabel('Mean HH/VV RMSE (dB)');axes[1].set_ylabel('Centered differential skill');axes[1].axhline(0,color='grey',linewidth=.7)
    for ax in axes: ax.set_xlabel('Whole-field adaptation (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'sensitivity.png',dpi=180);plt.close(fig)
    lines=['# Dielectric input-contract sensitivity','', 'Post-hoc audit; no model selection on SMEX02.','', 'The frozen baseline ALREADY used the same moisture-to-dielectric cubic in both campaigns. Earlier raw-table provenance statements must not be presented as the actual model input contract.','',f'Baseline reproduction maximum RMSE difference: {max(differences):.12g} dB.','', '|Arm|Fraction|Method|RMSE dB|Centered differential skill|','|---|---:|---|---:|---:|']
    for s in scores:
        if s['method'] in ['spm_only','spm_to_i2em','two_channel_means']: lines.append(f"|{s['arm']}|{s['fraction']}|{s['method']}|{s['rmse_mean_hh_vv_db']:.6f}|{s['centered_skill_differential']:.6f}|")
    lines+=['','This isolates the real-input contract with teachers held fixed. It does not establish which dielectric estimate is physically more accurate. Bootstrap intervals preserve repeated-field dependence but are conditional on these campaigns and fitted members. Existing prospective results remain unchanged.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(protocol['input_audit']),flush=True)
    print(f'COMPLETE: {out}',flush=True)

if __name__=='__main__': main()
