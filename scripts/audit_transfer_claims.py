"""Post-hoc diagnostics; frozen observations and predictions are read-only."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, default=Path('D:/research-pilots'))
parser.add_argument('--output', type=Path)
args = parser.parse_args()
BASE = args.project_root / 'outputs/scattering/rough_ground'
OUT = args.output or BASE / 'paper_submission_closeout_20260911_v1/paper_audit_supplement_v1'
OUT.mkdir(parents=True, exist_ok=True)
paths = {
 'predictions': BASE/'smex02_preregistered_few_shot_20260911_v1/heldout_predictions.csv',
 'target': BASE/'smex02_external_final_20260911_v1/smex02_field_day_model_ready.csv',
 'source': BASE/'portable_source_v1/smapvex12_portable_source.csv'}
p,t,s = [pd.read_csv(paths[k]) for k in ['predictions','target','source']]
Y=['sigma0_hh_db','sigma0_vv_db']
assert not t.duplicated(['field_id','acquisition_date']).any()
fields=sorted(t.field_id.unique()); fi={f:i for i,f in enumerate(fields)}
methods=['common_offset_source_mean','scratch','spm_only','spm_to_i2em','risk_spm_to_i2em','two_mean','two_mean_field_weighted']
rng=np.random.default_rng(20260911)
# One shared field multiplicity per draw, applied to every saved split.
weights=rng.multinomial(len(fields),np.ones(len(fields))/len(fields),size=4000)
results=[];checks=0
for fraction,part in p.groupby('requested_fraction'):
 counts=[];errors={m:[] for m in methods}; scores={m:[] for m in methods}
 for sid,g in part.groupby('split_repeat'):
  ids=set(g.adaptation_fields.iloc[0].split(','))
  assert g.adaptation_fields.nunique()==1 and not ids.intersection(g.field_id)
  assert ids | set(g.field_id)==set(fields)
  a=t[t.field_id.isin(ids)]
  assert len(ids)==round(float(g.actual_fraction.iloc[0])*len(fields))
  # Target-only constants use adaptation labels only; no held-out calibration.
  means={'two_mean':a[Y].mean().to_numpy(),
         'two_mean_field_weighted':a.groupby('field_id')[Y].mean().mean().to_numpy()}
  obs=g[Y].to_numpy(); assert np.isfinite(obs).all()
  verify=g.merge(t[['field_id','acquisition_date']+Y],on=['field_id','acquisition_date'],suffixes=('','_reference'),validate='many_to_one')
  assert np.allclose(verify[Y],verify[[x+'_reference' for x in Y]])
  n=np.zeros(len(fields)); idx=np.array([fi[f] for f in g.field_id]); np.add.at(n,idx,1)
  counts.append(n)
  for m in methods:
   pred=np.broadcast_to(means[m],obs.shape) if m in means else g[['hh_'+m,'vv_'+m]].to_numpy()
   assert np.isfinite(pred).all()
   sq=(pred-obs)**2; e=np.zeros((len(fields),2));np.add.at(e,idx,sq)
   errors[m].append(e);scores[m].append(float(np.sqrt(sq.mean(axis=0)).mean()))
  checks+=1
 counts=np.array(counts);den=weights@counts.T
 assert (den>0).all()
 boot={}
 for m in methods:
  e=np.array(errors[m])
  boot[m]=np.sqrt(np.einsum('bf,sfc->bsc',weights,e)/den[:,:,None]).mean(axis=(1,2))
  assert np.isclose(np.sqrt(e.sum(axis=1)/counts.sum(axis=1)[:,None]).mean(),np.mean(scores[m]))
 contrasts={}
 for first,second in [('spm_only','scratch'),('spm_only','two_mean'),('spm_to_i2em','spm_only'),('two_mean','common_offset_source_mean')]:
  d=boot[first]-boot[second]
  contrasts[first+'_minus_'+second]={'delta_db':float(np.mean(scores[first])-np.mean(scores[second])),
   'conditional_field_reweighting_interval':np.quantile(d,[.025,.975]).tolist()}
 results.append({'nominal_fraction':float(fraction),'actual_fraction':float(part.actual_fraction.iloc[0]),
 'rmse_db':{m:float(np.mean(scores[m])) for m in methods},'contrasts':contrasts})
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for m in ['common_offset_source_mean','two_mean','two_mean_field_weighted','scratch','spm_only','spm_to_i2em']:
 ax.plot([r['actual_fraction']*100 for r in results],[r['rmse_db'][m] for r in results],'-o',label=m)
ax.set(xlabel='Adaptation fields (%)',ylabel='Mean HH/VV RMSE (dB)',title='Post-hoc calibration controls; identical held-out splits')
ax.legend(fontsize=8);fig.savefig(OUT/'calibration_controls.png',dpi=180);plt.close(fig)
domain={};fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for name,df,acol in [('SMAPVEX12',s,'nominal_incidence_angle_deg'),('SMEX02',t,'incidence_angle_deg')]:
 common=df[Y].mean(axis=1);diff=df[Y[1]]-df[Y[0]];angle=df[acol]
 domain[name]={'n_rows':len(df),'n_fields':df.field_id.nunique(),'angle_definition':acol,
 'angle_min_median_max':angle.quantile([0,.5,1]).tolist(),'angle_missing':int(angle.isna().sum()),
 'common_mean_db':float(common.mean()),'vv_minus_hh_mean_db':float(diff.mean())}
 for ax,vals,label in zip(axes,[angle,common,diff],['Angle (degrees)','Common (dB)','VV - HH (dB)']):
  ax.hist(vals.dropna(),bins=15,alpha=.5,label=name);ax.set_xlabel(label);ax.set_ylabel('Field-day count')
axes[0].set_title('Source nominal / target measured');axes[1].legend(fontsize=8)
fig.savefig(OUT/'campaign_diagnostics.png',dpi=180);plt.close(fig)
summary={'analysis_type':'POST_HOC; not model selection or new prospective validation',
 'inference_limit':'Intervals reweight shared held-out field identities across all fixed splits. No refitting, no campaign resampling, no unconditional coverage claim. Repeated training dependence not fully propagated.',
 'checks_passed':checks,'bootstrap_draws':4000,'results':results,'domains':domain,
 'input_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
