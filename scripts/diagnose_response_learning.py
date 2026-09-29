"""Fixed-rule post-hoc response diagnostics; never select models on target scores."""
from pathlib import Path
import sys,json,warnings,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
parser=argparse.ArgumentParser()
parser.add_argument('--project-root',type=Path,default=Path('D:/research-pilots'))
parser.add_argument('--output',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.project_root/'src'))
from research_pilots.scattering.surfaces.spm import spm_backscatter_db
B=args.project_root/'outputs/scattering/rough_ground'
O=args.output or B/'paper_submission_closeout_20260911_v1/response_learning_v1';O.mkdir(parents=True,exist_ok=True)
F=['soil_moisture_m3_m3','soil_real_dielectric','pals_rms_height_cm','pals_correlation_length_cm'];Y=['sigma0_hh_db','sigma0_vv_db']
s=pd.read_csv(B/'portable_source_v1/smapvex12_portable_source.csv')
t=pd.read_csv(B/'smex02_external_final_20260911_v1/smex02_field_day_model_ready.csv')
p=pd.read_csv(B/'smex02_preregistered_few_shot_20260911_v1/heldout_predictions.csv')
def comp(y):return np.column_stack([y.mean(axis=1),y[:,1]-y[:,0]])
def measures(y,z):
 yc=y-y.mean();zc=z-z.mean();v=float(np.mean(yc**2));pv=float(np.mean(zc**2))
 return {'rmse':float(np.sqrt(np.mean((z-y)**2))),'centered_rmse':float(np.sqrt(np.mean((zc-yc)**2))),
 'observed_variance':v,'predicted_variance':pv,'variance_ratio':pv/v if v>0 else None,
 'centered_skill':float(1-np.mean((zc-yc)**2)/v) if v>0 else None}
rows=[]
for (f,r),g in p.groupby(['requested_fraction','split_repeat']):
 ids=g.adaptation_fields.iloc[0].split(',');a=t[t.field_id.isin(ids)]
 assert not set(ids)&set(g.field_id)
 ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(a[F],a[Y])
 preds={m:g[['hh_'+m,'vv_'+m]].to_numpy() for m in ['scratch','spm_only','spm_to_i2em','risk_spm_to_i2em']}
 preds['target_ridge_fixed_alpha1']=ridge.predict(g.merge(t[['field_id','acquisition_date']+F],on=['field_id','acquisition_date'],validate='many_to_one')[F])
 preds['two_mean']=np.tile(a[Y].mean().to_numpy(),(len(g),1))
 obs=comp(g[Y].to_numpy())
 for m,z in preds.items():
  z=comp(z)
  for j,c in enumerate(['common','differential']):rows.append({'fraction':f,'split':r,'method':m,'component':c,**measures(obs[:,j],z[:,j])})
df=pd.DataFrame(rows);df.to_csv(O/'response_metrics.csv',index=False)
agg=df.groupby(['fraction','method','component']).mean(numeric_only=True).drop(columns='split');agg.to_csv(O/'response_summary.csv')
# Source-only equal-capacity coordinate-control experiment (not a pretrained-head ablation).
# Orthonormal rotation preserves squared-error geometry. One scalar target scale in both arms.
Q=np.array([[1,-1],[1,1]])/np.sqrt(2);assert np.allclose(Q.T@Q,np.eye(2))
sr=[];warn=[]
for fold,(tr,te) in enumerate(GroupKFold(5).split(s,s.field_id,groups=s.field_id)):
 assert not set(s.iloc[tr].field_id)&set(s.iloc[te].field_id)
 xs=StandardScaler().fit(s.iloc[tr][F]);xt=xs.transform(s.iloc[tr][F]);xv=xs.transform(s.iloc[te][F])
 yt=s.iloc[tr][Y].to_numpy();yv=s.iloc[te][Y].to_numpy();mu=yt.mean(axis=0);scale=float(np.std(yt-mu));yt=(yt-mu)/scale
 for seed in [11,22,33]:
  for name,rot in [('direct_hh_vv',np.eye(2)),('orthonormal_components',Q)]:
   model=MLPRegressor(hidden_layer_sizes=(32,32),max_iter=100,early_stopping=False,tol=0,n_iter_no_change=101,batch_size=32,random_state=seed,alpha=.0001)
   with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always');model.fit(xt,yt@rot);warn.extend(str(x.message) for x in w)
   z=model.predict(xv)@rot.T*scale+mu
   sr.append({'fold':fold,'seed':seed,'method':name,'epochs':int(model.n_iter_),
    'parameters':sum(x.size for x in model.coefs_)+sum(x.size for x in model.intercepts_),
    'mean_channel_rmse':float(np.sqrt(np.mean((z-yv)**2,axis=0)).mean())})
pd.DataFrame(sr).to_csv(O/'source_coordinate_control.csv',index=False)
# Source-only fixed-parameter SPM scan: k*s <= 0.3 conservatively restricts height.
k=2*np.pi*1.26e9/299792458
valid=s[(k*s.pals_rms_height_cm/100<=.3)&(s.pals_rms_height_cm>0)&(s.pals_correlation_length_cm>0)].copy()
assert len(valid)>0
scan=[]
for loss in [0,.02,.05,.1]:
 for angle in [35,40,42.5,45,50]:
  z=spm_backscatter_db(valid.soil_real_dielectric.to_numpy()*(1-1j*loss),valid.pals_rms_height_cm.to_numpy()/100,valid.pals_correlation_length_cm.to_numpy()/100,1.26e9,angle,spectrum_model='exponential')
  c=comp(np.column_stack([z['hh_db'],z['vv_db']]))
  assert np.isfinite(c).all()
  for row,values in enumerate(c):scan.append({'source_row':int(valid.index[row]),'angle':angle,'loss_tangent':loss,'common':values[0],'differential':values[1]})
scan=pd.DataFrame(scan);scan.to_csv(O/'spm_controlled_angle_scan.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for j,c in enumerate(['common','differential']):
 for loss,g in scan.groupby('loss_tangent'):
  curve=g.groupby('angle')[c].mean();axes[j].plot(curve.index,curve-curve.loc[40],'-o',label=str(loss))
 axes[j].set(xlabel='Angle (degrees)',ylabel='Change from 40 degrees (dB)',title=c)
axes[1].legend(title='Loss tangent');fig.savefig(O/'spm_angle_response.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
for m in df.method.unique():
 g=df[(df.method==m)&(df.component=='differential')].groupby('fraction').centered_skill.mean();ax.plot(g.index*100,g,'-o',label=m)
ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Nominal adaptation percent (5 = actual 6.67)',ylabel='Centered differential skill vs constant',title='Post-hoc, test-mean removal for diagnostics only');ax.legend(fontsize=7)
fig.savefig(O/'centered_skill.png',dpi=180);plt.close(fig)
summary={'status':'post_hoc_fixed_rules','source_coordinate_control':pd.DataFrame(sr).groupby('method').mean(numeric_only=True).to_dict(),
 'scan_rows':len(valid),'scan_points':len(scan),'training_warning_count':len(warn),'warning_examples':list(set(warn))[:3],
 'limitations':['Centering uses held-out means only for scoring, not deployable correction.','Source MLP coordinate control is not equivalent to existing decoupled pretrained heads.','SPM scan is source-only sensitivity, not I2EM or measured angle correction.','No target model selection. Ridge alpha fixed at 1 without tuning.']}
(O/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2));print(agg[['variance_ratio','centered_skill','rmse']].to_string())
