"""Approved post-hoc E5 moisture-only OLS and E6 fixed-prediction uncertainty.

Run --freeze before --run, separately for each experiment. No manuscript writes.
Only E5 fits a new model. E6 conditions on saved predictions and calibration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd

KEY = ["field_id", "acquisition_date"]
SEED = 20261020
EPS = 1e-12
METHODS = ["source_training_mean", "ordinary_least_squares", "source_selected_ridge",
           "E1_selected_neural", "frozen_pretraining_neural",
           "frozen_source_inherited_neural", "frozen_source_reset_neural",
           "pretraining_source_intercept"]
QUANTITIES = ["bias_db", "rmse_db", "centered_mse_db2", "centered_skill",
              "correlation", "within_skill", "between_skill",
              "observed_variance_db2", "within_observed_variance_db2"]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    f = pd.read_csv(p, dtype={"field_id": str})
    f["field_id"] = f.field_id.str.strip()
    f["acquisition_date"] = pd.to_datetime(f.acquisition_date).dt.strftime("%Y-%m-%d")
    assert not f.duplicated(KEY).any(), f"Duplicate keys: {p}"
    return f.sort_values(KEY).reset_index(drop=True)

def paths(root, experiment):
    r = root / "reproducibility"
    p = {
        "target": r / "data/target/smex02_field_day_model_ready.csv",
        "stage": r / "results/stage_diagnostics/stage_predictions.csv",
        "E3_target": root / "outputs/review_E3_source_baselines/target_predictions.csv",
    }
    if experiment == "E5":
        p.update(source=r / "data/source/smapvex12_portable_source.csv",
                 splits=root / "outputs/review_E1_source_optimization/source_splits.csv",
                 E3_source_metrics=root / "outputs/review_E3_source_baselines/source_cv_metrics.csv")
    return p

def target_frame(p):
    target, pred, stage = read(p["target"]), read(p["E3_target"]), read(p["stage"])
    assert len(target) == 189 and target.field_id.nunique() == 30
    assert target.model_ready.astype(str).str.lower().eq("true").all()
    assert target[KEY].equals(pred[KEY])
    y = (target.sigma0_vv_db - target.sigma0_hh_db).to_numpy()
    assert np.allclose(y, pred.observed_diff_db, atol=1e-10, rtol=0)
    common = pred.common_138.astype(str).str.lower().eq("true").to_numpy()
    assert len(stage) == common.sum() == 138 and stage.field_id.nunique() == 22
    assert pred.loc[common, KEY].reset_index(drop=True).equals(stage[KEY])
    assert np.allclose(y[common], stage.sigma0_vv_db-stage.sigma0_hh_db, atol=1e-10, rtol=0)
    mapping = {"frozen_pretraining_neural": "differential_pretraining_only_surrogate",
               "frozen_source_inherited_neural": "differential_source_finetuned_surrogate",
               "frozen_source_reset_neural": "differential_source_reset_adam_003"}
    for a, b in mapping.items():
        assert np.allclose(pred.loc[common, a], stage[b], atol=1e-9, rtol=0), (a, b)
    offsets = (stage.differential_pretraining_source_offset -
               stage.differential_pretraining_only_surrogate).to_numpy()
    assert np.ptp(offsets) < 1e-10
    # Reuse the archived scalar, never re-estimate it from target observations.
    pred["pretraining_source_intercept"] = pred.frozen_pretraining_neural + offsets[0]
    pred["soil_moisture_m3_m3"] = target.soil_moisture_m3_m3
    pred["common_138"] = common
    assert np.isfinite(pred[METHODS + ["observed_diff_db", "soil_moisture_m3_m3"]]).all().all()
    return pred, float(offsets[0])

def divide(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan),
                     where=np.asarray(b) > EPS)

def stats(y, pred, w, field_index):
    """Rows of w are resampling weights; centers recomputed in every draw."""
    total = w.sum(axis=1)
    ym = divide(w @ y, total)
    pm = divide(w @ pred, total)
    residual = pred - y
    bias = divide(w @ residual, total)
    mse = divide(w @ residual**2, total)
    vy = np.maximum(0, divide(w @ y**2, total)-ym**2)
    vp = np.maximum(0, divide(w @ pred**2, total)-pm**2)
    cmse = np.maximum(0, mse-bias**2)
    cov = divide(w @ (y*pred), total)-ym*pm
    assert np.allclose(mse, bias**2+cmse, atol=1e-9, equal_nan=True)
    within_y = np.zeros(len(w))
    within_residual = np.zeros(len(w))
    for f in np.unique(field_index):
        mask = field_index == f
        wf, yf, rf = w[:, mask], y[mask], residual[mask]
        n = wf.sum(axis=1)
        # Absent fields have exactly zero within-field contribution.
        ysum, rsum = wf @ yf, wf @ rf
        yterm = np.divide(ysum**2, n, out=np.zeros(len(w)), where=n>0)
        rterm = np.divide(rsum**2, n, out=np.zeros(len(w)), where=n>0)
        within_y += np.maximum(0, wf @ yf**2-yterm)
        within_residual += np.maximum(0, wf @ rf**2-rterm)
    by = np.maximum(0, vy*total-within_y)
    br = np.maximum(0, cmse*total-within_residual)
    return dict(zip(QUANTITIES, [
        bias, np.sqrt(mse), cmse, 1-divide(cmse, vy),
        np.where((vy>EPS)&(vp>EPS),divide(cov, np.sqrt(vy*vp)),np.nan), 1-divide(within_residual, within_y),
        1-divide(br, by), vy, divide(within_y, total)]))

def counts(rng, n, draws):
    sample = rng.integers(0, n, size=(draws, n))
    c = np.zeros((draws, n), dtype=np.int16)
    np.add.at(c, (np.arange(draws)[:, None], sample), 1)
    return c

def weights(frame, scheme, draws):
    fields, fi = np.unique(frame.field_id, return_inverse=True)
    dates, di = np.unique(frame.acquisition_date, return_inverse=True)
    rng = np.random.default_rng(SEED)
    fc = counts(rng, len(fields), draws) if scheme != "date" else np.ones((draws,len(fields)), dtype=np.int16)
    dc = counts(rng, len(dates), draws) if scheme != "field" else np.ones((draws,len(dates)), dtype=np.int16)
    w = fc[:, fi].astype(float)*dc[:, di]
    represented_dates = np.stack([(w[:,di==d].sum(axis=1)>0) for d in range(len(dates))], axis=1).sum(axis=1)
    represented_fields = np.stack([(w[:,fi==f].sum(axis=1)>0) for f in range(len(fields))], axis=1).sum(axis=1)
    audit = pd.DataFrame({"draw": np.arange(draws), "total_weight": w.sum(axis=1),
                          "represented_fields": represented_fields, "represented_dates": represented_dates,
                          "fewer_than_three_dates": represented_dates<3})
    return w, fi, audit, dict(field_counts=fc, date_counts=dc,
                             field_labels=fields.astype(str), date_labels=dates.astype(str))

def interval(values):
    valid = np.isfinite(values)
    q = np.quantile(values[valid], [.025,.975]) if valid.any() else [np.nan,np.nan]
    return {"low": float(q[0]), "high": float(q[1]), "valid_draws": int(valid.sum()),
            "undefined_draws": int((~valid).sum()),
            "interpretation": "conditional_on_defined_draws" if not valid.all() else "all_draws"}

def check_math():
    y = np.array([1.,2.,4.,1.,3.,7.])
    p = np.array([2.,1.,5.,3.,4.,5.])
    fi = np.array([0,0,0,1,1,1])
    w = np.array([[2,1,0,2,0,1],[0,0,0,1,2,1],[1,0,0,1,0,0]],dtype=float)
    s = stats(y,p,w,fi)
    for i,row in enumerate(w):
        ix = np.repeat(np.arange(6),row.astype(int))
        yy,pp,ff = y[ix],p[ix],fi[ix]
        yc,pc = yy-yy.mean(),pp-pp.mean()
        expected_overall = 1-np.mean((pc-yc)**2)/np.mean(yc**2) if np.mean(yc**2)>EPS else np.nan
        assert np.allclose(s["centered_skill"][i],expected_overall,equal_nan=True)
        yr,pr = yy.copy(),pp.copy()
        for f in np.unique(ff):
            yr[ff==f] -= yy[ff==f].mean()
            pr[ff==f] -= pp[ff==f].mean()
        expected = 1-np.sum((pr-yr)**2)/np.sum(yr**2) if np.sum(yr**2)>EPS else np.nan
        assert np.allclose(s["within_skill"][i],expected,equal_nan=True)
    s2 = stats(y,p+3.17,w,fi)
    for q in ("centered_skill","within_skill","between_skill"):
        assert np.allclose(s[q],s2[q],atol=1e-10,equal_nan=True)
    constant = stats(y,np.ones(6),w,fi)
    defined = np.isfinite(constant["centered_skill"])
    assert np.allclose(constant["centered_skill"][defined],0,atol=1e-10)
    assert np.isnan(constant["correlation"]).all()

def freeze(root, out, experiment):
    assert not out.exists(), "Never overwrite an existing experiment directory"
    p = paths(root,experiment)
    protocol = {
        "experiment": experiment, "status": "FROZEN_BEFORE_EXECUTION", "post_hoc": True,
        "script_sha256": sha(__file__), "inputs": {k:str(v) for k,v in p.items()},
        "inputs_sha256": {k:sha(v) for k,v in p.items()},
        "seed": SEED, "draws": 5000 if experiment=="E5" else 10000,
        "response": "VV-HH in dB; supplied soil moisture is an inference input, not a retrieval target",
        "E5_specification": "OLS with intercept d=a+b*moisture; exact saved four source-field folds; no hyperparameter search. Final fit uses all 240 source rows. Target 138 and 189 cohorts separately. Other methods loaded unchanged. Only differential metrics; no HH/VV model ranking.",
        "E6_specification": "Common 138 records/22 fields/7 dates only; saved models and offsets fixed. Field, date, and crossed product weights. Each mode restarts RNG seed; crossed mode samples fields first then dates. Weighted overall and field means recomputed per draw. Missing cells remain missing.",
        "E6_primary_contrasts": [["pretraining_source_intercept","frozen_pretraining_neural","rmse_db"],
                                 ["frozen_source_inherited_neural","frozen_pretraining_neural","centered_skill"],
                                 ["frozen_source_reset_neural","frozen_pretraining_neural","centered_skill"],
                                 ["E1_selected_neural","source_training_mean","rmse_db"]],
        "E5_contrasts": "Moisture-only OLS minus every saved comparator, RMSE and centered skill; exploratory, unadjusted paired field percentiles.",
        "metrics": QUANTITIES,
        "undefined_rule": "Denominators <=1e-12 yield NaN. Keep all draws, report counts; percentiles on finite values explicitly conditional if any undefined. Fewer than 3 represented dates flagged, retained. Zero observed variance is not replaced by zero skill.",
        "scope": "Post-hoc sensitivity, not independent confirmation. Seven dates are not a reliable sample of all future dates. Bootstrap conditions on trained models, fixed offsets, source selection and observed missingness. Date exchangeability/independence is unverified. No E4 historical record-level inference.",
        "environment": {"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform()},
    }
    out.mkdir(parents=True)
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    print(experiment, "protocol frozen", flush=True)

def fit_ols(x,y):
    xc = x-x.mean()
    assert np.dot(xc,xc)>EPS
    b = np.dot(xc,y-y.mean())/np.dot(xc,xc)
    a = y.mean()-b*x.mean()
    independent = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
    assert np.allclose([a,b],independent,atol=1e-10,rtol=1e-10)
    return a,b

def evaluate(frame, methods, out, cohort, scheme, draws, pairs, save_draws):
    w,fi,audit,c = weights(frame,scheme,draws)
    y = frame.observed_diff_db.to_numpy()
    point,boot = {},{}
    rows=[]
    for name in methods:
        pred = frame[name].to_numpy()
        point[name] = {k:float(v[0]) for k,v in stats(y,pred,np.ones((1,len(y))),fi).items()}
        boot[name] = stats(y,pred,w,fi)
        for q in QUANTITIES:
            rows.append(dict(cohort=cohort,scheme=scheme,method=name,metric=q,
                             n=len(frame),fields=frame.field_id.nunique(),dates=frame.acquisition_date.nunique(),
                             estimate=point[name][q],**interval(boot[name][q])))
    contrasts=[]
    for first,second,q,role in pairs:
        contrasts.append(dict(cohort=cohort,scheme=scheme,first=first,second=second,metric=q,role=role,
                              estimate=point[first][q]-point[second][q],
                              **interval(boot[first][q]-boot[second][q])))
    prefix = f"{cohort}_{scheme}"
    np.savez_compressed(out/f"{prefix}_resampling.npz", **c)
    audit.to_csv(out/f"{prefix}_draw_audit.csv",index=False)
    if save_draws:
        pd.concat([pd.DataFrame({"draw":np.arange(draws),"method":name,**vals}) for name,vals in boot.items()],
                  ignore_index=True).to_csv(out/f"{prefix}_bootstrap_metrics.csv.gz",index=False,compression="gzip")
    # Frozen additive offsets must preserve all centered skills for every draw.
    for q in ("centered_skill","within_skill","between_skill"):
        assert np.allclose(boot["pretraining_source_intercept"][q],
                           boot["frozen_pretraining_neural"][q],atol=1e-8,rtol=1e-8,equal_nan=True)
    return rows,contrasts, {"cohort":cohort,"scheme":scheme,"draws":draws,
                             "fewer_than_three_dates":int(audit.fewer_than_three_dates.sum()),
                             "zero_total_weight":int((audit.total_weight==0).sum()),
                             "min_dates":int(audit.represented_dates.min()),
                             "max_dates":int(audit.represented_dates.max())}

def run(root,out,experiment):
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["experiment"]==experiment and not (out/"result_manifest.json").exists()
    assert sha(__file__)==protocol["script_sha256"]
    p=paths(root,experiment)
    assert {k:sha(v) for k,v in p.items()}==protocol["inputs_sha256"]
    check_math()
    metadata={"weighted_metrics_explicit_replication_test":"PASS","offset_skill_invariance":"PASS"}
    if experiment=="E5":
        source,splits=read(p["source"]),read(p["splits"])
        assert len(source)==240 and source.field_id.nunique()==24 and source[KEY].equals(splits[KEY])
        assert source.groupby("field_id").size().size==24
        x=source.soil_moisture_m3_m3.to_numpy()
        y=(source.sigma0_vv_db-source.sigma0_hh_db).to_numpy()
        assert source.model_ready.astype(str).str.lower().eq("true").all()
        assert np.isfinite(x).all() and np.isfinite(y).all()
        folds=splits.validation_fold.to_numpy()
        assert sorted(set(folds))==[1,2,3,4]
        oof=np.full(len(source),np.nan)
        coef=[]
        for fold in range(1,5):
            train,valid=folds!=fold,folds==fold
            assert not set(source.field_id[train]) & set(source.field_id[valid])
            assert source.field_id[valid].nunique()==6
            a,b=fit_ols(x[train],y[train])
            oof[valid]=a+b*x[valid]
            coef.append(dict(fit=f"fold_{fold}",intercept_db=a,slope_db_per_m3_m3=b,
                             training_records=int(train.sum()),validation_records=int(valid.sum())))
        a,b=fit_ols(x,y)
        coef.append(dict(fit="full_source",intercept_db=a,slope_db_per_m3_m3=b,
                         training_records=240,validation_records=0))
        pd.DataFrame(coef).to_csv(out/"coefficients.csv",index=False)
        sf=source[KEY].copy()
        sf["validation_fold"]=folds
        sf["soil_moisture_m3_m3"]=x
        sf["observed_diff_db"]=y
        sf["moisture_only_ols_oof"]=oof
        sf["moisture_only_ols_in_sample"]=a+b*x
        sf.to_csv(out/"source_predictions.csv",index=False)
        sm=[]
        fi=pd.factorize(source.field_id)[0]
        for role,pred in [("pooled_out_of_field",oof),("full_source_in_sample",a+b*x)]:
            sm.append({"role":role,**{k:float(v[0]) for k,v in stats(y,pred,np.ones((1,240)),fi).items()}})
        pd.DataFrame(sm).to_csv(out/"source_metrics.csv",index=False)
        # Persist all source fitting results before reading target labels.
        frame,offset=target_frame(p)
        frame["moisture_only_ols"]=a+b*frame.soil_moisture_m3_m3
        methods=METHODS+["moisture_only_ols"]
        pairs=[("moisture_only_ols",m,q,"exploratory") for m in METHODS for q in ("rmse_db","centered_skill")]
        jobs=[("common_138",frame[frame.common_138].copy(),"field"),
              ("full_189",frame.copy(),"field")]
    else:
        frame,offset=target_frame(p)
        frame=frame[frame.common_138].copy()
        methods=METHODS
        pairs=[(*v,"primary") for v in protocol["E6_primary_contrasts"]]
        pairs += [("source_selected_ridge","E1_selected_neural","centered_skill","exploratory")]
        jobs=[("common_138",frame.copy(),s) for s in ("field","date","crossed")]
    frame.to_csv(out/"analysis_predictions.csv",index=False)
    coverage=pd.crosstab(frame.field_id,frame.acquisition_date)
    coverage.to_csv(out/"field_date_coverage.csv")
    allrows,allpairs,diagnostics=[],[],[]
    for cohort,f,scheme in jobs:
        rr,pp,dd=evaluate(f,methods,out,cohort,scheme,protocol["draws"],pairs,True)
        allrows.extend(rr);allpairs.extend(pp);diagnostics.append(dd)
        print(experiment,cohort,scheme,"complete",flush=True)
    pd.DataFrame(allrows).to_csv(out/"metric_intervals.csv",index=False)
    pd.DataFrame(allpairs).to_csv(out/"paired_contrasts.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"resampling_summary.csv",index=False)
    metadata.update(status="COMPLETE",experiment=experiment,post_hoc=True,
                    source_intercept_reused_db=offset,model_training=(experiment=="E5"),
                    manuscript_modified=False,inputs_unchanged={k:sha(v) for k,v in p.items()}==protocol["inputs_sha256"],
                    output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file()})
    assert metadata["inputs_unchanged"]
    (out/"result_manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment",choices=["E5","E6"],required=True)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args()
    (freeze if args.freeze else run)(args.root.resolve(),args.output.resolve(),args.experiment)
