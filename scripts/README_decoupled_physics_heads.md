# Fully decoupled common/differential physics heads

This is the decisive ablation after shared-hidden-layer component models showed
cross-output coupling.  The common response and polarimetric differential are
modeled by separate estimators; only the differential estimator receives SPM
pretraining or an SPM constraint.

The protocol uses nested GroupKFold by field, repeated neural-network seeds,
and field-block bootstrap confidence intervals.

From `PROJECT_ROOT`:

```powershell
python scripts/test_decoupled_physics_heads.py

python scripts/evaluate_decoupled_physics_heads.py `
  --input "outputs/scattering/rough_ground/paper_diagnostics/cohort_common.csv" `
  --output "outputs/scattering/rough_ground/decoupled_physics_heads_20260909_v4" `
  --synthetic-samples 3000 `
  --pretrain-epochs 200 `
  --fine-tune-epochs 120 `
  --ridge-alphas "0.01,0.1,1,10,100" `
  --physics-weights "0,0.05,0.1,0.2,0.4,0.6" `
  --shrinkage-weights "0,0.1,0.25,0.5,0.75,1" `
  --outer-folds 5 `
  --inner-folds 4 `
  --repeats 5 `
  --bootstrap-iterations 4000
```

An improvement is claimed only when the field-block 95% interval excludes
zero.  The primary overall score is the mean of HH and VV RMSE; separate
channel results are retained to expose polarization trade-offs.  The experiment
is an ablation within one campaign, not independent
multi-site validation and not a causal scattering-mechanism decomposition.

Create the paper-oriented evidence table, figures, and interpretation after a
completed run:

```powershell
python scripts/summarize_decoupled_results.py `
  --input "outputs/scattering/rough_ground/decoupled_physics_heads_20260909_v4"
```
