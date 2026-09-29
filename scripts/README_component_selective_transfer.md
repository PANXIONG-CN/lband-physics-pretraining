# Response-component-selective SPM transfer

This experiment tests the next hypothesis after scalar confidence gating failed
to provide a significant unseen-field improvement.

The exact response transformation is:

```text
common = (HH + VV) / 2
differential = VV - HH
HH = common - differential / 2
VV = common + differential / 2
```

It compares direct HH/VV training with component-space training, a constraint
on both components, and an SPM constraint only on the polarimetric differential
component.  Constraint weights are selected inside nested GroupKFold by field.

From `PROJECT_ROOT`:

```powershell
python scripts/test_component_selective_transfer.py

python scripts/evaluate_component_selective_transfer.py `
  --input "outputs/scattering/rough_ground/paper_diagnostics/cohort_common.csv" `
  --output "outputs/scattering/rough_ground/component_selective_transfer_20260909" `
  --synthetic-samples 3000 `
  --pretrain-epochs 200 `
  --fine-tune-epochs 120 `
  --physics-weights "0,0.05,0.1,0.2,0.4" `
  --outer-folds 5 `
  --inner-folds 4 `
  --repeats 5 `
  --bootstrap-iterations 4000
```

Negative paired RMSE deltas favor the first method.  Do not claim improvement
when a 95% field-block bootstrap interval crosses zero.  The decomposition is
an exact response coordinate system, not proof that common and differential
responses correspond uniquely to separate scattering mechanisms.
