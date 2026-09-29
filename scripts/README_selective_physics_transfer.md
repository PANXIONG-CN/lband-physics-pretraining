# Selective physics-transfer evaluation

This experiment tests whether a reliability-gated SPM constraint avoids the
negative transfer observed with a single global physics weight.

## Methods compared

- training mean;
- offset-calibrated exponential SPM;
- MLP trained from scratch;
- SPM-pretrained MLP with observation-only fine-tuning;
- SPM-pretrained MLP with a fixed physics constraint;
- SPM-pretrained MLP with a selective physics constraint.

The selective gate is

```text
q = q_roughness * exp(-alpha * VWC) * q_provenance
effective physics weight = lambda_max * q
```

It uses no observed backscatter and no residual.  `lambda_max` and `alpha` are
selected inside nested field-grouped cross-validation.  The fixed and selective
models share the pretrained initialization and training order with the
observation-only model in each outer fold.

## Main run

From `PROJECT_ROOT` in the activated `research-pilots` environment:

```powershell
python scripts/test_selective_physics_transfer.py

python scripts/evaluate_selective_physics_transfer.py `
  --input "outputs/scattering/rough_ground/paper_diagnostics/cohort_common.csv" `
  --output "outputs/scattering/rough_ground/selective_physics_transfer_20260909" `
  --cohort all `
  --synthetic-samples 3000 `
  --pretrain-epochs 200 `
  --fine-tune-epochs 120 `
  --physics-weights "0,0.05,0.1,0.2" `
  --gate-alphas "0,0.8,1.6" `
  --outer-folds 5 `
  --inner-folds 4 `
  --repeats 5 `
  --bootstrap-iterations 4000
```

The program refuses to overwrite an existing completed output directory.

## Interpretation rule

Negative paired RMSE delta means the first method is better.  Claim an
improvement only when the field-block bootstrap 95% interval excludes zero.
An interval crossing zero is inconclusive.  This experiment evaluates
unseen-field prediction within SMAPVEX12; it does not validate SPM as physical
truth and does not constitute a causal vegetation-scattering decomposition.
