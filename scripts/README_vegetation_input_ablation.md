# Vegetation-input matched ablation

## Research question

Does adding in-situ vegetation water content (VWC, kg/m²) improve prediction
of observed PALS HH/VV backscatter when the samples, held-out fields, model
architecture and training budget are unchanged?

This is a controlled input ablation. It is intentionally completed before the
physics-pretraining and physics-constraint experiments.

## Fixed design

- Main cohort: every common-cohort record, with vegetation match gap ≤ 8 days.
- Sensitivity cohort: the prebuilt high-quality records with gap ≤ 2 days.
- Outer validation: five-fold `GroupKFold` by `field_id`. All dates from a held-out
  field stay in the test fold.
- Original inputs: soil moisture, measured real dielectric constant, PALS RMS
  height and PALS correlation length.
- Augmented inputs: the same four variables plus in-situ VWC.
- Targets: observed HH and VV sigma0 in dB.
- Ordinary baselines: Ridge and a small `(16, 8)` tanh MLP. The MLP uses Adam
  with training-fold-only early stopping and five fixed seeds; this avoids the
  non-convergence observed with an unregularized quasi-Newton fit.
- Fairness: feature and target scaling are fitted only on the outer training
  fold. The only changed experimental factor is the presence of VWC.
- Uncertainty: paired field-block bootstrap. Negative `delta_rmse` favors VWC.

The real dielectric constant and soil moisture are retained together because
they are part of the established original input set. Their strong collinearity
must be discussed later and can be examined in a separate sensitivity test.

## Exact Windows command

Run from `PROJECT_ROOT` with the `research-pilots` environment active:

```powershell
python scripts/evaluate_vegetation_inputs.py --main-input "outputs/scattering/rough_ground/paper_diagnostics/cohort_common.csv" --sensitivity-input "outputs/scattering/rough_ground/paper_diagnostics/cohort_high_quality.csv" --output "outputs/scattering/rough_ground/vegetation_input_ablation_20260908"
```

The script refuses to overwrite a non-empty output directory.

## Primary outputs

- `summary.json`: design audit and primary paired conclusion.
- `paired_vwc_effects.csv`: ΔRMSE and 95% field-block bootstrap intervals.
- `ensemble_metrics.csv`: held-out-field RMSE, MAE and R².
- `oof_predictions.csv`: every outer-fold prediction for reproducibility.
- `outer_fold_assignments.csv`: explicit leakage audit.
- `fieldwise_paired_effects.csv`: effect heterogeneity across fields.
- `figures/`: four figures suitable for method checking and group reporting.

## Interpretation boundary

An improvement shows that VWC contains useful predictive environment
information beyond the original four inputs under this campaign and split. It
does not establish a causal vegetation-scattering law. Physics pretraining and
physics constraints should be tested only after this input ablation, using the
same cohort and fold file.
