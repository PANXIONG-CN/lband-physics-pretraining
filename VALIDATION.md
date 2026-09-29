# Validation — 2026-09-29

## Version 0.2.1: editorial and publication revision

This revision updates author/funding information, repository metadata,
availability statements, language and table layout. Scientific source files,
processed data, frozen predictions, numerical summaries and figures are
unchanged. Models and original physics simulations were not rerun.
Current checks are recorded in `tests/editorial_0.2.1_validation.json`.

## Version 0.2.0: earlier scientific validation

The execution results below were supplied with the preceding revision.

### Completed

- Original 92-file SHA-256 manifest verified; processed cohorts and whole-field
  train/test separation checked from saved keys and splits.
- Final package: 52 unit tests and four standalone contract-check scripts passed.
  Editable installation with existing dependencies, frozen recalculation, corrected
  fairness evaluation and panel rendering passed from the staged public tree.
- Frozen point statistics and 10,000 paired field-bootstrap stage intervals
  recalculated. Original full-target few-shot numbers reproduced.
- Matched-information physical/neural mean baselines and 10,000 shared-field
  reweighting intervals recomputed without neural retraining.
- One original five-seed trajectory reconstruction, using unchanged training
  implementation. Maximum prediction discrepancies: pretraining 1.78e-15 dB,
  source-tuned 1.47e-14 dB, risk with archived coefficients 5.55e-15 dB.
- MSE decomposition, prediction/observation correlation, within/between-field
  variation, reported-box restriction and leave-one-field-out checks derived
  from frozen predictions.

### Material non-reproduction

Fresh source-only shrinkage selection produced [0,0,0,0,0], versus the archived
[0.1,0.1,0,0,0]. Fresh-risk output differs by at most 0.0640701 dB. Historical
Windows runtime and inner-fold evidence are not available in this execution;
no unique cause is assigned. Archived coefficients/predictions remain the
paper's risk result, while fresh selection is reported separately.

The optimizer state survives deepcopy and incremental training. Although the
model attribute is changed to 0.001, the reconstructed sequential Adam optimizer
retains initial base 0.003. This was documented, not silently changed.

### Not executed

- Restoration of the original Windows/Python 3.11 environment.
- New MATLAB I²EM calculations or official-raw-data collocation replay.
- Full simulation training sweeps, all cross-domain few-shot retraining, source
  structure/VWC nested-CV experiments (complete historical OOF inputs absent).
- Remote GitHub publication during that earlier scientific-validation step.

See `tests/release_validation.json` for the final package-level test results.
