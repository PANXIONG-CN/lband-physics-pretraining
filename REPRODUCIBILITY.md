# Reproducibility guide — version 0.2.0

## 1. Install and verify

```bash
python -m pip install -e .
python scripts/verify_release.py
python -m unittest discover -s scripts -p "test_*.py" -v
```

The core install avoids optional raw-product dependencies. For raw reconstruction:
`python -m pip install -e ".[raw]"`. This includes xlrd/rasterio/h5py; xlrd was
not installed or tested in the validation container. Never store Earthdata
credentials in the repository.

Original recorded runtime: Windows/Python 3.11.15, preserved in
`requirements-lock-windows-py311.txt`. Tested revision runtime: Linux/Python
3.13.5; core versions are in `requirements-validation-linux-py313.txt`.
The original environment was not restored in this execution.

## 2. Frozen result recalculation (no training)

```bash
python scripts/recalculate_frozen_results.py --bundle reproducibility --output outputs/recalculated
python scripts/evaluate_physics_offset_baseline.py --output outputs/fairness --bootstrap-iterations 10000
python scripts/render_revision_figures.py --output outputs/centered_figures
```

These commands read the compact release paths rather than an author's private
`outputs/` archive. Choose fresh output paths outside `reproducibility/`.
The first recalculates stage metrics, field bootstrap intervals, original
few-shot metrics, information-budget checks, error decompositions, within/between
field diagnostics, and reported-box coverage. The second is the authoritative
matched-information comparison and its 10,000-reweighting intervals. The third
renders four separate centered-response panels. Clipping <1e-12 floating noise
to zero is display-only.

`recalculate_frozen_results.py` retains the old unequal-budget contrast as a
**diagnostic** to identify the error, not as an eligible model claim. Revised
paper comparisons use `results/revision_20260929/fairness/paired_contrasts.csv`.
Original frozen inputs are never silently overwritten.

## 3. Original five-seed trajectory reconstruction (training)

```bash
python scripts/verify_stage_continuity.py --project-root . --output outputs/stage_continuity
```

Seeds: 20260910, 20360910, 20460910, 20560910, 20660910. Original SPM 200,
I²EM 100, source 120 epochs; unchanged internal seed offsets and training code.
The command also runs source-only four-fold shrinkage selection once. Target
labels do not select any setting. Pretraining, source and risk member outputs,
scalers and safe NPZ weight arrays are saved. NPZ arrays are reconstruction
artifacts, not recovered historical checkpoints.

Actual tested result: pretraining/source predictions match frozen output to
~1e-14 dB; applying archived member coefficients [0.1,0.1,0,0,0] also matches.
Fresh source-only selection in the tested runtime returns [0,0,0,0,0], not the
archived values; see `summary.json`. The latter is not substituted into paper
results. Do not tune against target results to force agreement.

The original incremental optimizer is intentionally unchanged. It is copied
with the model. Setting `learning_rate_init=0.001` does not recreate existing
Adam state; the reconstructed sequential optimizer retains base 0.003.

## 4. Full simulation experiments (provided, not rerun in this revision)

```bash
python scripts/evaluate_multifidelity_pretraining.py --spm reproducibility/data/teachers/spm_pretraining.csv --i2em-train-requests reproducibility/data/teachers/i2em_train_requests.csv --i2em-train-results reproducibility/data/teachers/i2em_train_results.csv --i2em-test-requests reproducibility/data/teachers/i2em_test_requests.csv --i2em-test-results reproducibility/data/teachers/i2em_test_results.csv --output outputs/multifidelity --spm-epochs 200 --i2em-epochs 100 --repeats 5 --seed 20260910
python scripts/evaluate_multifidelity_sample_efficiency.py --spm reproducibility/data/teachers/spm_pretraining.csv --i2em-train-requests reproducibility/data/teachers/i2em_train_requests.csv --i2em-train-results reproducibility/data/teachers/i2em_train_results.csv --i2em-test-requests reproducibility/data/teachers/i2em_test_requests.csv --i2em-test-results reproducibility/data/teachers/i2em_test_results.csv --output outputs/sample_efficiency --sample-sizes 32,64,128,256 --spm-epochs 200 --i2em-epochs 100 --repeats 10 --seed 20260910
python scripts/evaluate_simple_residual_baselines.py --data reproducibility/data/teachers --output outputs/residual_baselines --repeats 10 --seed 20260910
```

Equivalent label budgets are not equivalent optimization-update or compute
budgets. The numerical summaries were checked; these full training sweeps were
not rerun in the approved revision.

## 5. Raw data and extended controls (external inputs required)

Read `DATA.md`, obtain official products, and install the raw extras. The
collocation entry points are `inspect_smapvex.py`, `collocate_smapvex_ground.py`,
and `prepare_smex02_external.py`; use each `--help` for required paths. Source
matching uses same-day field centers. SMEX02 first matches all available nodes
spatially, then joins date/field observations. The same 500 m cutoff does not
imply the same observation support.

`review_source_structure.py`, `review_match_vegetation.py`, `review_angle_control.py`,
and the original stage/plot orchestration files remain as experiment source.
They may require historical output paths or raw controls not included here.
They are **not** the compact-release default workflows above. In particular,
source nested-CV OOF predictions, VWC matched/OOF tables, all physical anchors,
and all source-angle physical request/result files are absent. Frozen summary
and figure inspection is not claimed as a replay of those experiments.

New I²EM computation requires an independently obtained compatible solver and
MATLAB; only wrappers and frozen request/result tables are distributed.

## 6. Figure and result precedence

Revision comparisons: `results/revision_20260929/fairness/`.
New diagnostic panels: `results/revision_20260929/centered_figures/`.
Stage and decomposition evidence: `results/revision_20260929/statistics/` and
`stage_continuity/`. Frozen original figures remain for provenance. The old
common-cohort combined-ranking figure and tiny nonzero common-skill panel are
superseded; do not use them as the revised presentation. Full historical
plotters require their original extra inputs and are not presented as a
one-command regeneration of every figure.
