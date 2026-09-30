# L-band scattering: bias calibration and differential-response diagnostics

Companion code and processed data for *Bias Calibration and Differential-Response
Loss in Cross-Campaign Transfer of L-Band Scattering Surrogates*, version **0.2.5**.
The internal Python package is `research_pilots`. The fixed release and download
are available at [v0.2.5](https://github.com/PANXIONG-CN/lband-physics-pretraining/releases/tag/v0.2.5).

## Recalculate the paper results

Run from the repository root using Python 3.11–3.13. The following commands use
included tables, saved weights and predictions; they perform no model training.

```bash
uv venv --python 3.13.5 .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip sync requirements-reproducibility.txt
uv pip install --no-deps -e .
python scripts/diagnose_smex02_stage_retention.py --frozen-source-offset --output outputs/source_offset
python scripts/recalculate_frozen_results.py --bundle reproducibility --output outputs/recalculated
python scripts/render_tgrs_figures.py --compact-revision --output-root outputs/revised_figures
python scripts/evaluate_physics_offset_baseline.py --output outputs/matched_information --bootstrap-iterations 10000
python scripts/render_centered_panels.py --output outputs/centered_panels
```

The source-intercept command evaluates five saved pretraining members, fits
intercepts using source labels, and writes updated tables in the specified output
directory. The recalculation command reproduces metrics, paired field-bootstrap
intervals, error decomposition, within-field intervals, equal-field weighting
and leave-one-date-out scores. Plotting uses the supplied frozen summaries.
Run outputs are kept separate from `reproducibility/`.

Version 0.2.5 adds statistical comparisons, revised manuscript figures, and a
fresh replay of the two existing ancillary source-domain controls. It retains
the completed five-member Adam-reset results from version 0.2.4. The ancillary
replay uses the original model families, seeds, folds and training budgets; it
does not retrain the main target-stage experiments or add target labels.

## Numerical environment

Ordinary installation uses the dependency ranges in `pyproject.toml`. The
following versions were used for the numerical checks, Adam-reset control and
ancillary source-control replay:
Python **3.13.5**, NumPy **2.3.5**, SciPy **1.17.0**, pandas **2.2.3**,
scikit-learn **1.8.0**, Matplotlib **3.10.8**, threadpoolctl **3.6.0**.

The uv quick start installs all dependencies from the exact
`requirements-reproducibility.txt` snapshot. To pin only the principal numerical
libraries in another environment:

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 pandas==2.2.3 scikit-learn==1.8.0 matplotlib==3.10.8 threadpoolctl==3.6.0
```

Use CPU float64 and one BLAS thread when replaying the training trajectories.
The source-reset implementation uses scikit-learn internal initialization and
Adam interfaces; its validated version is 1.8.0. The original Windows experiment
environment is recorded separately in the supplementary material. It is not
combined with the environment above into an installation specification.

## Main results and interpretation

Source SMAPVEX12 contains **240 records from 24 fields**. Complete target SMEX02
contains **189 records from 30 fields**; the common physical-valid subset contains
**138 records from 22 fields**, covering seven observation dates.

| Common-cohort differential result | RMSE (dB) | Centered skill |
| --- | ---: | ---: |
| Physical pretraining | 5.987711 | 0.148175 |
| Source update, inherited Adam | 3.481578 | −0.471034 |
| Source update, reset Adam | 3.157751 | −0.155738 |
| Pretraining + source intercept | 2.653513 | 0.148175 |
| Source differential mean | 2.985482 | 0 |

The source intercept is **−3.807911 dB**. Its paired RMSE differences from reset
Adam and the source mean are respectively **−0.504238 dB** (95% field interval
**[−0.730476, −0.274752]**) and **−0.331969 dB** (**[−0.432649, −0.231267]**).
On all 189 target records, average HH/VV RMSE is **4.941184 dB**, versus
**4.981190 dB** for the source channel means; this 0.040006 dB improvement is
smaller than the historical 0.10 dB practical criterion.

Between-field differences explain 80.13% of observed differential variance.
Pretraining within-field skill is **0.018155**, with a field-bootstrap interval
**[−0.181655, 0.189509]**. Within-field skill remains negative after either source
update. Deleting each observation date and giving fields equal total weight
preserve positive overall pretraining skill and negative skill after both source
updates. These post-hoc calculations fix predictions, observed dates and source
calibration; they do not estimate uncertainty over new training or adaptation.

## Data and result locations

| Path | Contents |
| --- | --- |
| `src/research_pilots/` | Physical models, data readers and learning implementation |
| `scripts/` | Preparation, training, evaluation and figure entry points |
| `tests/` | Unit tests and standalone numerical checks |
| `configs/` | Existing experiment configuration examples |
| `reproducibility/data/` | Processed observations and frozen SPM/I²EM teacher tables |
| `reproducibility/results/` | Saved weights, predictions, metrics and field splits |
| `reproducibility/figures/` | PDF figure assets, retaining their existing filenames |

Under `reproducibility/results/`, `stage_diagnostics/` and
`trajectory_reconstruction/` hold the stage predictions and weights;
`statistical_checks/` holds independently recalculated metrics and intervals;
`matched_information/` holds comparisons with the same adaptation-label budget.
The added sensitivity values extend `exploratory_within_between_field.csv`,
and the new intercept contrasts extend `stage_pairwise_bootstrap_independent.csv`.
The common component is fixed to a source/adaptation mean by the model design;
its zero centered skill and variance ratio are structural, not learned collapse.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/test_external_domain_pipeline.py
python tests/test_multifidelity_pretraining.py
python tests/test_smex02_reader.py
```

The paper-result tests check saved NPZ predictions, source-only calibration,
Adam-reset settings, paired comparisons and response-sensitivity calculations.
The source-control tests also check nested field isolation, matching boundaries,
earlier-date tie breaking, and invariance of fits when audit records are saved.

## Replaying the completed Adam control

```bash
python scripts/verify_stage_continuity.py --reset-source-adam
```

This command **trains five source stages** and updates the existing result files
under `reproducibility/`. It restores each saved pretraining endpoint and its
scales, resets Adam moments and the update count, and runs 120 epochs, or 240
minibatch updates. The actual base learning rate is 0.003 for both inherited and
reset branches. Updating the estimator attribute to 0.001 in the original
incremental code did not replace the already initialized Adam base rate.
Saved reset arrays use `source_reset_adam_003_*` keys in the existing five NPZ
files. Source order uses member seed + 10102. Member seeds are 20260910,
20360910, 20460910, 20560910 and 20660910.

For the full original pretraining/source reconstruction:

```bash
python scripts/verify_stage_continuity.py --project-root . --output outputs/trajectory_reconstruction
```

This entry point runs 200 SPM, 100 I²EM and 120 source epochs plus source-only
four-fold shrinkage selection. Pretraining/source prediction reconstruction and
fresh shrinkage selection are separate outcomes. Archived coefficients are
`[0.1, 0.1, 0, 0, 0]`, whereas current reselection returned `[0, 0, 0, 0, 0]`.
Their maximum risk-prediction difference is 0.0640701 dB. Historical inner-fold
records are missing, so the selection discrepancy remains unresolved. Both
coefficient vectors are reported, and archived shrinkage is an auxiliary result.
For zero-shot evaluation the all-zero vector is exactly the source differential
mean. With few-shot adaptation it updates only the common mean, so it is not
equivalent to a target two-channel-mean baseline.

The existing simulation entry points are
`evaluate_multifidelity_pretraining.py`,
`evaluate_multifidelity_sample_efficiency.py`, and
`evaluate_simple_residual_baselines.py`; each accepts the supplied teacher files
and an output directory. Use `--help` for the original training options.

## Observation products and extended controls

Cite the original Version 1 NASA NSIDC DAAC products:

| Campaign | Product | DOI |
| --- | --- | --- |
| SMAPVEX12 | PALS backscatter, SV12PLBK | [10.5067/KQC0KOL4DK1I](https://doi.org/10.5067/KQC0KOL4DK1I) |
| SMAPVEX12 | Probe agricultural soil moisture, SV12PSMA | [10.5067/3ELRJAUL0A4G](https://doi.org/10.5067/3ELRJAUL0A4G) |
| SMAPVEX12 | Agricultural surface roughness, SV12SRA | [10.5067/QB4JHGKXH16O](https://doi.org/10.5067/QB4JHGKXH16O) |
| SMAPVEX12 | In situ agricultural vegetation, SV12VA | [10.5067/X2EF9ZKL0DGC](https://doi.org/10.5067/X2EF9ZKL0DGC) |
| SMEX02 | PALS data, NSIDC-0183 | [10.5067/D7IROA0Q6JEQ](https://doi.org/10.5067/D7IROA0Q6JEQ) |
| SMEX02 | Walnut Creek soil moisture, NSIDC-0143 | [10.5067/OAO4SU0XZLGR](https://doi.org/10.5067/OAO4SU0XZLGR) |
| SMEX02 | Geolocation, roughness and photographs, NSIDC-0204 | [10.5067/R9AA6FC58HES](https://doi.org/10.5067/R9AA6FC58HES) |

Use explicit local paths with `inspect_smapvex.py`, `collocate_smapvex_ground.py`
and `prepare_smex02_external.py`. Source matching uses same-day field centers;
SMEX02 first matches spatial nodes and then joins date/field observations. The
shared 500 m distance cutoff does not equalize the observation footprints.
The vegetation control uses the SV12VA field crop-biomass measurements.

The two ancillary controls were replayed using the packaged source cohort and
SV12VA Version 1 files retrieved from the official NASA NSIDC collection
`C3284167996-NSIDC_CPRD`. `reproducibility/data/source/sv12va_provenance.json`
records the product URLs and raw-file SHA-256 values; `vegetation_field_day.csv`
contains the derived field-day inputs. Original NSIDC terms and citations apply.

Complete outer and inner out-of-fold predictions, split membership, tuning
records, analysis inputs and manifests are supplied under
`reproducibility/results/source_controls/`. The original summarized values are
retained in `archived_summary.csv`. The replay does not exactly reproduce those
historical summaries, whose original complete records remain unavailable. The
supplement reports the fresh, auditable results and retains their ancillary role;
these controls do not establish the cause of the target-stage loss.

Audit the supplied control evidence without training:

```bash
python scripts/audit_source_controls.py --output outputs/source_controls_audit
```

Replay the source-structure experiment (five initializations, five outer field
folds, four inner folds, original 120/300/600-epoch candidates):

```bash
python scripts/review_source_structure.py --output outputs/source_structure
```

This command uses the 4096 SPM samples and 256 I²EM training labels supplied under
`reproducibility/data/teachers/`; `--teacher-dir` permits an explicit alternative.
The output includes checkpointed outer predictions and every inner validation
prediction. CPU float64 and one numerical-library thread are used. Each `fit`
resets Adam while retaining warm-start weights, as in the original control;
this differs from the inherited-Adam main experiment.

Replay the two original vegetation controls from the supplied derived table:

```bash
python scripts/review_match_vegetation.py --max-gap-days 2 --output outputs/vwc_gap2
python scripts/review_source_structure.py --classical-only --with-vwc --input outputs/vwc_gap2/matched_source.csv --output outputs/vwc_gap2_fit
python scripts/review_match_vegetation.py --max-gap-days 8 --output outputs/vwc_gap8
python scripts/review_source_structure.py --classical-only --with-vwc --input outputs/vwc_gap8/matched_source.csv --output outputs/vwc_gap8_fit
```

Matching retains 132 and 240 records respectively. The replayed average HH/VV
RMSEs for training mean / Ridge / Ridge+VWC are 3.11889 / 3.24112 / 3.17065 dB
at two days and 3.11355 / 3.26928 / 3.20564 dB at eight days. Adding VWC improves
the Ridge point estimate but does not outperform the matched training mean.
These values replace the historical vegetation summaries; they do not measure
the total vegetation contribution or establish statistical significance.

To rebuild the derived table
from the official biomass, height and coordinate files, pass
`--vegetation-root /path/to/SV12VA` instead of `--vegetation`. The original
same-field nearest-date rule, including earlier-date tie breaking, is retained.
The separate `evaluate_vegetation_inputs.py` workflow uses a different experiment
design and is not the replay of the supplement's Ridge control.

Some other extended control scripts require raw products, physical anchors or
source-angle request/result tables beyond the compact supplied inputs.

## New I²EM calculations

I²EM is a simulation teacher. The MATLAB adapters require a separately obtained
compatible IEEE-GRSS `I2EM_Backscatter_model.m`; reference-solver source is not
redistributed. `run_i2em_requests_matlab.m` and
`run_i2em_requests_matlab_com.ps1` provide MATLAB and Windows COM interfaces.
The optional `run_i2em_teacher.py` entry point supports `pyi2em==0.1.5` in a
compatible Linux/WSL Python 3.11 environment.

`prepare_i2em_validation_grid.py` prepares cohort-derived anchors, and
`prepare_i2em_loss_sensitivity.py` prepares 18 anchors at each loss tangent
0, 0.02, 0.05 and 0.10. The design checks `ks < 1` and RMS height/correlation
length ≤ 0.25. `evaluate_i2em_reference.py` and
`evaluate_i2em_loss_sensitivity.py` check paired solver outputs. New teacher
calculations are separate from the frozen-result quick start.

## Citation and licensing

> Chengyue Huang, Pan Xiong, Jing Liu, Roberto Battiston, Angelo De Santis,
> and Xuhui Shen (2026). *L-band physics pretraining: code and processed-data
> reproducibility bundle*. Version 0.2.5, 2026-09-30.
> https://github.com/PANXIONG-CN/lband-physics-pretraining/releases/tag/v0.2.5

### Data and third-party terms

Unless a file states otherwise, original project-derived tables, numerical
summaries and figures under `reproducibility/` are licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)
(CC BY 4.0), where contributors hold the necessary rights. Cite the software
and relevant source products when reusing these artifacts.

NASA/NSIDC observations retain their original terms and required citations.
The I²EM MATLAB reference source used by this project contained no explicit
redistribution license and is excluded from the repository. Only wrappers,
interface checks and frozen request/result tables are distributed; obtain a
properly licensed compatible solver separately. No third-party data or software
is relicensed by this repository.

### Code license

The following BSD-3-Clause license applies to original project code:

```text
BSD 3-Clause License

Copyright (c) 2026, research-pilots contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
