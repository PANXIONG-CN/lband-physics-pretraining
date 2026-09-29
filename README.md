# Quick start

Use Python 3.11 and run from the repository root. The internal package is
`research_pilots`; the companion software version is **0.2.2**.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
python scripts/recalculate_frozen_results.py --bundle reproducibility --output outputs/recalculated
python scripts/evaluate_physics_offset_baseline.py --output outputs/matched_information --bootstrap-iterations 10000
python scripts/render_centered_panels.py --output outputs/centered_panels
```

These commands use the included processed tables and frozen predictions. They
need no MATLAB, raw-data downloads or model training. Use a fresh output
directory for each run and keep outputs outside `reproducibility/`.

The first command checks required inputs, columns, sample keys, teacher pairs
and field splits, then recalculates metrics, bootstrap intervals, error
decompositions and domain coverage. The second computes the matched-information
physical and neural comparisons with 10,000 shared-field bootstrap reweightings
and renders two panels. The third renders four centered-response panels;
values below 1e-12 are rounded to zero for display only. Git records file versions.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/research_pilots/` | Physics models, data readers and training implementation |
| `scripts/` | Data preparation, training, evaluation and plotting entry points |
| `tests/` | Unit tests and standalone numerical checks |
| `configs/` | Experiment configuration examples |
| `reproducibility/data/` | Processed source/target tables and frozen teachers |
| `reproducibility/results/` | Scientific predictions, metrics, settings and splits |
| `reproducibility/figures/` | 16 final PDF assets named by paper figure number and content |

The supplied cohorts contain **240 source field-days**, **189 target field-days**
and **138 common-valid target field-days**. The workflow schematic is drawn in
the manuscript itself. Original observations and the external I²EM solver are
obtained separately; the private manuscript is maintained outside this repository.

Result directories are organized by purpose:

| Directory under `reproducibility/results/` | Role |
| --- | --- |
| `multifidelity/`, `sample_efficiency/`, `residual_baselines/` | Simulation comparisons and residual controls |
| `cross_domain/`, `common_cohort/` | Target predictions and fixed field splits |
| `matched_information/` | Authoritative matched-adaptation-label comparisons |
| `stage_diagnostics/` | Teacher, pretraining, source and risk-stage responses |
| `offset_diagnostics/` | Earlier unequal-budget control, retained as a diagnostic input |
| `trajectory_reconstruction/` | Five-seed reconstructed trajectories and NPZ weights |
| `statistical_checks/` | Recalculated statistics and exploratory decompositions |
| `diagnostics/`, `controlled_mismatch/` | Centered-response and mismatch controls |
| `paper_summaries/` | Aggregated paper results |

Use `matched_information/paired_contrasts.csv` for the paper's model comparisons.
Physical-offset and neural comparisons use their respective matched adaptation
label sets. The older unequal-budget contrast is retained to diagnose the budget
difference and is not an eligible model-ranking claim. Scientific protocol JSON
files retain settings, seeds and splits, including original provenance paths.

## Tests and environments

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/test_external_domain_pipeline.py
python tests/test_multifidelity_pretraining.py
python tests/test_smex02_reader.py
```

Dependencies are declared in `pyproject.toml`. Install
`python -m pip install -e '.[raw]'` for HDF5, raster and legacy Excel readers.
The core versions recorded for the original experiment and the earlier
numerical validation are:

| Component | Original Windows experiment | Linux numerical validation |
| --- | --- | --- |
| Python | 3.11.15 | 3.13.5 |
| NumPy | 2.4.6 | 2.3.5 |
| SciPy | 1.17.1 | 1.17.0 |
| pandas | 3.0.5 | 2.2.3 |
| scikit-learn | 1.9.0 | 1.8.0 |
| Matplotlib | 3.11.1 | 3.10.8 |
| threadpoolctl | 3.6.0 | 3.6.0 |

These are records of separate environments, not combined installation
requirements or a complete environment lock.
The 0.2.2 cleanup was checked with macOS/Python 3.11.4, including tests,
frozen-result recalculation, matched-information evaluation, panel rendering
and a short training check. The original Windows environment was not restored.

## Training from included inputs

Reconstruct the original five-seed trajectory with:

```bash
python scripts/verify_stage_continuity.py --project-root . --output outputs/trajectory_reconstruction
```

Seeds are 20260910, 20360910, 20460910, 20560910 and 20660910. The command uses
200 SPM, 100 I²EM and 120 source epochs, with the original internal seed offsets.
It also performs the original source-only four-fold shrinkage selection.
Target labels do not select settings. Saved NPZ arrays are reconstruction
artifacts, not recovered historical checkpoints.

In the recorded trajectory reconstruction, pretraining and source predictions
matched the frozen outputs to approximately 1e-14 dB. Applying archived member
coefficients `[0.1, 0.1, 0, 0, 0]` also matched. Fresh source-only selection
returned `[0, 0, 0, 0, 0]`, with a maximum risk-prediction difference of
0.0640701 dB. Archived coefficients and predictions remain the paper's risk result;
fresh selection is reported separately in `trajectory_reconstruction/summary.json`.
Historical inner-fold evidence is unavailable, so no unique cause is assigned.

The original incremental optimizer state is preserved. Setting the model's
`learning_rate_init` attribute to 0.001 does not recreate its existing Adam
state: sequential training retains the initial optimizer base step of 0.003.

The full simulation entry points accept the included teacher tables:

```bash
python scripts/evaluate_multifidelity_pretraining.py --spm reproducibility/data/teachers/spm_pretraining.csv --i2em-train-requests reproducibility/data/teachers/i2em_train_requests.csv --i2em-train-results reproducibility/data/teachers/i2em_train_results.csv --i2em-test-requests reproducibility/data/teachers/i2em_test_requests.csv --i2em-test-results reproducibility/data/teachers/i2em_test_results.csv --output outputs/multifidelity --spm-epochs 200 --i2em-epochs 100 --repeats 5 --seed 20260910
python scripts/evaluate_multifidelity_sample_efficiency.py --spm reproducibility/data/teachers/spm_pretraining.csv --i2em-train-requests reproducibility/data/teachers/i2em_train_requests.csv --i2em-train-results reproducibility/data/teachers/i2em_train_results.csv --i2em-test-requests reproducibility/data/teachers/i2em_test_requests.csv --i2em-test-results reproducibility/data/teachers/i2em_test_results.csv --output outputs/sample_efficiency --sample-sizes 32,64,128,256 --spm-epochs 200 --i2em-epochs 100 --repeats 10 --seed 20260910
python scripts/evaluate_simple_residual_baselines.py --data reproducibility/data/teachers --output outputs/residual_baselines --repeats 10 --seed 20260910
```

Equal label budgets do not imply equal optimization-update or compute budgets.
The full training sweeps and five-seed reconstruction were not rerun for this
structural cleanup; their supplied scientific outputs are preserved.

## Observation data and extended workflows

Obtain the original Version 1 products from NASA NSIDC DAAC and cite them:

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
and `prepare_smex02_external.py`; consult each script's `--help`. A convenient
local layout is `data/scattering/raw/SMAPVEX12/` and `data/scattering/raw/SMEX02/`.
Keep Earthdata credentials outside the repository.

Source matching uses same-day field centers. SMEX02 first matches all available
nodes spatially, then joins date/field observations; the same 500 m cutoff does
not imply the same observation support. The vegetation control uses SV12VA crop
biomass with same-field nearest-date matching, not an interchangeable VWC map.

Source nested-CV out-of-fold predictions, VWC matched/out-of-fold tables, all
physical anchors and all source-angle request/result tables are not included.
The retained extended-control and original plot orchestration scripts may
require those inputs or historical output directories. Supplied summaries and
figures support inspection; they do not provide a complete replay of those
experiments. The quick-start commands regenerate six panels, not every final
figure. Raw-data collocation and new solver calculations were not rerun for 0.2.2.

## New I²EM calculations

I²EM is a physics teacher, not observational ground truth. Request preparation
checks `ks < 1` and RMS-height/correlation-length `<= 0.25`. Use
`prepare_i2em_validation_grid.py` for cohort-derived anchors and
`prepare_i2em_loss_sensitivity.py` for fixed-anchor loss tangents
`0, 0.02, 0.05, 0.10`; the latter design has 18 anchors per level (72 requests).

The MATLAB adapter `run_i2em_requests_matlab.m` requires a separately obtained
compatible IEEE-GRSS `I2EM_Backscatter_model.m`. The PowerShell COM adapter
`run_i2em_requests_matlab_com.ps1` supports Windows account-based licensing.
Alternatively, `run_i2em_teacher.py` supports `pyi2em==0.1.5` in a compatible
Linux/WSL Python 3.11 environment; that backend is not a core dependency.

Check paired requests/results with `evaluate_i2em_reference.py` and
`evaluate_i2em_loss_sensitivity.py`. Confirm finite outputs, plausible anchor
trends and dielectric-loss sensitivity before new pretraining. I²EM-minus-SPM
differences quantify teacher disagreement, not observational error. External
reference-solver code is not distributed here.

## Contributing

Use focused changes and run the checks listed above. Add or update contract tests
when changing a data reader, physical model or metric, and describe changes to
sample definitions, seeds, splits, units and aggregation rules in the pull request.
Write new experiments to fresh output directories; preserve frozen results.
Keep raw third-party data, credentials, local absolute paths and generated outputs
out of commits. Distinguish exploratory work from locked or preregistered
evaluation; results already observed in the target campaign cannot be relabeled
as prospective.

## Citation and licensing

Cite the software as:

> Chengyue Huang, Pan Xiong, Jing Liu, Roberto Battiston, Angelo De Santis,
> and Xuhui Shen (2026). *L-band Physics Pretraining: Multi-Fidelity
> Rough-Surface Scattering*, version 0.2.2, released 2026-09-29.
> https://github.com/PANXIONG-CN/lband-physics-pretraining

Also cite the original observation products listed above.

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
