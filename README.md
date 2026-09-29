# L-band physics pretraining

Version **0.2.1** — 2026-09-29.

Code and processed-data companion for multi-fidelity L-band rough-surface
scattering experiments using SPM, I²EM, SMAPVEX12 and SMEX02.

Repository: https://github.com/PANXIONG-CN/lband-physics-pretraining

**Authors:** Chengyue Huang, Pan Xiong, Jing Liu, Roberto Battiston,
Angelo De Santis, and Xuhui Shen. Chengyue Huang and Jing Liu are co-first
authors. Corresponding authors: Pan Xiong and Xuhui Shen.

## Publication status

The complete version 0.2.1 code/data snapshot was published on 2026-09-29.
The [first complete snapshot](https://github.com/PANXIONG-CN/lband-physics-pretraining/commit/853a71dab9854034b97d1e9aad556bf42aa050ca)
contains all 279 release files, including processed data and archived validation
logs. An independent checkout passed both SHA-256 manifests and matched the
local release byte for byte. The existing repository history was preserved.

The commands below work from a complete repository checkout or the accompanying
archive. Validation reports under `tests/` preserve earlier scientific and
editorial sessions; their pending-upload statements describe those past sessions.

## Quick start

The internal Python package is `research_pilots`. Python 3.11 was the original
target; the recorded scientific validation used Linux/Python 3.13.5 (see
`VALIDATION.md`). Run from the complete companion root:

```bash
python -m pip install -e .
python scripts/verify_release.py --public-only
python scripts/recalculate_frozen_results.py --bundle reproducibility --output outputs/recalculated
python scripts/evaluate_physics_offset_baseline.py --output outputs/fairness --bootstrap-iterations 10000
python scripts/render_revision_figures.py --output outputs/centered_figures
```

Use a new output directory for each run. Frozen-statistic recalculation needs
neither MATLAB nor raw-data downloads. New physics simulations and raw-data
collocation require the external inputs listed in `REPRODUCIBILITY.md`.

## Contents

- `src/research_pilots/`: physics contracts, data readers and original training code.
- `scripts/`: training, evaluation, statistics, plotting and tests.
- `configs/`: experiment configuration examples.
- `reproducibility/data/`: source/target field-day tables and frozen teachers.
- `reproducibility/results/`: frozen experiments and versioned statistical corrections.
- `reproducibility/figures/`: figure assets.
- `DATA.md`, `REPRODUCIBILITY.md`: product provenance and runnable workflows.

The complete code/data snapshot is approximately 34 MB uncompressed; its
largest file is approximately 7.3 MB. Included cohorts contain 240 source
field-days, 189 target field-days, and 138 shared-valid target field-days.
All supplied processed data, numerical outputs, reconstructed NPZ weights,
and figures are included. Original NASA/NSIDC products are obtained through
the DOIs in `DATA.md`. New I²EM labels require a separately obtained solver.
The private manuscript, credentials, font files and external reference-solver
implementation are excluded from this public-ready directory.

## Version 0.2.1

Author/citation metadata, the repository address and the publication helper
were updated. The associated manuscript has updated authors/funding,
separate data and code availability statements, polished language, and
uniform-width tables. Scientific implementations, processed data, frozen
predictions, numerical results and figures are unchanged from version 0.2.0.

The matched-information comparisons from version 0.2.0 remain authoritative:
physical-offset and neural models use their respective matched adaptation-label
sets. Archived risk coefficients are preserved separately from fresh source-only
selection, which did not reproduce the archived coefficients in the recorded
validation runtime. Details are in `VALIDATION.md` and `CHANGELOG.md`.
Some historical control experiments are supplied as summaries and figures;
their full out-of-fold predictions and raw inputs are not included.

## Citation and licensing

Use `CITATION.cff` and cite the original data products in `DATA.md`.
Code: BSD-3-Clause. Original project-derived artifacts: CC BY 4.0 where the
contributors hold the necessary rights. Third-party products retain their
own terms (`LICENSE-DATA.md`, `THIRD_PARTY_NOTICES.md`).

## Publishing a verified snapshot

With Git and the GitHub CLI installed and authenticated as `PANXIONG-CN`,
run from this complete companion directory. Before publishing deliberate changes,
update the affected entries in `release_manifest.csv` so local verification passes:

```bash
python publish_github.py            # local verification; no remote changes
python publish_github.py --publish  # normal push preserving repository history
```

Only manifest-listed code/data files are copied and staged, including archived
validation logs explicitly listed in the manifest despite the general `*.log`
ignore rule. The sibling private `manuscript/` directory is never included.
The helper verifies the remote commit after pushing and prints its permanent URL.
