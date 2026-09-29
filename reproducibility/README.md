# Compact reproducibility bundle

This directory contains processed tables and frozen results needed to inspect
the paper-facing analyses without redistributing the 17.3 GB raw-data store.

- `data/source/` contains the 240-row SMAPVEX12 source table.
- `data/target/` contains the 189-row SMEX02 target table.
- `data/teachers/` contains SPM samples and paired I2EM request/result tables.
- `results/` contains frozen metrics, predictions used by diagnostics, and
  protocol summaries.
- `results/advisor_final/` contains the common-cohort training-stage diagnosis
  and the direct-physics few-shot channel-offset control.
- `figures/` contains the final paper-facing vector and raster figures.
- `sha256_manifest.csv` records file integrity.

The tables are derived from the official products listed in `DATA.md`. They do
not replace the source-product citations.
