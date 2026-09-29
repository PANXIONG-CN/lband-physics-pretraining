# Data access and release policy

Project repository: https://github.com/PANXIONG-CN/lband-physics-pretraining

This document describes the complete accompanying version 0.2.1 snapshot.
The approximately 34 MB code/data directory is prepared for ordinary Git
publication; the largest file is approximately 7.3 MB. The current upload
status is recorded in README.md. Original observation products are obtained
from the providers below.

## What is included

`reproducibility/` contains compact field-day tables, frozen teacher tables,
numerical summaries, final figures, and SHA-256 checksums needed to inspect and
repeat the reported analyses. These files are derived research artifacts, not
substitutes for the official source products.

## What is not included

Raw NSIDC products and the external I²EM MATLAB reference implementation are
not bundled. Download them from their original providers and preserve their
terms. The release does not contain unrelated MSTAR or AdaptSAPS data.

## Official products used by the rough-surface study

### SMAPVEX12 source campaign

- PALS backscatter, SV12PLBK, Version 1.
  DOI: https://doi.org/10.5067/KQC0KOL4DK1I
- Probe-based agricultural soil moisture, SV12PSMA, Version 1.
  DOI: https://doi.org/10.5067/3ELRJAUL0A4G
- Agricultural surface roughness, SV12SRA, Version 1.
  DOI: https://doi.org/10.5067/QB4JHGKXH16O
- In situ agricultural vegetation, SV12VA, Version 1 (VWC control reader).
  DOI: https://doi.org/10.5067/X2EF9ZKL0DGC
  The code uses crop biomass records and same-field nearest-date matching.
  Historical matched/OOF control tables are not included. SV12VWC map readers
  may also exist in helper code but are not substituted for this in situ product.

### SMEX02 independent campaign

- PALS data, NSIDC-0183, Version 1.
  DOI: https://doi.org/10.5067/D7IROA0Q6JEQ
- Walnut Creek watershed soil moisture, NSIDC-0143, Version 1.
  DOI: https://doi.org/10.5067/OAO4SU0XZLGR
- Geolocation, surface roughness, and photographs, NSIDC-0204, Version 1.
  DOI: https://doi.org/10.5067/R9AA6FC58HES

Official product terms and required citations govern the source data. The
repository's derived-artifact license does not relicense those products. No
blanket public-domain claim is made for all third-party inputs.

## Expected local layout

```text
data/scattering/raw/
├── SMAPVEX12/
└── SMEX02/
```

The readers accept explicit paths. Use `--help` on each script before running
the raw-data rebuild. Local Earthdata credentials must stay outside this
repository.

## Data license for repository artifacts

Original derived tables, summaries, and figures in `reproducibility/` are
offered under Creative Commons Attribution 4.0 where the contributors hold the
necessary rights. Source-product terms and required citations remain in force.
License text: https://creativecommons.org/licenses/by/4.0/legalcode
