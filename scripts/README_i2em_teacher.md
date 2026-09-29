# I2EM reference-teacher validation

This stage does not treat I2EM as ground truth.  It creates a traceable set of
cohort-derived anchors, enforces the published reference-code limits `ks < 1`
and `RMS height / correlation length <= 0.25`, and compares I2EM with the
existing SPM implementation before any I2EM samples enter pretraining.

Prepare the request grid on Windows:

```powershell
python scripts/prepare_i2em_validation_grid.py `
  --input "outputs/scattering/rough_ground/paper_diagnostics/cohort_common.csv" `
  --output "outputs/scattering/rough_ground/i2em_validation_20260909" `
  --frequency-ghz 1.26 `
  --incidence-angle-deg 40 `
  --loss-tangent 0
```

The current Windows MATLAB installation uses account-based online licensing;
a new `matlab -batch` process may not see a local licence file even though a
COM automation session is licensed.  The project therefore keeps two
interchangeable execution adapters:

1. `run_i2em_teacher.py` uses `pyi2em==0.1.5`.  That release provides CPython
   3.11 Linux wheels, not native Windows wheels.  Run it in a pinned Linux/WSL
   environment.
2. `run_i2em_requests_matlab.m` calls the official IEEE-GRSS
   `I2EM_Backscatter_model.m`; `run_i2em_requests_matlab_com.ps1` exposes that
   adapter to the Windows terminal when account-based licensing is used.  The
   official model remains isolated under `external/i2em_reference`.

After a backend produces `i2em_results.csv`, validate it with:

```powershell
python scripts/evaluate_i2em_reference.py `
  --requests "outputs/scattering/rough_ground/i2em_validation_20260909/i2em_requests.csv" `
  --results "outputs/scattering/rough_ground/i2em_validation_20260909/i2em_results.csv" `
  --output "outputs/scattering/rough_ground/i2em_validation_20260909/evaluation"
```

Do not begin I2EM pretraining until the anchor comparison is finite, trend
checks are physically plausible, and the dielectric-loss sensitivity is
reported.

## Dielectric-loss sensitivity experiment

The sensitivity design holds the cohort-derived physical anchors fixed and
varies only the dielectric loss tangent over `0, 0.02, 0.05, 0.10`.  It creates
72 requests: 18 anchors for each loss level.

```powershell
python scripts/prepare_i2em_loss_sensitivity.py `
  --input "outputs/scattering/rough_ground/paper_diagnostics/cohort_common.csv" `
  --output "outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1" `
  --frequency-ghz 1.26 `
  --incidence-angle-deg 40 `
  --loss-tangents 0 0.02 0.05 0.10
```

After MATLAB is activated and the official `I2EM_Backscatter_model.m` is in a
local reference-code directory, run all requests in one MATLAB batch:

```powershell
 matlab -batch "addpath('PROJECT_ROOT/scripts'); run_i2em_requests_matlab('PROJECT_ROOT/outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1/i2em_loss_sensitivity_requests.csv','PROJECT_ROOT/external/i2em_reference','PROJECT_ROOT/outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1/i2em_loss_sensitivity_results.csv')"
```

If the MATLAB installation uses account-based online licensing and
`matlab -batch` cannot see a local license file, use the COM adapter instead:

```powershell
& scripts/run_i2em_requests_matlab_com.ps1 `
  -RequestCsv "outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1/i2em_loss_sensitivity_requests.csv" `
  -OutputCsv "outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1/i2em_loss_sensitivity_results.csv" `
  -ModelDirectory "external/i2em_reference"
```

Then generate the sensitivity tables and figures:

```powershell
python scripts/evaluate_i2em_loss_sensitivity.py `
  --requests "outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1/i2em_loss_sensitivity_requests.csv" `
  --results "outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1/i2em_loss_sensitivity_results.csv" `
  --output "outputs/scattering/rough_ground/i2em_loss_sensitivity_20260909_v1/evaluation"
```

Interpret changes relative to the zero-loss case as parameter sensitivity.
Interpret I2EM-minus-SPM differences as disagreement between physics teachers,
not as I2EM error and not as agreement with observations.
