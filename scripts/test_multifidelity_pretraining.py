"""Fast numerical checks for the multi-fidelity preparation/evaluation code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    prepare = load_script("prepare_multifidelity_pretraining")
    evaluate = load_script("evaluate_multifidelity_pretraining")
    generator = np.random.default_rng(7)
    moisture = generator.uniform(0.08, 0.45, 64)
    cohort = pd.DataFrame(
        {
            "soil_moisture_m3_m3": moisture,
            "soil_real_dielectric": 3.5 + 65 * moisture,
            "pals_rms_height_cm": generator.uniform(0.4, 1.0, 64),
            "pals_correlation_length_cm": generator.uniform(6, 18, 64),
        }
    )
    domain = prepare.parameter_domain(cohort, 0.01, 0.99)
    first, _ = prepare.sample_valid_domain(32, domain, 1.26e9, 11, 0.3, 0.21)
    second, _ = prepare.sample_valid_domain(32, domain, 1.26e9, 11, 0.3, 0.21)
    assert np.allclose(first[list(prepare.FEATURE_NAMES)], second[list(prepare.FEATURE_NAMES)])
    with_targets = prepare.add_spm_targets(first, 1.26e9, 40.0, "exponential", 0.05)
    assert np.isfinite(with_targets[["spm_hh_db", "spm_vv_db"]]).all().all()
    requests = prepare.to_i2em_requests(with_targets, "test", 1.26, 40.0, "exponential", 0.05)
    assert requests.i2em_request_valid.all()
    channels = with_targets[["spm_hh_db", "spm_vv_db"]].to_numpy()
    assert np.allclose(evaluate.from_components(evaluate.to_components(channels)), channels)
    perturbed = requests.copy()
    perturbed["i2em_hh_db"] = perturbed["spm_hh_db"] + 0.2
    perturbed["i2em_vv_db"] = perturbed["spm_vv_db"] - 0.1
    diagnostics = evaluate.add_disagreement_diagnostics(perturbed)
    assert np.allclose(diagnostics.teacher_delta_hh_db, 0.2)
    assert np.allclose(diagnostics.teacher_delta_vv_db, -0.1)
    print("multi-fidelity checks passed: deterministic sampling, finite teachers, valid requests, reversible components, discrepancy diagnostics")


if __name__ == "__main__":
    main()
