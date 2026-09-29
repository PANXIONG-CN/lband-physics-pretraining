"""Prepare paired SPM/I2EM samples in the observed SMAPVEX12 parameter domain.

The dense SPM set and sparse I2EM train/test sets share the same parameter
box and validity rules.  I2EM values are intentionally produced by the
separate MATLAB reference runner; this script never substitutes SPM for I2EM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from research_pilots.scattering.surfaces.dielectric import (  # noqa: E402
    complex_relative_permittivity,
    topp_real_permittivity,
)
from research_pilots.scattering.surfaces.rough_surface import (  # noqa: E402
    spm_validity,
)
from research_pilots.scattering.surfaces.spm import spm_backscatter_db  # noqa: E402
from research_pilots.scattering.surfaces.teacher_contract import (  # noqa: E402
    validate_teacher_requests,
)


FEATURE_NAMES = (
    "soil_moisture_m3_m3",
    "soil_real_dielectric",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
)


def fingerprint(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read_cohort(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(FEATURE_NAMES).difference(frame.columns))
    if missing:
        raise ValueError(f"Input cohort is missing columns: {missing}")
    for column in FEATURE_NAMES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    finite = np.isfinite(frame[list(FEATURE_NAMES)].to_numpy(dtype=float)).all(axis=1)
    result = frame.loc[finite].copy()
    if len(result) < 20:
        raise ValueError("At least 20 finite cohort rows are required")
    return result


def parameter_domain(
    cohort: pd.DataFrame, lower_quantile: float, upper_quantile: float
) -> dict[str, tuple[float, float]]:
    moisture = cohort["soil_moisture_m3_m3"].to_numpy(dtype=float)
    topp = np.asarray(topp_real_permittivity(moisture), dtype=float)
    dielectric_factor = cohort["soil_real_dielectric"].to_numpy(dtype=float) / topp
    sources = {
        "soil_moisture_m3_m3": moisture,
        "dielectric_factor_relative_to_topp": dielectric_factor,
        "pals_rms_height_cm": cohort["pals_rms_height_cm"].to_numpy(dtype=float),
        "pals_correlation_length_cm": cohort[
            "pals_correlation_length_cm"
        ].to_numpy(dtype=float),
    }
    return {
        name: (
            float(np.quantile(values, lower_quantile)),
            float(np.quantile(values, upper_quantile)),
        )
        for name, values in sources.items()
    }


def sample_valid_domain(
    count: int,
    domain: dict[str, tuple[float, float]],
    frequency_hz: float,
    seed: int,
    ks_limit: float,
    slope_limit: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if count < 16:
        raise ValueError("sample count must be at least 16")
    accepted: list[np.ndarray] = []
    attempted = 0
    batch_index = 0
    while sum(len(batch) for batch in accepted) < count:
        batch_size = max(256, 2 * (count - sum(len(batch) for batch in accepted)))
        sampler = qmc.LatinHypercube(d=4, seed=seed + batch_index * 1009)
        unit = sampler.random(batch_size)
        names = list(domain)
        lower = np.array([domain[name][0] for name in names], dtype=float)
        upper = np.array([domain[name][1] for name in names], dtype=float)
        raw = qmc.scale(unit, lower, upper)
        moisture = raw[:, 0]
        dielectric = np.asarray(topp_real_permittivity(moisture)) * raw[:, 1]
        candidates = np.column_stack([moisture, dielectric, raw[:, 2], raw[:, 3]])
        validity = spm_validity(
            candidates[:, 2] / 100.0,
            candidates[:, 3] / 100.0,
            frequency_hz,
            ks_limit=ks_limit,
            slope_limit=slope_limit,
        )
        accepted.append(candidates[validity["valid"]])
        attempted += batch_size
        batch_index += 1
        if attempted > 5_000_000:
            raise RuntimeError("Could not draw enough points inside the shared validity domain")
    values = np.vstack(accepted)[:count]
    frame = pd.DataFrame(values, columns=FEATURE_NAMES)
    diagnostics = spm_validity(
        frame["pals_rms_height_cm"].to_numpy() / 100.0,
        frame["pals_correlation_length_cm"].to_numpy() / 100.0,
        frequency_hz,
        ks_limit=ks_limit,
        slope_limit=slope_limit,
    )
    frame["spm_k_rms_height"] = diagnostics["k_rms_height"]
    frame["spm_rms_slope_proxy"] = diagnostics["rms_slope_proxy"]
    frame["spm_valid"] = diagnostics["valid"]
    return frame, {
        "attempted": int(attempted),
        "accepted_returned": int(count),
        "acceptance_fraction": float(count / attempted),
    }


def add_spm_targets(
    frame: pd.DataFrame,
    frequency_hz: float,
    incidence_angle_deg: float,
    correlation_model: str,
    loss_tangent: float,
) -> pd.DataFrame:
    result = frame.copy()
    prediction = spm_backscatter_db(
        complex_relative_permittivity(
            result["soil_real_dielectric"].to_numpy(dtype=float),
            loss_tangent=loss_tangent,
        ),
        result["pals_rms_height_cm"].to_numpy(dtype=float) / 100.0,
        result["pals_correlation_length_cm"].to_numpy(dtype=float) / 100.0,
        frequency_hz,
        incidence_angle_deg,
        spectrum_model=correlation_model,
    )
    result["spm_hh_db"] = prediction["hh_db"]
    result["spm_vv_db"] = prediction["vv_db"]
    return result


def to_i2em_requests(
    frame: pd.DataFrame,
    split: str,
    frequency_ghz: float,
    incidence_angle_deg: float,
    correlation_model: str,
    loss_tangent: float,
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result.insert(0, "request_id", [f"{split}_{index:05d}" for index in range(len(result))])
    result.insert(1, "scenario", f"multifidelity_{split}")
    result.insert(2, "correlation_model", correlation_model)
    result["frequency_ghz"] = float(frequency_ghz)
    result["incidence_angle_deg"] = float(incidence_angle_deg)
    result["rms_height_m"] = result["pals_rms_height_cm"] / 100.0
    result["correlation_length_m"] = result["pals_correlation_length_cm"] / 100.0
    result["dielectric_real"] = result["soil_real_dielectric"]
    result["dielectric_loss_positive"] = result["dielectric_real"] * loss_tangent
    return validate_teacher_requests(result)


def save_domain_plot(
    cohort: pd.DataFrame,
    spm: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    output: Path,
) -> None:
    pairs = [
        ("soil_moisture_m3_m3", "soil_real_dielectric"),
        ("pals_correlation_length_cm", "pals_rms_height_cm"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, (x_name, y_name) in zip(axes, pairs):
        ax.scatter(spm[x_name], spm[y_name], s=8, alpha=0.12, label="dense SPM")
        ax.scatter(train[x_name], train[y_name], s=18, alpha=0.6, label="I2EM train")
        ax.scatter(test[x_name], test[y_name], s=22, marker="x", label="I2EM test")
        ax.scatter(cohort[x_name], cohort[y_name], s=14, c="black", alpha=0.45, label="observed cohort")
        ax.set(xlabel=x_name, ylabel=y_name)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare observed-domain multi-fidelity teacher samples")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spm-samples", type=int, default=4096)
    parser.add_argument("--i2em-train-samples", type=int, default=256)
    parser.add_argument("--i2em-test-samples", type=int, default=128)
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument("--loss-tangent", type=float, default=0.05)
    parser.add_argument("--correlation-model", choices=["exponential", "gaussian"], default="exponential")
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--spm-ks-limit", type=float, default=0.3)
    parser.add_argument("--spm-slope-limit", type=float, default=0.21)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")
    if args.loss_tangent < 0:
        raise ValueError("loss tangent must be non-negative")
    output = args.output.resolve()
    expected = output / "manifest.json"
    if expected.exists():
        raise FileExistsError("Output already exists; choose a new output directory")
    cohort_path = args.input.resolve()
    cohort = read_cohort(cohort_path)
    domain = parameter_domain(cohort, args.lower_quantile, args.upper_quantile)
    frequency_hz = args.frequency_ghz * 1e9
    spm, spm_sampling = sample_valid_domain(
        args.spm_samples, domain, frequency_hz, args.seed,
        args.spm_ks_limit, args.spm_slope_limit,
    )
    train, train_sampling = sample_valid_domain(
        args.i2em_train_samples, domain, frequency_hz, args.seed + 1,
        args.spm_ks_limit, args.spm_slope_limit,
    )
    test, test_sampling = sample_valid_domain(
        args.i2em_test_samples, domain, frequency_hz, args.seed + 2,
        args.spm_ks_limit, args.spm_slope_limit,
    )
    spm = add_spm_targets(spm, frequency_hz, args.incidence_angle_deg, args.correlation_model, args.loss_tangent)
    train = add_spm_targets(train, frequency_hz, args.incidence_angle_deg, args.correlation_model, args.loss_tangent)
    test = add_spm_targets(test, frequency_hz, args.incidence_angle_deg, args.correlation_model, args.loss_tangent)
    train_requests = to_i2em_requests(train, "train", args.frequency_ghz, args.incidence_angle_deg, args.correlation_model, args.loss_tangent)
    test_requests = to_i2em_requests(test, "test", args.frequency_ghz, args.incidence_angle_deg, args.correlation_model, args.loss_tangent)
    output.mkdir(parents=True, exist_ok=True)
    spm.to_csv(output / "spm_pretraining.csv", index=False)
    train_requests.to_csv(output / "i2em_train_requests.csv", index=False)
    test_requests.to_csv(output / "i2em_test_requests.csv", index=False)
    save_domain_plot(cohort, spm, train, test, output / "01_multifidelity_domain_coverage.png")
    manifest = {
        "research_question": "Does sparse I2EM refinement improve a dense SPM-pretrained surrogate inside the observed SMAPVEX12 parameter domain?",
        "input": fingerprint(cohort_path),
        "cohort": {
            "rows": int(len(cohort)),
            "fields": int(cohort["field_id"].nunique()) if "field_id" in cohort else None,
            "dates": int(cohort["acquisition_date"].nunique()) if "acquisition_date" in cohort else None,
        },
        "parameter_domain": {name: list(bounds) for name, bounds in domain.items()},
        "sampling": {"spm": spm_sampling, "i2em_train": train_sampling, "i2em_test": test_sampling},
        "settings": {
            "frequency_ghz": args.frequency_ghz,
            "incidence_angle_deg": args.incidence_angle_deg,
            "loss_tangent": args.loss_tangent,
            "correlation_model": args.correlation_model,
            "spm_ks_limit": args.spm_ks_limit,
            "spm_slope_limit": args.spm_slope_limit,
            "seed": args.seed,
        },
        "status": "awaiting_official_i2em_train_and_test_results",
        "guardrails": [
            "The 1st-99th percentile box is observed-domain interpolation support, not universal model validity.",
            "Moisture and dielectric are coupled through the observed Topp-ratio distribution.",
            "The exponential spectrum is the primary model form; Gaussian is a separate sensitivity experiment.",
            "I2EM is a higher-fidelity physics teacher, not measurement truth.",
            "Loss tangent 0.05 is an explicit nominal assumption supported only by the completed local sensitivity analysis.",
        ],
        "code": fingerprint(Path(__file__)),
    }
    expected.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(spm)} SPM, {len(train_requests)} I2EM-train, and {len(test_requests)} I2EM-test samples in {output}")


if __name__ == "__main__":
    main()
