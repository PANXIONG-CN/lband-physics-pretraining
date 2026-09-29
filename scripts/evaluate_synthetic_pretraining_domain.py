"""Validate the synthetic SPM pretraining surrogate on independent domains.

This experiment evaluates approximation of the same low-fidelity SPM teacher.
It does not validate SPM against field observations or a high-fidelity solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
from research_pilots.scattering.surrogate.physics_pretraining import (  # noqa: E402
    FEATURE_NAMES,
    generate_synthetic_spm_dataset,
    predict_physical_units,
    pretrain_physics_model,
)


TARGET_NAMES = ("HH", "VV")
TRAINING_BOX = {
    "soil_moisture_m3_m3": (0.03, 0.65),
    "soil_real_dielectric": (3.0, 70.0),
    "pals_rms_height_cm": (0.20, 1.50),
    "pals_correlation_length_cm": (3.0, 25.0),
}


def fingerprint(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def parse_sample_sizes(value: str) -> list[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or sizes[0] < 100:
        raise ValueError("sample sizes must be comma-separated integers >= 100")
    return sizes


def teacher_targets(
    features: np.ndarray,
    frequency_hz: float,
    incidence_angle_deg: float,
) -> np.ndarray:
    result = spm_backscatter_db(
        complex_relative_permittivity(features[:, 1]),
        features[:, 2] / 100.0,
        features[:, 3] / 100.0,
        frequency_hz,
        incidence_angle_deg,
        spectrum_model="exponential",
    )
    return np.column_stack([result["hh_db"], result["vv_db"]])


def sample_candidate_features(count: int, generator: np.random.Generator) -> np.ndarray:
    moisture = generator.uniform(0.01, 0.75, count)
    dielectric = np.clip(
        topp_real_permittivity(moisture) * generator.uniform(0.60, 1.70, count),
        2.5,
        80.0,
    )
    rms_height_cm = generator.uniform(0.10, 1.50, count)
    correlation_length_cm = generator.uniform(2.0, 35.0, count)
    return np.column_stack(
        [moisture, dielectric, rms_height_cm, correlation_length_cm]
    )


def sample_special_domain(
    sample_count: int,
    frequency_hz: float,
    incidence_angle_deg: float,
    seed: int,
    domain: str,
    ks_limit: float = 0.3,
    slope_limit: float = 0.21,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Draw SPM-valid points near its boundary or outside the training box."""
    if sample_count < 100:
        raise ValueError("sample_count must be at least 100")
    if domain not in {"near_validity_boundary", "range_extension"}:
        raise ValueError(f"unsupported domain: {domain}")
    generator = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    diagnostics: list[np.ndarray] = []
    attempted = 0
    accepted_count = 0
    while accepted_count < sample_count:
        batch_size = max(4000, 4 * (sample_count - accepted_count))
        candidates = sample_candidate_features(batch_size, generator)
        validity = spm_validity(
            candidates[:, 2] / 100.0,
            candidates[:, 3] / 100.0,
            frequency_hz,
            ks_limit=ks_limit,
            slope_limit=slope_limit,
        )
        normalized_margin = np.maximum(
            validity["k_rms_height"] / ks_limit,
            validity["rms_slope_proxy"] / slope_limit,
        )
        if domain == "near_validity_boundary":
            selector = validity["valid"] & (normalized_margin >= 0.80)
        else:
            outside = np.zeros(batch_size, dtype=bool)
            for index, name in enumerate(FEATURE_NAMES):
                lower, upper = TRAINING_BOX[name]
                outside |= (candidates[:, index] < lower) | (
                    candidates[:, index] > upper
                )
            selector = validity["valid"] & outside
        accepted.append(candidates[selector])
        diagnostics.append(normalized_margin[selector])
        accepted_count += int(selector.sum())
        attempted += batch_size
        if attempted > 20_000_000:
            raise RuntimeError(f"could not sample enough points for {domain}")
    features = np.vstack(accepted)[:sample_count]
    margin = np.concatenate(diagnostics)[:sample_count]
    targets = teacher_targets(features, frequency_hz, incidence_angle_deg)
    metadata = {
        "domain": domain,
        "sample_count": int(sample_count),
        "attempted_candidates": int(attempted),
        "acceptance_fraction": float(sample_count / attempted),
        "normalized_validity_margin_range": [float(margin.min()), float(margin.max())],
        "feature_ranges": {
            name: [float(features[:, i].min()), float(features[:, i].max())]
            for i, name in enumerate(FEATURE_NAMES)
        },
    }
    return features, targets, metadata


def error_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - reference
    denominator = float(np.sum((reference - reference.mean()) ** 2))
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "p95_absolute_error_db": float(np.quantile(np.abs(error), 0.95)),
        "maximum_absolute_error_db": float(np.max(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "r_squared": float(1.0 - np.sum(error**2) / denominator)
        if denominator > 0
        else float("nan"),
    }


def metric_rows(
    reference: np.ndarray,
    prediction: np.ndarray,
    domain: str,
    sample_size: int,
    repeat: int,
) -> list[dict[str, object]]:
    components = {
        "HH": (reference[:, 0], prediction[:, 0]),
        "VV": (reference[:, 1], prediction[:, 1]),
        "common_HH_VV": (reference.mean(axis=1), prediction.mean(axis=1)),
        "difference_VV_minus_HH": (
            reference[:, 1] - reference[:, 0],
            prediction[:, 1] - prediction[:, 0],
        ),
    }
    return [
        {
            "training_samples": int(sample_size),
            "repeat": int(repeat),
            "domain": domain,
            "component": component,
            "n": int(len(reference)),
            **error_metrics(y, p),
        }
        for component, (y, p) in components.items()
    ]


def save_plots(
    metrics: pd.DataFrame,
    ensemble: pd.DataFrame,
    output: Path,
) -> None:
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = {"interpolation": "#2878B5", "near_validity_boundary": "#C82423", "range_extension": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, component in zip(axes, TARGET_NAMES):
        subset = metrics[metrics.component == component]
        grouped = subset.groupby(["training_samples", "domain"]).rmse_db.agg(["mean", "std"]).reset_index()
        for domain, part in grouped.groupby("domain", sort=False):
            ax.errorbar(part.training_samples, part["mean"], yerr=part["std"].fillna(0), marker="o", capsize=3, label=domain.replace("_", " "), color=colors[domain])
        ax.set(xscale="log", xlabel="Synthetic training samples", ylabel="RMSE against SPM teacher (dB)", title=component)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "01_synthetic_learning_curve.png", dpi=180)
    plt.close(fig)

    maximum = metrics.training_samples.max()
    subset = metrics[(metrics.training_samples == maximum) & metrics.component.isin(["HH", "VV"])]
    summary = subset.groupby(["domain", "component"]).rmse_db.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    labels = [f"{row.domain.replace('_', ' ')}\n{row.component}" for row in summary.itertuples()]
    ax.bar(labels, summary["mean"], yerr=summary["std"].fillna(0), capsize=3, color=[colors[d] for d in summary.domain])
    ax.set(ylabel="RMSE against SPM teacher (dB)", title=f"Independent-domain validation at n={maximum}")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "02_independent_domain_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    for row, domain in enumerate(["interpolation", "near_validity_boundary", "range_extension"]):
        part = ensemble[ensemble.domain == domain]
        for column, component in enumerate(TARGET_NAMES):
            reference = part[f"reference_{component.lower()}_db"].to_numpy()
            prediction = part[f"prediction_{component.lower()}_db"].to_numpy()
            lower = min(reference.min(), prediction.min())
            upper = max(reference.max(), prediction.max())
            axes[row, column].scatter(reference, prediction, s=11, alpha=0.45, edgecolors="none")
            axes[row, column].plot([lower, upper], [lower, upper], "k--", lw=1)
            axes[row, column].set(xlabel="SPM teacher (dB)", ylabel="Surrogate (dB)", title=f"{domain.replace('_', ' ')} | {component}")
            axes[row, column].grid(alpha=0.2)
    fig.savefig(output / "03_teacher_vs_surrogate.png", dpi=180)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    sample_sizes = parse_sample_sizes(args.sample_sizes)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is nonempty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    frequency_hz = args.frequency_ghz * 1e9

    train_features, train_targets, train_metadata = generate_synthetic_spm_dataset(
        max(sample_sizes), frequency_hz, args.incidence_angle_deg, seed=args.seed
    )
    interpolation = generate_synthetic_spm_dataset(
        args.validation_samples,
        frequency_hz,
        args.incidence_angle_deg,
        seed=args.seed + 1,
    )
    boundary = sample_special_domain(
        args.validation_samples,
        frequency_hz,
        args.incidence_angle_deg,
        args.seed + 2,
        "near_validity_boundary",
    )
    extension = sample_special_domain(
        args.validation_samples,
        frequency_hz,
        args.incidence_angle_deg,
        args.seed + 3,
        "range_extension",
    )
    domains = {
        "interpolation": interpolation[:2],
        "near_validity_boundary": boundary[:2],
        "range_extension": extension[:2],
    }
    domain_metadata = {
        "interpolation": interpolation[2],
        "near_validity_boundary": boundary[2],
        "range_extension": extension[2],
    }

    rows: list[dict[str, object]] = []
    largest_predictions: dict[str, list[np.ndarray]] = {name: [] for name in domains}
    training_records: list[dict[str, object]] = []
    for sample_size in sample_sizes:
        for repeat in range(1, args.repeats + 1):
            seed = args.seed + repeat * 100_000 + sample_size
            bundle = pretrain_physics_model(
                train_features[:sample_size],
                train_targets[:sample_size],
                epochs=args.epochs,
                seed=seed,
            )
            training_records.append(
                {"training_samples": sample_size, "repeat": repeat, "seed": seed, **bundle.metadata}
            )
            for domain, (features, targets) in domains.items():
                prediction = predict_physical_units(bundle, bundle.model, features)
                rows.extend(metric_rows(targets, prediction, domain, sample_size, repeat))
                if sample_size == max(sample_sizes):
                    largest_predictions[domain].append(prediction)
            print(f"completed n={sample_size}, repeat={repeat}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["training_samples", "domain", "component"], as_index=False).agg(
        rmse_mean_db=("rmse_db", "mean"),
        rmse_std_db=("rmse_db", "std"),
        mae_mean_db=("mae_db", "mean"),
        p95_absolute_error_mean_db=("p95_absolute_error_db", "mean"),
    )
    ensemble_parts = []
    for domain, (features, targets) in domains.items():
        prediction = np.mean(largest_predictions[domain], axis=0)
        part = pd.DataFrame(features, columns=FEATURE_NAMES)
        part.insert(0, "domain", domain)
        part["reference_hh_db"] = targets[:, 0]
        part["reference_vv_db"] = targets[:, 1]
        part["prediction_hh_db"] = prediction[:, 0]
        part["prediction_vv_db"] = prediction[:, 1]
        ensemble_parts.append(part)
    ensemble = pd.concat(ensemble_parts, ignore_index=True)

    metrics.to_csv(output / "metrics_by_repeat.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    ensemble.to_csv(output / "largest_model_ensemble_predictions.csv", index=False)
    save_plots(metrics, ensemble, output)
    payload = {
        "research_question": "How many synthetic samples are needed to approximate the low-fidelity exponential-SPM teacher, and where does that surrogate lose fidelity?",
        "sample_sizes": sample_sizes,
        "repeats": int(args.repeats),
        "epochs": int(args.epochs),
        "frequency_hz": float(frequency_hz),
        "incidence_angle_deg": float(args.incidence_angle_deg),
        "training_domain": train_metadata,
        "validation_domains": domain_metadata,
        "training_records": training_records,
        "code": fingerprint(Path(__file__)),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "interpretation_guardrails": [
            "All references are generated by the same exponential SPM teacher; this is surrogate verification, not validation of physical truth.",
            "Range extension remains inside the configured SPM validity inequalities but outside at least one pretraining feature interval.",
            "Do not claim computational acceleration over SPM without a separate timing study against a genuinely expensive reference solver.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"outputs saved to {output}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent-domain validation of the synthetic SPM pretraining surrogate")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-sizes", default="500,1000,3000,5000")
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.validation_samples < 100 or args.repeats < 1 or args.epochs < 1:
        raise ValueError("validation-samples >= 100, repeats >= 1, and epochs >= 1 are required")
    run_experiment(args)


if __name__ == "__main__":
    main()
