"""Diagnose where differential-response skill is lost on the shared SMEX02 cohort.

The script reconstructs the pretraining-only sequential surrogate from frozen
simulation tables and the original seeds.  It does not fit on SMEX02 labels and
does not select a model from target-domain scores.  Source-fine-tuned and
risk-shrunk predictions are read from the frozen prospective evaluation.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def centered_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed
    observed_centered = observed - observed.mean()
    predicted_centered = predicted - predicted.mean()
    centered_error = predicted_centered - observed_centered
    observed_variance = float(np.mean(observed_centered**2))
    predicted_variance = float(np.mean(predicted_centered**2))
    centered_mse = float(np.mean(centered_error**2))
    return {
        "n": int(len(observed)),
        "bias_db": float(np.mean(error)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "centered_rmse_db": float(np.sqrt(centered_mse)),
        "centered_skill": float(1.0 - centered_mse / observed_variance),
        "variance_ratio": float(predicted_variance / observed_variance),
    }


def field_bootstrap_indices(
    fields: np.ndarray,
    iterations: int,
    seed: int,
) -> list[np.ndarray]:
    unique_fields = np.unique(fields.astype(str))
    members = {
        field: np.flatnonzero(fields.astype(str) == field)
        for field in unique_fields
    }
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(iterations):
        sampled = rng.choice(unique_fields, size=len(unique_fields), replace=True)
        draws.append(np.concatenate([members[field] for field in sampled]))
    return draws


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def save_figure(
    metrics: pd.DataFrame,
    output: Path,
) -> None:
    stage_order = [
        "i2em_fixed40_teacher",
        "pretraining_only_surrogate",
        "source_finetuned_surrogate",
        "risk_shrunk_surrogate",
    ]
    labels = [
        "I$^2$EM\nteacher",
        "Pretraining-only\nsurrogate",
        "Source-fine-tuned\nsurrogate",
        "Risk-shrunk\nsurrogate",
    ]
    colors = ["#E69F00", "#0072B2", "#D55E00", "#56B4E9"]
    selected = metrics.set_index("method").loc[stage_order]
    x = np.arange(len(stage_order))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True)

    skill = selected["centered_skill"].to_numpy(float)
    skill_low = selected["centered_skill_ci_low"].to_numpy(float)
    skill_high = selected["centered_skill_ci_high"].to_numpy(float)
    axes[0].errorbar(
        x,
        skill,
        yerr=np.vstack([skill - skill_low, skill_high - skill]),
        fmt="none",
        ecolor="#444444",
        capsize=2.5,
        zorder=1,
    )
    axes[0].scatter(x, skill, c=colors, marker="o", zorder=2)
    axes[0].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Centered differential skill")

    bias = selected["bias_db"].to_numpy(float)
    bias_low = selected["bias_db_ci_low"].to_numpy(float)
    bias_high = selected["bias_db_ci_high"].to_numpy(float)
    axes[1].errorbar(
        x,
        bias,
        yerr=np.vstack([bias - bias_low, bias_high - bias]),
        fmt="none",
        ecolor="#444444",
        capsize=2.5,
        zorder=1,
    )
    axes[1].scatter(x, bias, c=colors, marker="o", zorder=2)
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Mean differential error (dB)")

    for label, axis in zip(["(a)", "(b)"], axes):
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.text(
            0.01,
            0.98,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    stem = output / "Fig_stage_response_retention"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_angle_control_figure(metrics: pd.DataFrame, output: Path) -> None:
    order = ["i2em_fixed40_teacher", "i2em_actual_angle_control"]
    labels = [r"Fixed $40^{\circ}$", "Field-day angle"]
    selected = metrics.set_index("method").loc[order]
    x = np.arange(2)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), constrained_layout=True)
    for axis, metric_name, ylabel in [
        (axes[0], "centered_skill", "Centered differential skill"),
        (axes[1], "bias_db", "Mean differential error (dB)"),
    ]:
        values = selected[metric_name].to_numpy(float)
        low = selected[f"{metric_name}_ci_low"].to_numpy(float)
        high = selected[f"{metric_name}_ci_high"].to_numpy(float)
        axis.errorbar(
            x,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="o",
            color="#E69F00",
            capsize=2.5,
        )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0].text(0.01, 0.98, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].text(0.01, 0.98, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")

    stem = output / "Fig_i2em_observation_condition_control"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    scripts = root / "scripts"
    source_root = root / "src"
    sys.path[:0] = [str(scripts), str(source_root)]
    import evaluate_decoupled_physics_heads as base
    import evaluate_multifidelity_pretraining as multi

    common_path = root / "reproducibility/results/common_cohort/predictions.csv"
    physics_path = root / "reproducibility/results/stage_diagnostics/stage_predictions.csv"
    spm_path = root / "reproducibility/data/teachers/spm_pretraining.csv"
    i2em_request_path = root / "reproducibility/data/teachers/i2em_train_requests.csv"
    i2em_result_path = root / "reproducibility/data/teachers/i2em_train_results.csv"
    input_paths = [
        common_path,
        physics_path,
        spm_path,
        i2em_request_path,
        i2em_result_path,
    ]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    common = pd.read_csv(common_path, dtype={"field_id": str})
    physics = pd.read_csv(physics_path, dtype={"field_id": str})
    spm = pd.read_csv(spm_path)
    i2em = multi.load_paired(i2em_request_path, i2em_result_path)
    keys = ["field_id", "acquisition_date"]
    columns = [*keys, "i2em_fixed40_hh_db", "i2em_fixed40_vv_db", "i2em_actual_hh_db", "i2em_actual_vv_db"]
    frame = common.merge(physics[columns], on=keys, validate="one_to_one")
    if len(frame) != 138 or frame["field_id"].nunique() != 22:
        raise AssertionError("Expected 138 rows from 22 SMEX02 fields")

    features = frame[list(base.FEATURE_NAMES)].to_numpy(float)
    spm_features = multi.features_from(spm)
    spm_differential = base.to_components(
        spm[["spm_hh_db", "spm_vv_db"]].to_numpy(float)
    )[:, 1]
    i2em_features = multi.features_from(i2em)
    i2em_differential = base.to_components(
        i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(float)
    )[:, 1]

    reconstructed = []
    reconstruction_metadata = []
    for repeat in range(1, args.repeats + 1):
        seed = args.seed + (repeat - 1) * 100_000
        spm_bundle = base.pretrain_single_output(
            spm_features, spm_differential, args.spm_epochs, seed
        )
        sequential_model = base.fine_tune_single(
            spm_bundle,
            i2em_features,
            i2em_differential,
            args.i2em_epochs,
            seed + 2,
        )
        reconstructed.append(base.predict_single(spm_bundle, sequential_model, features))
        reconstruction_metadata.append(
            {
                "repeat": repeat,
                "seed": seed,
                "spm_epochs": args.spm_epochs,
                "i2em_epochs": args.i2em_epochs,
                "source_observation_fitting": False,
            }
        )
    frame["differential_pretraining_only"] = np.mean(reconstructed, axis=0)

    observed = frame["sigma0_vv_db"].to_numpy(float) - frame["sigma0_hh_db"].to_numpy(float)
    predictions = {
        "i2em_fixed40_teacher": frame["i2em_fixed40_vv_db"].to_numpy(float)
        - frame["i2em_fixed40_hh_db"].to_numpy(float),
        "pretraining_only_surrogate": frame["differential_pretraining_only"].to_numpy(float),
        "source_finetuned_surrogate": frame["vv_spm_to_i2em"].to_numpy(float)
        - frame["hh_spm_to_i2em"].to_numpy(float),
        "risk_shrunk_surrogate": frame["vv_risk_spm_to_i2em"].to_numpy(float)
        - frame["hh_risk_spm_to_i2em"].to_numpy(float),
        "i2em_actual_angle_control": frame["i2em_actual_vv_db"].to_numpy(float)
        - frame["i2em_actual_hh_db"].to_numpy(float),
    }
    for method, values in predictions.items():
        frame[f"differential_{method}"] = values

    fields = frame["field_id"].astype(str).to_numpy()
    draws = field_bootstrap_indices(fields, args.bootstrap_iterations, args.seed + 7_000_000)
    bootstrap = {
        method: {"centered_skill": [], "bias_db": []}
        for method in predictions
    }
    for index in draws:
        for method, predicted in predictions.items():
            values = centered_metrics(observed[index], predicted[index])
            bootstrap[method]["centered_skill"].append(values["centered_skill"])
            bootstrap[method]["bias_db"].append(values["bias_db"])

    metric_rows = []
    for method, predicted in predictions.items():
        values = centered_metrics(observed, predicted)
        skill_samples = np.asarray(bootstrap[method]["centered_skill"])
        bias_samples = np.asarray(bootstrap[method]["bias_db"])
        metric_rows.append(
            {
                "method": method,
                **values,
                "centered_skill_ci_low": interval(skill_samples)[0],
                "centered_skill_ci_high": interval(skill_samples)[1],
                "centered_skill_probability_above_zero": float(np.mean(skill_samples > 0)),
                "bias_db_ci_low": interval(bias_samples)[0],
                "bias_db_ci_high": interval(bias_samples)[1],
            }
        )
    metrics = pd.DataFrame(metric_rows)

    stage_order = [
        "i2em_fixed40_teacher",
        "pretraining_only_surrogate",
        "source_finetuned_surrogate",
        "risk_shrunk_surrogate",
    ]
    pair_rows = []
    for first, second in zip(stage_order[:-1], stage_order[1:]):
        first_skill = np.asarray(bootstrap[first]["centered_skill"])
        second_skill = np.asarray(bootstrap[second]["centered_skill"])
        first_bias = np.asarray(bootstrap[first]["bias_db"])
        second_bias = np.asarray(bootstrap[second]["bias_db"])
        point_first = metrics.set_index("method").loc[first]
        point_second = metrics.set_index("method").loc[second]
        pair_rows.append(
            {
                "comparison_type": "training_stage",
                "first_stage": first,
                "second_stage": second,
                "centered_skill_delta_second_minus_first": float(
                    point_second["centered_skill"] - point_first["centered_skill"]
                ),
                "centered_skill_delta_ci_low": interval(second_skill - first_skill)[0],
                "centered_skill_delta_ci_high": interval(second_skill - first_skill)[1],
                "bias_delta_second_minus_first_db": float(
                    point_second["bias_db"] - point_first["bias_db"]
                ),
                "bias_delta_ci_low_db": interval(second_bias - first_bias)[0],
                "bias_delta_ci_high_db": interval(second_bias - first_bias)[1],
            }
        )

    first = "i2em_fixed40_teacher"
    second = "i2em_actual_angle_control"
    first_skill = np.asarray(bootstrap[first]["centered_skill"])
    second_skill = np.asarray(bootstrap[second]["centered_skill"])
    first_bias = np.asarray(bootstrap[first]["bias_db"])
    second_bias = np.asarray(bootstrap[second]["bias_db"])
    point_first = metrics.set_index("method").loc[first]
    point_second = metrics.set_index("method").loc[second]
    pair_rows.append(
        {
            "comparison_type": "observation_condition_control",
            "first_stage": first,
            "second_stage": second,
            "centered_skill_delta_second_minus_first": float(
                point_second["centered_skill"] - point_first["centered_skill"]
            ),
            "centered_skill_delta_ci_low": interval(second_skill - first_skill)[0],
            "centered_skill_delta_ci_high": interval(second_skill - first_skill)[1],
            "bias_delta_second_minus_first_db": float(
                point_second["bias_db"] - point_first["bias_db"]
            ),
            "bias_delta_ci_low_db": interval(second_bias - first_bias)[0],
            "bias_delta_ci_high_db": interval(second_bias - first_bias)[1],
        }
    )
    pairwise = pd.DataFrame(pair_rows)

    frame.to_csv(output / "stage_predictions.csv", index=False)
    metrics.to_csv(output / "stage_metrics.csv", index=False)
    pairwise.to_csv(output / "stage_pairwise_deltas.csv", index=False)
    save_figure(metrics, output)
    save_angle_control_figure(metrics, output)

    manifest = {
        "status": "COMPLETE",
        "analysis_role": "post-hoc stage-wise response-retention diagnosis",
        "target_label_use": "metrics only; no target-domain fitting or selection",
        "historical_weight_artifact_available": False,
        "physics_values": "read from the archived stage table, not regenerated by a physical solver",
        "pretraining_only_stage": (
            "deterministically reconstructed from frozen simulation tables, "
            "the original architecture, epochs, and five source seeds"
        ),
        "source_finetuned_and_risk_stages": "read from frozen prospective predictions",
        "cohort": {"rows": len(frame), "fields": int(frame.field_id.nunique())},
        "bootstrap": {
            "unit": "field_id",
            "paired_across_methods": True,
            "iterations": args.bootstrap_iterations,
            "seed": args.seed + 7_000_000,
        },
        "reconstruction": reconstruction_metadata,
        "inputs": {str(path.relative_to(root)): sha256(path) for path in input_paths},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
