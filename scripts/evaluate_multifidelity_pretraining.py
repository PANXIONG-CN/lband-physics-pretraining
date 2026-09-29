"""Evaluate SPM-only, I2EM-only, and sequential SPM-to-I2EM pretraining.

All reported errors use an independent I2EM test set.  This establishes
physics-teacher approximation only; it does not establish field generalization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from research_pilots.scattering.surfaces.teacher_contract import (  # noqa: E402
    validate_teacher_results,
)
from research_pilots.scattering.surrogate.physics_pretraining import (  # noqa: E402
    FEATURE_NAMES,
    make_mlp,
    train_epochs,
)


METHODS = ["spm_teacher", "i2em_train_mean", "spm_only", "i2em_only", "spm_to_i2em"]
RESPONSES = ["HH", "VV", "common", "differential"]


def fingerprint(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def features_from(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(FEATURE_NAMES).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    values = frame[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    return values


def to_components(channels: np.ndarray) -> np.ndarray:
    channels = np.asarray(channels, dtype=float)
    return np.column_stack([(channels[:, 0] + channels[:, 1]) / 2.0, channels[:, 1] - channels[:, 0]])


def from_components(components: np.ndarray) -> np.ndarray:
    components = np.asarray(components, dtype=float)
    return np.column_stack([components[:, 0] - components[:, 1] / 2.0, components[:, 0] + components[:, 1] / 2.0])


def metric(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(reference, dtype=float)
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "p95_absolute_error_db": float(np.quantile(np.abs(error), 0.95)),
        "r_squared": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


def train_component_models(
    spm_features: np.ndarray,
    spm_components: np.ndarray,
    i2em_features: np.ndarray,
    i2em_components: np.ndarray,
    test_features: np.ndarray,
    spm_epochs: int,
    i2em_epochs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    feature_scaler = StandardScaler().fit(spm_features)
    target_scaler = StandardScaler().fit(spm_components)
    x_spm = feature_scaler.transform(spm_features)
    y_spm = target_scaler.transform(spm_components)
    x_i2em = feature_scaler.transform(i2em_features)
    y_i2em = target_scaler.transform(i2em_components)
    x_test = feature_scaler.transform(test_features)
    predictions = {name: np.zeros((len(test_features), 2), dtype=float) for name in ["spm_only", "i2em_only", "spm_to_i2em"]}
    for component_index in range(2):
        component_seed = seed + component_index * 10_000
        spm_model = make_mlp(component_seed, learning_rate=3e-3)
        train_epochs(spm_model, x_spm, y_spm[:, component_index], spm_epochs, component_seed)
        sequential_model = copy.deepcopy(spm_model)
        sequential_model.learning_rate_init = 1e-3
        train_epochs(sequential_model, x_i2em, y_i2em[:, component_index], i2em_epochs, component_seed + 1)
        i2em_model = make_mlp(component_seed, learning_rate=3e-3)
        train_epochs(i2em_model, x_i2em, y_i2em[:, component_index], spm_epochs + i2em_epochs, component_seed + 2)
        for name, model in [("spm_only", spm_model), ("i2em_only", i2em_model), ("spm_to_i2em", sequential_model)]:
            scaled = np.zeros((len(test_features), 2), dtype=float)
            scaled[:, component_index] = np.asarray(model.predict(x_test)).reshape(-1)
            # Inverse-transform one component without contaminating it with the other.
            predictions[name][:, component_index] = (
                scaled[:, component_index] * target_scaler.scale_[component_index]
                + target_scaler.mean_[component_index]
            )
    return predictions


def response_arrays(channels: np.ndarray) -> dict[str, np.ndarray]:
    components = to_components(channels)
    return {"HH": channels[:, 0], "VV": channels[:, 1], "common": components[:, 0], "differential": components[:, 1]}


def load_paired(request_path: Path, result_path: Path) -> pd.DataFrame:
    requests = pd.read_csv(request_path)
    results = pd.read_csv(result_path)
    return validate_teacher_results(requests, results)


def add_disagreement_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["i2em_validity_utilization"] = np.maximum(
        result["i2em_k_rms_height"].to_numpy(dtype=float),
        result["i2em_rms_to_correlation_ratio"].to_numpy(dtype=float) / 0.25,
    )
    for response in ["hh", "vv"]:
        result[f"teacher_delta_{response}_db"] = result[f"i2em_{response}_db"] - result[f"spm_{response}_db"]
        result[f"teacher_absolute_delta_{response}_db"] = result[f"teacher_delta_{response}_db"].abs()
    result["teacher_delta_common_db"] = (
        result["teacher_delta_hh_db"] + result["teacher_delta_vv_db"]
    ) / 2.0
    result["teacher_delta_differential_db"] = result["teacher_delta_vv_db"] - result["teacher_delta_hh_db"]
    result["teacher_joint_absolute_delta_db"] = np.sqrt(
        (result["teacher_delta_hh_db"] ** 2 + result["teacher_delta_vv_db"] ** 2) / 2.0
    )
    result["validity_stratum"] = pd.cut(
        result["i2em_validity_utilization"],
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=["interior_le_0.4", "middle_0.4_to_0.7", "nearer_boundary_gt_0.7"],
    ).astype("string")
    return result


def association_table(frame: pd.DataFrame) -> pd.DataFrame:
    parameters = [
        "soil_moisture_m3_m3", "soil_real_dielectric", "pals_rms_height_cm",
        "pals_correlation_length_cm", "i2em_validity_utilization",
    ]
    responses = ["teacher_absolute_delta_hh_db", "teacher_absolute_delta_vv_db", "teacher_joint_absolute_delta_db"]
    rows = []
    for parameter in parameters:
        for response in responses:
            statistic, pvalue = spearmanr(frame[parameter], frame[response])
            rows.append({"parameter": parameter, "disagreement": response, "spearman_rho": float(statistic), "p_value_unadjusted": float(pvalue), "n": int(len(frame))})
    return pd.DataFrame(rows)


def save_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    colors = {"spm_teacher": "#8C8C8C", "i2em_train_mean": "#B279A2", "spm_only": "#4C78A8", "i2em_only": "#F58518", "spm_to_i2em": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        part = metrics[metrics.response == response].groupby("method", as_index=False).rmse_db.mean()
        order = [name for name in METHODS if name in set(part.method)]
        values = [float(part.loc[part.method == name, "rmse_db"].iloc[0]) for name in order]
        ax.bar(range(len(order)), values, color=[colors[name] for name in order])
        ax.set_xticks(range(len(order)), [name.replace("_", "\n") for name in order], rotation=15)
        ax.set(ylabel="RMSE against independent I2EM (dB)", title=response)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "02_multifidelity_test_rmse.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, response in zip(axes.ravel(), RESPONSES):
        reference = predictions[f"reference_{response.lower()}_db"]
        for method in ["spm_only", "i2em_only", "spm_to_i2em"]:
            ax.scatter(reference, predictions[f"{method}_{response.lower()}_db"], s=12, alpha=0.4, label=method)
        lower = min(reference.min(), *(predictions[f"{m}_{response.lower()}_db"].min() for m in ["spm_only", "i2em_only", "spm_to_i2em"]))
        upper = max(reference.max(), *(predictions[f"{m}_{response.lower()}_db"].max() for m in ["spm_only", "i2em_only", "spm_to_i2em"]))
        ax.plot([lower, upper], [lower, upper], "k--", lw=1)
        ax.set(xlabel="I2EM reference (dB)", ylabel="Surrogate prediction (dB)", title=response)
        ax.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(output / "03_i2em_reference_vs_surrogates.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, response in zip(axes, ["hh", "vv"]):
        ax.scatter(paired["i2em_validity_utilization"], paired[f"teacher_absolute_delta_{response}_db"], s=16, alpha=0.55)
        ax.set(xlabel="I2EM validity-limit utilization", ylabel="|I2EM - SPM| (dB)", title=response.upper())
        ax.grid(alpha=0.2)
    fig.savefig(output / "04_teacher_disagreement_effective_domain.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate three multi-fidelity physics-pretraining paths")
    parser.add_argument("--spm", required=True, type=Path)
    parser.add_argument("--i2em-train-requests", required=True, type=Path)
    parser.add_argument("--i2em-train-results", required=True, type=Path)
    parser.add_argument("--i2em-test-requests", required=True, type=Path)
    parser.add_argument("--i2em-test-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if min(args.spm_epochs, args.i2em_epochs, args.repeats) < 1:
        raise ValueError("Epochs and repeats must be positive")
    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; choose a new directory")
    paths = [args.spm, args.i2em_train_requests, args.i2em_train_results, args.i2em_test_requests, args.i2em_test_results]
    paths = [path.resolve() for path in paths]
    spm = pd.read_csv(paths[0])
    train = add_disagreement_diagnostics(load_paired(paths[1], paths[2]))
    test = add_disagreement_diagnostics(load_paired(paths[3], paths[4]))
    spm_features = features_from(spm)
    train_features = features_from(train)
    test_features = features_from(test)
    spm_channels = spm[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float)
    train_i2em = train[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    test_i2em = test[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    test_spm = test[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float)
    all_prediction_rows = []
    metric_rows = []
    for repeat in range(1, args.repeats + 1):
        seed = args.seed + (repeat - 1) * 100_000
        component_predictions = train_component_models(
            spm_features, to_components(spm_channels), train_features, to_components(train_i2em), test_features,
            args.spm_epochs, args.i2em_epochs, seed,
        )
        channel_predictions = {name: from_components(values) for name, values in component_predictions.items()}
        channel_predictions["spm_teacher"] = test_spm
        channel_predictions["i2em_train_mean"] = np.tile(np.mean(train_i2em, axis=0), (len(test), 1))
        references = response_arrays(test_i2em)
        repeat_frame = test[["request_id", *FEATURE_NAMES, "i2em_validity_utilization", "validity_stratum"]].copy()
        repeat_frame.insert(0, "repeat", repeat)
        for response, reference in references.items():
            repeat_frame[f"reference_{response.lower()}_db"] = reference
        for method, channels in channel_predictions.items():
            arrays = response_arrays(channels)
            for response, predicted in arrays.items():
                repeat_frame[f"{method}_{response.lower()}_db"] = predicted
                metric_rows.append({"repeat": repeat, "method": method, "response": response, "n": len(test), **metric(references[response], predicted)})
        all_prediction_rows.append(repeat_frame)
        print(f"Completed repeat {repeat}/{args.repeats}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(all_prediction_rows, ignore_index=True)
    ensemble = predictions.groupby("request_id", as_index=False).agg({column: "first" if column in [*FEATURE_NAMES, "i2em_validity_utilization", "validity_stratum"] or column.startswith("reference_") else "mean" for column in predictions.columns if column not in ["request_id", "repeat"]})
    associations = association_table(pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True))
    stratum_summary = pd.concat([train.assign(split="train"), test.assign(split="test")]).groupby(["split", "validity_stratum"], dropna=False, as_index=False).agg(n=("request_id", "size"), hh_teacher_mae_db=("teacher_absolute_delta_hh_db", "mean"), vv_teacher_mae_db=("teacher_absolute_delta_vv_db", "mean"), joint_teacher_disagreement_db=("teacher_joint_absolute_delta_db", "mean"))
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics_by_repeat.csv", index=False)
    metrics.groupby(["method", "response"], as_index=False).agg(rmse_mean_db=("rmse_db", "mean"), rmse_std_db=("rmse_db", "std"), mae_mean_db=("mae_db", "mean"), r_squared_mean=("r_squared", "mean")).to_csv(output / "metrics_summary.csv", index=False)
    predictions.to_csv(output / "test_predictions_by_repeat.csv", index=False)
    ensemble.to_csv(output / "test_predictions_ensemble.csv", index=False)
    pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True).to_csv(output / "paired_teacher_disagreement.csv", index=False)
    associations.to_csv(output / "teacher_disagreement_associations.csv", index=False)
    stratum_summary.to_csv(output / "effective_domain_summary.csv", index=False)
    save_plots(metrics, ensemble, pd.concat([train, test], ignore_index=True), output)
    summary_metrics = pd.read_csv(output / "metrics_summary.csv")
    primary = summary_metrics[summary_metrics.response.isin(["HH", "VV"])].groupby("method").rmse_mean_db.mean().sort_values()
    summary = {
        "research_question": "Does sequential dense-SPM then sparse-I2EM pretraining outperform either single-fidelity path on independent I2EM samples?",
        "inputs": [fingerprint(path) for path in paths],
        "samples": {"spm": len(spm), "i2em_train": len(train), "i2em_test": len(test)},
        "protocol": {"coordinate_system": "fully decoupled common and VV-minus-HH heads", "spm_epochs": args.spm_epochs, "i2em_epochs": args.i2em_epochs, "repeats": args.repeats, "independent_i2em_test": True},
        "mean_hh_vv_rmse_ranking_db": {name: float(value) for name, value in primary.items()},
        "teacher_disagreement": {
            "train_joint_rmse_db": float(np.sqrt(np.mean((train_i2em - train[["spm_hh_db", "spm_vv_db"]].to_numpy()) ** 2))),
            "test_joint_rmse_db": float(np.sqrt(np.mean((test_i2em - test_spm) ** 2))),
        },
        "decision_rule": "Advance to real-data nested grouped ablation only if SPM-to-I2EM improves the independent I2EM test error over SPM-only and is competitive with I2EM-only; real-data superiority remains unproven until that next experiment.",
        "scope_limit": "This is physics-domain surrogate verification. I2EM is a higher-fidelity teacher, not observation truth, and the sampled box covers only the current cohort interpolation domain.",
        "code": fingerprint(Path(__file__)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(primary.to_string(), flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
