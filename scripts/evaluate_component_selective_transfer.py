"""Test response-component-selective SPM transfer on unseen fields.

The physical response is written as

    common = (HH + VV) / 2
    differential = VV - HH

and reconstructed without approximation.  The central comparison asks whether
an SPM constraint should act on both components or only on the polarimetric
differential component.  Hyperparameters are selected with nested GroupKFold
using field_id; test-field observations are never used for tuning.
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
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from research_pilots.scattering.surfaces.metrics import regression_metrics
from research_pilots.scattering.surrogate.physics_pretraining import (
    FEATURE_NAMES,
    blended_physics_target,
    fine_tune_pretrained,
    generate_synthetic_spm_dataset,
    predict_physical_units,
    pretrain_physics_model,
    train_from_scratch,
)


OBSERVED_COLUMNS = ["sigma0_hh_db", "sigma0_vv_db"]
PHYSICAL_COLUMNS = [
    "exponential_spm_hh_raw_db",
    "exponential_spm_vv_raw_db",
]
METHODS = [
    "training_mean",
    "direct_scratch",
    "direct_pretrained",
    "component_scratch",
    "component_pretrained",
    "both_component_constraint",
    "differential_only_constraint",
]
DISPLAY_NAMES = {
    "training_mean": "Training mean",
    "direct_scratch": "Direct scratch",
    "direct_pretrained": "Direct pretrained",
    "component_scratch": "Component scratch",
    "component_pretrained": "Component pretrained",
    "both_component_constraint": "Constrain common + diff.",
    "differential_only_constraint": "Constrain differential only",
}


def fingerprint(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    values = series.astype("string").str.strip().str.lower()
    unknown = values.notna() & ~values.isin(["true", "false", "1", "0"])
    if unknown.any():
        raise ValueError(f"Unrecognized Boolean values: {values[unknown].unique().tolist()}")
    return values.map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": "string"})
    required = {
        "acquisition_date",
        "field_id",
        "spm_valid",
        *FEATURE_NAMES,
        *OBSERVED_COLUMNS,
        *PHYSICAL_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")
    frame["acquisition_date"] = pd.to_datetime(frame["acquisition_date"], errors="raise")
    frame["field_id"] = frame["field_id"].str.strip()
    frame["spm_valid"] = parse_bool(frame["spm_valid"])
    numeric = [*FEATURE_NAMES, *OBSERVED_COLUMNS, *PHYSICAL_COLUMNS]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    finite = np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    selected = frame.loc[frame["spm_valid"] & finite].copy()
    if selected.empty:
        raise ValueError("No finite SPM-valid samples")
    if selected["field_id"].nunique() < 3:
        raise ValueError("At least three independent fields are required")
    if selected.duplicated(["field_id", "acquisition_date"]).any():
        raise ValueError("Duplicate field-date rows are not allowed")
    return selected.sort_values(["field_id", "acquisition_date"]).reset_index(drop=True)


def to_components(hh_vv: np.ndarray) -> np.ndarray:
    values = np.asarray(hh_vv, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected a two-column [HH, VV] array")
    common = (values[:, 0] + values[:, 1]) / 2.0
    differential = values[:, 1] - values[:, 0]
    return np.column_stack([common, differential])


def from_components(common_differential: np.ndarray) -> np.ndarray:
    values = np.asarray(common_differential, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected a two-column [common, differential] array")
    hh = values[:, 0] - values[:, 1] / 2.0
    vv = values[:, 0] + values[:, 1] / 2.0
    return np.column_stack([hh, vv])


def calibration_offsets(observed: np.ndarray, physical: np.ndarray) -> np.ndarray:
    return np.mean(observed - physical, axis=0)


def joint_channel_rmse(observed_components: np.ndarray, predicted_components: np.ndarray) -> float:
    observed = from_components(observed_components)
    predicted = from_components(predicted_components)
    return float(np.sqrt(np.mean((predicted - observed) ** 2, axis=0)).mean())


def component_constraint_target(
    observed_components: np.ndarray,
    calibrated_physics_components: np.ndarray,
    weight: float,
    mode: str,
) -> np.ndarray:
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    if mode == "both":
        return blended_physics_target(
            observed_components, calibrated_physics_components, weight
        )
    if mode != "differential_only":
        raise ValueError(f"Unknown constraint mode: {mode}")
    result = observed_components.copy()
    result[:, 1] = (
        (1.0 - weight) * observed_components[:, 1]
        + weight * calibrated_physics_components[:, 1]
    )
    return result


def tune_component_weights(
    bundle,
    features: np.ndarray,
    observed_components: np.ndarray,
    physical_components: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    weights: list[float],
    inner_folds: int,
    epochs: int,
    seed: int,
) -> tuple[float, float, list[dict[str, object]]]:
    local_groups = groups[outer_train]
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(local_groups))))
    scores = {
        "both": {weight: [] for weight in weights},
        "differential_only": {weight: [] for weight in weights},
    }
    records: list[dict[str, object]] = []
    for inner_fold, (train_local, validation_local) in enumerate(
        splitter.split(outer_train, groups=local_groups), start=1
    ):
        train = outer_train[train_local]
        validation = outer_train[validation_local]
        offsets = calibration_offsets(
            observed_components[train], physical_components[train]
        )
        calibrated = physical_components[train] + offsets
        training_seed = seed + inner_fold * 100
        for mode in scores:
            for weight in weights:
                target = component_constraint_target(
                    observed_components[train], calibrated, weight, mode
                )
                model = fine_tune_pretrained(
                    bundle,
                    features[train],
                    target,
                    epochs=epochs,
                    seed=training_seed,
                )
                predicted = predict_physical_units(
                    bundle, model, features[validation]
                )
                score = joint_channel_rmse(
                    observed_components[validation], predicted
                )
                scores[mode][weight].append(score)
                records.append(
                    {
                        "family": mode,
                        "inner_fold": inner_fold,
                        "physics_weight": weight,
                        "joint_hh_vv_validation_rmse_db": score,
                    }
                )
    means = {
        mode: {weight: float(np.mean(values)) for weight, values in by_weight.items()}
        for mode, by_weight in scores.items()
    }
    selected = {
        mode: min(values, key=lambda weight: (values[weight], weight))
        for mode, values in means.items()
    }
    for record in records:
        mode = str(record["family"])
        weight = float(record["physics_weight"])
        record["mean_rmse_for_candidate_db"] = means[mode][weight]
        record["candidate_selected"] = bool(weight == selected[mode])
    return float(selected["both"]), float(selected["differential_only"]), records


def run_outer_evaluation(
    frame: pd.DataFrame,
    direct_bundle,
    component_bundle,
    weights: list[float],
    outer_folds: int,
    inner_folds: int,
    epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    observed_channels = frame[OBSERVED_COLUMNS].to_numpy(dtype=float)
    physical_channels = frame[PHYSICAL_COLUMNS].to_numpy(dtype=float)
    observed_components = to_components(observed_channels)
    physical_components = to_components(physical_channels)
    groups = frame["field_id"].astype(str).to_numpy()
    predictions = {
        method: np.full_like(observed_channels, np.nan) for method in METHODS
    }
    fold_ids = np.full(len(frame), -1, dtype=int)
    both_weights = np.full(len(frame), np.nan)
    differential_weights = np.full(len(frame), np.nan)
    tuning_rows: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=min(outer_folds, len(np.unique(groups))))

    for outer_fold, (train, test) in enumerate(
        splitter.split(features, observed_channels, groups), start=1
    ):
        predictions["training_mean"][test] = observed_channels[train].mean(axis=0)
        direct_seed = seed + outer_fold * 10_000 + 7
        component_seed = seed + outer_fold * 10_000 + 11

        direct_scratch = train_from_scratch(
            direct_bundle,
            features[train],
            observed_channels[train],
            epochs=epochs,
            seed=direct_seed,
        )
        predictions["direct_scratch"][test] = predict_physical_units(
            direct_bundle, direct_scratch, features[test]
        )
        direct_pretrained = fine_tune_pretrained(
            direct_bundle,
            features[train],
            observed_channels[train],
            epochs=epochs,
            seed=direct_seed,
        )
        predictions["direct_pretrained"][test] = predict_physical_units(
            direct_bundle, direct_pretrained, features[test]
        )

        component_scratch = train_from_scratch(
            component_bundle,
            features[train],
            observed_components[train],
            epochs=epochs,
            seed=component_seed,
        )
        predictions["component_scratch"][test] = from_components(
            predict_physical_units(
                component_bundle, component_scratch, features[test]
            )
        )
        component_pretrained = fine_tune_pretrained(
            component_bundle,
            features[train],
            observed_components[train],
            epochs=epochs,
            seed=component_seed,
        )
        predictions["component_pretrained"][test] = from_components(
            predict_physical_units(
                component_bundle, component_pretrained, features[test]
            )
        )

        selected_both, selected_differential, records = tune_component_weights(
            component_bundle,
            features,
            observed_components,
            physical_components,
            groups,
            train,
            weights,
            inner_folds,
            epochs,
            seed + outer_fold * 100_000,
        )
        offsets = calibration_offsets(
            observed_components[train], physical_components[train]
        )
        calibrated_train = physical_components[train] + offsets
        for method, mode, weight in [
            ("both_component_constraint", "both", selected_both),
            (
                "differential_only_constraint",
                "differential_only",
                selected_differential,
            ),
        ]:
            target = component_constraint_target(
                observed_components[train], calibrated_train, weight, mode
            )
            model = fine_tune_pretrained(
                component_bundle,
                features[train],
                target,
                epochs=epochs,
                seed=component_seed,
            )
            predictions[method][test] = from_components(
                predict_physical_units(component_bundle, model, features[test])
            )

        fold_ids[test] = outer_fold
        both_weights[test] = selected_both
        differential_weights[test] = selected_differential
        for record in records:
            tuning_rows.append(
                {
                    "outer_fold": outer_fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_fields": len(np.unique(groups[train])),
                    "test_fields": len(np.unique(groups[test])),
                    "selected_both_weight": selected_both,
                    "selected_differential_weight": selected_differential,
                    **record,
                }
            )

    output = frame.copy()
    output["outer_fold"] = fold_ids
    output["selected_both_weight"] = both_weights
    output["selected_differential_weight"] = differential_weights
    for method, values in predictions.items():
        output[f"hh_{method}"] = values[:, 0]
        output[f"vv_{method}"] = values[:, 1]
        components = to_components(values)
        output[f"common_{method}"] = components[:, 0]
        output[f"differential_{method}"] = components[:, 1]
    output["observed_common_db"] = observed_components[:, 0]
    output["observed_differential_db"] = observed_components[:, 1]
    output["spm_common_db"] = physical_components[:, 0]
    output["spm_differential_db"] = physical_components[:, 1]
    return output, pd.DataFrame(tuning_rows)


def evaluate_channel_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for polarization, observed_column in zip(["HH", "VV"], OBSERVED_COLUMNS):
        observed = frame[observed_column].to_numpy(dtype=float)
        for method in METHODS:
            predicted = frame[f"{polarization.lower()}_{method}"].to_numpy(dtype=float)
            rows.append(
                {
                    "response": polarization,
                    "method": method,
                    **regression_metrics(observed, predicted),
                }
            )
    return pd.DataFrame(rows)


def evaluate_component_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for component in ["common", "differential"]:
        observed = frame[f"observed_{component}_db"].to_numpy(dtype=float)
        for method in METHODS:
            predicted = frame[f"{component}_{method}"].to_numpy(dtype=float)
            rows.append(
                {
                    "response": component,
                    "method": method,
                    **regression_metrics(observed, predicted),
                }
            )
    return pd.DataFrame(rows)


def aggregate_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0].drop(columns=["repeat"], errors="ignore").copy()
    for method in METHODS:
        for prefix in ["hh", "vv", "common", "differential"]:
            column = f"{prefix}_{method}"
            base[column] = np.mean(
                [frame[column].to_numpy(dtype=float) for frame in frames], axis=0
            )
    for column in ["selected_both_weight", "selected_differential_weight"]:
        base[column] = np.mean(
            [frame[column].to_numpy(dtype=float) for frame in frames], axis=0
        )
    return base


def grouped_bootstrap_delta(
    observed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    iterations: int,
    seed: int,
) -> dict[str, object]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    deltas = np.empty((iterations, observed.shape[1]))
    for iteration in range(iterations):
        sampled = generator.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        first_rmse = np.sqrt(
            np.mean((first[indices] - observed[indices]) ** 2, axis=0)
        )
        second_rmse = np.sqrt(
            np.mean((second[indices] - observed[indices]) ** 2, axis=0)
        )
        deltas[iteration] = first_rmse - second_rmse
    result: dict[str, object] = {}
    for index, label in enumerate(labels):
        point = np.sqrt(np.mean((first[:, index] - observed[:, index]) ** 2)) - np.sqrt(
            np.mean((second[:, index] - observed[:, index]) ** 2)
        )
        result[label] = {
            "rmse_delta_db": float(point),
            "ci_2_5_percent_db": float(np.quantile(deltas[:, index], 0.025)),
            "ci_97_5_percent_db": float(np.quantile(deltas[:, index], 0.975)),
            "probability_first_better": float(np.mean(deltas[:, index] < 0)),
        }
    return result


def repeat_metric_summary(metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (response, method), subset in metrics.groupby(["response", "method"], sort=False):
        rows.append(
            {
                "response": str(response),
                "method": str(method),
                "rmse_mean_db": float(subset["rmse_db"].mean()),
                "rmse_std_db": float(subset["rmse_db"].std(ddof=1))
                if len(subset) > 1
                else 0.0,
            }
        )
    return rows


def selected_weight_frequency(tuning: pd.DataFrame, family: str) -> list[dict[str, object]]:
    chosen = tuning.loc[
        (tuning["family"] == family) & tuning["candidate_selected"]
    ].drop_duplicates(["repeat", "outer_fold"])
    return (
        chosen["physics_weight"]
        .value_counts(normalize=True)
        .sort_index()
        .rename_axis("physics_weight")
        .reset_index(name="fraction")
        .to_dict(orient="records")
    )


def relabel_component_pretraining_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Replace generic two-output labels with the actual component names."""
    result = dict(metadata)
    raw = result.pop("synthetic_training_rmse_db")
    if not isinstance(raw, dict) or "HH" not in raw or "VV" not in raw:
        raise ValueError("Unexpected pretraining metadata schema")
    result["synthetic_training_rmse_db"] = {
        "common": float(raw["HH"]),
        "differential": float(raw["VV"]),
    }
    return result


def save_plots(
    frame: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    component_metrics: pd.DataFrame,
    comparisons: dict[str, object],
    output_dir: Path,
) -> None:
    colors = plt.cm.Set2(np.linspace(0, 1, len(METHODS)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        subset = channel_metrics[channel_metrics.response == response].set_index("method").loc[METHODS]
        ax.bar([DISPLAY_NAMES[m] for m in METHODS], subset.rmse_db, color=colors)
        ax.set(title=f"Unseen-field channel RMSE: {response}", ylabel="Grouped OOF RMSE (dB)")
        ax.tick_params(axis="x", rotation=43)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "01_channel_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, response in zip(axes, ["common", "differential"]):
        subset = component_metrics[component_metrics.response == response].set_index("method").loc[METHODS]
        ax.bar([DISPLAY_NAMES[m] for m in METHODS], subset.rmse_db, color=colors)
        ax.set(title=f"Unseen-field component RMSE: {response}", ylabel="Grouped OOF RMSE (dB)")
        ax.tick_params(axis="x", rotation=43)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "02_component_rmse.png", dpi=180)
    plt.close(fig)

    order = [
        "differential_vs_component_pretrained",
        "differential_vs_both",
        "differential_vs_direct_pretrained",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        points, low, high = [], [], []
        for name in order:
            stats = comparisons["channels"][name][response]
            points.append(stats["rmse_delta_db"])
            low.append(stats["rmse_delta_db"] - stats["ci_2_5_percent_db"])
            high.append(stats["ci_97_5_percent_db"] - stats["rmse_delta_db"])
        positions = np.arange(len(order))
        ax.errorbar(positions, points, yerr=[low, high], fmt="o", capsize=5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(positions, [name.replace("_", " ") for name in order], rotation=24)
        ax.set(title=f"Paired field bootstrap: {response}", ylabel="First - second RMSE (dB; <0 favors first)")
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "03_paired_channel_effects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for row, component in enumerate(["common", "differential"]):
        observed = frame[f"observed_{component}_db"].to_numpy(dtype=float)
        for column, method in enumerate(
            ["component_pretrained", "differential_only_constraint"]
        ):
            predicted = frame[f"{component}_{method}"].to_numpy(dtype=float)
            lower = float(min(observed.min(), predicted.min()))
            upper = float(max(observed.max(), predicted.max()))
            axes[row, column].scatter(observed, predicted, s=24, alpha=0.58, edgecolors="none")
            axes[row, column].plot([lower, upper], [lower, upper], "k--", linewidth=1)
            axes[row, column].set(
                xlabel=f"Observed {component} (dB)",
                ylabel="Grouped OOF prediction (dB)",
                title=f"{component} | {DISPLAY_NAMES[method]}",
                xlim=(lower, upper),
                ylim=(lower, upper),
            )
            axes[row, column].grid(alpha=0.22)
    fig.savefig(output_dir / "04_component_observed_vs_predicted.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested component-selective SPM-transfer experiment")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--synthetic-samples", type=int, default=3000)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--fine-tune-epochs", type=int, default=120)
    parser.add_argument("--physics-weights", default="0,0.05,0.1,0.2,0.4")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("repeats must be at least one")
    weights = sorted(set(float(value) for value in args.physics_weights.split(",")))
    if 0.0 not in weights or any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("physics weights must lie in [0,1] and include zero")
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    expected = [
        "summary.json",
        "channel_metrics.csv",
        "component_metrics.csv",
        "nested_tuning.csv",
        "oof_predictions.csv",
    ]
    if any((output_dir / name).exists() for name in expected):
        raise FileExistsError("Output files already exist; choose a new --output directory")
    frame = load_table(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_features, synthetic_channels, synthetic_metadata = generate_synthetic_spm_dataset(
        args.synthetic_samples,
        args.frequency_ghz * 1e9,
        args.incidence_angle_deg,
        seed=args.seed,
    )
    synthetic_components = to_components(synthetic_channels)
    evaluated_repeats: list[pd.DataFrame] = []
    tuning_repeats: list[pd.DataFrame] = []
    channel_metric_repeats: list[pd.DataFrame] = []
    component_metric_repeats: list[pd.DataFrame] = []
    pretraining_records: list[dict[str, object]] = []

    for repeat in range(1, args.repeats + 1):
        repeat_seed = args.seed + (repeat - 1) * 100_000
        direct_bundle = pretrain_physics_model(
            synthetic_features,
            synthetic_channels,
            epochs=args.pretrain_epochs,
            seed=repeat_seed,
        )
        component_bundle = pretrain_physics_model(
            synthetic_features,
            synthetic_components,
            epochs=args.pretrain_epochs,
            seed=repeat_seed,
        )
        evaluated, tuning = run_outer_evaluation(
            frame,
            direct_bundle,
            component_bundle,
            weights,
            args.outer_folds,
            args.inner_folds,
            args.fine_tune_epochs,
            repeat_seed,
        )
        evaluated["repeat"] = repeat
        tuning["repeat"] = repeat
        channel_metrics = evaluate_channel_metrics(evaluated)
        channel_metrics.insert(0, "repeat", repeat)
        component_metrics = evaluate_component_metrics(evaluated)
        component_metrics.insert(0, "repeat", repeat)
        evaluated_repeats.append(evaluated)
        tuning_repeats.append(tuning)
        channel_metric_repeats.append(channel_metrics)
        component_metric_repeats.append(component_metrics)
        pretraining_records.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "direct": direct_bundle.metadata,
                "component": relabel_component_pretraining_metadata(
                    component_bundle.metadata
                ),
            }
        )
        print(f"Completed repeat {repeat}/{args.repeats}", flush=True)

    ensemble = aggregate_predictions(evaluated_repeats)
    tuning = pd.concat(tuning_repeats, ignore_index=True)
    channel_metrics_by_repeat = pd.concat(channel_metric_repeats, ignore_index=True)
    component_metrics_by_repeat = pd.concat(component_metric_repeats, ignore_index=True)
    channel_metrics = evaluate_channel_metrics(ensemble)
    component_metrics = evaluate_component_metrics(ensemble)
    groups = ensemble["field_id"].astype(str).to_numpy()
    observed_channels = ensemble[OBSERVED_COLUMNS].to_numpy(dtype=float)
    observed_components = ensemble[["observed_common_db", "observed_differential_db"]].to_numpy(dtype=float)

    def channel_prediction(method: str) -> np.ndarray:
        return ensemble[[f"hh_{method}", f"vv_{method}"]].to_numpy(dtype=float)

    def component_prediction(method: str) -> np.ndarray:
        return ensemble[[f"common_{method}", f"differential_{method}"]].to_numpy(dtype=float)

    comparison_pairs = {
        "differential_vs_component_pretrained": (
            "differential_only_constraint",
            "component_pretrained",
        ),
        "differential_vs_both": (
            "differential_only_constraint",
            "both_component_constraint",
        ),
        "differential_vs_direct_pretrained": (
            "differential_only_constraint",
            "direct_pretrained",
        ),
        "component_pretrained_vs_direct_pretrained": (
            "component_pretrained",
            "direct_pretrained",
        ),
        "differential_vs_training_mean": (
            "differential_only_constraint",
            "training_mean",
        ),
    }
    comparisons = {"channels": {}, "components": {}}
    for index, (name, (first, second)) in enumerate(comparison_pairs.items()):
        comparisons["channels"][name] = grouped_bootstrap_delta(
            observed_channels,
            channel_prediction(first),
            channel_prediction(second),
            groups,
            ["HH", "VV"],
            args.bootstrap_iterations,
            args.seed + index * 100,
        )
        comparisons["components"][name] = grouped_bootstrap_delta(
            observed_components,
            component_prediction(first),
            component_prediction(second),
            groups,
            ["common", "differential"],
            args.bootstrap_iterations,
            args.seed + 1000 + index * 100,
        )

    summary = {
        "research_question": "Does constraining only the SPM-informed polarimetric differential component reduce negative transfer relative to constraining both absolute-response components?",
        "input": fingerprint(input_path),
        "code": fingerprint(Path(__file__)),
        "output": str(output_dir),
        "samples": int(len(frame)),
        "fields": int(frame["field_id"].nunique()),
        "dates": int(frame["acquisition_date"].nunique()),
        "response_definition": {
            "common": "(HH + VV) / 2",
            "differential": "VV - HH",
            "inverse": "HH = common - differential/2; VV = common + differential/2",
        },
        "protocol": {
            "outer_split": f"{args.outer_folds}-fold GroupKFold by field",
            "inner_split": f"up to {args.inner_folds}-fold GroupKFold by field",
            "repeats": args.repeats,
            "bootstrap": f"{args.bootstrap_iterations} field-block resamples",
            "selection_metric": "mean of reconstructed HH and VV RMSE",
            "same_initialization_control": "All component-pretrained variants share initialization and epoch-wise training order within each outer fold.",
        },
        "candidate_physics_weights": weights,
        "synthetic_domain": synthetic_metadata,
        "pretraining_by_repeat": pretraining_records,
        "selected_both_weight_frequency": selected_weight_frequency(tuning, "both"),
        "selected_differential_weight_frequency": selected_weight_frequency(tuning, "differential_only"),
        "ensemble_channel_metrics": channel_metrics.to_dict(orient="records"),
        "ensemble_component_metrics": component_metrics.to_dict(orient="records"),
        "repeat_channel_metric_summary": repeat_metric_summary(channel_metrics_by_repeat),
        "repeat_component_metric_summary": repeat_metric_summary(component_metrics_by_repeat),
        "paired_field_bootstrap": comparisons,
        "decision_rule": "Differential-only transfer is supported only if it improves reconstructed unseen-field HH/VV error over component-pretrained and both-component constraints with a field-bootstrap interval excluding zero, without a significant degradation in either channel.",
        "scope_limit": "This experiment tests predictive transfer within one campaign. The common/differential coordinates are an exact reparameterization, not a causal separation of scattering mechanisms, and SPM remains a low-fidelity teacher.",
    }

    ensemble_output = ensemble.copy()
    ensemble_output["acquisition_date"] = ensemble_output["acquisition_date"].dt.strftime("%Y-%m-%d")
    ensemble_output.to_csv(output_dir / "oof_predictions.csv", index=False)
    repeated_output = pd.concat(evaluated_repeats, ignore_index=True)
    repeated_output["acquisition_date"] = pd.to_datetime(repeated_output["acquisition_date"]).dt.strftime("%Y-%m-%d")
    repeated_output.to_csv(output_dir / "oof_predictions_by_repeat.csv", index=False)
    channel_metrics.to_csv(output_dir / "channel_metrics.csv", index=False)
    component_metrics.to_csv(output_dir / "component_metrics.csv", index=False)
    channel_metrics_by_repeat.to_csv(output_dir / "channel_metrics_by_repeat.csv", index=False)
    component_metrics_by_repeat.to_csv(output_dir / "component_metrics_by_repeat.csv", index=False)
    tuning.to_csv(output_dir / "nested_tuning.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    save_plots(ensemble, channel_metrics, component_metrics, comparisons, output_dir)

    print("\nComponent-selective transfer experiment completed.")
    print(f"Samples: {len(frame)}; fields: {frame['field_id'].nunique()}; dates: {frame['acquisition_date'].nunique()}")
    print("\nChannel metrics:")
    print(channel_metrics[["response", "method", "rmse_db", "mae_db", "r_squared_skill"]].to_string(index=False))
    print("\nComponent metrics:")
    print(component_metrics[["response", "method", "rmse_db", "mae_db", "r_squared_skill"]].to_string(index=False))
    print("\nPrimary paired comparisons:")
    print(json.dumps(comparisons["channels"], indent=2, ensure_ascii=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
