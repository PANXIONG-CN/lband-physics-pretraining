"""Evaluate confidence-gated transfer of an SPM physics constraint.

The experiment uses nested field-grouped cross-validation.  A small MLP is
pretrained on an exponential-SPM teacher and then fine-tuned on SMAPVEX12.
Three fine-tuning choices share the same pretrained initialization and sample
order in every outer fold:

1. observation-only fine-tuning;
2. a fixed (sample-independent) SPM constraint;
3. a selective SPM constraint gated by roughness validity, vegetation water
   content, and VWC-map provenance.

The gate never uses observed backscatter or residuals.  Its hyperparameters
are selected only inside the outer training split.
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
VWC_MAP_COLUMN = "vegetation_water_content_map_kg_m2"
VWC_INSITU_COLUMN = "vegetation_water_content_in_situ_kg_m2"
SOURCE_COLUMN = "vwc_map_source"
METHODS = [
    "training_mean",
    "spm_offset",
    "scratch_mlp",
    "pretrained_mlp",
    "fixed_constraint",
    "selective_constraint",
]
DISPLAY_NAMES = {
    "training_mean": "Training mean",
    "spm_offset": "SPM + offset",
    "scratch_mlp": "Scratch MLP",
    "pretrained_mlp": "SPM-pretrained",
    "fixed_constraint": "Fixed constraint",
    "selective_constraint": "Selective constraint",
}
SOURCE_QUALITY = {
    "direct_satellite": 1.0,
    "interpolated": 0.7,
    "extrapolated": 0.4,
}


def fingerprint(path: Path) -> dict[str, object]:
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


def load_table(path: Path, cohort: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": "string"})
    required = {
        "acquisition_date",
        "field_id",
        "spm_valid",
        "spm_k_rms_height",
        "spm_rms_slope_proxy",
        *FEATURE_NAMES,
        *OBSERVED_COLUMNS,
        *PHYSICAL_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")
    if VWC_MAP_COLUMN not in frame and VWC_INSITU_COLUMN not in frame:
        raise ValueError("Input must contain mapped or in-situ vegetation water content")

    if cohort != "all":
        flag = f"in_{cohort}_cohort"
        if flag not in frame:
            raise ValueError(f"--cohort {cohort} requires column {flag}")
        frame = frame.loc[parse_bool(frame[flag])].copy()

    frame["acquisition_date"] = pd.to_datetime(frame["acquisition_date"], errors="raise")
    frame["field_id"] = frame["field_id"].str.strip()
    frame["spm_valid"] = parse_bool(frame["spm_valid"])

    numeric = [
        *FEATURE_NAMES,
        *OBSERVED_COLUMNS,
        *PHYSICAL_COLUMNS,
        "spm_k_rms_height",
        "spm_rms_slope_proxy",
    ]
    for column in [VWC_MAP_COLUMN, VWC_INSITU_COLUMN]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    finite = np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    vwc = select_vwc(frame)
    selected = frame.loc[frame["spm_valid"] & finite & np.isfinite(vwc) & (vwc >= 0)].copy()
    if selected.empty:
        raise ValueError("No finite SPM-valid samples with vegetation information")
    if selected["field_id"].nunique() < 3:
        raise ValueError("At least three fields are required for grouped evaluation")
    if selected.duplicated(["field_id", "acquisition_date"]).any():
        raise ValueError("Duplicate field-date rows are not allowed")
    return selected.sort_values(["field_id", "acquisition_date"]).reset_index(drop=True)


def select_vwc(frame: pd.DataFrame) -> np.ndarray:
    if VWC_MAP_COLUMN in frame:
        mapped = pd.to_numeric(frame[VWC_MAP_COLUMN], errors="coerce")
    else:
        mapped = pd.Series(np.nan, index=frame.index)
    if VWC_INSITU_COLUMN in frame:
        in_situ = pd.to_numeric(frame[VWC_INSITU_COLUMN], errors="coerce")
    else:
        in_situ = pd.Series(np.nan, index=frame.index)
    return mapped.fillna(in_situ).to_numpy(dtype=float)


def source_quality(frame: pd.DataFrame) -> np.ndarray:
    if SOURCE_COLUMN not in frame:
        return np.full(len(frame), 0.5, dtype=float)
    values = frame[SOURCE_COLUMN].astype("string").str.strip().str.lower()
    unknown = sorted(set(values.dropna().unique()).difference(SOURCE_QUALITY))
    if unknown:
        raise ValueError(f"Unknown VWC map provenance labels: {unknown}")
    return values.map(SOURCE_QUALITY).fillna(0.5).to_numpy(dtype=float)


def physics_confidence(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Return a transparent, observation-independent SPM reliability gate."""
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    ks = frame["spm_k_rms_height"].to_numpy(dtype=float)
    slope = frame["spm_rms_slope_proxy"].to_numpy(dtype=float)
    roughness = np.minimum(
        np.clip((0.30 - ks) / 0.30, 0.0, 1.0),
        np.clip((0.21 - slope) / 0.21, 0.0, 1.0),
    )
    vwc = select_vwc(frame)
    vegetation = np.exp(-float(alpha) * np.clip(vwc, 0.0, None))
    provenance = source_quality(frame)
    confidence = np.clip(roughness * vegetation * provenance, 0.0, 1.0)
    return pd.DataFrame(
        {
            "gate_roughness": roughness,
            "gate_vegetation": vegetation,
            "gate_provenance": provenance,
            "gate_vwc_kg_m2": vwc,
            "physics_confidence": confidence,
        },
        index=frame.index,
    )


def selective_blended_target(
    observed: np.ndarray,
    calibrated_physics: np.ndarray,
    confidence: np.ndarray,
    maximum_weight: float,
) -> np.ndarray:
    weight = float(maximum_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("maximum_weight must be between zero and one")
    confidence = np.asarray(confidence, dtype=float).reshape(-1, 1)
    if len(confidence) != len(observed):
        raise ValueError("confidence length does not match targets")
    if np.any(~np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and lie in [0, 1]")
    effective = weight * confidence
    return (1.0 - effective) * observed + effective * calibrated_physics


def joint_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2, axis=0)).mean())


def calibration_offsets(observed: np.ndarray, physical: np.ndarray) -> np.ndarray:
    return np.mean(observed - physical, axis=0)


def candidate_pairs(weights: list[float], alphas: list[float]) -> list[tuple[float, float]]:
    pairs = [(0.0, 0.0)]
    pairs.extend((weight, alpha) for weight in weights if weight > 0 for alpha in alphas)
    return pairs


def tune_constraints(
    bundle,
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    weights: list[float],
    alphas: list[float],
    inner_folds: int,
    epochs: int,
    seed: int,
) -> tuple[float, tuple[float, float], list[dict[str, object]]]:
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    observed = frame[OBSERVED_COLUMNS].to_numpy(dtype=float)
    physical = frame[PHYSICAL_COLUMNS].to_numpy(dtype=float)
    groups = frame["field_id"].astype(str).to_numpy()
    local_groups = groups[train_indices]
    folds = min(int(inner_folds), len(np.unique(local_groups)))
    splitter = GroupKFold(n_splits=folds)
    fixed_scores = {weight: [] for weight in weights}
    selective_scores = {pair: [] for pair in candidate_pairs(weights, alphas)}
    records: list[dict[str, object]] = []

    for inner_fold, (inner_train_local, validation_local) in enumerate(
        splitter.split(train_indices, groups=local_groups), start=1
    ):
        inner_train = train_indices[inner_train_local]
        validation = train_indices[validation_local]
        offsets = calibration_offsets(observed[inner_train], physical[inner_train])
        calibrated_train = physical[inner_train] + offsets
        training_seed = seed + inner_fold * 100

        for weight in weights:
            target = blended_physics_target(observed[inner_train], calibrated_train, weight)
            model = fine_tune_pretrained(
                bundle, features[inner_train], target, epochs=epochs, seed=training_seed
            )
            prediction = predict_physical_units(bundle, model, features[validation])
            score = joint_rmse(observed[validation], prediction)
            fixed_scores[weight].append(score)
            records.append(
                {
                    "family": "fixed",
                    "inner_fold": inner_fold,
                    "maximum_physics_weight": weight,
                    "gate_alpha": np.nan,
                    "joint_validation_rmse_db": score,
                }
            )

        for weight, alpha in selective_scores:
            gate = physics_confidence(frame.iloc[inner_train], alpha)[
                "physics_confidence"
            ].to_numpy()
            target = selective_blended_target(
                observed[inner_train], calibrated_train, gate, weight
            )
            model = fine_tune_pretrained(
                bundle, features[inner_train], target, epochs=epochs, seed=training_seed
            )
            prediction = predict_physical_units(bundle, model, features[validation])
            score = joint_rmse(observed[validation], prediction)
            selective_scores[(weight, alpha)].append(score)
            records.append(
                {
                    "family": "selective",
                    "inner_fold": inner_fold,
                    "maximum_physics_weight": weight,
                    "gate_alpha": alpha,
                    "joint_validation_rmse_db": score,
                }
            )

    fixed_means = {key: float(np.mean(value)) for key, value in fixed_scores.items()}
    selective_means = {
        key: float(np.mean(value)) for key, value in selective_scores.items()
    }
    best_fixed = min(fixed_means, key=lambda key: (fixed_means[key], key))
    best_selective = min(
        selective_means,
        key=lambda key: (selective_means[key], key[0], key[1]),
    )
    for record in records:
        if record["family"] == "fixed":
            record["mean_rmse_for_candidate_db"] = fixed_means[
                float(record["maximum_physics_weight"])
            ]
            record["candidate_selected"] = bool(
                float(record["maximum_physics_weight"]) == best_fixed
            )
        else:
            pair = (
                float(record["maximum_physics_weight"]),
                float(record["gate_alpha"]),
            )
            record["mean_rmse_for_candidate_db"] = selective_means[pair]
            record["candidate_selected"] = bool(pair == best_selective)
    return float(best_fixed), best_selective, records


def run_outer_evaluation(
    frame: pd.DataFrame,
    bundle,
    weights: list[float],
    alphas: list[float],
    outer_folds: int,
    inner_folds: int,
    epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    observed = frame[OBSERVED_COLUMNS].to_numpy(dtype=float)
    physical = frame[PHYSICAL_COLUMNS].to_numpy(dtype=float)
    groups = frame["field_id"].astype(str).to_numpy()
    predictions = {method: np.full_like(observed, np.nan) for method in METHODS}
    fold_ids = np.full(len(frame), -1, dtype=int)
    selected_weight = np.full(len(frame), np.nan)
    selected_alpha = np.full(len(frame), np.nan)
    selected_confidence = np.full(len(frame), np.nan)
    selected_effective_weight = np.full(len(frame), np.nan)
    gate_rows: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=min(int(outer_folds), len(np.unique(groups))))

    for outer_fold, (train, test) in enumerate(
        splitter.split(features, observed, groups), start=1
    ):
        offsets = calibration_offsets(observed[train], physical[train])
        calibrated_train = physical[train] + offsets
        predictions["training_mean"][test] = observed[train].mean(axis=0)
        predictions["spm_offset"][test] = physical[test] + offsets

        scratch_seed = seed + outer_fold * 10_000 + 7
        fine_tune_seed = seed + outer_fold * 10_000 + 11
        scratch = train_from_scratch(
            bundle, features[train], observed[train], epochs=epochs, seed=scratch_seed
        )
        predictions["scratch_mlp"][test] = predict_physical_units(
            bundle, scratch, features[test]
        )
        pretrained = fine_tune_pretrained(
            bundle, features[train], observed[train], epochs=epochs, seed=fine_tune_seed
        )
        predictions["pretrained_mlp"][test] = predict_physical_units(
            bundle, pretrained, features[test]
        )

        best_fixed, best_selective, records = tune_constraints(
            bundle,
            frame,
            train,
            weights,
            alphas,
            inner_folds,
            epochs,
            seed + outer_fold * 100_000,
        )
        fixed_target = blended_physics_target(
            observed[train], calibrated_train, best_fixed
        )
        fixed_model = fine_tune_pretrained(
            bundle, features[train], fixed_target, epochs=epochs, seed=fine_tune_seed
        )
        predictions["fixed_constraint"][test] = predict_physical_units(
            bundle, fixed_model, features[test]
        )

        maximum_weight, alpha = best_selective
        train_gate = physics_confidence(frame.iloc[train], alpha)[
            "physics_confidence"
        ].to_numpy()
        selective_target = selective_blended_target(
            observed[train], calibrated_train, train_gate, maximum_weight
        )
        selective_model = fine_tune_pretrained(
            bundle,
            features[train],
            selective_target,
            epochs=epochs,
            seed=fine_tune_seed,
        )
        predictions["selective_constraint"][test] = predict_physical_units(
            bundle, selective_model, features[test]
        )

        test_gate = physics_confidence(frame.iloc[test], alpha).copy()
        test_gate["row_index"] = test
        test_gate["outer_fold"] = outer_fold
        test_gate["selected_maximum_weight"] = maximum_weight
        test_gate["selected_alpha"] = alpha
        test_gate["effective_physics_weight"] = maximum_weight * test_gate[
            "physics_confidence"
        ]
        gate_rows.append(test_gate)
        fold_ids[test] = outer_fold
        selected_weight[test] = maximum_weight
        selected_alpha[test] = alpha
        selected_confidence[test] = test_gate["physics_confidence"].to_numpy()
        selected_effective_weight[test] = test_gate[
            "effective_physics_weight"
        ].to_numpy()

        for record in records:
            tuning_rows.append(
                {
                    "outer_fold": outer_fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_fields": len(np.unique(groups[train])),
                    "test_fields": len(np.unique(groups[test])),
                    "selected_fixed_weight": best_fixed,
                    "selected_selective_weight": maximum_weight,
                    "selected_selective_alpha": alpha,
                    **record,
                }
            )

    result = frame.copy()
    result["outer_fold"] = fold_ids
    for method, values in predictions.items():
        result[f"hh_{method}"] = values[:, 0]
        result[f"vv_{method}"] = values[:, 1]
    result["selected_selective_weight"] = selected_weight
    result["selected_selective_alpha"] = selected_alpha
    result["selective_physics_confidence"] = selected_confidence
    result["selective_effective_weight"] = selected_effective_weight
    gates = pd.concat(gate_rows, ignore_index=True).sort_values("row_index")
    return result, pd.DataFrame(tuning_rows), gates


def evaluate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, polarization in enumerate(["HH", "VV"]):
        observed = frame[OBSERVED_COLUMNS[index]].to_numpy(dtype=float)
        for method in METHODS:
            prediction = frame[f"{polarization.lower()}_{method}"].to_numpy(dtype=float)
            rows.append(
                {
                    "polarization": polarization,
                    "method": method,
                    **regression_metrics(observed, prediction),
                }
            )
    return pd.DataFrame(rows)


def aggregate_repeat_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0].drop(columns=["repeat"], errors="ignore").copy()
    for polarization in ["hh", "vv"]:
        for method in METHODS:
            column = f"{polarization}_{method}"
            base[column] = np.mean(
                [frame[column].to_numpy(dtype=float) for frame in frames], axis=0
            )
    for column in [
        "selective_physics_confidence",
        "selective_effective_weight",
        "selected_selective_weight",
        "selected_selective_alpha",
    ]:
        base[column] = np.mean(
            [frame[column].to_numpy(dtype=float) for frame in frames], axis=0
        )
    return base


def grouped_bootstrap_delta(
    observed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
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
        rmse_first = np.sqrt(np.mean((first[indices] - observed[indices]) ** 2, axis=0))
        rmse_second = np.sqrt(np.mean((second[indices] - observed[indices]) ** 2, axis=0))
        deltas[iteration] = rmse_first - rmse_second
    result: dict[str, object] = {}
    for index, label in enumerate(["HH", "VV"]):
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


def metric_summary(metrics_by_repeat: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (polarization, method), subset in metrics_by_repeat.groupby(
        ["polarization", "method"], sort=False
    ):
        rows.append(
            {
                "polarization": str(polarization),
                "method": str(method),
                "rmse_mean_db": float(subset["rmse_db"].mean()),
                "rmse_std_db": float(subset["rmse_db"].std(ddof=1))
                if len(subset) > 1
                else 0.0,
            }
        )
    return rows


def selected_frequency(tuning: pd.DataFrame, family: str) -> list[dict[str, object]]:
    chosen = tuning.loc[
        (tuning["family"] == family) & tuning["candidate_selected"]
    ].drop_duplicates(["repeat", "outer_fold"])
    columns = ["maximum_physics_weight"]
    if family == "selective":
        columns.append("gate_alpha")
    frequency = chosen.value_counts(columns, normalize=True).reset_index(name="fraction")
    return frequency.to_dict(orient="records")


def save_plots(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    tuning: pd.DataFrame,
    comparisons: dict[str, object],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(METHODS)))
    for ax, polarization in zip(axes, ["HH", "VV"]):
        subset = metrics[metrics["polarization"] == polarization].set_index("method").loc[METHODS]
        ax.bar([DISPLAY_NAMES[m] for m in METHODS], subset["rmse_db"], color=colors)
        ax.set(title=f"Unseen-field performance: {polarization}", ylabel="Grouped OOF RMSE (dB)")
        ax.tick_params(axis="x", rotation=42)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "01_selective_constraint_rmse.png", dpi=180)
    plt.close(fig)

    comparison_order = [
        "selective_vs_pretrained",
        "selective_vs_fixed",
        "fixed_vs_pretrained",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, polarization in zip(axes, ["HH", "VV"]):
        points, lower, upper = [], [], []
        for comparison in comparison_order:
            stats = comparisons[comparison][polarization]
            points.append(stats["rmse_delta_db"])
            lower.append(stats["rmse_delta_db"] - stats["ci_2_5_percent_db"])
            upper.append(stats["ci_97_5_percent_db"] - stats["rmse_delta_db"])
        positions = np.arange(len(points))
        ax.errorbar(positions, points, yerr=[lower, upper], fmt="o", capsize=5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(positions, [name.replace("_", " ") for name in comparison_order], rotation=25)
        ax.set(title=f"Paired field bootstrap: {polarization}", ylabel="First - second RMSE (dB; <0 favors first)")
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "02_paired_constraint_effects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].hist(frame["selective_physics_confidence"], bins=14, color="#4C78A8", alpha=0.85)
    axes[0].set(xlabel="Mean selected physics confidence", ylabel="Samples", title="Reliability gate distribution")
    axes[0].grid(axis="y", alpha=0.22)
    axes[1].scatter(
        select_vwc(frame),
        frame["selective_effective_weight"],
        c=frame["spm_k_rms_height"],
        cmap="viridis",
        s=28,
        alpha=0.72,
    )
    axes[1].set(xlabel="Vegetation water content (kg/m2)", ylabel="Effective physics weight", title="Where the SPM constraint is active")
    axes[1].grid(alpha=0.22)
    colorbar = fig.colorbar(axes[1].collections[0], ax=axes[1])
    colorbar.set_label("k × RMS height")
    fig.savefig(output_dir / "03_gate_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for row, (polarization, observed_column) in enumerate(zip(["HH", "VV"], OBSERVED_COLUMNS)):
        observed = frame[observed_column].to_numpy(dtype=float)
        for column, method in enumerate(["pretrained_mlp", "selective_constraint"]):
            predicted = frame[f"{polarization.lower()}_{method}"].to_numpy(dtype=float)
            lower = float(min(observed.min(), predicted.min()))
            upper = float(max(observed.max(), predicted.max()))
            axes[row, column].scatter(observed, predicted, s=24, alpha=0.58, edgecolors="none")
            axes[row, column].plot([lower, upper], [lower, upper], "k--", linewidth=1)
            axes[row, column].set(
                xlabel="Observed sigma0 (dB)",
                ylabel="Grouped OOF prediction (dB)",
                title=f"{polarization} | {DISPLAY_NAMES[method]}",
                xlim=(lower, upper),
                ylim=(lower, upper),
            )
            axes[row, column].grid(alpha=0.22)
    fig.savefig(output_dir / "04_observed_vs_predicted.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested unseen-field test of selective SPM constraints")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cohort", choices=["all", "common", "high_quality"], default="all")
    parser.add_argument("--synthetic-samples", type=int, default=3000)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--fine-tune-epochs", type=int, default=120)
    parser.add_argument("--physics-weights", default="0,0.05,0.1,0.2")
    parser.add_argument("--gate-alphas", default="0,0.8,1.6")
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
    alphas = sorted(set(float(value) for value in args.gate_alphas.split(",")))
    if 0.0 not in weights or any(value < 0 or value > 1 for value in weights):
        raise ValueError("physics weights must lie in [0,1] and include zero")
    if any(value < 0 for value in alphas):
        raise ValueError("gate alphas must be nonnegative")

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    expected = [
        "summary.json",
        "metrics.csv",
        "metrics_by_repeat.csv",
        "oof_predictions.csv",
        "oof_predictions_by_repeat.csv",
        "nested_tuning.csv",
        "gate_diagnostics_by_repeat.csv",
    ]
    if any((output_dir / name).exists() for name in expected):
        raise FileExistsError("Output files already exist; choose a new --output directory")
    frame = load_table(input_path, args.cohort)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_features, synthetic_targets, synthetic_metadata = generate_synthetic_spm_dataset(
        args.synthetic_samples,
        args.frequency_ghz * 1e9,
        args.incidence_angle_deg,
        seed=args.seed,
    )
    evaluated_repeats: list[pd.DataFrame] = []
    tuning_repeats: list[pd.DataFrame] = []
    gate_repeats: list[pd.DataFrame] = []
    metric_repeats: list[pd.DataFrame] = []
    pretraining_records: list[dict[str, object]] = []
    for repeat in range(1, args.repeats + 1):
        repeat_seed = args.seed + (repeat - 1) * 100_000
        bundle = pretrain_physics_model(
            synthetic_features,
            synthetic_targets,
            epochs=args.pretrain_epochs,
            seed=repeat_seed,
        )
        evaluated, tuning, gates = run_outer_evaluation(
            frame,
            bundle,
            weights,
            alphas,
            args.outer_folds,
            args.inner_folds,
            args.fine_tune_epochs,
            repeat_seed,
        )
        evaluated["repeat"] = repeat
        tuning["repeat"] = repeat
        gates["repeat"] = repeat
        metrics = evaluate_metrics(evaluated)
        metrics.insert(0, "repeat", repeat)
        evaluated_repeats.append(evaluated)
        tuning_repeats.append(tuning)
        gate_repeats.append(gates)
        metric_repeats.append(metrics)
        pretraining_records.append({"repeat": repeat, "seed": repeat_seed, **bundle.metadata})
        print(f"Completed repeat {repeat}/{args.repeats}", flush=True)

    ensemble = aggregate_repeat_predictions(evaluated_repeats)
    tuning = pd.concat(tuning_repeats, ignore_index=True)
    gates = pd.concat(gate_repeats, ignore_index=True)
    metrics_by_repeat = pd.concat(metric_repeats, ignore_index=True)
    metrics = evaluate_metrics(ensemble)
    observed = ensemble[OBSERVED_COLUMNS].to_numpy(dtype=float)
    groups = ensemble["field_id"].astype(str).to_numpy()

    def predictions(method: str) -> np.ndarray:
        return ensemble[[f"hh_{method}", f"vv_{method}"]].to_numpy(dtype=float)

    comparisons = {
        "selective_vs_pretrained": grouped_bootstrap_delta(
            observed, predictions("selective_constraint"), predictions("pretrained_mlp"), groups, args.bootstrap_iterations, args.seed
        ),
        "selective_vs_fixed": grouped_bootstrap_delta(
            observed, predictions("selective_constraint"), predictions("fixed_constraint"), groups, args.bootstrap_iterations, args.seed + 100
        ),
        "fixed_vs_pretrained": grouped_bootstrap_delta(
            observed, predictions("fixed_constraint"), predictions("pretrained_mlp"), groups, args.bootstrap_iterations, args.seed + 200
        ),
        "selective_vs_training_mean": grouped_bootstrap_delta(
            observed, predictions("selective_constraint"), predictions("training_mean"), groups, args.bootstrap_iterations, args.seed + 300
        ),
    }
    summary = {
        "research_question": "Can an observation-independent reliability gate reduce negative transfer from a low-fidelity SPM constraint on unseen fields?",
        "input": fingerprint(input_path),
        "code": fingerprint(Path(__file__)),
        "output": str(output_dir),
        "cohort": args.cohort,
        "samples": int(len(frame)),
        "fields": int(frame["field_id"].nunique()),
        "dates": int(frame["acquisition_date"].nunique()),
        "protocol": {
            "outer_split": f"{args.outer_folds}-fold GroupKFold by field",
            "inner_split": f"up to {args.inner_folds}-fold GroupKFold by field",
            "repeats": args.repeats,
            "bootstrap": f"{args.bootstrap_iterations} field-block resamples",
            "same_initialization_control": "Pretrained, fixed, and selective variants use the same initialization and epoch-wise sample order within each outer fold.",
        },
        "gate": {
            "formula": "q = q_roughness * exp(-alpha * VWC) * q_provenance; effective weight = lambda_max * q",
            "roughness_limits": {"k_sigma": 0.30, "sigma_over_L": 0.21},
            "provenance_quality": SOURCE_QUALITY,
            "candidate_maximum_weights": weights,
            "candidate_alphas": alphas,
            "uses_observed_backscatter_or_residual": False,
        },
        "synthetic_domain": synthetic_metadata,
        "pretraining_by_repeat": pretraining_records,
        "repeat_metric_summary": metric_summary(metrics_by_repeat),
        "selected_fixed_frequency": selected_frequency(tuning, "fixed"),
        "selected_selective_frequency": selected_frequency(tuning, "selective"),
        "ensemble_metrics": metrics.to_dict(orient="records"),
        "paired_field_bootstrap": comparisons,
        "gate_summary": {
            "mean_confidence": float(ensemble["selective_physics_confidence"].mean()),
            "median_confidence": float(ensemble["selective_physics_confidence"].median()),
            "mean_effective_weight": float(ensemble["selective_effective_weight"].mean()),
            "fraction_effective_weight_below_0_01": float((ensemble["selective_effective_weight"] < 0.01).mean()),
        },
        "decision_rule": {
            "primary": "Selective constraint is supported only if its grouped-bootstrap RMSE delta versus both pretrained-only and fixed-constraint models is negative with a 95% interval excluding zero in at least one polarization and it does not significantly degrade the other polarization.",
            "secondary": "Intervals crossing zero are reported as uncertain, not as improvement.",
        },
        "scope_limit": "This is an unseen-field predictive test within one SMAPVEX12 campaign. SPM is a low-fidelity teacher, not physical truth; the gate is mechanistically motivated but not a causal vegetation-scattering decomposition.",
    }

    ensemble_output = ensemble.copy()
    ensemble_output["acquisition_date"] = ensemble_output["acquisition_date"].dt.strftime("%Y-%m-%d")
    ensemble_output.to_csv(output_dir / "oof_predictions.csv", index=False)
    repeated_output = pd.concat(evaluated_repeats, ignore_index=True)
    repeated_output["acquisition_date"] = pd.to_datetime(repeated_output["acquisition_date"]).dt.strftime("%Y-%m-%d")
    repeated_output.to_csv(output_dir / "oof_predictions_by_repeat.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    metrics_by_repeat.to_csv(output_dir / "metrics_by_repeat.csv", index=False)
    tuning.to_csv(output_dir / "nested_tuning.csv", index=False)
    gates.to_csv(output_dir / "gate_diagnostics_by_repeat.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    save_plots(ensemble, metrics, tuning, comparisons, output_dir)

    print("\nSelective physics-transfer experiment completed.")
    print(f"Samples: {len(frame)}; fields: {frame['field_id'].nunique()}; dates: {frame['acquisition_date'].nunique()}")
    print(
        metrics[
            [
                "polarization",
                "method",
                "rmse_db",
                "mae_db",
                "r_squared_skill",
            ]
        ].to_string(index=False)
    )
    for name, value in comparisons.items():
        print(f"\n{name}: {json.dumps(value, ensure_ascii=False)}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
