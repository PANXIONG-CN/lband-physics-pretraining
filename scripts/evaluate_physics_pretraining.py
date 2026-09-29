"""Evaluate SPM pretraining and weak physics-constrained fine-tuning."""

from __future__ import annotations

import argparse
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

try:
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
except ImportError:  # standalone validation
    MODULE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(MODULE_DIR))
    from metrics import regression_metrics  # type: ignore[no-redef]
    from physics_pretraining import (  # type: ignore[no-redef]
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
    "spm_offset",
    "scratch_mlp",
    "pretrained_mlp",
    "physics_constrained_pretrained_mlp",
]
DISPLAY_NAMES = {
    "training_mean": "Training mean",
    "spm_offset": "SPM + offset",
    "scratch_mlp": "Scratch MLP",
    "pretrained_mlp": "SPM-pretrained MLP",
    "physics_constrained_pretrained_mlp": "Pretrained + physics constraint",
}


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
    if frame["spm_valid"].dtype != bool:
        frame["spm_valid"] = frame["spm_valid"].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    numeric = [*FEATURE_NAMES, *OBSERVED_COLUMNS, *PHYSICAL_COLUMNS]
    finite = np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    selected = frame.loc[frame["spm_valid"].fillna(False) & finite].copy()
    if selected.empty:
        raise ValueError("No finite SPM-valid samples")
    return selected.reset_index(drop=True)


def joint_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    per_channel = np.sqrt(np.mean((predicted - observed) ** 2, axis=0))
    return float(np.mean(per_channel))


def calibration_offsets(observed: np.ndarray, physical: np.ndarray) -> np.ndarray:
    return np.mean(observed - physical, axis=0)


def tune_physics_weight(
    bundle,
    features: np.ndarray,
    observed: np.ndarray,
    physical: np.ndarray,
    groups: np.ndarray,
    candidate_weights: list[float],
    inner_folds: int,
    fine_tune_epochs: int,
    seed: int,
) -> tuple[float, list[dict[str, object]]]:
    fold_count = min(int(inner_folds), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=fold_count)
    records: list[dict[str, object]] = []
    weight_scores: dict[float, list[float]] = {weight: [] for weight in candidate_weights}
    for inner_fold, (train, validation) in enumerate(
        splitter.split(features, observed, groups), start=1
    ):
        offsets = calibration_offsets(observed[train], physical[train])
        calibrated_train = physical[train] + offsets
        for weight in candidate_weights:
            target = blended_physics_target(
                observed[train], calibrated_train, weight
            )
            model = fine_tune_pretrained(
                bundle,
                features[train],
                target,
                epochs=fine_tune_epochs,
                seed=seed + inner_fold * 100 + int(weight * 1000),
            )
            prediction = predict_physical_units(bundle, model, features[validation])
            score = joint_rmse(observed[validation], prediction)
            weight_scores[weight].append(score)
            records.append(
                {
                    "inner_fold": int(inner_fold),
                    "physics_weight": float(weight),
                    "joint_validation_rmse_db": float(score),
                }
            )
    mean_scores = {
        weight: float(np.mean(scores)) for weight, scores in weight_scores.items()
    }
    best_weight = min(mean_scores, key=mean_scores.get)
    for record in records:
        record["mean_rmse_for_weight_db"] = mean_scores[
            float(record["physics_weight"])
        ]
        record["selected_weight"] = float(best_weight)
    return float(best_weight), records


def grouped_bootstrap_delta(
    observed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    unique_groups = np.unique(groups)
    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    generator = np.random.default_rng(seed)
    deltas = np.empty((iterations, observed.shape[1]), dtype=float)
    for iteration in range(iterations):
        sampled = generator.choice(unique_groups, len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        first_rmse = np.sqrt(np.mean((first[indices] - observed[indices]) ** 2, axis=0))
        second_rmse = np.sqrt(np.mean((second[indices] - observed[indices]) ** 2, axis=0))
        deltas[iteration] = first_rmse - second_rmse
    labels = ["HH", "VV"]
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


def run_outer_evaluation(
    frame: pd.DataFrame,
    bundle,
    candidate_weights: list[float],
    outer_folds: int,
    inner_folds: int,
    fine_tune_epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    observed = frame[OBSERVED_COLUMNS].to_numpy(dtype=float)
    physical = frame[PHYSICAL_COLUMNS].to_numpy(dtype=float)
    groups = frame["field_id"].astype(str).to_numpy()
    output = frame.copy()
    predictions = {
        method: np.full_like(observed, np.nan, dtype=float) for method in METHODS
    }
    tuning_rows: list[dict[str, object]] = []
    outer = GroupKFold(n_splits=min(outer_folds, len(np.unique(groups))))
    fold_ids = np.full(len(frame), -1, dtype=int)

    for outer_fold, (train, test) in enumerate(
        outer.split(features, observed, groups), start=1
    ):
        offsets = calibration_offsets(observed[train], physical[train])
        calibrated_train = physical[train] + offsets
        predictions["training_mean"][test] = np.mean(observed[train], axis=0)
        predictions["spm_offset"][test] = physical[test] + offsets

        scratch = train_from_scratch(
            bundle,
            features[train],
            observed[train],
            epochs=fine_tune_epochs,
            seed=seed + outer_fold * 1000,
        )
        predictions["scratch_mlp"][test] = predict_physical_units(
            bundle, scratch, features[test]
        )

        pretrained = fine_tune_pretrained(
            bundle,
            features[train],
            observed[train],
            epochs=fine_tune_epochs,
            seed=seed + outer_fold * 1000 + 1,
        )
        predictions["pretrained_mlp"][test] = predict_physical_units(
            bundle, pretrained, features[test]
        )

        best_weight, inner_records = tune_physics_weight(
            bundle,
            features[train],
            observed[train],
            physical[train],
            groups[train],
            candidate_weights,
            inner_folds,
            fine_tune_epochs,
            seed + outer_fold * 10000,
        )
        constrained_target = blended_physics_target(
            observed[train], calibrated_train, best_weight
        )
        constrained = fine_tune_pretrained(
            bundle,
            features[train],
            constrained_target,
            epochs=fine_tune_epochs,
            seed=seed + outer_fold * 1000 + 2,
        )
        predictions["physics_constrained_pretrained_mlp"][test] = (
            predict_physical_units(bundle, constrained, features[test])
        )
        fold_ids[test] = outer_fold
        for record in inner_records:
            tuning_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_fields": int(len(np.unique(groups[train]))),
                    "test_fields": int(len(np.unique(groups[test]))),
                    **record,
                }
            )

    output["outer_fold"] = fold_ids
    for method, values in predictions.items():
        output[f"hh_{method}"] = values[:, 0]
        output[f"vv_{method}"] = values[:, 1]
    return output, pd.DataFrame(tuning_rows)


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


def aggregate_repeat_predictions(
    repeated_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Average stochastic model predictions while retaining one row per sample."""
    if not repeated_frames:
        raise ValueError("At least one repeat is required")
    prediction_columns = [
        f"{polarization}_{method}"
        for polarization in ["hh", "vv"]
        for method in METHODS
    ]
    base = repeated_frames[0].drop(columns=["repeat"], errors="ignore").copy()
    for column in prediction_columns:
        base[column] = np.mean(
            [frame[column].to_numpy(dtype=float) for frame in repeated_frames],
            axis=0,
        )
    return base


def repeat_metric_summary(metrics_by_repeat: pd.DataFrame) -> list[dict[str, object]]:
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


def repeat_pairwise_summary(metrics_by_repeat: pd.DataFrame) -> dict[str, object]:
    indexed = metrics_by_repeat.set_index(["repeat", "polarization", "method"])
    comparisons = {
        "pretraining_vs_scratch": ("pretrained_mlp", "scratch_mlp"),
        "physics_constraint_vs_pretrained": (
            "physics_constrained_pretrained_mlp",
            "pretrained_mlp",
        ),
    }
    result: dict[str, object] = {}
    repeats = sorted(metrics_by_repeat["repeat"].unique())
    for name, (first, second) in comparisons.items():
        channel_result: dict[str, object] = {}
        for polarization in ["HH", "VV"]:
            deltas = np.asarray(
                [
                    indexed.loc[(repeat, polarization, first), "rmse_db"]
                    - indexed.loc[(repeat, polarization, second), "rmse_db"]
                    for repeat in repeats
                ],
                dtype=float,
            )
            channel_result[polarization] = {
                "rmse_delta_mean_db": float(deltas.mean()),
                "rmse_delta_std_db": float(deltas.std(ddof=1))
                if len(deltas) > 1
                else 0.0,
                "fraction_first_better": float(np.mean(deltas < 0.0)),
            }
        result[name] = channel_result
    return result


def save_plots(frame: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, polarization in zip(axes, ["HH", "VV"]):
        subset = metrics[metrics["polarization"] == polarization].set_index("method").loc[METHODS]
        ax.bar([DISPLAY_NAMES[method] for method in METHODS], subset["rmse_db"], color=plt.cm.Set2(np.linspace(0, 1, len(METHODS))))
        ax.set(ylabel="Outer field-held-out RMSE (dB)", title=f"Pretraining comparison: {polarization}")
        ax.tick_params(axis="x", rotation=48)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "01_pretraining_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    plot_methods = ["scratch_mlp", "pretrained_mlp", "physics_constrained_pretrained_mlp"]
    for row, (polarization, observed_column) in enumerate(zip(["HH", "VV"], OBSERVED_COLUMNS)):
        observed = frame[observed_column].to_numpy(dtype=float)
        for column, method in enumerate(plot_methods):
            prediction = frame[f"{polarization.lower()}_{method}"].to_numpy(dtype=float)
            lower = float(min(observed.min(), prediction.min()))
            upper = float(max(observed.max(), prediction.max()))
            axes[row, column].scatter(observed, prediction, s=23, alpha=0.56, edgecolors="none")
            axes[row, column].plot([lower, upper], [lower, upper], "k--", linewidth=1)
            axes[row, column].set(xlabel="Observed sigma0 (dB)", ylabel="OOF prediction (dB)", title=f"{polarization} | {DISPLAY_NAMES[method]}")
            axes[row, column].set_xlim(lower, upper)
            axes[row, column].set_ylim(lower, upper)
            axes[row, column].grid(alpha=0.22)
    fig.savefig(output_dir / "02_pretraining_observed_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (polarization, observed_column) in zip(axes, zip(["HH", "VV"], OBSERVED_COLUMNS)):
        observed = frame[observed_column]
        for method in ["scratch_mlp", "pretrained_mlp", "physics_constrained_pretrained_mlp"]:
            ax.scatter(
                frame["soil_moisture_m3_m3"],
                frame[f"{polarization.lower()}_{method}"] - observed,
                s=20,
                alpha=0.42,
                label=DISPLAY_NAMES[method],
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Volumetric soil moisture (m3/m3)", ylabel="Prediction - observation (dB)", title=f"Residual domain shift: {polarization}")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "03_pretraining_residuals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Field-held-out physics pretraining and constrained fine-tuning"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--synthetic-samples", type=int, default=3000)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--fine-tune-epochs", type=int, default=120)
    parser.add_argument("--physics-weights", default="0,0.05,0.1,0.2,0.4")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Random-seed repeats; predictions are also averaged as an ensemble",
    )
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")

    candidate_weights = [float(item) for item in args.physics_weights.split(",")]
    if 0.0 not in candidate_weights:
        raise ValueError("physics-weights must include 0 for an unconstrained option")
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_table(input_path)

    synthetic_features, synthetic_targets, synthetic_metadata = (
        generate_synthetic_spm_dataset(
            args.synthetic_samples,
            args.frequency_ghz * 1e9,
            args.incidence_angle_deg,
            seed=args.seed,
        )
    )
    repeated_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    repeat_metrics: list[pd.DataFrame] = []
    pretraining_records: list[dict[str, object]] = []
    for repeat in range(1, args.repeats + 1):
        repeat_seed = args.seed + (repeat - 1) * 100_000
        bundle = pretrain_physics_model(
            synthetic_features,
            synthetic_targets,
            epochs=args.pretrain_epochs,
            seed=repeat_seed,
        )
        evaluated_repeat, tuning_repeat = run_outer_evaluation(
            frame,
            bundle,
            candidate_weights,
            args.outer_folds,
            args.inner_folds,
            args.fine_tune_epochs,
            repeat_seed,
        )
        evaluated_repeat["repeat"] = repeat
        tuning_repeat["repeat"] = repeat
        metrics_repeat = evaluate_metrics(evaluated_repeat)
        metrics_repeat.insert(0, "repeat", repeat)
        repeated_frames.append(evaluated_repeat)
        tuning_frames.append(tuning_repeat)
        repeat_metrics.append(metrics_repeat)
        pretraining_records.append(
            {"repeat": repeat, "seed": repeat_seed, **bundle.metadata}
        )

    evaluated = aggregate_repeat_predictions(repeated_frames)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    metrics_by_repeat = pd.concat(repeat_metrics, ignore_index=True)
    metrics = evaluate_metrics(evaluated)
    observed = evaluated[OBSERVED_COLUMNS].to_numpy(dtype=float)
    groups = evaluated["field_id"].astype(str).to_numpy()

    comparisons = {
        "pretraining_vs_scratch": grouped_bootstrap_delta(
            observed,
            evaluated[["hh_pretrained_mlp", "vv_pretrained_mlp"]].to_numpy(),
            evaluated[["hh_scratch_mlp", "vv_scratch_mlp"]].to_numpy(),
            groups,
            args.bootstrap_iterations,
            args.seed,
        ),
        "physics_constraint_vs_pretrained": grouped_bootstrap_delta(
            observed,
            evaluated[["hh_physics_constrained_pretrained_mlp", "vv_physics_constrained_pretrained_mlp"]].to_numpy(),
            evaluated[["hh_pretrained_mlp", "vv_pretrained_mlp"]].to_numpy(),
            groups,
            args.bootstrap_iterations,
            args.seed + 100,
        ),
        "constrained_vs_training_mean": grouped_bootstrap_delta(
            observed,
            evaluated[["hh_physics_constrained_pretrained_mlp", "vv_physics_constrained_pretrained_mlp"]].to_numpy(),
            evaluated[["hh_training_mean", "vv_training_mean"]].to_numpy(),
            groups,
            args.bootstrap_iterations,
            args.seed + 200,
        ),
    }
    selected_weights = (
        tuning.groupby(["repeat", "outer_fold"])["selected_weight"]
        .first()
        .reset_index()
    )
    weight_frequency = (
        selected_weights["selected_weight"].value_counts(normalize=True).sort_index()
    )
    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "research_question": (
            "Does exponential-SPM pretraining plus a weak, inner-CV-selected "
            "physics constraint improve unseen-field generalization under scarce data?"
        ),
        "samples": int(len(evaluated)),
        "fields": int(evaluated["field_id"].nunique()),
        "dates": int(evaluated["acquisition_date"].nunique()),
        "synthetic_domain": synthetic_metadata,
        "repeat_count": int(args.repeats),
        "pretraining_by_repeat": pretraining_records,
        "repeat_metric_summary": repeat_metric_summary(metrics_by_repeat),
        "repeat_pairwise_summary": repeat_pairwise_summary(metrics_by_repeat),
        "candidate_physics_weights": candidate_weights,
        "selected_physics_weight_frequency": {
            str(float(key)): float(value) for key, value in weight_frequency.items()
        },
        "paired_field_bootstrap": comparisons,
        "innovation_guardrail": (
            "Pretraining is evidence-supported only if it improves outer grouped "
            "OOF error over scratch training. Physics constraint is supported only "
            "if it further improves pretraining without negative transfer."
        ),
    }

    output_frame = evaluated.copy()
    output_frame["acquisition_date"] = output_frame["acquisition_date"].dt.strftime("%Y-%m-%d")
    output_frame.to_csv(output_dir / "pretraining_oof_predictions.csv", index=False)
    repeated_output = pd.concat(repeated_frames, ignore_index=True)
    repeated_output["acquisition_date"] = pd.to_datetime(
        repeated_output["acquisition_date"]
    ).dt.strftime("%Y-%m-%d")
    repeated_output.to_csv(
        output_dir / "pretraining_oof_predictions_by_repeat.csv", index=False
    )
    metrics.to_csv(output_dir / "pretraining_metrics.csv", index=False)
    metrics_by_repeat.to_csv(
        output_dir / "pretraining_metrics_by_repeat.csv", index=False
    )
    tuning.to_csv(output_dir / "physics_weight_nested_tuning.csv", index=False)
    pd.DataFrame(synthetic_features, columns=FEATURE_NAMES).assign(
        synthetic_sigma0_hh_db=synthetic_targets[:, 0],
        synthetic_sigma0_vv_db=synthetic_targets[:, 1],
    ).to_csv(output_dir / "synthetic_pretraining_dataset.csv", index=False)
    with (output_dir / "pretraining_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_plots(evaluated, metrics, output_dir)

    print("Physics pretraining experiment completed.")
    print(f"Real samples: {len(evaluated)}; fields: {evaluated['field_id'].nunique()}")
    print(f"Synthetic pretraining samples: {len(synthetic_features)}")
    print(f"Random-seed repeats: {args.repeats}")
    print("\nOuter field-held-out metrics:")
    print(metrics.to_string(index=False))
    print("\nSelected physics-weight frequency:")
    print(weight_frequency.to_string())
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
