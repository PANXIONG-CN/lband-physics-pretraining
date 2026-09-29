"""Compare direct AI and physics-residual AI on SMAPVEX12 field-day data."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

try:
    from research_pilots.scattering.surfaces.metrics import regression_metrics
    from research_pilots.scattering.surrogate.hybrid_surrogate import (
        grouped_mean_and_offset_baselines,
        nested_group_oof_predictions,
    )
except ImportError:  # standalone validation before copying to the repository
    MODULE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(MODULE_DIR))
    from hybrid_surrogate import (  # type: ignore[no-redef]
        grouped_mean_and_offset_baselines,
        nested_group_oof_predictions,
    )
    from metrics import regression_metrics  # type: ignore[no-redef]


FEATURE_COLUMNS = [
    "soil_moisture_m3_m3",
    "soil_real_dielectric",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
]
OBSERVED_COLUMNS = {"HH": "sigma0_hh_db", "VV": "sigma0_vv_db"}
PHYSICAL_COLUMNS = {
    "HH": "exponential_spm_hh_raw_db",
    "VV": "exponential_spm_vv_raw_db",
}
MODEL_ORDER = [
    "training_mean",
    "spm_raw",
    "spm_offset",
    "direct_rbf",
    "residual_rbf",
    "direct_mlp",
    "residual_mlp",
]
DISPLAY_NAMES = {
    "training_mean": "Training mean",
    "spm_raw": "Raw SPM",
    "spm_offset": "SPM + offset",
    "direct_rbf": "Direct RBF",
    "residual_rbf": "SPM + RBF residual",
    "direct_mlp": "Direct MLP",
    "residual_mlp": "SPM + MLP residual",
}


def load_common_valid_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": "string"})
    required = {
        "acquisition_date",
        "field_id",
        "spm_valid",
        *FEATURE_COLUMNS,
        *OBSERVED_COLUMNS.values(),
        *PHYSICAL_COLUMNS.values(),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")
    frame["acquisition_date"] = pd.to_datetime(
        frame["acquisition_date"], errors="raise"
    )
    frame["field_id"] = frame["field_id"].str.strip()
    if frame["spm_valid"].dtype != bool:
        frame["spm_valid"] = (
            frame["spm_valid"].astype(str).str.lower().map({"true": True, "false": False})
        )
    finite_columns = [
        *FEATURE_COLUMNS,
        *OBSERVED_COLUMNS.values(),
        *PHYSICAL_COLUMNS.values(),
    ]
    finite = np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all(axis=1)
    selected = frame.loc[frame["spm_valid"].fillna(False) & finite].copy()
    if selected.empty:
        raise ValueError("No common finite SPM-valid samples remain")
    return selected.reset_index(drop=True)


def grouped_bootstrap_rmse_delta(
    observed,
    first_prediction,
    second_prediction,
    groups,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap fields; negative delta means the first method is better."""
    observed = np.asarray(observed, dtype=float)
    first_prediction = np.asarray(first_prediction, dtype=float)
    second_prediction = np.asarray(second_prediction, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    generator = np.random.default_rng(seed)
    deltas = np.empty(int(iterations), dtype=float)
    for iteration in range(int(iterations)):
        sampled_groups = generator.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        first_rmse = np.sqrt(
            np.mean((first_prediction[indices] - observed[indices]) ** 2)
        )
        second_rmse = np.sqrt(
            np.mean((second_prediction[indices] - observed[indices]) ** 2)
        )
        deltas[iteration] = first_rmse - second_rmse
    point_first = np.sqrt(np.mean((first_prediction - observed) ** 2))
    point_second = np.sqrt(np.mean((second_prediction - observed) ** 2))
    return {
        "rmse_delta_db": float(point_first - point_second),
        "bootstrap_mean_delta_db": float(np.mean(deltas)),
        "ci_2_5_percent_db": float(np.quantile(deltas, 0.025)),
        "ci_97_5_percent_db": float(np.quantile(deltas, 0.975)),
        "probability_first_better": float(np.mean(deltas < 0.0)),
    }


def evaluate_all(
    frame: pd.DataFrame,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    features = result[FEATURE_COLUMNS].to_numpy(dtype=float)
    groups = result["field_id"].astype(str).to_numpy()
    metric_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    offset_records: dict[str, object] = {}

    for polarization in OBSERVED_COLUMNS:
        observed = result[OBSERVED_COLUMNS[polarization]].to_numpy(dtype=float)
        physical = result[PHYSICAL_COLUMNS[polarization]].to_numpy(dtype=float)
        mean_prediction, offset_prediction, fold_ids, offsets = (
            grouped_mean_and_offset_baselines(
                observed, physical, groups, outer_splits=outer_folds
            )
        )
        result["outer_fold"] = fold_ids
        result[f"{polarization.lower()}_training_mean"] = mean_prediction
        result[f"{polarization.lower()}_spm_raw"] = physical
        result[f"{polarization.lower()}_spm_offset"] = offset_prediction
        offset_records[polarization] = {
            "fold_offsets_db": [float(value) for value in offsets],
            "mean_offset_db": float(np.mean(offsets)),
        }

        for model_name in ("rbf", "mlp"):
            for strategy in ("direct", "residual"):
                nested = nested_group_oof_predictions(
                    features,
                    observed,
                    physical,
                    groups,
                    model_name=model_name,
                    strategy=strategy,
                    outer_splits=outer_folds,
                    inner_splits=inner_folds,
                    random_state=seed,
                )
                column = f"{polarization.lower()}_{strategy}_{model_name}"
                result[column] = nested.predictions
                for record in nested.tuning_records:
                    tuning_rows.append({"polarization": polarization, **record})

        for method in MODEL_ORDER:
            prediction = result[f"{polarization.lower()}_{method}"].to_numpy(
                dtype=float
            )
            metric_rows.append(
                {
                    "polarization": polarization,
                    "method": method,
                    **regression_metrics(observed, prediction),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    for polarization in OBSERVED_COLUMNS:
        reference = float(
            metrics.loc[
                (metrics["polarization"] == polarization)
                & (metrics["method"] == "spm_offset"),
                "rmse_db",
            ].iloc[0]
        )
        mask = metrics["polarization"] == polarization
        metrics.loc[mask, "rmse_improvement_vs_spm_offset_db"] = (
            reference - metrics.loc[mask, "rmse_db"]
        )

    field_rows: list[dict[str, object]] = []
    for polarization, observed_column in OBSERVED_COLUMNS.items():
        for field_id, field_frame in result.groupby("field_id"):
            observed = field_frame[observed_column].to_numpy(dtype=float)
            for method in MODEL_ORDER:
                predicted = field_frame[
                    f"{polarization.lower()}_{method}"
                ].to_numpy(dtype=float)
                field_rows.append(
                    {
                        "polarization": polarization,
                        "field_id": str(field_id),
                        "method": method,
                        "n": int(len(field_frame)),
                        "rmse_db": float(
                            np.sqrt(np.mean((predicted - observed) ** 2))
                        ),
                        "mae_db": float(np.mean(np.abs(predicted - observed))),
                    }
                )
    return (
        result,
        metrics,
        pd.DataFrame(tuning_rows),
        {"offsets": offset_records, "field_metrics": field_rows},
    )


def save_plots(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    field_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
    for ax, polarization in zip(axes, OBSERVED_COLUMNS):
        subset = metrics[metrics["polarization"] == polarization].set_index("method").loc[MODEL_ORDER]
        ax.bar([DISPLAY_NAMES[item] for item in MODEL_ORDER], subset["rmse_db"], color=colors)
        ax.set(ylabel="Outer field-held-out RMSE (dB)", title=f"Model comparison: {polarization}")
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "01_grouped_oof_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    plot_methods = ["direct_rbf", "residual_rbf", "direct_mlp", "residual_mlp"]
    for row, polarization in enumerate(OBSERVED_COLUMNS):
        observed = frame[OBSERVED_COLUMNS[polarization]].to_numpy(dtype=float)
        for column, method in enumerate(plot_methods):
            predicted = frame[f"{polarization.lower()}_{method}"].to_numpy(dtype=float)
            lower = float(min(observed.min(), predicted.min()))
            upper = float(max(observed.max(), predicted.max()))
            axes[row, column].scatter(observed, predicted, s=23, alpha=0.57, edgecolors="none")
            axes[row, column].plot([lower, upper], [lower, upper], "k--", linewidth=1)
            axes[row, column].set(xlabel="Observed sigma0 (dB)", ylabel="OOF predicted sigma0 (dB)", title=f"{polarization} | {DISPLAY_NAMES[method]}")
            axes[row, column].set_xlim(lower, upper)
            axes[row, column].set_ylim(lower, upper)
            axes[row, column].grid(alpha=0.22)
    fig.savefig(output_dir / "02_observed_vs_oof_predictions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    compare_methods = ["spm_offset", "direct_rbf", "residual_rbf", "direct_mlp", "residual_mlp"]
    for ax, polarization in zip(axes, OBSERVED_COLUMNS):
        data = [
            field_metrics.loc[
                (field_metrics["polarization"] == polarization)
                & (field_metrics["method"] == method),
                "rmse_db",
            ].to_numpy()
            for method in compare_methods
        ]
        ax.boxplot(data, tick_labels=[DISPLAY_NAMES[item] for item in compare_methods], showmeans=True)
        ax.set(ylabel="Per-field RMSE (dB)", title=f"Generalization stability: {polarization}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "03_per_field_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, polarization in zip(axes, OBSERVED_COLUMNS):
        observed = frame[OBSERVED_COLUMNS[polarization]]
        for method in ["spm_offset", "direct_rbf", "residual_rbf", "direct_mlp", "residual_mlp"]:
            residual = frame[f"{polarization.lower()}_{method}"] - observed
            ax.scatter(frame["soil_moisture_m3_m3"], residual, s=18, alpha=0.37, label=DISPLAY_NAMES[method])
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Volumetric soil moisture (m3/m3)", ylabel="OOF prediction - observation (dB)", title=f"Residual structure: {polarization}")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "04_residuals_vs_soil_moisture.png", dpi=180)
    plt.close(fig)

    date_means = frame.groupby("acquisition_date", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, polarization in zip(axes, OBSERVED_COLUMNS):
        best_method = metrics.loc[
            metrics["polarization"] == polarization
        ].sort_values("rmse_db").iloc[0]["method"]
        ax.plot(date_means["acquisition_date"], date_means[OBSERVED_COLUMNS[polarization]], "o-", linewidth=2, label="Observed")
        ax.plot(date_means["acquisition_date"], date_means[f"{polarization.lower()}_{best_method}"], "s--", linewidth=1.7, label=DISPLAY_NAMES[str(best_method)])
        ax.set(ylabel="Mean sigma0 (dB)", title=f"Date-mean diagnostic: {polarization}")
        ax.tick_params(axis="x", rotation=40)
        ax.grid(alpha=0.22)
        ax.legend()
    fig.savefig(output_dir / "05_date_mean_diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested field-held-out direct and physics-residual surrogates"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_common_valid_table(input_path)
    evaluated, metrics, tuning, extra = evaluate_all(
        frame,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    field_metrics = pd.DataFrame(extra.pop("field_metrics"))

    bootstrap: dict[str, object] = {}
    groups = evaluated["field_id"].astype(str).to_numpy()
    for polarization, observed_column in OBSERVED_COLUMNS.items():
        observed = evaluated[observed_column].to_numpy(dtype=float)
        for model in ("rbf", "mlp"):
            bootstrap[f"{polarization}_residual_vs_direct_{model}"] = (
                grouped_bootstrap_rmse_delta(
                    observed,
                    evaluated[f"{polarization.lower()}_residual_{model}"],
                    evaluated[f"{polarization.lower()}_direct_{model}"],
                    groups,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + (0 if model == "rbf" else 100),
                )
            )
        best_method = str(
            metrics.loc[metrics["polarization"] == polarization]
            .sort_values("rmse_db")
            .iloc[0]["method"]
        )
        bootstrap[f"{polarization}_best_vs_spm_offset"] = (
            grouped_bootstrap_rmse_delta(
                observed,
                evaluated[f"{polarization.lower()}_{best_method}"],
                evaluated[f"{polarization.lower()}_spm_offset"],
                groups,
                iterations=args.bootstrap_iterations,
                seed=args.seed + 200,
            )
        )

    best_records = (
        metrics.sort_values(["polarization", "rmse_db"])
        .groupby("polarization", as_index=False)
        .first()
        .to_dict(orient="records")
    )
    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "evaluation_design": {
            "outer_split": f"{args.outer_folds}-fold GroupKFold by field_id",
            "inner_split": f"up to {args.inner_folds}-fold GroupKFold by field_id",
            "features": FEATURE_COLUMNS,
            "field_id_used_as_feature": False,
            "date_used_as_feature": False,
            "physical_prior": "raw exponential-spectrum first-order SPM",
            "bootstrap_unit": "field_id",
            "bootstrap_iterations": int(args.bootstrap_iterations),
        },
        "samples": int(len(evaluated)),
        "fields": int(evaluated["field_id"].nunique()),
        "dates": int(evaluated["acquisition_date"].nunique()),
        "best_outer_oof_records": best_records,
        "paired_group_bootstrap": bootstrap,
        **extra,
        "interpretation_rule": (
            "A hybrid model is supported only when outer field-held-out error "
            "improves, R2 becomes useful, and the paired field bootstrap interval "
            "does not indicate an unstable comparison."
        ),
    }

    output_frame = evaluated.copy()
    output_frame["acquisition_date"] = output_frame["acquisition_date"].dt.strftime("%Y-%m-%d")
    output_frame.to_csv(output_dir / "hybrid_oof_predictions.csv", index=False)
    metrics.to_csv(output_dir / "hybrid_metrics.csv", index=False)
    tuning.to_csv(output_dir / "hybrid_nested_tuning.csv", index=False)
    field_metrics.to_csv(output_dir / "hybrid_field_metrics.csv", index=False)
    with (output_dir / "hybrid_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_plots(evaluated, metrics, field_metrics, output_dir)

    print("Nested field-held-out surrogate comparison completed.")
    print(f"Samples: {len(evaluated)}; fields: {evaluated['field_id'].nunique()}; dates: {evaluated['acquisition_date'].nunique()}")
    print("\nOuter OOF metrics:")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
