"""Matched vegetation-input ablation for SMAPVEX12 ground backscatter.

The experiment answers one narrow question: when samples, held-out fields,
models and training budgets are fixed, does adding in-situ vegetation water
content improve prediction of observed PALS HH/VV sigma0?

The script deliberately keeps the source cohort immutable. All scaling is fit
inside each outer training fold. Fields, rather than rows, define the folds.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


KEY_COLUMNS = ["acquisition_date", "field_id"]
ORIGINAL_FEATURES = [
    "soil_moisture_m3_m3",
    "soil_real_dielectric",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
]
VWC_COLUMN = "vegetation_water_content_in_situ_kg_m2"
TARGET_COLUMNS = ["sigma0_hh_db", "sigma0_vv_db"]
SPM_COLUMNS = ["exponential_spm_hh_raw_db", "exponential_spm_vv_raw_db"]
REQUIRED_COLUMNS = (
    KEY_COLUMNS
    + ORIGINAL_FEATURES
    + [VWC_COLUMN, "vegetation_time_offset_days"]
    + TARGET_COLUMNS
    + SPM_COLUMNS
)
INPUT_SETS = {
    "original": ORIGINAL_FEATURES,
    "original_plus_vwc": ORIGINAL_FEATURES + [VWC_COLUMN],
}
MODEL_CONFIG = {
    "ridge": {"alpha": 1.0, "repeats": 1},
    "mlp_tanh": {
        "alpha": 1.0e-2,
        "hidden_layer_sizes": [16, 8],
        "activation": "tanh",
        "solver": "adam",
        "learning_rate_init": 1.0e-3,
        "max_iter": 5000,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 100,
        "tol": 1.0e-5,
        "repeats": 5,
    },
}
SEEDS = [42, 10042, 20042, 30042, 40042]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the incremental value of in-situ VWC with matched field folds."
    )
    parser.add_argument("--main-input", required=True, type=Path)
    parser.add_argument("--sensitivity-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(np.var(y_true), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def prepare_cohort(path: Path, cohort_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Cohort not found: {path}")

    frame = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{cohort_name}: missing columns: {missing_columns}")

    data = frame[REQUIRED_COLUMNS].copy()
    data["acquisition_date"] = pd.to_datetime(
        data["acquisition_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    data["field_id"] = data["field_id"].astype(str)

    numeric_columns = [c for c in REQUIRED_COLUMNS if c not in KEY_COLUMNS]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[REQUIRED_COLUMNS].isna().any().any():
        counts = data[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(
            f"{cohort_name}: missing required values: "
            f"{counts[counts > 0].to_dict()}"
        )
    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{cohort_name}: non-finite required numeric values")
    duplicates = data.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        examples = data.loc[duplicates, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"{cohort_name}: duplicate sample keys: {examples}")
    if data["field_id"].nunique() < 5:
        raise ValueError(f"{cohort_name}: fewer than five fields")

    data["cohort"] = cohort_name
    data["record_id"] = (
        data["acquisition_date"] + "__field_" + data["field_id"]
    )
    data["vegetation_match_abs_gap_days"] = data[
        "vegetation_time_offset_days"
    ].abs()
    return data.sort_values(KEY_COLUMNS).reset_index(drop=True)


def assign_outer_folds(data: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    n_groups = data["field_id"].nunique()
    if n_splits < 2 or n_splits > n_groups:
        raise ValueError(f"outer folds must be in [2, {n_groups}]")

    assigned = data.copy()
    assigned["outer_fold"] = -1
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((len(assigned), 1), dtype=float)
    for fold, (_, test_index) in enumerate(
        splitter.split(dummy_x, groups=assigned["field_id"])
    ):
        assigned.loc[test_index, "outer_fold"] = fold

    per_field = assigned.groupby("field_id")["outer_fold"].nunique()
    if not (per_field == 1).all() or (assigned["outer_fold"] < 0).any():
        raise RuntimeError("A field was split across outer folds")
    return assigned


def fit_predict_scaled(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, bool]:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_train_scaled = y_scaler.transform(y_train)

    if model_name == "ridge":
        model = Ridge(alpha=float(MODEL_CONFIG["ridge"]["alpha"]))
    elif model_name == "mlp_tanh":
        config = MODEL_CONFIG["mlp_tanh"]
        model = MLPRegressor(
            hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
            activation=str(config["activation"]),
            solver=str(config["solver"]),
            alpha=float(config["alpha"]),
            learning_rate_init=float(config["learning_rate_init"]),
            max_iter=int(config["max_iter"]),
            early_stopping=bool(config["early_stopping"]),
            validation_fraction=float(config["validation_fraction"]),
            n_iter_no_change=int(config["n_iter_no_change"]),
            tol=float(config["tol"]),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train_scaled, y_train_scaled)
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    prediction = y_scaler.inverse_transform(model.predict(x_test_scaled))
    return np.asarray(prediction, dtype=float), warned


def run_cohort(
    data: pd.DataFrame, n_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assigned = assign_outer_folds(data, n_splits=n_splits)
    prediction_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []

    for fold in range(n_splits):
        test_mask = assigned["outer_fold"].eq(fold)
        train = assigned.loc[~test_mask]
        test = assigned.loc[test_mask]
        y_train = train[TARGET_COLUMNS].to_numpy(dtype=float)
        y_test = test[TARGET_COLUMNS].to_numpy(dtype=float)

        # Two references use exactly the same held-out fields.
        mean_prediction = np.repeat(
            y_train.mean(axis=0, keepdims=True), len(test), axis=0
        )
        spm_prediction = test[SPM_COLUMNS].to_numpy(dtype=float)
        for reference_name, reference_prediction in [
            ("training_mean", mean_prediction),
            ("raw_exponential_spm", spm_prediction),
        ]:
            for local_i, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    prediction_record(
                        row,
                        input_set="reference",
                        model=reference_name,
                        repeat=0,
                        seed=-1,
                        observed=y_test[local_i],
                        predicted=reference_prediction[local_i],
                        convergence_warning=False,
                    )
                )

        for input_set, feature_columns in INPUT_SETS.items():
            x_train = train[feature_columns].to_numpy(dtype=float)
            x_test = test[feature_columns].to_numpy(dtype=float)
            for model_name in MODEL_CONFIG:
                repeats = int(MODEL_CONFIG[model_name]["repeats"])
                for repeat in range(repeats):
                    seed = SEEDS[repeat]
                    predicted, warned = fit_predict_scaled(
                        model_name, x_train, y_train, x_test, seed
                    )
                    warning_rows.append(
                        {
                            "cohort": str(test["cohort"].iloc[0]),
                            "outer_fold": fold,
                            "input_set": input_set,
                            "model": model_name,
                            "repeat": repeat,
                            "seed": seed,
                            "convergence_warning": warned,
                        }
                    )
                    for local_i, (_, row) in enumerate(test.iterrows()):
                        prediction_rows.append(
                            prediction_record(
                                row,
                                input_set=input_set,
                                model=model_name,
                                repeat=repeat,
                                seed=seed,
                                observed=y_test[local_i],
                                predicted=predicted[local_i],
                                convergence_warning=warned,
                            )
                        )

    predictions = pd.DataFrame(prediction_rows)
    fold_columns = [
        "cohort",
        "record_id",
        "acquisition_date",
        "field_id",
        "outer_fold",
        VWC_COLUMN,
        "vegetation_match_abs_gap_days",
    ]
    folds = assigned[fold_columns].copy()
    warning_table = pd.DataFrame(warning_rows)
    validate_prediction_pairing(predictions, assigned)
    return predictions, folds, warning_table


def prediction_record(
    row: pd.Series,
    input_set: str,
    model: str,
    repeat: int,
    seed: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    convergence_warning: bool,
) -> dict[str, object]:
    return {
        "cohort": row["cohort"],
        "record_id": row["record_id"],
        "acquisition_date": row["acquisition_date"],
        "field_id": row["field_id"],
        "outer_fold": int(row["outer_fold"]),
        VWC_COLUMN: float(row[VWC_COLUMN]),
        "vegetation_match_abs_gap_days": float(
            row["vegetation_match_abs_gap_days"]
        ),
        "input_set": input_set,
        "model": model,
        "repeat": repeat,
        "seed": seed,
        "observed_hh_db": float(observed[0]),
        "observed_vv_db": float(observed[1]),
        "predicted_hh_db": float(predicted[0]),
        "predicted_vv_db": float(predicted[1]),
        "convergence_warning": bool(convergence_warning),
    }


def validate_prediction_pairing(
    predictions: pd.DataFrame, assigned: pd.DataFrame
) -> None:
    expected = set(assigned["record_id"])
    for (input_set, model, repeat), group in predictions.groupby(
        ["input_set", "model", "repeat"], sort=False
    ):
        observed = set(group["record_id"])
        if observed != expected or group["record_id"].duplicated().any():
            raise RuntimeError(
                f"Unpaired predictions for {input_set}/{model}/repeat={repeat}"
            )
        field_folds = group.groupby("field_id")["outer_fold"].nunique()
        if not (field_folds == 1).all():
            raise RuntimeError("Prediction table split a field across folds")


def metric_rows(
    group: pd.DataFrame,
    cohort: str,
    input_set: str,
    model: str,
    repeat: int | str,
) -> list[dict[str, object]]:
    observed = group[["observed_hh_db", "observed_vv_db"]].to_numpy(float)
    predicted = group[["predicted_hh_db", "predicted_vv_db"]].to_numpy(float)
    rows: list[dict[str, object]] = []
    for index, target in enumerate(["HH", "VV"]):
        rows.append(
            metric_record(
                cohort,
                input_set,
                model,
                repeat,
                target,
                observed[:, index],
                predicted[:, index],
            )
        )
    rows.append(
        metric_record(
            cohort,
            input_set,
            model,
            repeat,
            "polarization_difference_VV_minus_HH",
            observed[:, 1] - observed[:, 0],
            predicted[:, 1] - predicted[:, 0],
        )
    )
    rows.append(
        metric_record(
            cohort,
            input_set,
            model,
            repeat,
            "joint_HH_VV",
            observed.ravel(),
            predicted.ravel(),
        )
    )
    return rows


def metric_record(
    cohort: str,
    input_set: str,
    model: str,
    repeat: int | str,
    target: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    return {
        "cohort": cohort,
        "input_set": input_set,
        "model": model,
        "repeat": repeat,
        "target": target,
        "n": int(len(observed)),
        "rmse_db": rmse(observed, predicted),
        "mae_db": mae(observed, predicted),
        "r2": safe_r2(observed, predicted),
        "mean_error_db": float(np.mean(predicted - observed)),
    }


def calculate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(
        ["cohort", "input_set", "model", "repeat"], sort=False
    ):
        repeat_rows.extend(metric_rows(group, *keys))
    metrics_by_repeat = pd.DataFrame(repeat_rows)

    key_columns = [
        "cohort",
        "record_id",
        "acquisition_date",
        "field_id",
        "outer_fold",
        VWC_COLUMN,
        "vegetation_match_abs_gap_days",
        "input_set",
        "model",
    ]
    ensemble = (
        predictions.groupby(key_columns, as_index=False)
        .agg(
            observed_hh_db=("observed_hh_db", "first"),
            observed_vv_db=("observed_vv_db", "first"),
            predicted_hh_db=("predicted_hh_db", "mean"),
            predicted_vv_db=("predicted_vv_db", "mean"),
            repeat_count=("repeat", "nunique"),
        )
        .sort_values(["cohort", "model", "input_set", "record_id"])
    )
    ensemble_rows: list[dict[str, object]] = []
    for keys, group in ensemble.groupby(
        ["cohort", "input_set", "model"], sort=False
    ):
        ensemble_rows.extend(metric_rows(group, *keys, repeat="ensemble"))
    ensemble_metrics = pd.DataFrame(ensemble_rows)
    return metrics_by_repeat, ensemble, ensemble_metrics


def target_arrays(frame: pd.DataFrame, target: str, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "HH":
        return (
            frame["observed_hh_db"].to_numpy(float),
            frame[f"predicted_hh_db_{suffix}"].to_numpy(float),
        )
    if target == "VV":
        return (
            frame["observed_vv_db"].to_numpy(float),
            frame[f"predicted_vv_db_{suffix}"].to_numpy(float),
        )
    if target == "polarization_difference_VV_minus_HH":
        observed = frame["observed_vv_db"].to_numpy(float) - frame[
            "observed_hh_db"
        ].to_numpy(float)
        predicted = frame[f"predicted_vv_db_{suffix}"].to_numpy(float) - frame[
            f"predicted_hh_db_{suffix}"
        ].to_numpy(float)
        return observed, predicted
    if target == "joint_HH_VV":
        observed = frame[["observed_hh_db", "observed_vv_db"]].to_numpy(float)
        predicted = frame[
            [f"predicted_hh_db_{suffix}", f"predicted_vv_db_{suffix}"]
        ].to_numpy(float)
        return observed.ravel(), predicted.ravel()
    raise ValueError(target)


def paired_frames(ensemble: pd.DataFrame, cohort: str, model: str) -> pd.DataFrame:
    subset = ensemble.loc[
        ensemble["cohort"].eq(cohort)
        & ensemble["model"].eq(model)
        & ensemble["input_set"].isin(INPUT_SETS)
    ].copy()
    base = subset.loc[subset["input_set"].eq("original")].copy()
    augmented = subset.loc[subset["input_set"].eq("original_plus_vwc")].copy()
    merge_columns = ["record_id", "field_id", "outer_fold"]
    keep = merge_columns + [
        "observed_hh_db",
        "observed_vv_db",
        "predicted_hh_db",
        "predicted_vv_db",
    ]
    paired = base[keep].merge(
        augmented[keep],
        on=merge_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_augmented"),
    )
    if len(paired) != len(base) or len(paired) != len(augmented):
        raise RuntimeError(f"Incomplete paired comparison for {cohort}/{model}")
    for target in ["hh", "vv"]:
        left = paired[f"observed_{target}_db_original"]
        right = paired[f"observed_{target}_db_augmented"]
        if not np.allclose(left, right):
            raise RuntimeError("Observed targets changed between input sets")
        paired[f"observed_{target}_db"] = left
    return paired


def calculate_paired_effects(
    ensemble: pd.DataFrame, bootstrap_repeats: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    targets = [
        "HH",
        "VV",
        "polarization_difference_VV_minus_HH",
        "joint_HH_VV",
    ]
    for cohort in sorted(ensemble["cohort"].unique()):
        for model in ["ridge", "mlp_tanh"]:
            paired = paired_frames(ensemble, cohort, model)
            fields = np.array(sorted(paired["field_id"].unique()))
            row_index_by_field = {
                field: np.flatnonzero(paired["field_id"].to_numpy() == field)
                for field in fields
            }
            for target in targets:
                observed_original, predicted_original = target_arrays(
                    paired, target, "original"
                )
                observed_augmented, predicted_augmented = target_arrays(
                    paired, target, "augmented"
                )
                if not np.allclose(observed_original, observed_augmented):
                    raise RuntimeError("Target mismatch in paired effect")
                point_delta = rmse(
                    observed_augmented, predicted_augmented
                ) - rmse(observed_original, predicted_original)

                boot = np.empty(bootstrap_repeats, dtype=float)
                for b in range(bootstrap_repeats):
                    sampled_fields = rng.choice(fields, size=len(fields), replace=True)
                    indices = np.concatenate(
                        [row_index_by_field[field] for field in sampled_fields]
                    )
                    if target == "joint_HH_VV":
                        indices = np.concatenate([2 * indices, 2 * indices + 1])
                    boot[b] = rmse(
                        observed_augmented[indices], predicted_augmented[indices]
                    ) - rmse(
                        observed_original[indices], predicted_original[indices]
                    )

                per_field_deltas: list[float] = []
                for field in fields:
                    indices = row_index_by_field[field]
                    target_indices = indices
                    if target == "joint_HH_VV":
                        target_indices = np.concatenate([2 * indices, 2 * indices + 1])
                    delta = rmse(
                        observed_augmented[target_indices],
                        predicted_augmented[target_indices],
                    ) - rmse(
                        observed_original[target_indices],
                        predicted_original[target_indices],
                    )
                    per_field_deltas.append(delta)
                    field_rows.append(
                        {
                            "cohort": cohort,
                            "model": model,
                            "target": target,
                            "field_id": field,
                            "n": int(len(indices)),
                            "delta_rmse_db_augmented_minus_original": delta,
                            "improved_with_vwc": bool(delta < 0.0),
                        }
                    )

                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                if ci_high < 0.0:
                    verdict = "improvement_supported"
                elif ci_low > 0.0:
                    verdict = "degradation_supported"
                else:
                    verdict = "uncertain"
                effect_rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "target": target,
                        "n_rows": int(len(paired)),
                        "n_fields": int(len(fields)),
                        "delta_rmse_db_augmented_minus_original": point_delta,
                        "field_block_bootstrap_ci95_low_db": float(ci_low),
                        "field_block_bootstrap_ci95_high_db": float(ci_high),
                        "bootstrap_probability_improved": float(np.mean(boot < 0.0)),
                        "fields_improved": int(np.sum(np.asarray(per_field_deltas) < 0.0)),
                        "verdict": verdict,
                    }
                )
    return pd.DataFrame(effect_rows), pd.DataFrame(field_rows)


def plot_rmse_comparison(ensemble_metrics: pd.DataFrame, output: Path) -> None:
    main = ensemble_metrics.loc[
        ensemble_metrics["cohort"].eq("main_gap_le_8d")
        & ensemble_metrics["model"].isin(["ridge", "mlp_tanh"])
        & ensemble_metrics["target"].isin(["HH", "VV", "joint_HH_VV"])
    ].copy()
    targets = ["HH", "VV", "joint_HH_VV"]
    target_labels = ["HH", "VV", "Joint HH/VV"]
    models = ["ridge", "mlp_tanh"]
    colors = {"original": "#6B7280", "original_plus_vwc": "#2563EB"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    legend_handles = None
    legend_labels = None
    for axis, model in zip(axes, models):
        x = np.arange(len(targets))
        width = 0.34
        for offset, input_set in [(-width / 2, "original"), (width / 2, "original_plus_vwc")]:
            values = []
            for target in targets:
                row = main.loc[
                    main["model"].eq(model)
                    & main["input_set"].eq(input_set)
                    & main["target"].eq(target),
                    "rmse_db",
                ]
                values.append(float(row.iloc[0]))
            label = "Original inputs" if input_set == "original" else "+ in-situ VWC"
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=colors[input_set],
                label=label,
            )
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        axis.set_xticks(x, target_labels)
        axis.set_title("Ridge" if model == "ridge" else "Small tanh MLP")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        legend_handles, legend_labels = axis.get_legend_handles_labels()
    axes[0].set_ylabel("Held-out-field RMSE (dB)")
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Incremental value of vegetation input on the same samples and folds")
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_effects(effects: pd.DataFrame, output: Path) -> None:
    selected = effects.loc[effects["target"].eq("joint_HH_VV")].copy()
    order = [
        ("main_gap_le_8d", "ridge"),
        ("main_gap_le_8d", "mlp_tanh"),
        ("sensitivity_gap_le_2d", "ridge"),
        ("sensitivity_gap_le_2d", "mlp_tanh"),
    ]
    labels = [
        "Main, Ridge",
        "Main, small MLP",
        "≤2-day sensitivity, Ridge",
        "≤2-day sensitivity, small MLP",
    ]
    y = np.arange(len(order))[::-1]
    points, low, high = [], [], []
    for cohort, model in order:
        row = selected.loc[
            selected["cohort"].eq(cohort) & selected["model"].eq(model)
        ].iloc[0]
        point = float(row["delta_rmse_db_augmented_minus_original"])
        points.append(point)
        low.append(point - float(row["field_block_bootstrap_ci95_low_db"]))
        high.append(float(row["field_block_bootstrap_ci95_high_db"]) - point)
    fig, axis = plt.subplots(figsize=(8.2, 4.2))
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.errorbar(
        points,
        y,
        xerr=np.vstack([low, high]),
        fmt="o",
        color="#2563EB",
        ecolor="#64748B",
        capsize=4,
    )
    axis.set_yticks(y, labels)
    axis.set_xlabel("ΔRMSE = (+VWC) − original (dB); negative is better")
    axis.set_title("Paired field-block bootstrap effect on joint HH/VV error")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_observed_vs_predicted(ensemble: pd.DataFrame, output: Path) -> None:
    subset = ensemble.loc[
        ensemble["cohort"].eq("main_gap_le_8d")
        & ensemble["model"].eq("mlp_tanh")
        & ensemble["input_set"].isin(INPUT_SETS)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.1), sharex=True, sharey=True)
    for row_i, input_set in enumerate(["original", "original_plus_vwc"]):
        current = subset.loc[subset["input_set"].eq(input_set)]
        for col_i, target in enumerate(["hh", "vv"]):
            axis = axes[row_i, col_i]
            observed = current[f"observed_{target}_db"].to_numpy(float)
            predicted = current[f"predicted_{target}_db"].to_numpy(float)
            low = min(observed.min(), predicted.min())
            high = max(observed.max(), predicted.max())
            axis.scatter(observed, predicted, s=18, alpha=0.65, color="#2563EB")
            axis.plot([low, high], [low, high], "--", color="#111827", linewidth=1)
            axis.text(
                0.04,
                0.95,
                f"RMSE = {rmse(observed, predicted):.2f} dB\nR² = {safe_r2(observed, predicted):.2f}",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
            )
            axis.grid(alpha=0.2)
            if row_i == 0:
                axis.set_title(target.upper())
            if col_i == 0:
                label = "Original inputs" if input_set == "original" else "+ in-situ VWC"
                axis.set_ylabel(f"{label}\nPredicted σ⁰ (dB)")
            if row_i == 1:
                axis.set_xlabel("Observed σ⁰ (dB)")
    fig.suptitle("Held-out-field small-MLP predictions")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fieldwise_effects(field_effects: pd.DataFrame, output: Path) -> None:
    selected = field_effects.loc[
        field_effects["cohort"].eq("main_gap_le_8d")
        & field_effects["model"].eq("mlp_tanh")
        & field_effects["target"].eq("joint_HH_VV")
    ].sort_values("delta_rmse_db_augmented_minus_original")
    colors = np.where(selected["improved_with_vwc"], "#2563EB", "#DC2626")
    fig, axis = plt.subplots(figsize=(10.2, 4.5))
    axis.bar(
        selected["field_id"].astype(str),
        selected["delta_rmse_db_augmented_minus_original"],
        color=colors,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Held-out field")
    axis.set_ylabel("ΔRMSE (dB); negative is better")
    axis.set_title("Field heterogeneity of the vegetation-input effect (small MLP)")
    axis.tick_params(axis="x", rotation=60)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_summary(
    cohorts: dict[str, pd.DataFrame],
    effects: pd.DataFrame,
    warning_table: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    main_joint = effects.loc[
        effects["cohort"].eq("main_gap_le_8d")
        & effects["target"].eq("joint_HH_VV")
    ]
    sensitivity_joint = effects.loc[
        effects["cohort"].eq("sensitivity_gap_le_2d")
        & effects["target"].eq("joint_HH_VV")
    ]
    model_results: dict[str, object] = {}
    for model in ["ridge", "mlp_tanh"]:
        main = main_joint.loc[main_joint["model"].eq(model)].iloc[0]
        sensitivity = sensitivity_joint.loc[
            sensitivity_joint["model"].eq(model)
        ].iloc[0]
        signs_agree = math.copysign(1, main["delta_rmse_db_augmented_minus_original"]) == math.copysign(
            1, sensitivity["delta_rmse_db_augmented_minus_original"]
        )
        model_results[model] = {
            "main_joint_delta_rmse_db": float(
                main["delta_rmse_db_augmented_minus_original"]
            ),
            "main_joint_ci95_db": [
                float(main["field_block_bootstrap_ci95_low_db"]),
                float(main["field_block_bootstrap_ci95_high_db"]),
            ],
            "main_verdict": str(main["verdict"]),
            "sensitivity_joint_delta_rmse_db": float(
                sensitivity["delta_rmse_db_augmented_minus_original"]
            ),
            "sensitivity_joint_ci95_db": [
                float(sensitivity["field_block_bootstrap_ci95_low_db"]),
                float(sensitivity["field_block_bootstrap_ci95_high_db"]),
            ],
            "sensitivity_verdict": str(sensitivity["verdict"]),
            "effect_direction_consistent_across_cohorts": bool(signs_agree),
        }
    return {
        "question": "Does in-situ VWC improve held-out-field HH/VV sigma0 prediction when only the input set changes?",
        "design": {
            "outer_split": f"{args.outer_folds}-fold GroupKFold by field_id",
            "same_samples_folds_models_and_budget": True,
            "scaling_fit_on_outer_training_fold_only": True,
            "original_features": ORIGINAL_FEATURES,
            "augmented_features": INPUT_SETS["original_plus_vwc"],
            "targets_db": TARGET_COLUMNS,
            "model_config": MODEL_CONFIG,
            "mlp_seeds": SEEDS,
            "field_block_bootstrap_repeats": args.bootstrap_repeats,
        },
        "cohorts": {
            name: {
                "rows": int(len(frame)),
                "fields": int(frame["field_id"].nunique()),
                "dates": int(frame["acquisition_date"].nunique()),
                "maximum_abs_vegetation_match_gap_days": float(
                    frame["vegetation_match_abs_gap_days"].max()
                ),
            }
            for name, frame in cohorts.items()
        },
        "paired_joint_results": model_results,
        "convergence_warning_fits": int(
            warning_table["convergence_warning"].sum()
        ),
        "interpretation_rule": "A negative delta RMSE favors adding VWC. A 95% field-block bootstrap interval wholly below zero supports improvement; an interval crossing zero is inconclusive.",
        "scope_limit": "Predictive ablation on one campaign; it is not a causal estimate of vegetation scattering and not yet the physics-pretraining/physics-constraint experiment.",
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    cohorts = {
        "main_gap_le_8d": prepare_cohort(args.main_input, "main_gap_le_8d"),
        "sensitivity_gap_le_2d": prepare_cohort(
            args.sensitivity_input, "sensitivity_gap_le_2d"
        ),
    }
    all_predictions, all_folds, all_warnings = [], [], []
    for cohort in cohorts.values():
        predictions, folds, warning_table = run_cohort(
            cohort, n_splits=args.outer_folds
        )
        all_predictions.append(predictions)
        all_folds.append(folds)
        all_warnings.append(warning_table)

    predictions = pd.concat(all_predictions, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    warning_table = pd.concat(all_warnings, ignore_index=True)
    metrics_by_repeat, ensemble, ensemble_metrics = calculate_metrics(predictions)
    paired_effects, field_effects = calculate_paired_effects(
        ensemble, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed
    )
    summary = create_summary(cohorts, paired_effects, warning_table, args)

    predictions.to_csv(output / "oof_predictions.csv", index=False)
    folds.to_csv(output / "outer_fold_assignments.csv", index=False)
    warning_table.to_csv(output / "fit_diagnostics.csv", index=False)
    metrics_by_repeat.to_csv(output / "metrics_by_repeat.csv", index=False)
    ensemble.to_csv(output / "ensemble_oof_predictions.csv", index=False)
    ensemble_metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    paired_effects.to_csv(output / "paired_vwc_effects.csv", index=False)
    field_effects.to_csv(output / "fieldwise_paired_effects.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_rmse_comparison(
        ensemble_metrics, figures / "01_main_rmse_comparison.png"
    )
    plot_paired_effects(
        paired_effects, figures / "02_paired_joint_effects.png"
    )
    plot_observed_vs_predicted(
        ensemble, figures / "03_main_mlp_observed_vs_predicted.png"
    )
    plot_fieldwise_effects(
        field_effects, figures / "04_main_mlp_fieldwise_effects.png"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved matched vegetation ablation to: {output}")


if __name__ == "__main__":
    main()
