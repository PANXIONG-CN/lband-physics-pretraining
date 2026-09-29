"""Evaluate fully decoupled common and differential response heads.

The experiment isolates the effect of physics transfer to the polarimetric
differential response.  Common-response predictions never share parameters or
gradients with the differential model.  All tuning is nested and grouped by
field_id.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from research_pilots.scattering.surfaces.metrics import regression_metrics
from research_pilots.scattering.surrogate.physics_pretraining import (
    FEATURE_NAMES,
    fine_tune_pretrained,
    generate_synthetic_spm_dataset,
    make_mlp,
    predict_physical_units,
    pretrain_physics_model,
    train_epochs,
)


OBSERVED_COLUMNS = ["sigma0_hh_db", "sigma0_vv_db"]
PHYSICAL_COLUMNS = [
    "exponential_spm_hh_raw_db",
    "exponential_spm_vv_raw_db",
]
METHODS = [
    "training_mean",
    "direct_pretrained",
    "mean_common_diff_scratch",
    "mean_common_diff_pretrained",
    "mean_common_diff_shrunk_pretrained",
    "mean_common_diff_constrained",
    "ridge_common_diff_scratch",
    "ridge_common_diff_pretrained",
    "ridge_common_diff_constrained",
]
DISPLAY_NAMES = {
    "training_mean": "Training mean",
    "direct_pretrained": "Direct pretrained",
    "mean_common_diff_scratch": "Mean common + diff scratch",
    "mean_common_diff_pretrained": "Mean common + diff pretrained",
    "mean_common_diff_shrunk_pretrained": "Mean common + risk-controlled diff",
    "mean_common_diff_constrained": "Mean common + constrained diff",
    "ridge_common_diff_scratch": "Ridge common + diff scratch",
    "ridge_common_diff_pretrained": "Ridge common + diff pretrained",
    "ridge_common_diff_constrained": "Ridge common + constrained diff",
}


@dataclass
class SingleOutputBundle:
    model: object
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    metadata: dict[str, object]


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
        raise ValueError("At least three fields are required")
    if selected.duplicated(["field_id", "acquisition_date"]).any():
        raise ValueError("Duplicate field-date rows are not allowed")
    return selected.sort_values(["field_id", "acquisition_date"]).reset_index(drop=True)


def to_components(hh_vv: np.ndarray) -> np.ndarray:
    values = np.asarray(hh_vv, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected [HH, VV]")
    return np.column_stack(
        [(values[:, 0] + values[:, 1]) / 2.0, values[:, 1] - values[:, 0]]
    )


def from_components(common: np.ndarray, differential: np.ndarray) -> np.ndarray:
    common = np.asarray(common, dtype=float).reshape(-1)
    differential = np.asarray(differential, dtype=float).reshape(-1)
    if len(common) != len(differential):
        raise ValueError("Common and differential lengths differ")
    return np.column_stack(
        [common - differential / 2.0, common + differential / 2.0]
    )


def verify_error_identity(
    observed_channels: np.ndarray, predicted_channels: np.ndarray
) -> None:
    channel_errors = predicted_channels - observed_channels
    observed_components = to_components(observed_channels)
    predicted_components = to_components(predicted_channels)
    component_errors = predicted_components - observed_components
    left = np.mean(channel_errors**2, axis=1)
    right = component_errors[:, 0] ** 2 + component_errors[:, 1] ** 2 / 4.0
    if not np.allclose(left, right, atol=1e-12, rtol=1e-12):
        raise AssertionError("HH/VV and common/differential error identity failed")


def pretrain_single_output(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    seed: int,
) -> SingleOutputBundle:
    targets = np.asarray(targets, dtype=float).reshape(-1, 1)
    feature_scaler = StandardScaler().fit(features)
    target_scaler = StandardScaler().fit(targets)
    scaled_features = feature_scaler.transform(features)
    scaled_targets = target_scaler.transform(targets).ravel()
    model = make_mlp(seed, learning_rate=3e-3)
    train_epochs(model, scaled_features, scaled_targets, epochs, seed)
    predicted = target_scaler.inverse_transform(
        np.asarray(model.predict(scaled_features)).reshape(-1, 1)
    ).ravel()
    return SingleOutputBundle(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        metadata={
            "pretrain_epochs": int(epochs),
            "synthetic_training_rmse_db": float(
                np.sqrt(np.mean((predicted - targets.ravel()) ** 2))
            ),
        },
    )


def fine_tune_single(
    bundle: SingleOutputBundle,
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    seed: int,
) -> object:
    model = copy.deepcopy(bundle.model)
    model.learning_rate_init = 1e-3
    scaled_features = bundle.feature_scaler.transform(features)
    scaled_targets = bundle.target_scaler.transform(
        np.asarray(targets).reshape(-1, 1)
    ).ravel()
    return train_epochs(model, scaled_features, scaled_targets, epochs, seed)


def scratch_single(
    bundle: SingleOutputBundle,
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    seed: int,
) -> object:
    model = make_mlp(seed, learning_rate=1e-3)
    scaled_features = bundle.feature_scaler.transform(features)
    scaled_targets = bundle.target_scaler.transform(
        np.asarray(targets).reshape(-1, 1)
    ).ravel()
    return train_epochs(model, scaled_features, scaled_targets, epochs, seed)


def predict_single(
    bundle: SingleOutputBundle, model: object, features: np.ndarray
) -> np.ndarray:
    scaled = np.asarray(model.predict(bundle.feature_scaler.transform(features))).reshape(-1, 1)
    return bundle.target_scaler.inverse_transform(scaled).ravel()


def constrained_differential_target(
    observed: np.ndarray,
    calibrated_physics: np.ndarray,
    weight: float,
) -> np.ndarray:
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0,1]")
    return (1.0 - weight) * np.asarray(observed) + weight * np.asarray(
        calibrated_physics
    )


def shrink_prediction_to_mean(
    predicted: np.ndarray, training_mean: float, weight: float
) -> np.ndarray:
    """Shrink a learned differential response toward the outer-training mean."""
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0,1]")
    predicted = np.asarray(predicted, dtype=float)
    return float(training_mean) + weight * (predicted - float(training_mean))


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(features), targets)
    return scaler, model


def tune_common_ridge_alpha(
    features: np.ndarray,
    common: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    alphas: list[float],
    inner_folds: int,
) -> tuple[float, list[dict[str, object]]]:
    local_groups = groups[outer_train]
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(local_groups))))
    scores = {alpha: [] for alpha in alphas}
    records: list[dict[str, object]] = []
    for inner_fold, (train_local, validation_local) in enumerate(
        splitter.split(outer_train, groups=local_groups), start=1
    ):
        train = outer_train[train_local]
        validation = outer_train[validation_local]
        for alpha in alphas:
            scaler, model = fit_ridge(features[train], common[train], alpha)
            predicted = model.predict(scaler.transform(features[validation]))
            score = float(np.sqrt(np.mean((predicted - common[validation]) ** 2)))
            scores[alpha].append(score)
            records.append(
                {
                    "family": "common_ridge",
                    "inner_fold": inner_fold,
                    "candidate": alpha,
                    "validation_rmse_db": score,
                }
            )
    means = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
    selected = min(means, key=lambda alpha: (means[alpha], alpha))
    for record in records:
        candidate = float(record["candidate"])
        record["mean_rmse_for_candidate_db"] = means[candidate]
        record["candidate_selected"] = bool(candidate == selected)
    return float(selected), records


def tune_differential_weight(
    bundle: SingleOutputBundle,
    features: np.ndarray,
    observed: np.ndarray,
    physical: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    weights: list[float],
    inner_folds: int,
    epochs: int,
    seed: int,
) -> tuple[float, list[dict[str, object]]]:
    local_groups = groups[outer_train]
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(local_groups))))
    scores = {weight: [] for weight in weights}
    records: list[dict[str, object]] = []
    for inner_fold, (train_local, validation_local) in enumerate(
        splitter.split(outer_train, groups=local_groups), start=1
    ):
        train = outer_train[train_local]
        validation = outer_train[validation_local]
        offset = float(np.mean(observed[train] - physical[train]))
        calibrated = physical[train] + offset
        training_seed = seed + inner_fold * 100
        for weight in weights:
            target = constrained_differential_target(
                observed[train], calibrated, weight
            )
            model = fine_tune_single(
                bundle,
                features[train],
                target,
                epochs=epochs,
                seed=training_seed,
            )
            predicted = predict_single(bundle, model, features[validation])
            score = float(
                np.sqrt(np.mean((predicted - observed[validation]) ** 2))
            )
            scores[weight].append(score)
            records.append(
                {
                    "family": "differential_constraint",
                    "inner_fold": inner_fold,
                    "candidate": weight,
                    "validation_rmse_db": score,
                }
            )
    means = {weight: float(np.mean(values)) for weight, values in scores.items()}
    selected = min(means, key=lambda weight: (means[weight], weight))
    for record in records:
        candidate = float(record["candidate"])
        record["mean_rmse_for_candidate_db"] = means[candidate]
        record["candidate_selected"] = bool(candidate == selected)
    return float(selected), records


def tune_differential_shrinkage(
    bundle: SingleOutputBundle,
    features: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    weights: list[float],
    inner_folds: int,
    epochs: int,
    seed: int,
) -> tuple[float, list[dict[str, object]]]:
    """Select transfer strength using only inner held-out fields."""
    local_groups = groups[outer_train]
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(local_groups))))
    scores = {weight: [] for weight in weights}
    records: list[dict[str, object]] = []
    for inner_fold, (train_local, validation_local) in enumerate(
        splitter.split(outer_train, groups=local_groups), start=1
    ):
        train = outer_train[train_local]
        validation = outer_train[validation_local]
        training_mean = float(np.mean(observed[train]))
        model = fine_tune_single(
            bundle,
            features[train],
            observed[train],
            epochs=epochs,
            seed=seed + inner_fold * 100,
        )
        raw_prediction = predict_single(bundle, model, features[validation])
        for weight in weights:
            predicted = shrink_prediction_to_mean(
                raw_prediction, training_mean, weight
            )
            score = float(
                np.sqrt(np.mean((predicted - observed[validation]) ** 2))
            )
            scores[weight].append(score)
            records.append(
                {
                    "family": "differential_shrinkage",
                    "inner_fold": inner_fold,
                    "candidate": weight,
                    "validation_rmse_db": score,
                }
            )
    means = {weight: float(np.mean(values)) for weight, values in scores.items()}
    selected = min(means, key=lambda weight: (means[weight], weight))
    for record in records:
        candidate = float(record["candidate"])
        record["mean_rmse_for_candidate_db"] = means[candidate]
        record["candidate_selected"] = bool(candidate == selected)
    return float(selected), records


def run_outer_evaluation(
    frame: pd.DataFrame,
    direct_bundle,
    differential_bundle: SingleOutputBundle,
    ridge_alphas: list[float],
    physics_weights: list[float],
    shrinkage_weights: list[float],
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
    common = observed_components[:, 0]
    differential = observed_components[:, 1]
    physical_differential = physical_components[:, 1]
    groups = frame["field_id"].astype(str).to_numpy()
    predictions = {
        method: np.full_like(observed_channels, np.nan) for method in METHODS
    }
    fold_ids = np.full(len(frame), -1, dtype=int)
    selected_ridge = np.full(len(frame), np.nan)
    selected_physics = np.full(len(frame), np.nan)
    selected_shrinkage = np.full(len(frame), np.nan)
    tuning_rows: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=min(outer_folds, len(np.unique(groups))))

    for outer_fold, (train, test) in enumerate(
        splitter.split(features, observed_channels, groups), start=1
    ):
        common_mean = float(np.mean(common[train]))
        differential_mean = float(np.mean(differential[train]))
        predictions["training_mean"][test] = from_components(
            np.full(len(test), common_mean),
            np.full(len(test), differential_mean),
        )

        direct_seed = seed + outer_fold * 10_000 + 3
        direct_model = fine_tune_pretrained(
            direct_bundle,
            features[train],
            observed_channels[train],
            epochs=epochs,
            seed=direct_seed,
        )
        predictions["direct_pretrained"][test] = predict_physical_units(
            direct_bundle, direct_model, features[test]
        )

        ridge_alpha, ridge_records = tune_common_ridge_alpha(
            features,
            common,
            groups,
            train,
            ridge_alphas,
            inner_folds,
        )
        common_scaler, common_model = fit_ridge(
            features[train], common[train], ridge_alpha
        )
        common_mean_prediction = np.full(len(test), common_mean)
        common_ridge_prediction = common_model.predict(
            common_scaler.transform(features[test])
        )

        differential_seed = seed + outer_fold * 10_000 + 11
        differential_scratch = scratch_single(
            differential_bundle,
            features[train],
            differential[train],
            epochs=epochs,
            seed=differential_seed,
        )
        scratch_prediction = predict_single(
            differential_bundle, differential_scratch, features[test]
        )
        differential_pretrained = fine_tune_single(
            differential_bundle,
            features[train],
            differential[train],
            epochs=epochs,
            seed=differential_seed,
        )
        pretrained_prediction = predict_single(
            differential_bundle, differential_pretrained, features[test]
        )
        shrinkage_weight, shrinkage_records = tune_differential_shrinkage(
            differential_bundle,
            features,
            differential,
            groups,
            train,
            shrinkage_weights,
            inner_folds,
            epochs,
            seed + outer_fold * 200_000,
        )
        shrunk_pretrained_prediction = shrink_prediction_to_mean(
            pretrained_prediction, differential_mean, shrinkage_weight
        )

        physics_weight, physics_records = tune_differential_weight(
            differential_bundle,
            features,
            differential,
            physical_differential,
            groups,
            train,
            physics_weights,
            inner_folds,
            epochs,
            seed + outer_fold * 100_000,
        )
        offset = float(
            np.mean(differential[train] - physical_differential[train])
        )
        constrained_target = constrained_differential_target(
            differential[train],
            physical_differential[train] + offset,
            physics_weight,
        )
        differential_constrained = fine_tune_single(
            differential_bundle,
            features[train],
            constrained_target,
            epochs=epochs,
            seed=differential_seed,
        )
        constrained_prediction = predict_single(
            differential_bundle, differential_constrained, features[test]
        )

        for common_name, common_prediction in [
            ("mean_common", common_mean_prediction),
            ("ridge_common", common_ridge_prediction),
        ]:
            differential_variants = [
                ("diff_scratch", scratch_prediction),
                ("diff_pretrained", pretrained_prediction),
                ("diff_constrained", constrained_prediction),
            ]
            if common_name == "mean_common":
                differential_variants.insert(
                    2, ("diff_shrunk_pretrained", shrunk_pretrained_prediction)
                )
            for suffix, differential_prediction in differential_variants:
                method = f"{common_name}_{suffix}"
                predictions[method][test] = from_components(
                    common_prediction, differential_prediction
                )

        # Decoupling assertion: common predictions must be bitwise identical
        # across differential-head variants within a common-head family.
        for common_name in ["mean_common", "ridge_common"]:
            suffixes = ["diff_scratch", "diff_pretrained", "diff_constrained"]
            if common_name == "mean_common":
                suffixes.insert(2, "diff_shrunk_pretrained")
            common_predictions = [
                to_components(predictions[f"{common_name}_{suffix}"][test])[:, 0]
                for suffix in suffixes
            ]
            if not all(
                np.allclose(
                    common_predictions[0],
                    values,
                    atol=1e-12,
                    rtol=1e-12,
                )
                for values in common_predictions[1:]
            ):
                raise AssertionError("Differential head leaked into common prediction")

        fold_ids[test] = outer_fold
        selected_ridge[test] = ridge_alpha
        selected_physics[test] = physics_weight
        selected_shrinkage[test] = shrinkage_weight
        for record in [*ridge_records, *physics_records, *shrinkage_records]:
            tuning_rows.append(
                {
                    "outer_fold": outer_fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_fields": len(np.unique(groups[train])),
                    "test_fields": len(np.unique(groups[test])),
                    "selected_ridge_alpha": ridge_alpha,
                    "selected_differential_weight": physics_weight,
                    "selected_shrinkage_weight": shrinkage_weight,
                    **record,
                }
            )

    output = frame.copy()
    output["outer_fold"] = fold_ids
    output["selected_ridge_alpha"] = selected_ridge
    output["selected_differential_weight"] = selected_physics
    output["selected_shrinkage_weight"] = selected_shrinkage
    output["observed_common_db"] = common
    output["observed_differential_db"] = differential
    output["spm_differential_db"] = physical_differential
    for method, values in predictions.items():
        verify_error_identity(observed_channels, values)
        output[f"hh_{method}"] = values[:, 0]
        output[f"vv_{method}"] = values[:, 1]
        components = to_components(values)
        output[f"common_{method}"] = components[:, 0]
        output[f"differential_{method}"] = components[:, 1]
    return output, pd.DataFrame(tuning_rows)


def evaluate_metrics(frame: pd.DataFrame, response_space: str) -> pd.DataFrame:
    if response_space == "channels":
        responses = [("HH", "sigma0_hh_db", "hh"), ("VV", "sigma0_vv_db", "vv")]
    elif response_space == "components":
        responses = [
            ("common", "observed_common_db", "common"),
            ("differential", "observed_differential_db", "differential"),
        ]
    else:
        raise ValueError("Unknown response space")
    rows: list[dict[str, object]] = []
    for response, observed_column, prefix in responses:
        observed = frame[observed_column].to_numpy(dtype=float)
        for method in METHODS:
            predicted = frame[f"{prefix}_{method}"].to_numpy(dtype=float)
            rows.append(
                {
                    "response": response,
                    "method": method,
                    **regression_metrics(observed, predicted),
                }
            )
    return pd.DataFrame(rows)


def joint_channel_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate HH and VV jointly without hiding a channel trade-off."""
    observed = frame[OBSERVED_COLUMNS].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        predicted = frame[[f"hh_{method}", f"vv_{method}"]].to_numpy(dtype=float)
        channel_rmse = np.sqrt(np.mean((predicted - observed) ** 2, axis=0))
        rows.append(
            {
                "method": method,
                "mean_channel_rmse_db": float(np.mean(channel_rmse)),
                "pooled_channel_rmse_db": float(
                    np.sqrt(np.mean((predicted - observed) ** 2))
                ),
                "hh_rmse_db": float(channel_rmse[0]),
                "vv_rmse_db": float(channel_rmse[1]),
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
    for column in [
        "selected_ridge_alpha",
        "selected_differential_weight",
        "selected_shrinkage_weight",
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
        first_rmse = np.sqrt(np.mean((first[indices] - observed[indices]) ** 2, axis=0))
        second_rmse = np.sqrt(np.mean((second[indices] - observed[indices]) ** 2, axis=0))
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


def grouped_bootstrap_joint_delta(
    observed: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Field-block bootstrap for mean(HH RMSE, VV RMSE)."""
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(seed)
    deltas = np.empty(iterations)
    for iteration in range(iterations):
        sampled = generator.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        first_rmse = np.sqrt(
            np.mean((first[indices] - observed[indices]) ** 2, axis=0)
        )
        second_rmse = np.sqrt(
            np.mean((second[indices] - observed[indices]) ** 2, axis=0)
        )
        deltas[iteration] = float(np.mean(first_rmse) - np.mean(second_rmse))
    first_point = np.sqrt(np.mean((first - observed) ** 2, axis=0))
    second_point = np.sqrt(np.mean((second - observed) ** 2, axis=0))
    point = float(np.mean(first_point) - np.mean(second_point))
    return {
        "mean_channel_rmse_delta_db": point,
        "ci_2_5_percent_db": float(np.quantile(deltas, 0.025)),
        "ci_97_5_percent_db": float(np.quantile(deltas, 0.975)),
        "probability_first_better": float(np.mean(deltas < 0)),
    }


def selected_frequency(
    tuning: pd.DataFrame, family: str
) -> list[dict[str, object]]:
    selected = tuning.loc[
        (tuning["family"] == family) & tuning["candidate_selected"]
    ].drop_duplicates(["repeat", "outer_fold"])
    return (
        selected["candidate"]
        .value_counts(normalize=True)
        .sort_index()
        .rename_axis("candidate")
        .reset_index(name="fraction")
        .to_dict(orient="records")
    )


def repeat_summary(metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (response, method), subset in metrics.groupby(["response", "method"], sort=False):
        rows.append(
            {
                "response": str(response),
                "method": str(method),
                "rmse_mean_db": float(subset.rmse_db.mean()),
                "rmse_std_db": float(subset.rmse_db.std(ddof=1)) if len(subset) > 1 else 0.0,
            }
        )
    return rows


def save_plots(
    frame: pd.DataFrame,
    channel_metrics: pd.DataFrame,
    component_metrics: pd.DataFrame,
    comparisons: dict[str, object],
    output_dir: Path,
) -> None:
    colors = plt.cm.Set2(np.linspace(0, 1, len(METHODS)))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        subset = channel_metrics[channel_metrics.response == response].set_index("method").loc[METHODS]
        ax.bar([DISPLAY_NAMES[m] for m in METHODS], subset.rmse_db, color=colors)
        ax.set(title=f"Decoupled unseen-field RMSE: {response}", ylabel="Grouped OOF RMSE (dB)")
        ax.tick_params(axis="x", rotation=46)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "01_decoupled_channel_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for ax, response in zip(axes, ["common", "differential"]):
        subset = component_metrics[component_metrics.response == response].set_index("method").loc[METHODS]
        ax.bar([DISPLAY_NAMES[m] for m in METHODS], subset.rmse_db, color=colors)
        ax.set(title=f"Decoupled component RMSE: {response}", ylabel="Grouped OOF RMSE (dB)")
        ax.tick_params(axis="x", rotation=46)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "02_decoupled_component_rmse.png", dpi=180)
    plt.close(fig)

    order = [
        "constraint_vs_pretrained",
        "pretraining_vs_scratch",
        "constraint_vs_training_mean",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        points, low, high = [], [], []
        for name in order:
            stats = comparisons["mean_common_channels"][name][response]
            points.append(stats["rmse_delta_db"])
            low.append(stats["rmse_delta_db"] - stats["ci_2_5_percent_db"])
            high.append(stats["ci_97_5_percent_db"] - stats["rmse_delta_db"])
        positions = np.arange(len(order))
        ax.errorbar(positions, points, yerr=[low, high], fmt="o", capsize=5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(positions, [value.replace("_", " ") for value in order], rotation=25)
        ax.set(title=f"Isolated differential-head effects: {response}", ylabel="First - second RMSE (dB; <0 favors first)")
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "03_isolated_differential_effects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    observed = frame["observed_differential_db"].to_numpy(dtype=float)
    for ax, method in zip(
        axes,
        ["mean_common_diff_pretrained", "mean_common_diff_constrained"],
    ):
        predicted = frame[f"differential_{method}"].to_numpy(dtype=float)
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        ax.scatter(observed, predicted, s=25, alpha=0.58, edgecolors="none")
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set(
            xlabel="Observed differential VV-HH (dB)",
            ylabel="Grouped OOF prediction (dB)",
            title=DISPLAY_NAMES[method],
            xlim=(lower, upper),
            ylim=(lower, upper),
        )
        ax.grid(alpha=0.22)
    fig.savefig(output_dir / "04_differential_observed_vs_predicted.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fully decoupled common/differential physics-transfer experiment")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--synthetic-samples", type=int, default=3000)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--fine-tune-epochs", type=int, default=120)
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100")
    parser.add_argument("--physics-weights", default="0,0.05,0.1,0.2,0.4,0.6")
    parser.add_argument("--shrinkage-weights", default="0,0.1,0.25,0.5,0.75,1")
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
    ridge_alphas = sorted(set(float(value) for value in args.ridge_alphas.split(",")))
    physics_weights = sorted(set(float(value) for value in args.physics_weights.split(",")))
    shrinkage_weights = sorted(
        set(float(value) for value in args.shrinkage_weights.split(","))
    )
    if any(alpha <= 0 for alpha in ridge_alphas):
        raise ValueError("Ridge alphas must be positive")
    if 0.0 not in physics_weights or any(weight < 0 or weight > 1 for weight in physics_weights):
        raise ValueError("Physics weights must lie in [0,1] and include zero")
    if 0.0 not in shrinkage_weights or 1.0 not in shrinkage_weights or any(
        weight < 0 or weight > 1 for weight in shrinkage_weights
    ):
        raise ValueError("Shrinkage weights must lie in [0,1] and include zero and one")

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    expected = [
        "summary.json",
        "channel_metrics.csv",
        "component_metrics.csv",
        "joint_channel_metrics.csv",
        "oof_predictions.csv",
    ]
    if any((output_dir / name).exists() for name in expected):
        raise FileExistsError("Output files already exist; choose a new directory")
    frame = load_table(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_features, synthetic_channels, synthetic_metadata = generate_synthetic_spm_dataset(
        args.synthetic_samples,
        args.frequency_ghz * 1e9,
        args.incidence_angle_deg,
        seed=args.seed,
    )
    synthetic_differential = to_components(synthetic_channels)[:, 1]

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
        differential_bundle = pretrain_single_output(
            synthetic_features,
            synthetic_differential,
            epochs=args.pretrain_epochs,
            seed=repeat_seed,
        )
        evaluated, tuning = run_outer_evaluation(
            frame,
            direct_bundle,
            differential_bundle,
            ridge_alphas,
            physics_weights,
            shrinkage_weights,
            args.outer_folds,
            args.inner_folds,
            args.fine_tune_epochs,
            repeat_seed,
        )
        evaluated["repeat"] = repeat
        tuning["repeat"] = repeat
        channel_metrics = evaluate_metrics(evaluated, "channels")
        channel_metrics.insert(0, "repeat", repeat)
        component_metrics = evaluate_metrics(evaluated, "components")
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
                "differential": differential_bundle.metadata,
            }
        )
        print(f"Completed repeat {repeat}/{args.repeats}", flush=True)

    ensemble = aggregate_predictions(evaluated_repeats)
    tuning = pd.concat(tuning_repeats, ignore_index=True)
    channel_metrics_by_repeat = pd.concat(channel_metric_repeats, ignore_index=True)
    component_metrics_by_repeat = pd.concat(component_metric_repeats, ignore_index=True)
    channel_metrics = evaluate_metrics(ensemble, "channels")
    component_metrics = evaluate_metrics(ensemble, "components")
    joint_metrics = joint_channel_metrics(ensemble)
    groups = ensemble["field_id"].astype(str).to_numpy()
    observed_channels = ensemble[OBSERVED_COLUMNS].to_numpy(dtype=float)
    observed_components = ensemble[["observed_common_db", "observed_differential_db"]].to_numpy(dtype=float)

    def channel_prediction(method: str) -> np.ndarray:
        return ensemble[[f"hh_{method}", f"vv_{method}"]].to_numpy(dtype=float)

    def component_prediction(method: str) -> np.ndarray:
        return ensemble[[f"common_{method}", f"differential_{method}"]].to_numpy(dtype=float)

    comparison_pairs = {
        "constraint_vs_pretrained": (
            "mean_common_diff_constrained",
            "mean_common_diff_pretrained",
        ),
        "pretraining_vs_scratch": (
            "mean_common_diff_pretrained",
            "mean_common_diff_scratch",
        ),
        "constraint_vs_training_mean": (
            "mean_common_diff_constrained",
            "training_mean",
        ),
        "risk_control_vs_training_mean": (
            "mean_common_diff_shrunk_pretrained",
            "training_mean",
        ),
        "risk_control_vs_pretrained": (
            "mean_common_diff_shrunk_pretrained",
            "mean_common_diff_pretrained",
        ),
        "risk_control_vs_direct_pretrained": (
            "mean_common_diff_shrunk_pretrained",
            "direct_pretrained",
        ),
        "pretraining_vs_training_mean": (
            "mean_common_diff_pretrained",
            "training_mean",
        ),
        "pretraining_vs_direct_pretrained": (
            "mean_common_diff_pretrained",
            "direct_pretrained",
        ),
        "constraint_vs_direct_pretrained": (
            "mean_common_diff_constrained",
            "direct_pretrained",
        ),
        "ridge_vs_mean_common_constrained": (
            "ridge_common_diff_constrained",
            "mean_common_diff_constrained",
        ),
        "ridge_constrained_vs_direct_pretrained": (
            "ridge_common_diff_constrained",
            "direct_pretrained",
        ),
    }
    comparisons = {
        "mean_common_channels": {},
        "components": {},
        "joint_channels": {},
    }
    for index, (name, (first, second)) in enumerate(comparison_pairs.items()):
        comparisons["mean_common_channels"][name] = grouped_bootstrap_delta(
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
        comparisons["joint_channels"][name] = grouped_bootstrap_joint_delta(
            observed_channels,
            channel_prediction(first),
            channel_prediction(second),
            groups,
            args.bootstrap_iterations,
            args.seed + 2000 + index * 100,
        )

    summary = {
        "research_question": "When common and differential responses are fully decoupled, does SPM pretraining and weak constraint improve the differential head and reconstructed unseen-field HH/VV predictions?",
        "input": fingerprint(input_path),
        "code": fingerprint(Path(__file__)),
        "output": str(output_dir),
        "samples": int(len(frame)),
        "fields": int(frame.field_id.nunique()),
        "dates": int(frame.acquisition_date.nunique()),
        "error_identity": "(e_HH^2 + e_VV^2)/2 = e_common^2 + e_differential^2/4",
        "protocol": {
            "outer_split": f"{args.outer_folds}-fold GroupKFold by field",
            "inner_split": f"up to {args.inner_folds}-fold GroupKFold by field",
            "repeats": args.repeats,
            "bootstrap": f"{args.bootstrap_iterations} field-block resamples",
            "common_heads": ["outer-training common mean", "inner-CV-tuned Ridge"],
            "differential_heads": ["scratch MLP", "SPM-pretrained MLP", "SPM-pretrained plus calibrated constraint"],
            "parameter_sharing_between_heads": False,
        },
        "ridge_alpha_candidates": ridge_alphas,
        "physics_weight_candidates": physics_weights,
        "shrinkage_weight_candidates": shrinkage_weights,
        "synthetic_domain": synthetic_metadata,
        "pretraining_by_repeat": pretraining_records,
        "selected_ridge_alpha_frequency": selected_frequency(tuning, "common_ridge"),
        "selected_differential_weight_frequency": selected_frequency(tuning, "differential_constraint"),
        "selected_shrinkage_weight_frequency": selected_frequency(
            tuning, "differential_shrinkage"
        ),
        "ensemble_channel_metrics": channel_metrics.to_dict(orient="records"),
        "ensemble_component_metrics": component_metrics.to_dict(orient="records"),
        "ensemble_joint_channel_metrics": joint_metrics.to_dict(orient="records"),
        "repeat_channel_metric_summary": repeat_summary(channel_metrics_by_repeat),
        "repeat_component_metric_summary": repeat_summary(component_metrics_by_repeat),
        "paired_field_bootstrap": comparisons,
        "decision_rule": "Physics transfer is supported only if constrained differential improves over matched pretrained-only and scratch differential heads with field-bootstrap intervals excluding zero, and reconstructed HH/VV do not significantly degrade relative to the training-mean baseline.",
        "scope_limit": "This is an unseen-field predictive evaluation within one campaign. SPM is a low-fidelity teacher; common/differential coordinates do not uniquely identify physical mechanisms.",
    }

    ensemble_output = ensemble.copy()
    ensemble_output["acquisition_date"] = ensemble_output.acquisition_date.dt.strftime("%Y-%m-%d")
    ensemble_output.to_csv(output_dir / "oof_predictions.csv", index=False)
    repeated_output = pd.concat(evaluated_repeats, ignore_index=True)
    repeated_output["acquisition_date"] = pd.to_datetime(repeated_output.acquisition_date).dt.strftime("%Y-%m-%d")
    repeated_output.to_csv(output_dir / "oof_predictions_by_repeat.csv", index=False)
    channel_metrics.to_csv(output_dir / "channel_metrics.csv", index=False)
    component_metrics.to_csv(output_dir / "component_metrics.csv", index=False)
    joint_metrics.to_csv(output_dir / "joint_channel_metrics.csv", index=False)
    channel_metrics_by_repeat.to_csv(output_dir / "channel_metrics_by_repeat.csv", index=False)
    component_metrics_by_repeat.to_csv(output_dir / "component_metrics_by_repeat.csv", index=False)
    tuning.to_csv(output_dir / "nested_tuning.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    save_plots(ensemble, channel_metrics, component_metrics, comparisons, output_dir)

    print("\nDecoupled physics-head experiment completed.")
    print(f"Samples: {len(frame)}; fields: {frame.field_id.nunique()}; dates: {frame.acquisition_date.nunique()}")
    print(channel_metrics[["response", "method", "rmse_db", "mae_db", "r_squared_skill"]].to_string(index=False))
    print("\nPrimary comparisons:")
    print(json.dumps(comparisons, indent=2, ensure_ascii=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
