"""Leakage-resistant grouped evaluation for direct and physics-residual models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUPPORTED_MODELS = ("rbf", "mlp")
SUPPORTED_STRATEGIES = ("direct", "residual")


@dataclass(frozen=True)
class NestedPredictionResult:
    predictions: np.ndarray
    fold_ids: np.ndarray
    tuning_records: list[dict[str, object]]


def estimator_and_grid(model_name: str, random_state: int):
    """Return a scaled estimator and a deliberately small, auditable grid."""
    normalized = model_name.strip().lower()
    if normalized == "rbf":
        estimator = KernelRidge(kernel="rbf")
        grid = {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": [0.03, 0.1, 0.3, 1.0],
        }
    elif normalized == "mlp":
        estimator = MLPRegressor(
            solver="lbfgs",
            max_iter=5000,
            tol=1e-3,
            random_state=int(random_state),
        )
        grid = {
            "model__hidden_layer_sizes": [(4,), (8,)],
            "model__alpha": [1.0],
        }
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {SUPPORTED_MODELS}")
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", estimator)]
    )
    return pipeline, grid


def nested_group_oof_predictions(
    features,
    observed,
    physical_prediction,
    groups,
    model_name: str,
    strategy: str,
    outer_splits: int = 5,
    inner_splits: int = 4,
    random_state: int = 42,
) -> NestedPredictionResult:
    """Produce outer group-held-out predictions with group-aware tuning.

    ``direct`` learns observed sigma0. ``residual`` learns
    observed - physical_prediction and adds the physical prediction back only
    after predicting the held-out fold.
    """
    features = np.asarray(features, dtype=float)
    observed = np.asarray(observed, dtype=float)
    physical_prediction = np.asarray(physical_prediction, dtype=float)
    groups = np.asarray(groups)
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {SUPPORTED_STRATEGIES}"
        )
    if not (
        len(features) == len(observed) == len(physical_prediction) == len(groups)
    ):
        raise ValueError("features, targets, physical predictions and groups differ in length")
    if not (
        np.isfinite(features).all()
        and np.isfinite(observed).all()
        and np.isfinite(physical_prediction).all()
    ):
        raise ValueError("Nested evaluation received non-finite values")

    unique_groups = np.unique(groups)
    outer_count = min(int(outer_splits), len(unique_groups))
    if outer_count < 2:
        raise ValueError("At least two groups are required for outer evaluation")

    oof = np.full(len(observed), np.nan, dtype=float)
    fold_ids = np.full(len(observed), -1, dtype=int)
    records: list[dict[str, object]] = []
    outer = GroupKFold(n_splits=outer_count)

    for fold, (train, test) in enumerate(
        outer.split(features, observed, groups), start=1
    ):
        train_groups = groups[train]
        inner_count = min(int(inner_splits), len(np.unique(train_groups)))
        if inner_count < 2:
            raise ValueError("At least two training groups are required for tuning")

        target_train = observed[train]
        if normalized_strategy == "residual":
            target_train = target_train - physical_prediction[train]

        estimator, parameter_grid = estimator_and_grid(
            model_name, random_state=random_state + fold
        )
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=inner_count),
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(features[train], target_train, groups=train_groups)
        model_prediction = search.predict(features[test])
        if normalized_strategy == "residual":
            model_prediction = physical_prediction[test] + model_prediction
        oof[test] = model_prediction
        fold_ids[test] = fold
        records.append(
            {
                "outer_fold": int(fold),
                "model": model_name,
                "strategy": normalized_strategy,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_fields": int(len(np.unique(groups[train]))),
                "test_fields": int(len(np.unique(groups[test]))),
                "inner_folds": int(inner_count),
                "inner_best_rmse_db": float(-search.best_score_),
                "best_parameters": str(search.best_params_),
            }
        )

    if np.isnan(oof).any() or np.any(fold_ids < 0):
        raise RuntimeError("Outer grouped evaluation did not predict every row")
    return NestedPredictionResult(oof, fold_ids, records)


def grouped_mean_and_offset_baselines(
    observed,
    physical_prediction,
    groups,
    outer_splits: int = 5,
):
    """Return training-mean and physical-plus-training-offset OOF baselines."""
    observed = np.asarray(observed, dtype=float)
    physical_prediction = np.asarray(physical_prediction, dtype=float)
    groups = np.asarray(groups)
    split_count = min(int(outer_splits), len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("At least two groups are required")
    mean_prediction = np.full(len(observed), np.nan, dtype=float)
    offset_prediction = np.full(len(observed), np.nan, dtype=float)
    fold_ids = np.full(len(observed), -1, dtype=int)
    offsets: list[float] = []
    splitter = GroupKFold(n_splits=split_count)
    dummy = np.zeros((len(observed), 1))
    for fold, (train, test) in enumerate(
        splitter.split(dummy, observed, groups), start=1
    ):
        mean_prediction[test] = float(np.mean(observed[train]))
        offset = float(
            np.mean(observed[train] - physical_prediction[train])
        )
        offset_prediction[test] = physical_prediction[test] + offset
        fold_ids[test] = fold
        offsets.append(offset)
    return mean_prediction, offset_prediction, fold_ids, offsets
