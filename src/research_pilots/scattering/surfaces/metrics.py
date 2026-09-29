"""Evaluation helpers for physical scattering baselines."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    y = observed[valid]
    p = predicted[valid]
    if len(y) == 0:
        raise ValueError("No finite observation-prediction pairs")
    residual = p - y
    denominator = np.sum((y - np.mean(y)) ** 2)
    correlation = (
        float(pearsonr(y, p).statistic)
        if len(y) >= 3 and np.ptp(y) > 0.0 and np.ptp(p) > 0.0
        else None
    )
    return {
        "n": int(len(y)),
        "rmse_db": float(np.sqrt(np.mean(residual**2))),
        "mae_db": float(np.mean(np.abs(residual))),
        "bias_pred_minus_obs_db": float(np.mean(residual)),
        "pearson_r": correlation,
        "r_squared_skill": float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0.0
        else None,
    }


def grouped_oof_bias_calibration(
    observed,
    predicted,
    groups,
    n_splits: int = 5,
):
    """Estimate only an additive dB offset using group-held-out folds."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    groups = np.asarray(groups)
    if len(observed) != len(predicted) or len(observed) != len(groups):
        raise ValueError("observed, predicted, and groups must have equal length")
    unique_groups = np.unique(groups)
    folds = min(int(n_splits), len(unique_groups))
    if folds < 2:
        raise ValueError("At least two independent groups are required")

    calibrated = np.full(len(observed), np.nan, dtype=float)
    offsets: list[float] = []
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros((len(observed), 1))
    for train, test in splitter.split(dummy, observed, groups):
        offset = float(np.mean(observed[train] - predicted[train]))
        calibrated[test] = predicted[test] + offset
        offsets.append(offset)
    return calibrated, offsets
