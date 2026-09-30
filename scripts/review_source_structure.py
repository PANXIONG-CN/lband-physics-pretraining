"""Source-domain grouped validation and response-structure controls."""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_pilots.scattering.surfaces.teacher_contract import (
    validate_teacher_results,
)

DATA = ROOT / "reproducibility/data"

FEATURES = [
    "soil_moisture_m3_m3",
    "soil_real_dielectric",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
]
TARGETS = ["sigma0_hh_db", "sigma0_vv_db"]
VWC = "vegetation_water_content_in_situ_kg_m2"

# Row-vector transform: [HH, VV] @ ROT.
ROT = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)


def topp(m):
    return 3.03 + 9.3 * m + 146.0 * m**2 - 76.7 * m**3


def fingerprint(path):
    path = Path(path)
    return {
        "path": str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_real(path):
    frame = pd.read_csv(path, dtype={"field_id": str})
    frame["acquisition_date"] = pd.to_datetime(
        frame["acquisition_date"], errors="raise"
    )
    frame["soil_real_dielectric"] = topp(
        frame["soil_moisture_m3_m3"].to_numpy(float)
    )
    values = frame[FEATURES + TARGETS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite source inputs or responses.")
    if (frame[FEATURES[2:]] <= 0).any().any():
        raise ValueError("Roughness height and length must be positive.")
    if frame.duplicated(["field_id", "acquisition_date"]).any():
        raise ValueError("Duplicate field-date rows.")
    return frame.sort_values(
        ["field_id", "acquisition_date"]
    ).reset_index(drop=True)


def mean_channel_rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2, axis=0)).mean())


def components(y):
    return np.column_stack([
        y[:, 0],
        y[:, 1],
        y.mean(axis=1),
        y[:, 1] - y[:, 0],
    ])


def response_metrics(y, p):
    observed = components(y)
    predicted = components(p)
    rows = []
    for j, name in enumerate(["HH", "VV", "common", "differential"]):
        a, b = observed[:, j], predicted[:, j]
        err = b - a
        variance = np.var(a)
        centered_error = (b - b.mean()) - (a - a.mean())
        rows.append({
            "response": name,
            "rmse_db": np.sqrt(np.mean(err**2)),
            "bias_db": err.mean(),
            "centered_rmse_db": np.sqrt(np.mean(centered_error**2)),
            "centered_skill": (
                1.0 - np.mean(centered_error**2) / variance
                if variance > 1e-12 else np.nan
            ),
            "variance_ratio": (
                np.var(b) / variance if variance > 1e-12 else np.nan
            ),
        })
    return rows


def new_head(seed, epochs, lr):
    return MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="tanh",
        solver="adam",
        alpha=1e-3,
        batch_size="auto",
        learning_rate_init=lr,
        max_iter=epochs,
        warm_start=True,
        shuffle=True,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=epochs + 1,
        random_state=seed,
    )


def fit_heads(initial, x, z, epochs, seed, lr):
    models = []
    for j in range(2):
        model = (
            new_head(seed + j, epochs, lr)
            if initial is None else copy.deepcopy(initial[j])
        )
        model.set_params(
            max_iter=epochs,
            n_iter_no_change=epochs + 1,
            learning_rate_init=lr,
            random_state=seed + j,
        )
        # Fixed-epoch training. Selection occurs in grouped inner CV.
        # warm_start retains weights; fit restarts Adam at each stage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x, z[:, j])
        if not np.isfinite(model.loss_):
            raise FloatingPointError("Non-finite training loss.")
        models.append(model)
    return models


def predict_heads(models, x, rotation, mean, scale):
    z = np.column_stack([m.predict(x) for m in models])
    return (z @ rotation.T) * scale + mean


def fixed_common(prediction, training_y):
    p = prediction.copy()
    d = p[:, 1] - p[:, 0]
    c = training_y.mean(axis=1).mean()
    p[:, 0] = c - d / 2.0
    p[:, 1] = c + d / 2.0
    return p


def fit_ridge(x, y, groups, record_inner=None):
    candidates = [1e-4, 1e-2, 1.0, 100.0, 10000.0]
    splits = list(GroupKFold(4).split(x, groups=groups))
    losses = []
    for alpha in candidates:
        oof = np.full_like(y, np.nan)
        for inner_id, (tr, va) in enumerate(splits, start=1):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[tr], y[tr])
            oof[va] = model.predict(x[va])
            if record_inner is not None:
                record_inner(inner_id, va, oof[va], candidate_alpha=alpha)
        losses.append(mean_channel_rmse(y, oof))
    selected = candidates[int(np.argmin(losses))]
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected))
    model.fit(x, y)
    return model, selected


def grouped_assignments(frame):
    """Explicit held-out membership for the original deterministic group folds."""
    groups = frame["field_id"].to_numpy()
    outer = list(GroupKFold(5).split(frame, groups=groups))
    outer_rows, inner_rows = [], []
    for fold, (train, test) in enumerate(outer, start=1):
        part = frame.loc[test, ["field_id", "acquisition_date"]].copy()
        part["row_id"], part["fold"] = test, fold
        outer_rows.append(part)
        for inner_fold, (_, valid) in enumerate(
            GroupKFold(4).split(frame.iloc[train], groups=groups[train]), start=1
        ):
            part = frame.loc[train[valid], ["field_id", "acquisition_date"]].copy()
            part["row_id"] = train[valid]
            part["fold"], part["inner_fold"] = fold, inner_fold
            inner_rows.append(part)
    return outer, pd.concat(outer_rows, ignore_index=True), pd.concat(inner_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=DATA / "source/smapvex12_portable_source.csv",
    )
    parser.add_argument("--teacher-dir", type=Path, default=DATA / "teachers")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, nargs="+", default=[120, 300, 600])
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--classical-only", action="store_true")
    parser.add_argument("--with-vwc", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()

    checkpoints = sorted(set(args.epochs))
    if min(checkpoints) < 1 or args.repeats < 1:
        raise ValueError("Epochs and repeats must be positive.")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory.")

    frame = load_real(args.input)
    x = frame[FEATURES].to_numpy(float)
    y = frame[TARGETS].to_numpy(float)
    groups = frame["field_id"].to_numpy()
    if len(np.unique(groups)) < 6:
        raise ValueError("At least six fields are required.")

    x_vwc = None
    if args.with_vwc:
        x_vwc = frame[FEATURES + [VWC]].to_numpy(float)
        if not np.isfinite(x_vwc).all():
            raise ValueError("VWC comparison requires a fully matched cohort.")

    args.output.mkdir(parents=True)
    paths = [args.input]
    if not args.classical_only:
        spm_path = args.teacher_dir / "spm_pretraining.csv"
        req_path = args.teacher_dir / "i2em_train_requests.csv"
        res_path = args.teacher_dir / "i2em_train_results.csv"
        paths += [spm_path, req_path, res_path]

        spm = pd.read_csv(spm_path)
        i2em = validate_teacher_results(
            pd.read_csv(req_path), pd.read_csv(res_path)
        )
        xs = spm[FEATURES].to_numpy(float)
        ys = spm[["spm_hh_db", "spm_vv_db"]].to_numpy(float)
        xi = i2em[FEATURES].to_numpy(float)
        yi = i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(float)

        if not all(np.isfinite(v).all() for v in [xs, ys, xi, yi]):
            raise ValueError("Non-finite synthetic training data.")

        scaler = StandardScaler().fit(xs)
        xs, xi, xn = map(scaler.transform, [xs, xi, x])
        y_mean = ys.mean(axis=0)
        y_scale = float(np.sqrt(np.mean((ys - y_mean) ** 2)))
        if y_scale <= 0:
            raise ValueError("Synthetic target scale must be positive.")

    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [fingerprint(p) for p in paths],
        "script": fingerprint(__file__),
        "sklearn_version": sklearn.__version__,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "threadpools": [{k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()],
        "rows": len(frame),
        "fields": int(frame.field_id.nunique()),
        "dielectric": "topp_both",
        "outer_folds": 5,
        "inner_folds": 4,
        "epochs": checkpoints,
        "repeats": 1 if args.classical_only else args.repeats,
        "design": "post-hoc source-domain structural control",
        "centering": "one common scalar target scale; orthogonal rotation",
        "fixed_common_control": "same fitted differential, common replaced",
        "classical_only": args.classical_only,
        "with_vwc": args.with_vwc,
        "seeds": [20260917 + i * 10000 for i in range(1 if args.classical_only else args.repeats)],
        "ridge_alphas": [1e-4, 1e-2, 1.0, 100.0, 10000.0],
        "neural_settings": {"hidden_layer_sizes": [16, 8], "activation": "tanh",
                            "alpha": 0.001, "spm_epochs": args.spm_epochs,
                            "i2em_epochs": args.i2em_epochs,
                            "spm_learning_rate": 0.003, "other_learning_rate": 0.001,
                            "optimizer": "Adam reset at each fit; warm-start weights",
                            "early_stopping": False, "shuffle": True},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    predictions, tuning, diagnostics = [], [], []
    outer, outer_assignments, inner_assignments = grouped_assignments(frame)
    outer_assignments.to_csv(args.output / "outer_fold_assignments.csv", index=False)
    inner_assignments.to_csv(args.output / "inner_fold_assignments.csv", index=False)
    frame.to_csv(args.output / "analysis_input.csv", index=False)
    inner_path = args.output / "inner_oof_predictions.csv"
    repeats = 1 if args.classical_only else args.repeats

    for repeat in range(repeats):
        seed = 20260917 + repeat * 10000
        bank = {}
        if not args.classical_only:
            for coord, rotation in [("direct", np.eye(2)), ("components", ROT)]:
                zs = ((ys - y_mean) / y_scale) @ rotation
                zi = ((yi - y_mean) / y_scale) @ rotation
                spm_heads = fit_heads(
                    None, xs, zs, args.spm_epochs, seed, 3e-3
                )
                mf_heads = fit_heads(
                    spm_heads, xi, zi, args.i2em_epochs, seed + 100, 1e-3
                )
                bank[coord] = {
                    "scratch": None,
                    "spm": spm_heads,
                    "multifidelity": mf_heads,
                }

        for fold, (train, test) in enumerate(outer, start=1):
            def record_inner(method, inner_id, valid, predicted, **candidate):
                rows = train[valid]
                part = frame.loc[rows, ["field_id", "acquisition_date"]].copy()
                part["row_id"] = rows
                part["repeat"], part["fold"], part["inner_fold"] = repeat + 1, fold, inner_id
                part["method"] = method
                part["candidate_epochs"] = candidate.get("candidate_epochs", np.nan)
                part["candidate_alpha"] = candidate.get("candidate_alpha", np.nan)
                part["observed_hh"], part["observed_vv"] = y[rows, 0], y[rows, 1]
                part["predicted_hh"], part["predicted_vv"] = predicted[:, 0], predicted[:, 1]
                part.to_csv(inner_path, mode="a", header=not inner_path.exists(), index=False)

            fold_predictions = {
                "training_mean": np.tile(y[train].mean(axis=0), (len(test), 1))
            }
            for name, data in [("ridge", x), ("ridge_vwc", x_vwc)]:
                if data is None:
                    continue
                model, alpha = fit_ridge(
                    data[train], y[train], groups[train],
                    lambda inner_id, valid, p, **candidate: record_inner(name, inner_id, valid, p, **candidate),
                )
                fold_predictions[name] = model.predict(data[test])
                tuning.append({
                    "repeat": repeat + 1, "fold": fold,
                    "method": name, "selected_alpha": alpha,
                })

            if not args.classical_only:
                inner = list(GroupKFold(4).split(
                    xn[train], groups=groups[train]
                ))
                for coord, rotation in [
                    ("direct", np.eye(2)), ("components", ROT)
                ]:
                    z = ((y - y_mean) / y_scale) @ rotation
                    for init, initial in bank[coord].items():
                        method = f"{coord}_{init}"
                        scores = []
                        for epochs in checkpoints:
                            inner_pred = np.full_like(y[train], np.nan)
                            for inner_id, (a, b) in enumerate(inner):
                                model = fit_heads(
                                    initial, xn[train[a]], z[train[a]],
                                    epochs, seed + fold * 100 + inner_id, 1e-3,
                                )
                                inner_pred[b] = predict_heads(
                                    model, xn[train[b]], rotation, y_mean, y_scale
                                )
                                record_inner(method, inner_id + 1, b, inner_pred[b], candidate_epochs=epochs)
                            score = mean_channel_rmse(y[train], inner_pred)
                            scores.append(score)
                            tuning.append({
                                "repeat": repeat + 1, "fold": fold,
                                "method": method, "candidate_epochs": epochs,
                                "inner_rmse_db": score,
                            })

                        selected = checkpoints[int(np.argmin(scores))]
                        tuning.append({"repeat": repeat + 1, "fold": fold,
                                       "method": method, "selected_epochs": selected})
                        model = fit_heads(
                            initial, xn[train], z[train],
                            selected, seed + fold * 1000, 1e-3,
                        )
                        p = predict_heads(
                            model, xn[test], rotation, y_mean, y_scale
                        )
                        fold_predictions[method] = p
                        if coord == "components":
                            fold_predictions[f"fixed_common_{init}"] = (
                                fixed_common(p, y[train])
                            )

                        for head, fitted in enumerate(model):
                            curve = np.asarray(fitted.loss_curve_[-selected:])
                            diagnostics.append({
                                "repeat": repeat + 1, "fold": fold,
                                "method": method, "head": head,
                                "selected_epochs": selected,
                                "selected_maximum": selected == max(checkpoints),
                                "last_loss": float(curve[-1]),
                                "loss_first": float(curve[0]),
                                "parameter_count": sum(
                                    a.size for a in fitted.coefs_ + fitted.intercepts_
                                ),
                            })

            for method, p in fold_predictions.items():
                if not np.isfinite(p).all():
                    raise FloatingPointError(f"Invalid prediction: {method}")
                part = frame.loc[test, ["field_id", "acquisition_date"]].copy()
                part["row_id"] = test
                part["repeat"] = repeat + 1
                part["fold"] = fold
                part["method"] = method
                part["observed_hh"] = y[test, 0]
                part["observed_vv"] = y[test, 1]
                part["predicted_hh"] = p[:, 0]
                part["predicted_vv"] = p[:, 1]
                predictions.append(part)

            print(f"Completed repeat {repeat + 1}, fold {fold}", flush=True)
            # Checkpoints preserve completed folds if a later fit is interrupted.
            pd.concat(predictions, ignore_index=True).to_csv(args.output / "oof_predictions.csv", index=False)
            pd.DataFrame(tuning).to_csv(args.output / "inner_tuning.csv", index=False)
            pd.DataFrame(diagnostics).to_csv(args.output / "training_diagnostics.csv", index=False)

    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output / "inner_tuning.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        args.output / "training_diagnostics.csv", index=False
    )

    fold_rows, overall_rows = [], []
    for (rep, fold, method), part in pred.groupby(["repeat", "fold", "method"]):
        a = part[["observed_hh", "observed_vv"]].to_numpy()
        b = part[["predicted_hh", "predicted_vv"]].to_numpy()
        for record in response_metrics(a, b):
            fold_rows.append({
                "repeat": rep, "fold": fold, "method": method,
                "n": len(part), **record,
            })

    for (rep, method), part in pred.groupby(["repeat", "method"]):
        a = part[["observed_hh", "observed_vv"]].to_numpy()
        b = part[["predicted_hh", "predicted_vv"]].to_numpy()
        overall_rows.append({
            "repeat": rep, "method": method,
            "mean_channel_rmse_db": mean_channel_rmse(a, b),
        })

    fold_metrics = pd.DataFrame(fold_rows)
    overall = pd.DataFrame(overall_rows)
    fold_metrics.to_csv(args.output / "metrics_by_fold.csv", index=False)
    overall.to_csv(args.output / "metrics_by_repeat.csv", index=False)
    summary = overall.groupby("method").mean_channel_rmse_db.agg(["mean", "std"])
    summary.sort_values("mean").to_csv(args.output / "summary.csv")

    # Center within each held-out fold, not across unrelated fitted models.
    fold_metrics.groupby(["method", "response"])[
        ["centered_skill", "centered_rmse_db", "variance_ratio"]
    ].mean().to_csv(args.output / "centered_summary.csv")

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = time.monotonic() - started
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.sort_values("mean").round(5))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
