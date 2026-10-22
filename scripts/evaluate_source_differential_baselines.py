#!/usr/bin/env python3
"""E3 source-only empirical baselines for observed VV-HH backscatter.

Freeze the protocol before fitting. Target labels are opened only after the
ridge penalty has been selected by source-field validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
E1 = ROOT / "outputs/review_E1_source_optimization"
FEATURES = ["soil_moisture_m3_m3", "soil_real_dielectric",
            "pals_rms_height_cm", "pals_correlation_length_cm"]
KEY = ["field_id", "acquisition_date"]
ALPHAS = (0.1, 1.0, 10.0, 100.0)
BOOTSTRAPS = 5000
SOURCE_SHA = "7b52a128dfbdd49c6270bc80203227cb63daa7f7e312a1f40a50c6ecbe15e785"
TARGET_SHA = "e984a1cf6b39a35f28e3087b7f13f22e7aaf0de0a3a4b2e843b7dc00f44fb408"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs() -> dict[str, Path]:
    return {
        "source": REPRO / "data/source/smapvex12_portable_source.csv",
        "target": REPRO / "data/target/smex02_field_day_model_ready.csv",
        "common_138": REPRO / "results/stage_diagnostics/stage_predictions.csv",
        "e1_source_splits": E1 / "source_splits.csv",
        "e1_target_predictions": E1 / "target_ensemble_predictions.csv",
        "e1_manifest": E1 / "result_manifest.json",
    }


def read_observations(path: Path, expected_rows: int, expected_fields: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": str})
    needed = set(KEY + ["soil_moisture_m3_m3", "pals_rms_height_cm",
                        "pals_correlation_length_cm", "sigma0_hh_db",
                        "sigma0_vv_db", "model_ready"])
    if needed - set(frame):
        raise ValueError(f"Required observational columns are missing: {needed - set(frame)}")
    if not frame.model_ready.astype(str).str.lower().eq("true").all():
        raise ValueError("Unexpected non-ready observation")
    frame["field_id"] = frame.field_id.str.strip()
    frame["acquisition_date"] = pd.to_datetime(frame.acquisition_date).dt.strftime("%Y-%m-%d")
    moisture = frame.soil_moisture_m3_m3.to_numpy(float)
    frame["soil_real_dielectric"] = 3.03 + 9.3 * moisture + 146.0 * moisture**2 - 76.7 * moisture**3
    numeric = FEATURES + ["sigma0_hh_db", "sigma0_vv_db"]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite observation input or response")
    frame = frame.sort_values(KEY).reset_index(drop=True)
    if len(frame) != expected_rows or frame.field_id.nunique() != expected_fields or frame.duplicated(KEY).any():
        raise ValueError("Observed cohort differs from the recorded version")
    return frame


def source_folds(source: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    split = pd.read_csv(inputs()["e1_source_splits"], dtype={"field_id": str})
    if len(split) != len(source) or split[KEY].reset_index(drop=True).equals(source[KEY]) is False:
        raise ValueError("E1 source split keys do not match E3 source data")
    folds = list(GroupKFold(n_splits=4).split(np.arange(len(source)),
                                               groups=source.field_id.to_numpy()))
    assignment = [next(j + 1 for j, (_, valid) in enumerate(folds) if i in valid)
                  for i in range(len(source))]
    if split.validation_fold.tolist() != assignment:
        raise ValueError("E1 fold assignment changed")
    for train, valid in folds:
        if set(source.field_id.iloc[train]) & set(source.field_id.iloc[valid]):
            raise AssertionError("Field leakage")
    return folds, assignment


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    residual = pred - y
    yc = y - y.mean()
    pc = pred - pred.mean()
    mse = float(np.mean(residual**2))
    bias = float(np.mean(residual))
    yc_var = float(np.mean(yc**2))
    corr_denom = float(np.sqrt(yc_var * np.mean(pc**2)))
    centered_mse = float(np.mean((pc - yc)**2))
    answer = {
        "bias_db": bias,
        "rmse_db": float(np.sqrt(mse)),
        "centered_rmse_db": float(np.sqrt(centered_mse)),
        "centered_skill": float(1 - centered_mse / yc_var),
        "correlation": float(np.mean(yc * pc) / corr_denom) if corr_denom > 1e-15 else None,
    }
    if not np.isclose(mse, bias**2 + centered_mse, atol=1e-10):
        raise AssertionError("Bias and centered error decomposition failed")
    return answer


def freeze(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    paths = inputs()
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes["source"] != SOURCE_SHA or hashes["target"] != TARGET_SHA:
        raise ValueError("Source or target data changed")
    e1_manifest = json.loads(paths["e1_manifest"].read_text(encoding="utf-8"))
    if e1_manifest["status"] != "COMPLETE" or e1_manifest["target_labels_used_for_selection"]:
        raise ValueError("E1 comparison is not a completed source-selected result")
    source = read_observations(paths["source"], 240, 24)
    folds, assignment = source_folds(source)
    protocol = {
        "experiment": "E3_source_only_observational_differential_baselines",
        "status": "FROZEN_BEFORE_FITTING",
        "post_hoc": True,
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": hashes,
        "target": "Observed VV-HH backscatter difference in dB, not a moisture retrieval or temporal change.",
        "inference_inputs": FEATURES,
        "feature_policy": "Topp dielectric computed from observed soil moisture on both campaigns; no response, date, field ID, or target label is an input feature.",
        "baseline_candidates": ["source_training_mean", "ordinary_least_squares",
                                *[f"ridge_alpha_{a:g}" for a in ALPHAS]],
        "ridge_alphas": list(ALPHAS),
        "model_selection": "Choose ridge alpha minimizing pooled source out-of-field RMSE; exact ties choose larger alpha. OLS and source mean are fixed, separately reported baselines.",
        "scaling": "StandardScaler fitted on each source training fold only; final scaler fitted on 240 source records.",
        "folds": "Exact four field-disjoint folds saved by E1, six validation fields each.",
        "source_validation_sizes": [len(valid) for _, valid in folds],
        "target_scoring": "After source-only ridge selection, fit on all 240 source rows and score 189 target rows and historical 138 common rows once. Compare to E1 frozen source-selected and original neural predictions on identical rows.",
        "metrics": ["bias_db", "rmse_db", "centered_rmse_db", "centered_skill", "correlation"],
        "uncertainty": "5000 paired field bootstrap draws on each target cohort, seed 27261021; model fixed.",
        "selection_uses_target_labels": False,
        "no_experiment_changes": "No source data revision, target calibration, physical teacher retraining, or replacement of historical results.",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{name: importlib.metadata.version(name)
                           for name in ("numpy", "pandas", "scikit-learn")}},
        "limitations": [
            "This forward task does not support a backscatter-change-to-moisture-change baseline.",
            "The target campaign was examined previously; this is a post-hoc comparison.",
            "The original fixed-incidence model and these baselines do not use target angle as an input.",
            "Moisture is supplied at inference as in the original forward problem."
        ],
        "output_directory": str(out),
    }
    membership = source[KEY].copy()
    membership.insert(0, "row_index", np.arange(len(source)))
    membership["validation_fold"] = assignment
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    membership.to_csv(out / "source_splits.csv", index=False)
    print(json.dumps({"status": protocol["status"], "source_rows": len(source),
                      "source_fields": source.field_id.nunique(),
                      "fold_sizes": protocol["source_validation_sizes"]}), flush=True)


def fit_predict(kind: str, x_train: np.ndarray, y_train: np.ndarray,
                x_eval: np.ndarray) -> np.ndarray:
    if kind == "source_training_mean":
        return np.full(len(x_eval), float(y_train.mean()))
    if kind == "ordinary_least_squares":
        model = make_pipeline(StandardScaler(), LinearRegression())
    else:
        alpha = float(kind.split("ridge_alpha_", 1)[1])
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
    model.fit(x_train, y_train)
    prediction = model.predict(x_eval)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite predictions for {kind}")
    return prediction


def bootstrap_arrays(y: np.ndarray, predictions: dict[str, np.ndarray],
                     fields: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    unique, inverse = np.unique(fields, return_inverse=True)
    rng = np.random.default_rng(27261021)
    sampled = rng.integers(0, len(unique), size=(BOOTSTRAPS, len(unique)))
    counts = np.zeros((BOOTSTRAPS, len(unique)), dtype=np.int32)
    np.add.at(counts, (np.arange(BOOTSTRAPS)[:, None], sampled), 1)
    weights = counts[:, inverse].astype(float)
    totals = weights.sum(axis=1)
    y_mean = (weights @ y) / totals
    y_var = (weights @ (y**2)) / totals - y_mean**2
    if (y_var <= 0).any():
        raise ValueError("Bootstrap draw has zero observed variance")
    result = {}
    for name, pred in predictions.items():
        residual = pred - y
        bias = (weights @ residual) / totals
        mse = (weights @ (residual**2)) / totals
        result[name] = {
            "bias_db": bias,
            "rmse_db": np.sqrt(mse),
            "centered_skill": 1 - (mse - bias**2) / y_var,
        }
    return result


def target_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    target = read_observations(inputs()["target"], 189, 30)
    stage = pd.read_csv(inputs()["common_138"], dtype={"field_id": str})
    stage.acquisition_date = pd.to_datetime(stage.acquisition_date).dt.strftime("%Y-%m-%d")
    common_keys = set(map(tuple, stage[KEY].to_numpy()))
    if len(stage) != 138 or stage.field_id.nunique() != 22:
        raise ValueError("Historical common target cohort changed")
    common = np.array([tuple(row) in common_keys for row in target[KEY].to_numpy()])
    if int(common.sum()) != 138 or target.field_id[common].nunique() != 22:
        raise ValueError("Historical common cohort does not match target table")
    e1 = pd.read_csv(inputs()["e1_target_predictions"], dtype={"field_id": str})
    e1.acquisition_date = pd.to_datetime(e1.acquisition_date).dt.strftime("%Y-%m-%d")
    joined = target[KEY].merge(e1, on=KEY, how="left", validate="one_to_one", sort=False)
    y = target.sigma0_vv_db.to_numpy(float) - target.sigma0_hh_db.to_numpy(float)
    if len(e1) != len(target) or not np.array_equal(joined.common_138.to_numpy(bool), common):
        raise ValueError("E1 comparator cohort changed")
    if np.max(np.abs(joined.observed_diff_db.to_numpy(float) - y)) > 1e-10:
        raise ValueError("E1 comparator targets differ from current target table")
    return target, common, joined


def run(out: Path) -> None:
    started = time.monotonic()
    protocol_path = out / "protocol.json"
    if not protocol_path.is_file() or (out / "result_manifest.json").exists():
        raise ValueError("An unused frozen E3 protocol is required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_FITTING" or protocol["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Protocol or script differs from frozen version")
    if {name: sha(path) for name, path in inputs().items()} != protocol["inputs_sha256"]:
        raise ValueError("An input changed after E3 protocol freeze")
    source = read_observations(inputs()["source"], 240, 24)
    folds, assignment = source_folds(source)
    recorded = pd.read_csv(out / "source_splits.csv", dtype={"field_id": str})
    if recorded.validation_fold.tolist() != assignment:
        raise ValueError("Saved E3 fold assignment differs from protocol")
    x = source[FEATURES].to_numpy(float)
    y = (source.sigma0_vv_db - source.sigma0_hh_db).to_numpy(float)
    names = protocol["baseline_candidates"]
    predictions = {name: np.full(len(source), np.nan) for name in names}
    fold_rows = []
    for fold, (train, validation) in enumerate(folds, 1):
        for name in names:
            value = fit_predict(name, x[train], y[train], x[validation])
            predictions[name][validation] = value
            fold_rows.append({"fold": fold, "method": name, "n_train": len(train),
                              "n_validation": len(validation),
                              "train_fields": source.field_id.iloc[train].nunique(),
                              "validation_fields": source.field_id.iloc[validation].nunique(),
                              **metric(y[validation], value)})
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise AssertionError("Incomplete source validation predictions")
    pd.DataFrame(fold_rows).to_csv(out / "source_fold_metrics.csv", index=False)
    source_metrics = pd.DataFrame([
        {"method": name, "n": len(source), "fields": source.field_id.nunique(),
         **metric(y, predictions[name])}
        for name in names
    ])
    source_metrics.to_csv(out / "source_cv_metrics.csv", index=False)
    pd.DataFrame([
        {"field_id": source.at[i, "field_id"],
         "acquisition_date": source.at[i, "acquisition_date"],
         "validation_fold": assignment[i], "observed_diff_db": y[i],
         **{name: float(predictions[name][i]) for name in names}}
        for i in range(len(source))
    ]).to_csv(out / "source_cv_predictions.csv", index=False)
    ridge = source_metrics[source_metrics.method.str.startswith("ridge_alpha_")].copy()
    ridge["alpha"] = ridge.method.str.removeprefix("ridge_alpha_").astype(float)
    selected = ridge.sort_values(["rmse_db", "alpha"], ascending=[True, False]).iloc[0]
    ridge_name = str(selected.method)
    selection = {"selected_ridge": ridge_name, "alpha": float(selected.alpha),
                 "source_oof_rmse_db": float(selected.rmse_db),
                 "source_oof_centered_skill": float(selected.centered_skill),
                 "target_labels_opened_for_selection": False}
    (out / "source_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection), flush=True)

    # Target observations are read only after source_selection.json is saved.
    target, common, e1 = target_data()
    tx = target[FEATURES].to_numpy(float)
    ty = (target.sigma0_vv_db - target.sigma0_hh_db).to_numpy(float)
    methods = {
        "source_training_mean": fit_predict("source_training_mean", x, y, tx),
        "ordinary_least_squares": fit_predict("ordinary_least_squares", x, y, tx),
        "source_selected_ridge": fit_predict(ridge_name, x, y, tx),
        "E1_selected_neural": e1.selected_E1.to_numpy(float),
        "frozen_pretraining_neural": e1.pretraining.to_numpy(float),
        "frozen_source_inherited_neural": e1.source.to_numpy(float),
        "frozen_source_reset_neural": e1.source_reset_adam_003.to_numpy(float),
    }
    pd.DataFrame([
        {"field_id": target.at[i, "field_id"],
         "acquisition_date": target.at[i, "acquisition_date"],
         "common_138": bool(common[i]), "observed_diff_db": ty[i],
         **{name: float(value[i]) for name, value in methods.items()}}
        for i in range(len(target))
    ]).to_csv(out / "target_predictions.csv", index=False)
    metrics_rows, paired_rows = [], []
    pairs = [
        ("source_selected_ridge", "E1_selected_neural"),
        ("source_selected_ridge", "frozen_pretraining_neural"),
        ("source_selected_ridge", "frozen_source_reset_neural"),
        ("ordinary_least_squares", "E1_selected_neural"),
        ("source_training_mean", "E1_selected_neural"),
    ]
    for cohort, mask in (("full_189", np.ones(len(target), dtype=bool)),
                         ("common_138", common)):
        yy = ty[mask]
        pp = {name: value[mask] for name, value in methods.items()}
        fields = target.field_id.to_numpy(str)[mask]
        boot = bootstrap_arrays(yy, pp, fields)
        point = {name: metric(yy, value) for name, value in pp.items()}
        for name in methods:
            row = {"cohort": cohort, "method": name, "n": int(mask.sum()),
                   "fields": len(set(fields)), **point[name]}
            for quantity in ("bias_db", "rmse_db", "centered_skill"):
                low, high = np.quantile(boot[name][quantity], [.025, .975])
                row[f"{quantity}_ci_low"] = float(low)
                row[f"{quantity}_ci_high"] = float(high)
            metrics_rows.append(row)
        for first, second in pairs:
            row = {"cohort": cohort, "first": first, "second": second,
                   "n": int(mask.sum()), "fields": len(set(fields))}
            for quantity in ("rmse_db", "centered_skill"):
                values = boot[first][quantity] - boot[second][quantity]
                low, high = np.quantile(values, [.025, .975])
                row[f"{quantity}_first_minus_second"] = point[first][quantity] - point[second][quantity]
                row[f"{quantity}_delta_ci_low"] = float(low)
                row[f"{quantity}_delta_ci_high"] = float(high)
            paired_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(out / "target_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "target_paired_contrasts.csv", index=False)
    manifest = {
        "status": "COMPLETE",
        "experiment": protocol["experiment"],
        "source_only_selected_ridge": selection,
        "source_rows": len(source), "source_fields": source.field_id.nunique(),
        "target_rows": len(target), "target_fields": target.field_id.nunique(),
        "target_labels_used_for_selection": False,
        "post_hoc": True,
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {p.name: sha(p) for p in out.iterdir()
                          if p.is_file() and p.name != "result_manifest.json"},
    }
    (out / "result_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "selection": selection,
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--freeze-protocol", action="store_true")
    choice.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/review_E3_source_baselines")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.freeze_protocol:
        freeze(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
