#!/usr/bin/env python3
"""Post-hoc E1 source optimization control. New outputs only.

Freeze a protocol first, then run it. Target labels are opened only after
source-grouped validation has selected one configuration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import audit_external_domain as audit
import evaluate_decoupled_physics_heads as base
import evaluate_multifidelity_pretraining as multi
from research_pilots.scattering.surrogate.physics_pretraining import train_epochs

REPRO = ROOT / "reproducibility"
TRAJ = REPRO / "results/trajectory_reconstruction"
SEEDS = (20260910, 20360910, 20460910, 20560910, 20660910)
RATES = (0.001, 0.003)
STATES = ("inherited", "reset")
EPOCHS = (30, 60, 120)
BOOTSTRAPS = 5000
KEY = ["field_id", "acquisition_date"]
EXPECTED_SOURCE = "7b52a128dfbdd49c6270bc80203227cb63daa7f7e312a1f40a50c6ecbe15e785"
EXPECTED_TARGET = "e984a1cf6b39a35f28e3087b7f13f22e7aaf0de0a3a4b2e843b7dc00f44fb408"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def inputs() -> dict[str, Path]:
    result = {
        "source": REPRO / "data/source/smapvex12_portable_source.csv",
        "target": REPRO / "data/target/smex02_field_day_model_ready.csv",
        "spm": REPRO / "data/teachers/spm_pretraining.csv",
        "i2em_requests": REPRO / "data/teachers/i2em_train_requests.csv",
        "i2em_results": REPRO / "data/teachers/i2em_train_results.csv",
        "common_138": REPRO / "results/stage_diagnostics/stage_predictions.csv",
    }
    result.update({f"member_{j}": TRAJ / f"rebuilt_member_{j}.npz" for j in range(1, 6)})
    return result


def source_data() -> pd.DataFrame:
    p = inputs()["source"]
    source = audit.prepare_table(p, "SMAPVEX12", "topp_both")
    source = source.loc[source.finite_model_row].sort_values(KEY).reset_index(drop=True)
    if len(source) != 240 or source.field_id.nunique() != 24 or source.duplicated(KEY).any():
        raise ValueError("Source cohort is not the recorded 240 rows / 24 fields.")
    source["field_id"] = source.field_id.astype(str)
    source["acquisition_date"] = source.acquisition_date.dt.strftime("%Y-%m-%d")
    return source


def make_splits(source: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = source.field_id.to_numpy()
    folds = list(GroupKFold(n_splits=4).split(np.arange(len(source)), groups=groups))
    for train, valid in folds:
        if set(groups[train]) & set(groups[valid]):
            raise AssertionError("Field leakage in grouped split.")
    if sorted(i for _, valid in folds for i in valid) != list(range(len(source))):
        raise AssertionError("Incomplete source validation coverage.")
    return folds


def freeze(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Output directory exists: {out}")
    source = source_data()
    files = inputs()
    hashes = {name: sha(path) for name, path in files.items()}
    if hashes["source"] != EXPECTED_SOURCE or hashes["target"] != EXPECTED_TARGET:
        raise ValueError("Source/target file version changed.")
    folds = make_splits(source)
    membership = [
        {"row_index": i, "field_id": row.field_id,
         "acquisition_date": row.acquisition_date,
         "validation_fold": next(j + 1 for j, (_, valid) in enumerate(folds) if i in valid)}
        for i, row in source.iterrows()
    ]
    protocol = {
        "experiment": "E1_same_initialization_source_optimization",
        "status": "FROZEN_BEFORE_TRAINING",
        "post_hoc": True,
        "main_result_policy": "Existing frozen manuscript results are not replaced.",
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": hashes,
        "source": {"rows": 240, "fields": 24, "partition": "four-fold GroupKFold by field_id",
                   "validation_rows_per_fold": [len(v) for _, v in folds],
                   "validation_fields_per_fold": [len(set(source.field_id.iloc[v])) for _, v in folds]},
        "starting_checkpoint": "Five rebuilt_member_1..5.npz pretraining arrays; reconstructed Adam state must match each checkpoint within 1e-10.",
        "reconstruction": "SPM 200 epochs followed by I2EM 100 epochs with seeds 20260910 + 100000*j, j=0..4; original fixed scalers.",
        "architecture": [4, 16, 8, 1],
        "activation": "tanh",
        "loss": "scikit-learn MLPRegressor squared error with alpha=0.001, trained on original standardized differential VV-HH",
        "optimizer": "Adam beta_1=0.9 beta_2=0.999 epsilon=1e-8; inherited versus zero moments and clock; both model and optimizer effective learning rates set to candidate value.",
        "learning_rates": list(RATES),
        "optimizer_states": list(STATES),
        "epochs": list(EPOCHS),
        "checkpoint_rule": "One continuous 120-epoch training path per fold, member, state, rate; record epochs 30/60/120.",
        "source_training_order_seed": "member seed + 10102, restarted identically for all arms and folds.",
        "randomness": {"member_seeds": list(SEEDS), "cv_folds": 4, "threads": 1, "bootstrap_seed": 27261020},
        "selection": "Minimum pooled out-of-field source validation RMSE (dB) of five-member ensemble across 12 candidates. Exact ties: fewer epochs, lower learning rate, inherited before reset.",
        "target": "Read target observations only after source selection. Score selected full-source refit once on full 189 and historical common 138 cohorts; compare frozen pretraining/inherited/reset weights. No target-based selection.",
        "metrics": ["bias_db", "rmse_db", "centered_rmse_db", "centered_skill", "correlation"],
        "uncertainty": "Fixed-model field bootstrap on target at 5000 draws; post-selection source uncertainty is descriptive.",
        "output_directory": str(out),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{p: importlib.metadata.version(p) for p in
                           ("numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl")}},
        "limitations": ["Exploratory post-hoc control; target cohort was used in earlier project work.",
                        "Source simulation range was informed by source covariates.",
                        "Only five initialization seeds and four source field folds."],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    pd.DataFrame(membership).to_csv(out / "source_splits.csv", index=False)
    print(json.dumps({"status": protocol["status"], "output": str(out),
                      "split_sizes": protocol["source"]["validation_rows_per_fold"]}))


def reconstruct(seed: int, spm_x: np.ndarray, spm_y: np.ndarray,
                i2em_x: np.ndarray, i2em_y: np.ndarray, checkpoint: Path):
    low = base.pretrain_single_output(spm_x, spm_y, 200, seed)
    model = base.fine_tune_single(low, i2em_x, i2em_y, 100, seed + 2)
    with np.load(checkpoint, allow_pickle=False) as saved:
        differences = [
            np.max(np.abs(arr - saved[f"pretraining_{kind}_{j}"]))
            for kind, arrays in (("coefs", model.coefs_), ("intercepts", model.intercepts_))
            for j, arr in enumerate(arrays)
        ]
        differences.extend([
            np.max(np.abs(low.feature_scaler.mean_ - saved["x_mean"])),
            np.max(np.abs(low.feature_scaler.scale_ - saved["x_scale"])),
            np.max(np.abs(low.target_scaler.mean_ - saved["y_mean"])),
            np.max(np.abs(low.target_scaler.scale_ - saved["y_scale"])),
        ])
    difference = float(max(differences))
    if difference > 1e-10 or model._optimizer.t != 4400:
        raise ValueError(f"Member {seed} pretraining reconstruction mismatch: {difference}")
    return model, difference


def candidate_model(start, state: str, rate: float):
    model = copy.deepcopy(start)
    model.learning_rate_init = float(rate)
    if state == "reset":
        model._optimizer = AdamOptimizer(model.coefs_ + model.intercepts_,
                                         learning_rate_init=float(rate),
                                         beta_1=model.beta_1, beta_2=model.beta_2,
                                         epsilon=model.epsilon)
        model.t_ = 0
        model.n_iter_ = 0
        model.loss_curve_ = []
    else:
        model._optimizer.learning_rate_init = float(rate)
    if model._optimizer.learning_rate_init != rate:
        raise AssertionError("Requested effective learning rate not installed.")
    return model


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    yc = y - y.mean()
    pc = pred - pred.mean()
    denominator = float(np.mean(yc**2))
    c_mse = float(np.mean((pc - yc)**2))
    corr_denom = float(np.sqrt(np.mean(yc**2) * np.mean(pc**2)))
    return {"bias_db": float(error.mean()),
            "rmse_db": float(np.sqrt(np.mean(error**2))),
            "centered_rmse_db": float(np.sqrt(c_mse)),
            "centered_skill": float(1 - c_mse / denominator),
            "correlation": float(np.mean(yc * pc) / corr_denom) if corr_denom > 0 else None}


def predict(model, x: np.ndarray, scales: dict) -> np.ndarray:
    xx = (x - scales["x_mean"]) / scales["x_scale"]
    return (model.predict(xx).reshape(-1) * scales["y_scale"] + scales["y_mean"].ravel()[0])


def checkpoint_predict(saved, stage: str, x: np.ndarray) -> np.ndarray:
    hidden = (x - saved["x_mean"]) / saved["x_scale"]
    for layer in (0, 1):
        hidden = np.tanh(hidden @ saved[f"{stage}_coefs_{layer}"] + saved[f"{stage}_intercepts_{layer}"])
    scaled = hidden @ saved[f"{stage}_coefs_2"] + saved[f"{stage}_intercepts_2"]
    return (scaled * saved["y_scale"] + saved["y_mean"]).ravel()


def target_data() -> tuple[pd.DataFrame, np.ndarray]:
    target = audit.prepare_table(inputs()["target"], "SMEX02", "topp_both")
    target = target.loc[target.finite_model_row].sort_values(KEY).reset_index(drop=True)
    target["field_id"] = target.field_id.astype(str)
    target["acquisition_date"] = target.acquisition_date.dt.strftime("%Y-%m-%d")
    if len(target) != 189 or target.field_id.nunique() != 30 or target.duplicated(KEY).any():
        raise ValueError("Target cohort differs from 189 rows / 30 fields.")
    stage = pd.read_csv(inputs()["common_138"], dtype={"field_id": str})
    stage.acquisition_date = pd.to_datetime(stage.acquisition_date).dt.strftime("%Y-%m-%d")
    common_keys = set(map(tuple, stage[KEY].to_numpy()))
    if len(stage) != 138 or stage.field_id.nunique() != 22 or not common_keys.issubset(set(map(tuple, target[KEY].to_numpy()))):
        raise ValueError("Historical common cohort changed.")
    common = np.array([tuple(row) in common_keys for row in target[KEY].to_numpy()])
    return target, common


def target_ci(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict:
    rng = np.random.default_rng(27261020)
    fields, inv = np.unique(groups, return_inverse=True)
    values = []
    for _ in range(BOOTSTRAPS):
        draw = rng.integers(0, len(fields), len(fields))
        counts = np.bincount(draw, minlength=len(fields))
        weights = counts[inv]
        yc = y - np.average(y, weights=weights)
        pc = pred - np.average(pred, weights=weights)
        denominator = np.average(yc**2, weights=weights)
        if denominator > 0:
            values.append(1 - np.average((pc - yc)**2, weights=weights) / denominator)
    low, high = np.quantile(values, [0.025, 0.975])
    return {"centered_skill_ci_low": float(low), "centered_skill_ci_high": float(high),
            "bootstrap_valid_draws": len(values)}


def run(out: Path) -> None:
    start_time = time.monotonic()
    protocol_path = out / "protocol.json"
    if not protocol_path.is_file() or (out / "result_manifest.json").exists():
        raise ValueError("A frozen, unused E1 protocol is required.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_TRAINING" or protocol["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Protocol or script changed after freeze.")
    if {name: sha(path) for name, path in inputs().items()} != protocol["inputs_sha256"]:
        raise ValueError("Input hashes changed after protocol freeze.")
    source = source_data()
    splits = make_splits(source)
    membership = pd.read_csv(out / "source_splits.csv", dtype={"field_id": str})
    expected = [
        next(j + 1 for j, (_, valid) in enumerate(splits) if i in valid)
        for i in range(len(source))
    ]
    if membership.validation_fold.tolist() != expected:
        raise ValueError("Frozen field folds changed.")
    sx = source[list(base.FEATURE_NAMES)].to_numpy(float)
    sy = (source.sigma0_vv_db - source.sigma0_hh_db).to_numpy(float)
    spm = pd.read_csv(inputs()["spm"])
    i2em = multi.load_paired(inputs()["i2em_requests"], inputs()["i2em_results"])
    spm_x = multi.features_from(spm)
    spm_y = base.to_components(spm[["spm_hh_db", "spm_vv_db"]].to_numpy(float))[:, 1]
    i2em_x = multi.features_from(i2em)
    i2em_y = base.to_components(i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(float))[:, 1]
    starts = []
    differences = []
    with threadpool_limits(limits=1):
        for member, seed in enumerate(SEEDS, 1):
            model, difference = reconstruct(seed, spm_x, spm_y, i2em_x, i2em_y,
                                            inputs()[f"member_{member}"])
            starts.append(model)
            differences.append(difference)
            print(f"Reconstructed member {member}/5; max checkpoint difference {difference:.3g}", flush=True)

        oof = {(state, rate, epoch): np.full((5, len(source)), np.nan)
               for state in STATES for rate in RATES for epoch in EPOCHS}
        curves = []
        for member, (seed, initial) in enumerate(zip(SEEDS, starts), 1):
            with np.load(inputs()[f"member_{member}"], allow_pickle=False) as saved:
                scales = {name: saved[name].copy() for name in ("x_mean", "x_scale", "y_mean", "y_scale")}
            for fold, (train, valid) in enumerate(splits, 1):
                xtrain = (sx[train] - scales["x_mean"]) / scales["x_scale"]
                ytrain = ((sy[train] - scales["y_mean"].ravel()[0]) / scales["y_scale"].ravel()[0])
                for state in STATES:
                    for rate in RATES:
                        model = candidate_model(initial, state, rate)
                        rng = np.random.default_rng(seed + 10102)
                        for epoch in range(1, max(EPOCHS) + 1):
                            order = rng.permutation(len(train))
                            model.partial_fit(xtrain[order], ytrain[order])
                            if not all(np.isfinite(arr).all() for arr in model.coefs_ + model.intercepts_):
                                raise ValueError(f"Nonfinite training state: member={member} fold={fold} {state} {rate} epoch={epoch}")
                            pred_train = predict(model, sx[train], scales)
                            pred_valid = predict(model, sx[valid], scales)
                            curves.append({
                                "member": member, "fold": fold, "optimizer_state": state,
                                "effective_learning_rate": rate, "epoch": epoch,
                                "train_rows": len(train), "validation_rows": len(valid),
                                "train_rmse_db": metric(sy[train], pred_train)["rmse_db"],
                                "validation_rmse_db": metric(sy[valid], pred_valid)["rmse_db"],
                                "train_scaled_objective": float(model.loss_curve_[-1]),
                                "optimizer_t": int(model._optimizer.t),
                            })
                            if epoch in EPOCHS:
                                oof[(state, rate, epoch)][member - 1, valid] = pred_valid
                        print(f"CV member {member}/5 fold {fold}/4 {state} lr={rate} complete", flush=True)

        if any(not np.isfinite(values).all() for values in oof.values()):
            raise AssertionError("Incomplete source out-of-fold predictions.")
        curve_path = out / "source_learning_curves.csv"
        pd.DataFrame(curves).to_csv(curve_path, index=False)
        rows, raw = [], []
        for state in STATES:
            for rate in RATES:
                for epoch in EPOCHS:
                    vals = oof[(state, rate, epoch)]
                    ensemble = vals.mean(axis=0)
                    row = {"optimizer_state": state, "effective_learning_rate": rate,
                           "epochs": epoch, "evaluation": "out_of_field_pooled",
                           "members": 5, "n": len(source), "fields": source.field_id.nunique(),
                           **metric(sy, ensemble)}
                    rows.append(row)
                    for member in range(5):
                        mm = metric(sy, vals[member])
                        rows.append({**row, "evaluation": f"member_{member+1}", **mm})
                        for i in range(len(source)):
                            raw.append({"field_id": source.at[i, "field_id"],
                                        "acquisition_date": source.at[i, "acquisition_date"],
                                        "fold": expected[i], "member": member + 1,
                                        "optimizer_state": state, "effective_learning_rate": rate,
                                        "epochs": epoch, "observed_diff_db": sy[i],
                                        "predicted_diff_db": vals[member, i]})
        score = pd.DataFrame(rows)
        score.to_csv(out / "source_validation_metrics.csv", index=False)
        pd.DataFrame(raw).to_csv(out / "source_validation_predictions.csv", index=False)
        candidates = score.loc[score.evaluation.eq("out_of_field_pooled")].copy()
        candidates["tie_state"] = candidates.optimizer_state.map({"inherited": 0, "reset": 1})
        candidates = candidates.sort_values(["rmse_db", "epochs", "effective_learning_rate", "tie_state"])
        selected = candidates.iloc[0]
        state = str(selected.optimizer_state)
        rate = float(selected.effective_learning_rate)
        endpoint = int(selected.epochs)
        decision = {
            "selected": {"optimizer_state": state, "effective_learning_rate": rate,
                         "epochs": endpoint, "source_oof_rmse_db": float(selected.rmse_db),
                         "source_oof_centered_skill": float(selected.centered_skill)},
            "selection_input": "source_validation_metrics.csv only; target labels not opened",
            "selection_rule": protocol["selection"],
        }
        (out / "source_selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        print(f"Source-selected: {state} lr={rate} epochs={endpoint}, OOF RMSE={selected.rmse_db:.6f}", flush=True)

        # The target table is deliberately opened only after source_selection.json exists.
        target, common = target_data()
        tx = target[list(base.FEATURE_NAMES)].to_numpy(float)
        ty = (target.sigma0_vv_db - target.sigma0_hh_db).to_numpy(float)
        selected_pred = np.empty((5, len(target)))
        selected_source_pred = np.empty((5, len(source)))
        baselines = {name: np.empty((5, len(target))) for name in
                     ("pretraining", "source", "source_reset_adam_003")}
        full_curves = []
        for member, (seed, initial) in enumerate(zip(SEEDS, starts), 1):
            with np.load(inputs()[f"member_{member}"], allow_pickle=False) as saved:
                scales = {name: saved[name].copy() for name in ("x_mean", "x_scale", "y_mean", "y_scale")}
                for stage in baselines:
                    baselines[stage][member - 1] = checkpoint_predict(saved, stage, tx)
            model = candidate_model(initial, state, rate)
            xx = (sx - scales["x_mean"]) / scales["x_scale"]
            yy = ((sy - scales["y_mean"].ravel()[0]) / scales["y_scale"].ravel()[0])
            rng = np.random.default_rng(seed + 10102)
            for epoch in range(1, endpoint + 1):
                order = rng.permutation(len(source))
                model.partial_fit(xx[order], yy[order])
                source_pred = predict(model, sx, scales)
                full_curves.append({"member": member, "epoch": epoch,
                                    "source_train_rmse_db": metric(sy, source_pred)["rmse_db"],
                                    "train_scaled_objective": float(model.loss_curve_[-1]),
                                    "optimizer_t": int(model._optimizer.t)})
            selected_source_pred[member - 1] = source_pred
            selected_pred[member - 1] = predict(model, tx, scales)
            if state == "inherited" and rate == 0.003 and endpoint == 120:
                with np.load(inputs()[f"member_{member}"], allow_pickle=False) as saved:
                    replay_diff = max(float(np.max(np.abs(arr - saved[f"source_{kind}_{j}"])))
                                      for kind, arrays in (("coefs", model.coefs_),
                                                           ("intercepts", model.intercepts_))
                                      for j, arr in enumerate(arrays))
                if replay_diff > 1e-8:
                    raise ValueError(f"Historical source replay mismatch: {replay_diff}")
            print(f"Full-source fit member {member}/5 complete", flush=True)
        pd.DataFrame(full_curves).to_csv(out / "selected_full_source_learning_curves.csv", index=False)

        stage = pd.read_csv(inputs()["common_138"], dtype={"field_id": str})
        stage.acquisition_date = pd.to_datetime(stage.acquisition_date).dt.strftime("%Y-%m-%d")
        merged = target[KEY].merge(stage[KEY + ["differential_pretraining_only_surrogate",
                                                "differential_source_finetuned_surrogate",
                                                "differential_source_reset_adam_003"]],
                                   on=KEY, how="left", validate="one_to_one", sort=False)
        expected_columns = {
            "pretraining": "differential_pretraining_only_surrogate",
            "source": "differential_source_finetuned_surrogate",
            "source_reset_adam_003": "differential_source_reset_adam_003",
        }
        for name, col in expected_columns.items():
            difference = np.max(np.abs(baselines[name].mean(axis=0)[common] -
                                       merged[col].to_numpy(float)[common]))
            if difference > 1e-8:
                raise ValueError(f"Frozen target baseline mismatch {name}: {difference}")
        all_predictions = {"selected_E1": selected_pred, **baselines}
        pred_rows, target_metrics = [], []
        for name, values in all_predictions.items():
            ensemble = values.mean(axis=0)
            for member in range(5):
                for i in range(len(target)):
                    pred_rows.append({"field_id": target.at[i, "field_id"],
                                      "acquisition_date": target.at[i, "acquisition_date"],
                                      "common_138": bool(common[i]), "method": name,
                                      "member": member + 1, "observed_diff_db": ty[i],
                                      "predicted_diff_db": values[member, i]})
            for cohort, mask in (("full_189", np.ones(len(target), dtype=bool)),
                                 ("common_138", common)):
                row = {"method": name, "cohort": cohort, "n": int(mask.sum()),
                       "fields": int(target.field_id[mask].nunique()),
                       **metric(ty[mask], ensemble[mask]),
                       **target_ci(ty[mask], ensemble[mask],
                                   target.field_id.to_numpy()[mask])}
                target_metrics.append(row)
        pd.DataFrame(pred_rows).to_csv(out / "target_predictions_by_member.csv", index=False)
        pd.DataFrame(target_metrics).to_csv(out / "target_metrics.csv", index=False)
        pd.DataFrame([
            {"field_id": target.at[i, "field_id"],
             "acquisition_date": target.at[i, "acquisition_date"],
             "common_138": bool(common[i]), "observed_diff_db": ty[i],
             **{name: float(values.mean(axis=0)[i]) for name, values in all_predictions.items()}}
            for i in range(len(target))
        ]).to_csv(out / "target_ensemble_predictions.csv", index=False)
        pd.DataFrame([
            {"field_id": source.at[i, "field_id"],
             "acquisition_date": source.at[i, "acquisition_date"],
             "observed_diff_db": sy[i],
             **{f"member_{j+1}": float(selected_source_pred[j, i]) for j in range(5)},
             "ensemble_prediction_db": float(selected_source_pred[:, i].mean())}
            for i in range(len(source))
        ]).to_csv(out / "selected_full_source_predictions.csv", index=False)
    manifest = {
        "status": "COMPLETE", "experiment": protocol["experiment"],
        "selected_source_only": decision["selected"],
        "max_pretraining_checkpoint_difference": max(differences),
        "source_oof_rows": len(source), "source_oof_fields": source.field_id.nunique(),
        "target_rows": len(target), "target_fields": target.field_id.nunique(),
        "target_labels_used_for_selection": False,
        "source_final_fit_is_in_sample": True,
        "post_hoc": True, "elapsed_seconds": time.monotonic() - start_time,
        "outputs_sha256": {p.name: sha(p) for p in out.iterdir()
                           if p.is_file() and p.name != "result_manifest.json"},
    }
    (out / "result_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "selected": decision["selected"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-protocol", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/review_E1_source_optimization")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.freeze_protocol:
        freeze(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
