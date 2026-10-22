#!/usr/bin/env python3
"""E4 frozen-weight target intercept control on archived whole-field splits.

Two scalar offsets are fitted on adaptation fields: common response and
VV-HH differential. Network weights, scalers, and source mean remain fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULT = REPRO / "results"
KEY = ["field_id", "acquisition_date"]
FRACTIONS = (0.05, 0.1, 0.2)
ADAPT_FIELDS = {0.05: 2, 0.1: 3, 0.2: 6}
STAGES = ("pretraining", "source")
METHODS = ("pretraining_no_target", "pretraining_two_intercepts",
           "source_no_target", "source_two_intercepts")
RESPONSES = ("HH", "VV", "common", "differential")
SOURCE_SHA = "7b52a128dfbdd49c6270bc80203227cb63daa7f7e312a1f40a50c6ecbe15e785"
TARGET_SHA = "e984a1cf6b39a35f28e3087b7f13f22e7aaf0de0a3a4b2e843b7dc00f44fb408"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs() -> dict[str, Path]:
    answer = {
        "source": REPRO / "data/source/smapvex12_portable_source.csv",
        "target": REPRO / "data/target/smex02_field_day_model_ready.csv",
        "field_assignments": RESULT / "cross_domain/field_split_assignments.csv",
        "historical_metrics": RESULT / "cross_domain/few_shot_metrics_by_split.csv",
        "historical_summary": RESULT / "cross_domain/few_shot_summary.json",
        "frozen_zero_shot": RESULT / "cross_domain/zero_shot_predictions.csv",
    }
    answer.update({f"member_{j}": RESULT / f"trajectory_reconstruction/rebuilt_member_{j}.npz"
                   for j in range(1, 6)})
    return answer


def read_groups(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": str})
    needed = {"split_repeat", "requested_fraction", "actual_fraction", "field_id", "role"}
    if needed - set(frame):
        raise ValueError(f"Missing assignment columns: {needed - set(frame)}")
    frame["field_id"] = frame.field_id.str.strip()
    if len(frame) != 1800 or frame.duplicated(["split_repeat", "requested_fraction", "field_id"]).any():
        raise ValueError("Historical assignment table is not the expected 20 x 3 x 30 design")
    if set(frame.split_repeat) != set(range(1, 21)) or set(frame.requested_fraction) != set(FRACTIONS):
        raise ValueError("Historical split repeats or fractions changed")
    for (split, fraction), group in frame.groupby(["split_repeat", "requested_fraction"]):
        if len(group) != 30 or group.field_id.nunique() != 30 or set(group.role) != {"adaptation", "test"}:
            raise ValueError(f"Invalid assignment repeat {split}, fraction {fraction}")
        if group.role.eq("adaptation").sum() != ADAPT_FIELDS[float(fraction)]:
            raise ValueError("Unexpected number of whole adaptation fields")
        expected_fraction = ADAPT_FIELDS[float(fraction)] / 30
        if not np.allclose(group.actual_fraction, expected_fraction, atol=1e-12):
            raise ValueError("Actual adaptation fraction differs from field count")
    return frame


def metadata_table(path: Path, expected_rows: int, expected_fields: int) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=KEY + ["model_ready"], dtype={"field_id": str})
    frame["field_id"] = frame.field_id.str.strip()
    frame["acquisition_date"] = pd.to_datetime(frame.acquisition_date).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(KEY).reset_index(drop=True)
    if (len(frame) != expected_rows or frame.field_id.nunique() != expected_fields
            or frame.duplicated(KEY).any()
            or not frame.model_ready.astype(str).str.lower().eq("true").all()):
        raise ValueError("Source or target cohort changed")
    return frame


def observed_table(path: Path, expected_rows: int, expected_fields: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": str})
    needed = set(KEY + ["model_ready", "soil_moisture_m3_m3", "pals_rms_height_cm",
                        "pals_correlation_length_cm", "sigma0_hh_db", "sigma0_vv_db"])
    if needed - set(frame):
        raise ValueError(f"Missing observational columns: {needed - set(frame)}")
    frame["field_id"] = frame.field_id.str.strip()
    frame["acquisition_date"] = pd.to_datetime(frame.acquisition_date).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(KEY).reset_index(drop=True)
    if (len(frame) != expected_rows or frame.field_id.nunique() != expected_fields
            or frame.duplicated(KEY).any()
            or not frame.model_ready.astype(str).str.lower().eq("true").all()):
        raise ValueError("Observation cohort changed")
    moisture = frame.soil_moisture_m3_m3.to_numpy(float)
    frame["soil_real_dielectric_topp"] = (
        3.03 + 9.3 * moisture + 146.0 * moisture**2 - 76.7 * moisture**3
    )
    numeric = ["soil_moisture_m3_m3", "soil_real_dielectric_topp",
               "pals_rms_height_cm", "pals_correlation_length_cm",
               "sigma0_hh_db", "sigma0_vv_db"]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("Nonfinite input or observation")
    return frame


def source_common(frame: pd.DataFrame) -> float:
    channels = frame[["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(float)
    return float(channels.mean())


def features(frame: pd.DataFrame) -> np.ndarray:
    return frame[["soil_moisture_m3_m3", "soil_real_dielectric_topp",
                  "pals_rms_height_cm", "pals_correlation_length_cm"]].to_numpy(float)


def differential(checkpoint, stage: str, x: np.ndarray) -> np.ndarray:
    hidden = (x - checkpoint["x_mean"]) / checkpoint["x_scale"]
    for layer in (0, 1):
        hidden = np.tanh(hidden @ checkpoint[f"{stage}_coefs_{layer}"]
                         + checkpoint[f"{stage}_intercepts_{layer}"])
    scaled = hidden @ checkpoint[f"{stage}_coefs_2"] + checkpoint[f"{stage}_intercepts_2"]
    return (scaled * checkpoint["y_scale"] + checkpoint["y_mean"]).ravel()


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - y
    bias = float(error.mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    yc = y - y.mean()
    pc = pred - pred.mean()
    variance = float(np.mean(yc**2))
    centered_mse = float(np.mean((pc - yc)**2))
    if not np.isclose(rmse**2, bias**2 + centered_mse, atol=1e-10):
        raise AssertionError("Error decomposition failed")
    return {
        "bias_db": bias, "rmse_db": rmse,
        "centered_rmse_db": float(np.sqrt(centered_mse)),
        "centered_skill": float(1 - centered_mse / variance),
        "observed_variance_db2": variance,
    }


def response_arrays(hh: np.ndarray, vv: np.ndarray) -> dict[str, np.ndarray]:
    return {"HH": hh, "VV": vv, "common": (hh + vv) / 2,
            "differential": vv - hh}


def check_historical_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keys = ["split_repeat", "requested_fraction", "method", "response"]
    if len(frame) != 1920 or frame.duplicated(keys).any():
        raise ValueError("Historical few-shot metric table changed")
    if set(frame.split_repeat) != set(range(1, 21)) or set(frame.requested_fraction) != set(FRACTIONS):
        raise ValueError("Historical metric split plan changed")
    if "spm_to_i2em" not in set(frame.method) or set(RESPONSES) - set(frame.response):
        raise ValueError("Historical fine-tuning comparator absent")
    return frame


def freeze(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Output directory exists: {out}")
    paths = inputs()
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes["source"] != SOURCE_SHA or hashes["target"] != TARGET_SHA:
        raise ValueError("Observational data version changed")
    source = metadata_table(paths["source"], 240, 24)
    target = metadata_table(paths["target"], 189, 30)
    groups = read_groups(paths["field_assignments"])
    old = check_historical_metrics(paths["historical_metrics"])
    if set(groups.field_id) != set(target.field_id):
        raise ValueError("Target fields differ from historical splits")
    summary = json.loads(paths["historical_summary"].read_text(encoding="utf-8"))
    if summary["status"] != "PREREGISTERED_FEW_SHOT_COMPLETE":
        raise ValueError("Historical full adaptation is not complete")
    split_rows = []
    for (split, fraction), group in groups.groupby(["split_repeat", "requested_fraction"], sort=True):
        adapt_fields = set(group.loc[group.role.eq("adaptation"), "field_id"])
        test_fields = set(group.loc[group.role.eq("test"), "field_id"])
        adapt = target.field_id.isin(adapt_fields)
        test = target.field_id.isin(test_fields)
        if adapt.sum() + test.sum() != len(target) or adapt.sum() == 0 or test.sum() == 0:
            raise ValueError("Incomplete target split")
        old_count = old.loc[(old.split_repeat.eq(split))
                            & (old.requested_fraction.eq(fraction))
                            & (old.method.eq("spm_to_i2em"))
                            & (old.response.eq("differential")), "n"]
        if len(old_count) != 1 or int(old_count.iloc[0]) != int(test.sum()):
            raise ValueError("Historical adaptation test row count differs")
        split_rows.append({
            "split_repeat": int(split), "requested_fraction": float(fraction),
            "adaptation_fields": len(adapt_fields), "test_fields": len(test_fields),
            "adaptation_rows": int(adapt.sum()), "test_rows": int(test.sum()),
            "adaptation_dates": int(target.acquisition_date[adapt].nunique()),
            "test_dates": int(target.acquisition_date[test].nunique()),
            "shared_date_count": len(set(target.acquisition_date[adapt])
                                     & set(target.acquisition_date[test])),
        })
    protocol = {
        "experiment": "E4_frozen_weights_target_two_intercepts",
        "status": "FROZEN_BEFORE_CALIBRATION",
        "post_hoc": True,
        "script_sha256": sha(Path(__file__)),
        "inputs_sha256": hashes,
        "source": {"rows": len(source), "fields": 24,
                   "role": "fixed source common response and historical weight training only"},
        "target": {"rows": len(target), "fields": 30,
                   "task": "HH and VV backscatter, common response and VV-HH differential in dB"},
        "frozen_model_stages": list(STAGES),
        "member_checkpoints": [f"rebuilt_member_{j}.npz" for j in range(1, 6)],
        "adaptation": "For each checkpoint, fit two scalars using adaptation rows only: mean target common minus fixed source common, and mean observed differential minus frozen predicted differential. No gradient updates, source refit, or model selection.",
        "aggregation": "Offsets are row-weighted means, matching the historical common-component adaptation rule; five member predictions are averaged.",
        "field_split": "Reuse 20 archived repeats at 2, 3, and 6 whole adaptation fields; score all rows from untouched fields.",
        "budgets_fields": [2, 3, 6],
        "metrics": ["bias_db", "rmse_db", "centered_rmse_db", "centered_skill"],
        "historical_comparator": "Archived spm_to_i2em 30-epoch differential-head target fine-tuning with the same source start and a target common mean. Only per-split archived metrics are available.",
        "comparison_rule": "Pair by split_repeat, fraction, and response; describe mean, median, range and proportion of split scores favoring two intercepts. Repeated splits overlap, so do not treat 20 repeats as independent tests.",
        "frozen_zero_shot_check": "Numerically compare both no-target ensembles with existing 189-row frozen zero-shot predictions before estimating any target offsets.",
        "target_test_rule": "Adaptation and test fields are disjoint in each split. Target test labels are used only for scoring after offsets are computed from adaptation rows.",
        "date_scope": "Target dates may be shared between adaptation and test fields; this is a field-holdout result, not a new-date result.",
        "historical_results_policy": "No old metric, checkpoint, source, target, or manuscript file is replaced.",
        "output_directory": str(out),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    pd.DataFrame(split_rows).to_csv(out / "split_integrity.csv", index=False)
    print(json.dumps({"status": protocol["status"], "splits": len(split_rows),
                      "adaptation_budgets_fields": protocol["budgets_fields"]}), flush=True)


def run(out: Path) -> None:
    started = time.monotonic()
    protocol_path = out / "protocol.json"
    if not protocol_path.is_file() or (out / "result_manifest.json").exists():
        raise ValueError("An unused frozen E4 protocol is required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_CALIBRATION" or protocol["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Protocol or script changed after freeze")
    paths = inputs()
    if {name: sha(path) for name, path in paths.items()} != protocol["inputs_sha256"]:
        raise ValueError("An E4 input changed after freeze")
    source = observed_table(paths["source"], 240, 24)
    target = observed_table(paths["target"], 189, 30)
    groups = read_groups(paths["field_assignments"])
    historical = check_historical_metrics(paths["historical_metrics"])
    recorded_splits = pd.read_csv(out / "split_integrity.csv")
    common_source = source_common(source)
    x = features(target)
    fixed = {stage: np.empty((5, len(target))) for stage in STAGES}
    for member in range(1, 6):
        with np.load(paths[f"member_{member}"], allow_pickle=False) as checkpoint:
            for stage in STAGES:
                fixed[stage][member - 1] = differential(checkpoint, stage, x)
    if any(not np.isfinite(value).all() for value in fixed.values()):
        raise ValueError("Nonfinite frozen checkpoint prediction")
    zero = pd.read_csv(paths["frozen_zero_shot"], dtype={"field_id": str})
    zero.acquisition_date = pd.to_datetime(zero.acquisition_date).dt.strftime("%Y-%m-%d")
    joined = target[KEY].merge(
        zero[KEY + ["hh_pretraining_only", "vv_pretraining_only",
                    "hh_spm_to_i2em", "vv_spm_to_i2em"]],
        on=KEY, how="left", validate="one_to_one", sort=False)
    if len(zero) != len(target) or joined.isna().any().any():
        raise ValueError("Frozen zero-shot cohort changed")
    checks = {}
    for stage, label in (("pretraining", "pretraining_only"),
                         ("source", "spm_to_i2em")):
        d = fixed[stage].mean(axis=0)
        hh = common_source - d / 2
        vv = common_source + d / 2
        difference = max(float(np.max(np.abs(hh - joined[f"hh_{label}"].to_numpy(float)))),
                         float(np.max(np.abs(vv - joined[f"vv_{label}"].to_numpy(float)))))
        if difference > 1e-8:
            raise ValueError(f"Frozen {stage} predictions differ from archived zero-shot: {difference}")
        checks[stage] = difference
    pd.DataFrame([
        {"field_id": target.at[i, "field_id"],
         "acquisition_date": target.at[i, "acquisition_date"], "member": member,
         "pretraining_diff_db": float(fixed["pretraining"][member - 1, i]),
         "source_diff_db": float(fixed["source"][member - 1, i])}
        for member in range(1, 6) for i in range(len(target))
    ]).to_csv(out / "fixed_member_predictions.csv", index=False)
    yhh = target.sigma0_hh_db.to_numpy(float)
    yvv = target.sigma0_vv_db.to_numpy(float)
    truth = response_arrays(yhh, yvv)
    offsets, raw, metrics = [], [], []
    for (split, fraction), assignment in groups.groupby(["split_repeat", "requested_fraction"], sort=True):
        adapt_fields = set(assignment.loc[assignment.role.eq("adaptation"), "field_id"])
        test_fields = set(assignment.loc[assignment.role.eq("test"), "field_id"])
        adapt = target.field_id.isin(adapt_fields).to_numpy()
        test = target.field_id.isin(test_fields).to_numpy()
        record = recorded_splits.loc[(recorded_splits.split_repeat.eq(split))
                                     & (recorded_splits.requested_fraction.eq(fraction))]
        if (len(record) != 1 or int(record.iloc[0].adaptation_rows) != int(adapt.sum())
                or int(record.iloc[0].test_rows) != int(test.sum())
                or set(target.field_id[adapt]) & set(target.field_id[test])):
            raise ValueError("Frozen target split differs at execution")
        common_offset = float(np.mean(truth["common"][adapt] - common_source))
        differential_offsets = {}
        for stage in STAGES:
            differential_offsets[stage] = np.array([
                float(np.mean(truth["differential"][adapt] - member_pred[adapt]))
                for member_pred in fixed[stage]
            ])
            for member, offset in enumerate(differential_offsets[stage], 1):
                offsets.append({
                    "split_repeat": int(split), "requested_fraction": float(fraction),
                    "adaptation_fields": len(adapt_fields), "adaptation_rows": int(adapt.sum()),
                    "member": member, "stage": stage,
                    "common_offset_db": common_offset, "differential_offset_db": float(offset),
                })
        predictions = {}
        for stage in STAGES:
            diff = fixed[stage].mean(axis=0)
            for calibrated in (False, True):
                method = f"{stage}_{'two_intercepts' if calibrated else 'no_target'}"
                common = np.full(len(target), common_source + (common_offset if calibrated else 0))
                value = diff + (float(differential_offsets[stage].mean()) if calibrated else 0)
                predictions[method] = response_arrays(common - value / 2,
                                                      common + value / 2)
        if set(predictions) != set(METHODS):
            raise AssertionError("Unexpected E4 method family")
        test_index = np.flatnonzero(test)
        for method, response in predictions.items():
            for i in test_index:
                raw.append({
                    "split_repeat": int(split), "requested_fraction": float(fraction),
                    "adaptation_fields": len(adapt_fields), "adaptation_rows": int(adapt.sum()),
                    "field_id": target.at[i, "field_id"],
                    "acquisition_date": target.at[i, "acquisition_date"], "method": method,
                    "observed_hh_db": float(yhh[i]), "observed_vv_db": float(yvv[i]),
                    "predicted_hh_db": float(response["HH"][i]),
                    "predicted_vv_db": float(response["VV"][i]),
                    "observed_common_db": float(truth["common"][i]),
                    "predicted_common_db": float(response["common"][i]),
                    "observed_differential_db": float(truth["differential"][i]),
                    "predicted_differential_db": float(response["differential"][i]),
                })
            for name in RESPONSES:
                metrics.append({
                    "split_repeat": int(split), "requested_fraction": float(fraction),
                    "adaptation_fields": len(adapt_fields), "adaptation_rows": int(adapt.sum()),
                    "test_fields": len(test_fields), "test_rows": int(test.sum()),
                    "method": method, "response": name,
                    **metric(truth[name][test], response[name][test]),
                })
        print(f"E4 repeat {int(split)}/20, adaptation fields {len(adapt_fields)} complete", flush=True)
    pd.DataFrame(offsets).to_csv(out / "offsets_by_member.csv", index=False)
    pd.DataFrame(raw).to_csv(out / "heldout_predictions.csv", index=False)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(out / "metrics_by_split.csv", index=False)
    summary = metrics_frame.groupby(["requested_fraction", "adaptation_fields", "method", "response"],
                                    as_index=False).agg(
        split_repeats=("split_repeat", "nunique"),
        mean_rmse_db=("rmse_db", "mean"), median_rmse_db=("rmse_db", "median"),
        min_rmse_db=("rmse_db", "min"), max_rmse_db=("rmse_db", "max"),
        mean_bias_db=("bias_db", "mean"),
        mean_centered_rmse_db=("centered_rmse_db", "mean"),
        mean_centered_skill=("centered_skill", "mean"),
    )
    summary.to_csv(out / "metrics_summary.csv", index=False)
    old = historical.loc[historical.method.eq("spm_to_i2em")].copy()
    comparison = metrics_frame.loc[metrics_frame.method.eq("source_two_intercepts")].merge(
        old[["split_repeat", "requested_fraction", "response", "n",
             "rmse_db", "bias_db", "r_squared_skill"]],
        on=["split_repeat", "requested_fraction", "response"],
        how="left", validate="one_to_one", suffixes=("_two_intercepts", "_historical_finetune"))
    if comparison.rmse_db_historical_finetune.isna().any():
        raise ValueError("Historical fine-tuning metric missing for a split")
    if not np.array_equal(comparison.test_rows.to_numpy(int), comparison.n.to_numpy(int)):
        raise ValueError("Historical comparison denominator differs")
    historical_centered_mse = (
        comparison.rmse_db_historical_finetune.to_numpy(float)**2
        - comparison.bias_db_historical_finetune.to_numpy(float)**2
    )
    if (historical_centered_mse < -1e-9).any():
        raise ValueError("Historical metric decomposition invalid")
    comparison["historical_centered_rmse_db"] = np.sqrt(np.maximum(0, historical_centered_mse))
    comparison["historical_centered_skill"] = (
        1 - historical_centered_mse / comparison.observed_variance_db2.to_numpy(float)
    )
    comparison["historical_r_squared_recomputed"] = (
        1 - comparison.rmse_db_historical_finetune.to_numpy(float)**2
        / comparison.observed_variance_db2.to_numpy(float)
    )
    if np.max(np.abs(comparison.historical_r_squared_recomputed
                     - comparison.r_squared_skill)) > 1e-7:
        raise ValueError("Historical test response variance is not aligned")
    comparison["rmse_delta_two_minus_finetune_db"] = (
        comparison.rmse_db_two_intercepts - comparison.rmse_db_historical_finetune
    )
    comparison["centered_skill_delta_two_minus_finetune"] = (
        comparison.centered_skill - comparison.historical_centered_skill
    )
    comparison.to_csv(out / "historical_comparison_by_split.csv", index=False)
    comp_summary = comparison.groupby(["requested_fraction", "adaptation_fields", "response"],
                                      as_index=False).agg(
        split_repeats=("split_repeat", "nunique"),
        mean_rmse_delta_db=("rmse_delta_two_minus_finetune_db", "mean"),
        median_rmse_delta_db=("rmse_delta_two_minus_finetune_db", "median"),
        min_rmse_delta_db=("rmse_delta_two_minus_finetune_db", "min"),
        max_rmse_delta_db=("rmse_delta_two_minus_finetune_db", "max"),
        fraction_two_intercepts_lower_rmse=("rmse_delta_two_minus_finetune_db",
                                            lambda value: float(np.mean(value < 0))),
        mean_centered_skill_delta=("centered_skill_delta_two_minus_finetune", "mean"),
    )
    comp_summary.to_csv(out / "historical_comparison_summary.csv", index=False)
    manifest = {
        "status": "COMPLETE", "experiment": protocol["experiment"],
        "source_rows": len(source), "source_fields": source.field_id.nunique(),
        "target_rows": len(target), "target_fields": target.field_id.nunique(),
        "field_split_count": 60, "member_count": 5,
        "adaptation_parameters_per_member": 2,
        "model_weights_updated": False, "target_test_labels_used_for_calibration": False,
        "max_frozen_zero_shot_difference_db": checks,
        "historical_raw_predictions_available": False,
        "post_hoc": True, "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {path.name: sha(path) for path in out.iterdir()
                          if path.is_file() and path.name != "result_manifest.json"},
    }
    (out / "result_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"],
                      "zero_shot_checks": checks,
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-protocol", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/review_E4_target_intercept")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.freeze_protocol:
        freeze(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
