"""One-time prospective SMEX02 zero-shot evaluation under the frozen Stage-6 rule.

All fitting and shrinkage selection use SMAPVEX12 only. SMEX02 targets are
read only after the source-locked predictions have been constructed and are
used solely for final scoring and uncertainty estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_external_domain as audit  # noqa: E402
import evaluate_decoupled_physics_heads as base  # noqa: E402
import evaluate_multifidelity_pretraining as multi  # noqa: E402
import evaluate_multifidelity_real_transfer as transfer  # noqa: E402


METHODS = [
    "source_mean",
    "source_ridge",
    "scratch",
    "spm_only",
    "i2em_only",
    "spm_to_i2em",
    "risk_spm_to_i2em",
    "ridge_common_risk_spm_to_i2em",
]
SIMPLE_BASELINES = ["source_mean", "source_ridge", "scratch"]
RESPONSES = ["HH", "VV", "common", "differential"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = frame[list(base.FEATURE_NAMES)].to_numpy(dtype=float)
    channels = frame[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
    return features, channels, base.to_components(channels)


def response_dict(channels: np.ndarray) -> dict[str, np.ndarray]:
    components = base.to_components(channels)
    return {
        "HH": channels[:, 0],
        "VV": channels[:, 1],
        "common": components[:, 0],
        "differential": components[:, 1],
    }


def metric(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(observed)
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "r_squared_skill": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan")
        ),
    }


def fit_predict_repeat(
    source: pd.DataFrame,
    external_features: np.ndarray,
    spm: pd.DataFrame,
    i2em: pd.DataFrame,
    weights: list[float],
    spm_epochs: int,
    i2em_epochs: int,
    source_epochs: int,
    inner_folds: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    source_features, source_channels, source_components = arrays(source)
    common = source_components[:, 0]
    differential = source_components[:, 1]
    groups = source.field_id.astype(str).to_numpy()
    common_mean = float(np.mean(common))
    differential_mean = float(np.mean(differential))

    bundles, bundle_metadata = transfer.build_bundles(
        multi.features_from(spm),
        spm[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float),
        multi.features_from(i2em),
        i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float),
        spm_epochs,
        i2em_epochs,
        seed,
    )

    channel_ridge = Ridge(alpha=1.0).fit(source_features, source_channels)
    common_ridge = Ridge(alpha=1.0).fit(source_features, common)
    scratch_model = base.scratch_single(
        bundles["spm_only"], source_features, differential, source_epochs, seed + 10_007
    )
    differential_models = {
        scheme: base.fine_tune_single(
            bundles[scheme],
            source_features,
            differential,
            source_epochs,
            seed + 10_100 + index,
        )
        for index, scheme in enumerate(["spm_only", "i2em_only", "spm_to_i2em"])
    }

    all_source = np.arange(len(source), dtype=int)
    selected_weight, records = base.tune_differential_shrinkage(
        bundles["spm_to_i2em"],
        source_features,
        differential,
        groups,
        all_source,
        weights,
        inner_folds,
        source_epochs,
        seed + 500_000,
    )
    tuning = pd.DataFrame(records)
    tuning.insert(0, "selected_weight", selected_weight)

    n_external = len(external_features)
    common_primary = np.full(n_external, common_mean)
    result: dict[str, np.ndarray] = {
        "source_mean": np.tile(np.mean(source_channels, axis=0), (n_external, 1)),
        "source_ridge": np.asarray(channel_ridge.predict(external_features), dtype=float),
    }
    scratch_diff = base.predict_single(
        bundles["spm_only"], scratch_model, external_features
    )
    result["scratch"] = base.from_components(common_primary, scratch_diff)
    for scheme in ["spm_only", "i2em_only", "spm_to_i2em"]:
        predicted_diff = base.predict_single(
            bundles[scheme], differential_models[scheme], external_features
        )
        result[scheme] = base.from_components(common_primary, predicted_diff)

    raw_diff = base.to_components(result["spm_to_i2em"])[:, 1]
    risk_diff = base.shrink_prediction_to_mean(
        raw_diff, differential_mean, selected_weight
    )
    result["risk_spm_to_i2em"] = base.from_components(common_primary, risk_diff)
    result["ridge_common_risk_spm_to_i2em"] = base.from_components(
        np.asarray(common_ridge.predict(external_features), dtype=float), risk_diff
    )
    metadata = {
        "seed": seed,
        "selected_source_only_shrinkage": selected_weight,
        "source_common_mean_db": common_mean,
        "source_differential_mean_db": differential_mean,
        "pretraining": bundle_metadata,
    }
    return result, tuning, metadata


def ensemble_predictions(
    predictions: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        method: np.mean([repeat[method] for repeat in predictions], axis=0)
        for method in METHODS
    }


def metric_table(observed: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    reference = response_dict(observed)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        predicted = response_dict(predictions[method])
        for response in RESPONSES:
            rows.append(
                {
                    "method": method,
                    "response": response,
                    "n": len(observed),
                    **metric(reference[response], predicted[response]),
                }
            )
    return pd.DataFrame(rows)


def joint_table(observed: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        errors = predictions[method] - observed
        channel_rmse = np.sqrt(np.mean(errors**2, axis=0))
        rows.append(
            {
                "method": method,
                "mean_hh_vv_rmse_db": float(np.mean(channel_rmse)),
                "pooled_hh_vv_rmse_db": float(np.sqrt(np.mean(errors**2))),
                "hh_rmse_db": float(channel_rmse[0]),
                "vv_rmse_db": float(channel_rmse[1]),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_hh_vv_rmse_db").reset_index(drop=True)


def field_subgroups(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    minimum_rows: int,
) -> pd.DataFrame:
    rows = []
    fields = frame.field_id.astype(str).to_numpy()
    for field_id in np.unique(fields):
        selected = fields == field_id
        if int(np.sum(selected)) < minimum_rows:
            continue
        risk_rmse = np.sqrt(
            np.mean((predictions["risk_spm_to_i2em"][selected] - observed[selected]) ** 2, axis=0)
        )
        mean_rmse = np.sqrt(
            np.mean((predictions["source_mean"][selected] - observed[selected]) ** 2, axis=0)
        )
        rows.append(
            {
                "field_id": field_id,
                "rows": int(np.sum(selected)),
                "risk_mean_hh_vv_rmse_db": float(np.mean(risk_rmse)),
                "source_mean_hh_vv_rmse_db": float(np.mean(mean_rmse)),
                "risk_minus_source_mean_db": float(np.mean(risk_rmse) - np.mean(mean_rmse)),
                "noninferior_within_0_10_db": bool(np.mean(risk_rmse) - np.mean(mean_rmse) <= 0.10),
            }
        )
    return pd.DataFrame(rows).sort_values("field_id").reset_index(drop=True)


def save_plots(
    joint: pd.DataFrame,
    frame: pd.DataFrame,
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    subgroups: pd.DataFrame,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    part = joint.sort_values("mean_hh_vv_rmse_db", ascending=False)
    colors = ["#B279A2" if name == "risk_spm_to_i2em" else "#4C78A8" for name in part.method]
    ax.barh(part.method, part.mean_hh_vv_rmse_db, color=colors)
    ax.set(xlabel="Mean of HH and VV RMSE (dB)", title="Prospective SMEX02 zero-shot ranking")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "01_zero_shot_ranking.png", dpi=190)
    plt.close(fig)

    observed_diff = base.to_components(observed)[:, 1]
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    for method, color in [
        ("source_mean", "#999999"),
        ("spm_to_i2em", "#54A24B"),
        ("risk_spm_to_i2em", "#B279A2"),
    ]:
        predicted_diff = base.to_components(predictions[method])[:, 1]
        ax.scatter(observed_diff, predicted_diff, s=18, alpha=0.38, label=method, color=color)
    lower = float(min(observed_diff.min(), *(base.to_components(predictions[m])[:, 1].min() for m in ["source_mean", "spm_to_i2em", "risk_spm_to_i2em"])))
    upper = float(max(observed_diff.max(), *(base.to_components(predictions[m])[:, 1].max() for m in ["source_mean", "spm_to_i2em", "risk_spm_to_i2em"])))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    ax.set(xlabel="Observed VV-HH (dB)", ylabel="Predicted VV-HH (dB)", title="Locked differential-head transfer")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "02_differential_observed_vs_predicted.png", dpi=190)
    plt.close(fig)

    if not subgroups.empty:
        fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
        ordered = subgroups.sort_values("risk_minus_source_mean_db")
        colors = np.where(ordered.risk_minus_source_mean_db <= 0.10, "#54A24B", "#E45756")
        ax.bar(ordered.field_id.astype(str), ordered.risk_minus_source_mean_db, color=colors)
        ax.axhline(0.10, color="black", linestyle="--", linewidth=1, label="+0.10 dB margin")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set(xlabel="SMEX02 field", ylabel="Risk method minus source mean RMSE (dB)", title="Field-level non-inferiority diagnostic")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.savefig(output / "03_field_noninferiority.png", dpi=190)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen prospective SMEX02 zero-shot evaluation")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--spm", required=True, type=Path)
    parser.add_argument("--i2em-requests", required=True, type=Path)
    parser.add_argument("--i2em-results", required=True, type=Path)
    parser.add_argument("--freeze-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--source-epochs", type=int, default=120)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shrinkage-weights", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--minimum-field-subgroup-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_TARGET_SCORING":
        raise ValueError("Freeze manifest is not valid")
    expected_external_hash = freeze["files"]["external_table"]["sha256"]
    if sha256(args.external.resolve()) != expected_external_hash:
        raise ValueError("External table changed after freezing")
    expected_evaluator_hash = freeze["files"]["evaluator"]["sha256"]
    if sha256(Path(__file__).resolve()) != expected_evaluator_hash:
        raise ValueError("Evaluator changed after freezing")

    weights = sorted({float(value) for value in args.shrinkage_weights.split(",")})
    if weights != [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("Shrinkage candidates differ from the frozen contract")

    source = audit.prepare_table(args.source.resolve(), "SMAPVEX12", "topp_both")
    external = audit.prepare_table(args.external.resolve(), "SMEX02", "topp_both")
    source = source.loc[source.finite_model_row].sort_values(["field_id", "acquisition_date"]).reset_index(drop=True)
    external = external.loc[external.finite_model_row].sort_values(["field_id", "acquisition_date"]).reset_index(drop=True)
    if set(source.campaign_id.astype(str)) & set(external.campaign_id.astype(str)):
        raise ValueError("Source and external campaign IDs overlap")

    spm = pd.read_csv(args.spm.resolve())
    i2em = multi.load_paired(args.i2em_requests.resolve(), args.i2em_results.resolve())
    external_features, observed, _ = arrays(external)
    repeat_predictions = []
    tuning_frames = []
    repeat_metadata = []
    repeat_metric_frames = []
    for repeat in range(1, args.repeats + 1):
        seed = args.seed + (repeat - 1) * 100_000
        predictions, tuning, metadata = fit_predict_repeat(
            source,
            external_features,
            spm,
            i2em,
            weights,
            args.spm_epochs,
            args.i2em_epochs,
            args.source_epochs,
            args.inner_folds,
            seed,
        )
        for prediction in predictions.values():
            if not np.isfinite(prediction).all():
                raise ValueError("Non-finite zero-shot prediction")
        repeat_predictions.append(predictions)
        tuning.insert(0, "repeat", repeat)
        tuning_frames.append(tuning)
        repeat_metadata.append({"repeat": repeat, **metadata})
        repeated_metrics = metric_table(observed, predictions)
        repeated_metrics.insert(0, "repeat", repeat)
        repeat_metric_frames.append(repeated_metrics)
        print(f"Completed frozen source-only repeat {repeat}/{args.repeats}", flush=True)

    predictions = ensemble_predictions(repeat_predictions)
    for prediction in predictions.values():
        base.verify_error_identity(observed, prediction)
    metrics = metric_table(observed, predictions)
    joint = joint_table(observed, predictions)
    groups = external.field_id.astype(str).to_numpy()

    comparisons = {}
    for comparator in ["source_mean", "source_ridge", "scratch", "spm_only", "i2em_only", "spm_to_i2em"]:
        comparisons[f"risk_spm_to_i2em_minus_{comparator}"] = base.grouped_bootstrap_joint_delta(
            observed,
            predictions["risk_spm_to_i2em"],
            predictions[comparator],
            groups,
            args.bootstrap_iterations,
            args.seed + 900_000 + len(comparisons) * 100,
        )

    simple = joint[joint.method.isin(SIMPLE_BASELINES)].sort_values("mean_hh_vv_rmse_db")
    strongest_simple = str(simple.iloc[0].method)
    primary_delta = comparisons[f"risk_spm_to_i2em_minus_{strongest_simple}"]
    subgroups = field_subgroups(
        external, observed, predictions, args.minimum_field_subgroup_rows
    )
    criterion_improvement = primary_delta["mean_channel_rmse_delta_db"] <= -0.10
    criterion_ci = primary_delta["ci_97_5_percent_db"] < 0.0
    criterion_subgroups = bool(
        len(subgroups) > 0 and subgroups.noninferior_within_0_10_db.all()
    )
    claim_supported = bool(criterion_improvement and criterion_ci and criterion_subgroups)

    output.mkdir(parents=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    metrics_by_repeat = pd.concat(repeat_metric_frames, ignore_index=True)
    tuning.to_csv(output / "source_only_shrinkage_tuning.csv", index=False)
    metrics_by_repeat.to_csv(output / "metrics_by_repeat.csv", index=False)
    metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    joint.to_csv(output / "joint_channel_metrics.csv", index=False)
    subgroups.to_csv(output / "field_subgroup_noninferiority.csv", index=False)

    prediction_frame = external[["campaign_id", "acquisition_date", "field_id", *base.FEATURE_NAMES, *base.OBSERVED_COLUMNS]].copy()
    for method in METHODS:
        prediction_frame[f"hh_{method}"] = predictions[method][:, 0]
        prediction_frame[f"vv_{method}"] = predictions[method][:, 1]
    prediction_frame.to_csv(output / "zero_shot_predictions.csv", index=False)

    repeated_rows = []
    for repeat, predicted in enumerate(repeat_predictions, start=1):
        frame = external[["campaign_id", "acquisition_date", "field_id"]].copy()
        frame.insert(0, "repeat", repeat)
        for method in METHODS:
            frame[f"hh_{method}"] = predicted[method][:, 0]
            frame[f"vv_{method}"] = predicted[method][:, 1]
        repeated_rows.append(frame)
    pd.concat(repeated_rows, ignore_index=True).to_csv(
        output / "zero_shot_predictions_by_repeat.csv", index=False
    )
    save_plots(joint, external, observed, predictions, subgroups, output)

    summary = {
        "status": "PROSPECTIVE_ZERO_SHOT_COMPLETE",
        "claim_verdict": "SUPPORTED" if claim_supported else "NOT_SUPPORTED",
        "research_question": "Does the frozen component-decoupled, risk-controlled multi-fidelity method improve HH/VV backscatter prediction in independent SMEX02?",
        "protocol": {
            "source_campaign": "SMAPVEX12",
            "external_campaign": "SMEX02",
            "source_only_model_selection": True,
            "external_target_use": "final scoring and field-block uncertainty only",
            "common_head_primary": "source-training mean",
            "differential_head_primary": "SPM-to-I2EM plus source-real fit and source-only grouped-CV shrinkage",
            "common_ridge_role": "secondary pre-registered ablation",
            "repeats": args.repeats,
            "bootstrap_unit": "SMEX02 field_id",
            "bootstrap_iterations": args.bootstrap_iterations,
            "minimum_field_subgroup_rows": args.minimum_field_subgroup_rows,
        },
        "samples": {
            "source_rows": len(source),
            "source_fields": int(source.field_id.nunique()),
            "external_rows": len(external),
            "external_fields": int(external.field_id.nunique()),
            "external_dates": int(external.acquisition_date.nunique()),
        },
        "source_only_shrinkage_by_repeat": [
            {
                "repeat": item["repeat"],
                "seed": item["seed"],
                "selected_weight": item["selected_source_only_shrinkage"],
            }
            for item in repeat_metadata
        ],
        "mean_hh_vv_rmse_ranking_db": {
            str(row.method): float(row.mean_hh_vv_rmse_db)
            for row in joint.itertuples(index=False)
        },
        "paired_field_bootstrap": comparisons,
        "primary_comparator": strongest_simple,
        "primary_comparison": primary_delta,
        "pre_registered_claim_checks": {
            "improvement_at_least_0_10_db": bool(criterion_improvement),
            "bootstrap_ci_entirely_below_zero": bool(criterion_ci),
            "all_adequate_fields_noninferior_to_source_mean_within_0_10_db": bool(criterion_subgroups),
            "adequate_fields": len(subgroups),
            "fields_failing_noninferiority": (
                subgroups.loc[~subgroups.noninferior_within_0_10_db, "field_id"].astype(str).tolist()
                if not subgroups.empty
                else []
            ),
        },
        "interpretation_rule": (
            "A superiority claim is permitted only when all three checks are true; otherwise report a limitation or negative-transfer result."
        ),
        "inputs": {
            "source_sha256": sha256(args.source.resolve()),
            "external_sha256": sha256(args.external.resolve()),
            "freeze_manifest_sha256": sha256(args.freeze_manifest.resolve()),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(joint.to_string(index=False), flush=True)
    print(json.dumps(summary["pre_registered_claim_checks"], indent=2, ensure_ascii=False), flush=True)
    print(f"Claim verdict: {summary['claim_verdict']}", flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
