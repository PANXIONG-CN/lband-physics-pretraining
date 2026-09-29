"""Strict unseen-field ablation for multi-fidelity differential pretraining.

The common response is always the outer-training mean.  Only the fully
separate VV-minus-HH head changes, so any difference among methods is
attributable to its initialization and nested risk-control strength.
"""

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
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_decoupled_physics_heads as base  # noqa: E402
from research_pilots.scattering.surfaces.teacher_contract import (  # noqa: E402
    validate_teacher_results,
)


PRIMARY_METHODS = [
    "training_mean",
    "scratch",
    "spm_only",
    "i2em_only",
    "spm_to_i2em",
]
RISK_METHODS = [
    "risk_spm_only",
    "risk_i2em_only",
    "risk_spm_to_i2em",
]
ALL_METHODS = PRIMARY_METHODS + RISK_METHODS
SCHEMES = ["spm_only", "i2em_only", "spm_to_i2em"]


def features_from(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(base.FEATURE_NAMES).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return frame[list(base.FEATURE_NAMES)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)


def build_bundles(
    spm_features: np.ndarray,
    spm_channels: np.ndarray,
    i2em_features: np.ndarray,
    i2em_channels: np.ndarray,
    spm_epochs: int,
    i2em_epochs: int,
    seed: int,
) -> tuple[dict[str, base.SingleOutputBundle], dict[str, object]]:
    spm_differential = base.to_components(spm_channels)[:, 1]
    i2em_differential = base.to_components(i2em_channels)[:, 1]
    spm_bundle = base.pretrain_single_output(spm_features, spm_differential, spm_epochs, seed)
    i2em_bundle = base.pretrain_single_output(
        i2em_features, i2em_differential, spm_epochs + i2em_epochs, seed + 1
    )
    sequential_model = base.fine_tune_single(
        spm_bundle, i2em_features, i2em_differential, i2em_epochs, seed + 2
    )
    sequential_bundle = base.SingleOutputBundle(
        model=sequential_model,
        feature_scaler=spm_bundle.feature_scaler,
        target_scaler=spm_bundle.target_scaler,
        metadata={
            "spm_epochs": spm_epochs,
            "i2em_refinement_epochs": i2em_epochs,
            "coordinate": "VV_minus_HH",
        },
    )
    bundles = {
        "spm_only": spm_bundle,
        "i2em_only": i2em_bundle,
        "spm_to_i2em": sequential_bundle,
    }
    metadata = {
        "spm_only": spm_bundle.metadata,
        "i2em_only": i2em_bundle.metadata,
        "spm_to_i2em": sequential_bundle.metadata,
    }
    return bundles, metadata


def run_repeat(
    frame: pd.DataFrame,
    bundles: dict[str, base.SingleOutputBundle],
    shrinkage_weights: list[float],
    outer_folds: int,
    inner_folds: int,
    fine_tune_epochs: int,
    seed: int,
    repeat: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features_from(frame)
    observed_channels = frame[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
    observed_components = base.to_components(observed_channels)
    common = observed_components[:, 0]
    differential = observed_components[:, 1]
    groups = frame["field_id"].astype(str).to_numpy()
    predictions = {method: np.full_like(observed_channels, np.nan) for method in ALL_METHODS}
    fold_ids = np.full(len(frame), -1, dtype=int)
    selected = {scheme: np.full(len(frame), np.nan) for scheme in SCHEMES}
    tuning_rows: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=min(outer_folds, len(np.unique(groups))))
    for outer_fold, (train, test) in enumerate(splitter.split(features, groups=groups), start=1):
        common_mean = float(np.mean(common[train]))
        differential_mean = float(np.mean(differential[train]))
        predictions["training_mean"][test] = base.from_components(
            np.full(len(test), common_mean), np.full(len(test), differential_mean)
        )
        scratch_model = base.scratch_single(
            bundles["spm_only"], features[train], differential[train], fine_tune_epochs,
            seed + outer_fold * 10_000 + 7,
        )
        scratch_differential = base.predict_single(
            bundles["spm_only"], scratch_model, features[test]
        )
        predictions["scratch"][test] = base.from_components(
            np.full(len(test), common_mean), scratch_differential
        )
        for scheme_index, scheme in enumerate(SCHEMES):
            bundle = bundles[scheme]
            fit_seed = seed + outer_fold * 10_000 + scheme_index * 100 + 11
            model = base.fine_tune_single(
                bundle, features[train], differential[train], fine_tune_epochs, fit_seed
            )
            raw = base.predict_single(bundle, model, features[test])
            predictions[scheme][test] = base.from_components(
                np.full(len(test), common_mean), raw
            )
            weight, records = base.tune_differential_shrinkage(
                bundle, features, differential, groups, train, shrinkage_weights,
                inner_folds, fine_tune_epochs,
                seed + outer_fold * 200_000 + scheme_index * 10_000,
            )
            risk = base.shrink_prediction_to_mean(raw, differential_mean, weight)
            predictions[f"risk_{scheme}"][test] = base.from_components(
                np.full(len(test), common_mean), risk
            )
            selected[scheme][test] = weight
            for record in records:
                tuning_rows.append(
                    {
                        "repeat": repeat,
                        "scheme": scheme,
                        "outer_fold": outer_fold,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_fields": len(np.unique(groups[train])),
                        "test_fields": len(np.unique(groups[test])),
                        **record,
                    }
                )
        fold_ids[test] = outer_fold
    output = frame[["acquisition_date", "field_id", *base.FEATURE_NAMES, *base.OBSERVED_COLUMNS]].copy()
    output.insert(0, "row_id", np.arange(len(frame)))
    output.insert(0, "repeat", repeat)
    output["outer_fold"] = fold_ids
    output["observed_common_db"] = common
    output["observed_differential_db"] = differential
    for scheme in SCHEMES:
        output[f"selected_shrinkage_{scheme}"] = selected[scheme]
    for method, channels in predictions.items():
        base.verify_error_identity(observed_channels, channels)
        components = base.to_components(channels)
        output[f"hh_{method}"] = channels[:, 0]
        output[f"vv_{method}"] = channels[:, 1]
        output[f"common_{method}"] = components[:, 0]
        output[f"differential_{method}"] = components[:, 1]
    return output, pd.DataFrame(tuning_rows)


def regression_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    baseline = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "r_squared_skill": float(1.0 - np.sum(error**2) / baseline) if baseline > 0 else float("nan"),
    }


def metrics_for(frame: pd.DataFrame) -> pd.DataFrame:
    response_specs = [
        ("HH", "sigma0_hh_db", "hh"),
        ("VV", "sigma0_vv_db", "vv"),
        ("common", "observed_common_db", "common"),
        ("differential", "observed_differential_db", "differential"),
    ]
    rows = []
    for response, observed_name, prefix in response_specs:
        observed = frame[observed_name].to_numpy(dtype=float)
        for method in ALL_METHODS:
            predicted = frame[f"{prefix}_{method}"].to_numpy(dtype=float)
            rows.append({"response": response, "method": method, "n": len(frame), **regression_row(observed, predicted)})
    return pd.DataFrame(rows)


def ensemble_predictions(repeats: pd.DataFrame) -> pd.DataFrame:
    first_columns = [
        "acquisition_date", "field_id", *base.FEATURE_NAMES, *base.OBSERVED_COLUMNS,
        "observed_common_db", "observed_differential_db",
    ]
    prediction_columns = [
        f"{prefix}_{method}"
        for method in ALL_METHODS
        for prefix in ["hh", "vv", "common", "differential"]
    ]
    first = repeats.sort_values("repeat").drop_duplicates("row_id").set_index("row_id")[first_columns]
    averaged = repeats.groupby("row_id")[prediction_columns].mean()
    return first.join(averaged).reset_index()


def save_plots(metrics: pd.DataFrame, ensemble: pd.DataFrame, tuning: pd.DataFrame, output: Path) -> None:
    labels = {
        "training_mean": "Mean", "scratch": "Scratch", "spm_only": "SPM",
        "i2em_only": "I2EM", "spm_to_i2em": "SPM→I2EM",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        part = metrics[(metrics.response == response) & metrics.method.isin(PRIMARY_METHODS)]
        values = [float(part.loc[part.method == name, "rmse_db"].iloc[0]) for name in PRIMARY_METHODS]
        ax.bar(range(len(values)), values, color=["#999999", "#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
        ax.set_xticks(range(len(values)), [labels[name] for name in PRIMARY_METHODS], rotation=15)
        ax.set(ylabel="Grouped OOF RMSE (dB)", title=response)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "01_real_domain_primary_ablation.png", dpi=190)
    plt.close(fig)

    part = metrics[metrics.response == "differential"].copy()
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(range(len(part)), part.rmse_db, color=["#999999" if method in PRIMARY_METHODS else "#B279A2" for method in part.method])
    ax.set_xticks(range(len(part)), [name.replace("_", "\n") for name in part.method], rotation=15)
    ax.set(ylabel="VV-HH grouped OOF RMSE (dB)", title="Differential-head ablation and nested risk control")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "02_differential_ablation.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 5.6), constrained_layout=True)
    observed = ensemble.observed_differential_db
    for method, color in [("scratch", "#4C78A8"), ("spm_only", "#72B7B2"), ("i2em_only", "#F58518"), ("spm_to_i2em", "#54A24B"), ("risk_spm_to_i2em", "#B279A2")]:
        ax.scatter(observed, ensemble[f"differential_{method}"], s=14, alpha=0.35, label=method, color=color)
    lower = float(min(observed.min(), ensemble[[f"differential_{m}" for m in ["scratch", "spm_only", "i2em_only", "spm_to_i2em", "risk_spm_to_i2em"]]].min().min()))
    upper = float(max(observed.max(), ensemble[[f"differential_{m}" for m in ["scratch", "spm_only", "i2em_only", "spm_to_i2em", "risk_spm_to_i2em"]]].max().max()))
    ax.plot([lower, upper], [lower, upper], "k--", lw=1)
    ax.set(xlabel="Observed VV-HH (dB)", ylabel="Grouped OOF prediction (dB)", title="Unseen-field differential response")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(output / "03_differential_observed_vs_predicted.png", dpi=190)
    plt.close(fig)

    selected = tuning[tuning.candidate_selected].drop_duplicates(["repeat", "scheme", "outer_fold"])
    frequencies = selected.groupby(["scheme", "candidate"]).size().rename("count").reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True, sharey=True)
    for ax, scheme in zip(axes, SCHEMES):
        part = frequencies[frequencies.scheme == scheme]
        ax.bar(part.candidate.astype(str), part["count"], color="#4C78A8")
        ax.set(title=scheme, xlabel="Selected shrinkage")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Outer-fold selections")
    fig.savefig(output / "04_nested_shrinkage_selection.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested unseen-field multi-fidelity transfer ablation")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--spm", required=True, type=Path)
    parser.add_argument("--i2em-requests", required=True, type=Path)
    parser.add_argument("--i2em-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--fine-tune-epochs", type=int, default=120)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shrinkage-weights", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; choose a new output directory")
    weights = sorted(set(float(value) for value in args.shrinkage_weights.split(",")))
    if 0.0 not in weights or 1.0 not in weights or any(value < 0 or value > 1 for value in weights):
        raise ValueError("Shrinkage weights must be in [0,1] and include 0 and 1")
    real = base.load_table(args.input.resolve())
    spm = pd.read_csv(args.spm.resolve())
    i2em = validate_teacher_results(pd.read_csv(args.i2em_requests.resolve()), pd.read_csv(args.i2em_results.resolve()))
    spm_features = features_from(spm)
    spm_channels = spm[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float)
    i2em_features = features_from(i2em)
    i2em_channels = i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    repeat_frames = []
    tuning_frames = []
    pretraining = []
    metric_repeats = []
    for repeat in range(1, args.repeats + 1):
        seed = args.seed + (repeat - 1) * 100_000
        bundles, metadata = build_bundles(
            spm_features, spm_channels, i2em_features, i2em_channels,
            args.spm_epochs, args.i2em_epochs, seed,
        )
        evaluated, tuning = run_repeat(
            real, bundles, weights, args.outer_folds, args.inner_folds,
            args.fine_tune_epochs, seed, repeat,
        )
        repeated_metrics = metrics_for(evaluated)
        repeated_metrics.insert(0, "repeat", repeat)
        repeat_frames.append(evaluated)
        tuning_frames.append(tuning)
        metric_repeats.append(repeated_metrics)
        pretraining.append({"repeat": repeat, "seed": seed, "metadata": metadata})
        print(f"Completed real-domain repeat {repeat}/{args.repeats}", flush=True)
    repeated = pd.concat(repeat_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    metrics_by_repeat = pd.concat(metric_repeats, ignore_index=True)
    ensemble = ensemble_predictions(repeated)
    metrics = metrics_for(ensemble)
    groups = ensemble.field_id.astype(str).to_numpy()
    observed_channels = ensemble[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
    observed_components = ensemble[["observed_common_db", "observed_differential_db"]].to_numpy(dtype=float)
    comparisons = {}
    pairs = {
        "multifidelity_vs_scratch": ("spm_to_i2em", "scratch"),
        "multifidelity_vs_spm": ("spm_to_i2em", "spm_only"),
        "multifidelity_vs_i2em": ("spm_to_i2em", "i2em_only"),
        "risk_multifidelity_vs_mean": ("risk_spm_to_i2em", "training_mean"),
        "risk_multifidelity_vs_raw": ("risk_spm_to_i2em", "spm_to_i2em"),
    }
    for index, (name, (first, second)) in enumerate(pairs.items()):
        first_channels = ensemble[[f"hh_{first}", f"vv_{first}"]].to_numpy()
        second_channels = ensemble[[f"hh_{second}", f"vv_{second}"]].to_numpy()
        first_components = ensemble[[f"common_{first}", f"differential_{first}"]].to_numpy()
        second_components = ensemble[[f"common_{second}", f"differential_{second}"]].to_numpy()
        comparisons[name] = {
            "joint_channels": base.grouped_bootstrap_joint_delta(observed_channels, first_channels, second_channels, groups, args.bootstrap_iterations, args.seed + index * 100),
            "components": base.grouped_bootstrap_delta(observed_components, first_components, second_components, groups, ["common", "differential"], args.bootstrap_iterations, args.seed + 1000 + index * 100),
        }
    output.mkdir(parents=True, exist_ok=True)
    repeated.to_csv(output / "oof_predictions_by_repeat.csv", index=False)
    ensemble.to_csv(output / "oof_predictions.csv", index=False)
    tuning.to_csv(output / "nested_shrinkage_tuning.csv", index=False)
    metrics_by_repeat.to_csv(output / "metrics_by_repeat.csv", index=False)
    metrics.to_csv(output / "ensemble_metrics.csv", index=False)
    save_plots(metrics, ensemble, tuning, output)
    joint = metrics[metrics.response.isin(["HH", "VV"])].groupby("method").rmse_db.mean().sort_values()
    summary = {
        "research_question": "Does higher-fidelity or sequential physics initialization improve fully decoupled differential-response transfer to unseen SMAPVEX12 fields?",
        "samples": len(real), "fields": int(real.field_id.nunique()), "dates": int(real.acquisition_date.nunique()),
        "protocol": {"outer": f"{args.outer_folds}-fold GroupKFold by field", "inner": f"up to {args.inner_folds}-fold GroupKFold by field", "repeats": args.repeats, "bootstrap_field_blocks": args.bootstrap_iterations, "common_head": "outer-training mean for every method", "differential_head": "fully separate VV-minus-HH MLP", "risk_control": "inner-field-CV shrinkage toward outer-training differential mean"},
        "five_primary_baselines": PRIMARY_METHODS,
        "risk_controlled_variants": RISK_METHODS,
        "mean_hh_vv_rmse_ranking_db": {name: float(value) for name, value in joint.items()},
        "paired_field_bootstrap": comparisons,
        "pretraining": pretraining,
        "decision_rule": "Multi-fidelity transfer supports the paper claim only if it improves over scratch and both single-fidelity initializations on unseen fields, while the risk-controlled variant is non-inferior to the training mean.",
        "scope_limit": "One campaign and one nominal incidence angle; I2EM is a physics teacher, not observation truth. Cross-campaign generalization remains future evidence.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(joint.to_string(), flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
