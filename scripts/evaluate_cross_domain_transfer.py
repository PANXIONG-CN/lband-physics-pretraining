"""Locked zero-shot and field-grouped few-shot external-domain evaluation.

Model and tuning choices must be finalized on the source campaign.  External
targets are used only for final scoring and for explicitly labelled 5/10/20%
few-shot adaptation; held-out external fields remain untouched in each split.
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
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_external_domain as audit  # noqa: E402
import evaluate_decoupled_physics_heads as base  # noqa: E402
import evaluate_multifidelity_real_transfer as transfer  # noqa: E402
import evaluate_multifidelity_pretraining as multi  # noqa: E402


SCHEMES = ["spm_only", "i2em_only", "spm_to_i2em"]
RESPONSES = ["HH", "VV", "common", "differential"]


def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = frame[list(base.FEATURE_NAMES)].to_numpy(dtype=float)
    channels = frame[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
    return features, channels, base.to_components(channels)


def metric(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "bias_db": float(np.mean(error)),
        "r_squared_skill": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


def response_dict(channels: np.ndarray) -> dict[str, np.ndarray]:
    components = base.to_components(channels)
    return {"HH": channels[:, 0], "VV": channels[:, 1], "common": components[:, 0], "differential": components[:, 1]}


def fit_source_models(
    source: pd.DataFrame,
    bundles: dict[str, base.SingleOutputBundle],
    fine_tune_epochs: int,
    seed: int,
) -> dict[str, object]:
    features, channels, components = arrays(source)
    common_ridge = Ridge(alpha=1.0).fit(features, components[:, 0])
    channel_ridge = Ridge(alpha=1.0).fit(features, channels)
    scratch = base.scratch_single(bundles["spm_only"], features, components[:, 1], fine_tune_epochs, seed + 7)
    differential_models = {
        scheme: base.fine_tune_single(bundle, features, components[:, 1], fine_tune_epochs, seed + 100 + index)
        for index, (scheme, bundle) in enumerate(bundles.items())
    }
    return {"common_ridge": common_ridge, "channel_ridge": channel_ridge, "scratch": scratch, "differential_models": differential_models}


def predict_source_locked(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    bundles: dict[str, base.SingleOutputBundle],
    fitted: dict[str, object],
) -> dict[str, np.ndarray]:
    features, _, _ = arrays(frame)
    source_features, source_channels, source_components = arrays(source)
    common = np.asarray(fitted["common_ridge"].predict(features), dtype=float)
    result = {
        "source_mean": np.tile(np.mean(source_channels, axis=0), (len(frame), 1)),
        "source_ridge": np.asarray(fitted["channel_ridge"].predict(features), dtype=float),
    }
    scratch_diff = base.predict_single(bundles["spm_only"], fitted["scratch"], features)
    result["scratch"] = base.from_components(common, scratch_diff)
    for scheme in SCHEMES:
        differential = base.predict_single(bundles[scheme], fitted["differential_models"][scheme], features)
        result[scheme] = base.from_components(common, differential)
    for prediction in result.values():
        base.verify_error_identity(frame[base.OBSERVED_COLUMNS].to_numpy(dtype=float), prediction)
    return result


def adapt_predictions(
    train_external: pd.DataFrame,
    test_external: pd.DataFrame,
    source: pd.DataFrame,
    bundles: dict[str, base.SingleOutputBundle],
    fitted: dict[str, object],
    adaptation_epochs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    zero_train = predict_source_locked(train_external, source, bundles, fitted)
    zero_test = predict_source_locked(test_external, source, bundles, fitted)
    observed_train = train_external[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
    train_features, _, train_components = arrays(train_external)
    test_features, _, _ = arrays(test_external)
    adapted: dict[str, np.ndarray] = {}
    for method in ["source_mean", "source_ridge"]:
        offset = np.mean(observed_train - zero_train[method], axis=0)
        adapted[method] = zero_test[method] + offset
    for index, scheme in enumerate(["scratch", *SCHEMES]):
        bundle_key = "spm_only" if scheme == "scratch" else scheme
        source_model = fitted["scratch"] if scheme == "scratch" else fitted["differential_models"][scheme]
        source_bundle = base.SingleOutputBundle(model=source_model, feature_scaler=bundles[bundle_key].feature_scaler, target_scaler=bundles[bundle_key].target_scaler, metadata={"stage": "source_fitted"})
        adapted_model = base.fine_tune_single(source_bundle, train_features, train_components[:, 1], adaptation_epochs, seed + index * 100)
        differential = base.predict_single(source_bundle, adapted_model, test_features)
        zero_common_train = base.to_components(zero_train[scheme])[:, 0]
        common_offset = float(np.mean(train_components[:, 0] - zero_common_train))
        common = base.to_components(zero_test[scheme])[:, 0] + common_offset
        adapted[scheme] = base.from_components(common, differential)
    return adapted


def rows_for_predictions(reference: np.ndarray, predictions: dict[str, np.ndarray], stage: str, fraction: float, repeat: int, n_adapt_fields: int, n_test_fields: int) -> list[dict[str, object]]:
    refs = response_dict(reference)
    rows: list[dict[str, object]] = []
    for method, channels in predictions.items():
        preds = response_dict(channels)
        for response in RESPONSES:
            rows.append({"stage": stage, "adaptation_fraction": fraction, "repeat": repeat, "method": method, "response": response, "n": len(reference), "adaptation_fields": n_adapt_fields, "test_fields": n_test_fields, **metric(refs[response], preds[response])})
    return rows


def save_plot(metrics: pd.DataFrame, output: Path) -> None:
    part = metrics[(metrics.response.isin(["HH", "VV"]))].groupby(["adaptation_fraction", "method"], as_index=False).rmse_db.mean()
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for method in ["source_mean", "source_ridge", "scratch", *SCHEMES]:
        curve = part[part.method == method].sort_values("adaptation_fraction")
        if not curve.empty:
            ax.plot(curve.adaptation_fraction * 100, curve.rmse_db, marker="o", label=method)
    ax.set(xlabel="External fields used for adaptation (%)", ylabel="Mean HH/VV RMSE on untouched external fields (dB)", title="Cross-domain zero-shot and few-shot adaptation")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.savefig(output / "01_cross_domain_adaptation_curve.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked cross-domain transfer evaluation")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--spm", required=True, type=Path)
    parser.add_argument("--i2em-train-requests", required=True, type=Path)
    parser.add_argument("--i2em-train-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--source-epochs", type=int, default=100)
    parser.add_argument("--adaptation-epochs", type=int, default=30)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output exists; use a new versioned directory")
    source = audit.prepare_table(args.source.resolve(), "SMAPVEX12", "topp_both")
    external = audit.prepare_table(args.external.resolve(), "external", "topp_both")
    source = source.loc[source.finite_model_row].reset_index(drop=True)
    external = external.loc[external.finite_model_row].reset_index(drop=True)
    if set(source.campaign_id.astype(str)) & set(external.campaign_id.astype(str)):
        raise ValueError("Source and external campaign_id values overlap")
    spm = pd.read_csv(args.spm.resolve())
    i2em = multi.load_paired(args.i2em_train_requests.resolve(), args.i2em_train_results.resolve())
    bundles, bundle_metadata = transfer.build_bundles(
        multi.features_from(spm),
        spm[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float),
        multi.features_from(i2em),
        i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float),
        args.spm_epochs,
        args.i2em_epochs,
        args.seed,
    )
    fitted = fit_source_models(source, bundles, args.source_epochs, args.seed + 10_000)
    zero = predict_source_locked(external, source, bundles, fitted)
    metric_rows = rows_for_predictions(external[base.OBSERVED_COLUMNS].to_numpy(dtype=float), zero, "zero_shot", 0.0, 0, 0, int(external.field_id.nunique()))
    prediction_rows: list[pd.DataFrame] = []
    groups = external.field_id.astype(str).to_numpy()
    unique_groups = np.unique(groups)
    split_rows: list[dict[str, object]] = []
    requested_fractions = sorted({float(value) for value in args.fractions.split(",")})
    if any(value <= 0 or value >= 1 for value in requested_fractions):
        raise ValueError("Few-shot fractions must lie strictly between zero and one")
    # With only a few external fields, several nominal percentages can round to
    # the same number of adaptation groups (for example 5% and 10% of 9 fields).
    # Collapse those duplicates and report the actually used field fraction.
    plan: dict[int, list[float]] = {}
    for requested in requested_fractions:
        count = max(
            1,
            min(
                len(unique_groups) - 1,
                int(np.ceil(requested * len(unique_groups))),
            ),
        )
        plan.setdefault(count, []).append(requested)
    for repeat in range(1, args.repeats + 1):
        shuffled = np.random.default_rng(args.seed + repeat * 100_000).permutation(unique_groups)
        for count, nominal_fractions in sorted(plan.items()):
            fraction = count / len(unique_groups)
            adapt_groups = set(shuffled[:count])
            adapt_mask = np.array([value in adapt_groups for value in groups])
            train_external = external.loc[adapt_mask].reset_index(drop=True)
            test_external = external.loc[~adapt_mask].reset_index(drop=True)
            predictions = adapt_predictions(train_external, test_external, source, bundles, fitted, args.adaptation_epochs, args.seed + repeat * 1_000_000 + count * 1_000)
            metric_rows.extend(rows_for_predictions(test_external[base.OBSERVED_COLUMNS].to_numpy(dtype=float), predictions, "few_shot", fraction, repeat, count, int(test_external.field_id.nunique())))
            for group in unique_groups:
                split_rows.append({"repeat": repeat, "adaptation_fraction": fraction, "requested_fractions": ",".join(f"{value:.2f}" for value in nominal_fractions), "adaptation_field_count": count, "total_external_fields": len(unique_groups), "field_id": group, "role": "adaptation" if group in adapt_groups else "test"})
    metrics = pd.DataFrame(metric_rows)
    summary_metrics = metrics.groupby(["stage", "adaptation_fraction", "method", "response"], as_index=False).agg(rmse_mean_db=("rmse_db", "mean"), rmse_std_db=("rmse_db", "std"), mae_mean_db=("mae_db", "mean"), bias_mean_db=("bias_db", "mean"), repeats=("repeat", "nunique"))
    observed = external[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
    comparisons = {}
    for comparator in ["source_mean", "source_ridge", "spm_only", "i2em_only"]:
        comparisons[f"spm_to_i2em_minus_{comparator}"] = base.grouped_bootstrap_joint_delta(observed, zero["spm_to_i2em"], zero[comparator], groups, args.bootstrap_iterations, args.seed + len(comparisons))
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics_by_split.csv", index=False)
    summary_metrics.to_csv(output / "metrics_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "external_field_split_assignments.csv", index=False)
    zero_frame = external[["campaign_id", "acquisition_date", "field_id", *base.FEATURE_NAMES, *base.OBSERVED_COLUMNS]].copy()
    for method, channels in zero.items():
        zero_frame[f"hh_{method}"] = channels[:, 0]
        zero_frame[f"vv_{method}"] = channels[:, 1]
    zero_frame.to_csv(output / "zero_shot_predictions.csv", index=False)
    save_plot(metrics, output)
    summary = {
        "research_question": "Does selective multi-fidelity physics pretraining transfer across campaign, region, and year, and how much target-domain data is required?",
        "protocol": {"source_only_model_selection": True, "zero_shot_first": True, "requested_adaptation_fractions": requested_fractions, "realized_adaptation_plan": [{"adaptation_fields": count, "actual_fraction": count / len(unique_groups), "requested_fractions_collapsed": values} for count, values in sorted(plan.items())], "adaptation_unit": "whole field_id groups", "external_test_unit": "untouched field_id groups", "repeats": args.repeats, "bootstrap_unit": "external field_id"},
        "portable_feature_contract": "Topp-derived real dielectric is recomputed from soil moisture in both source and external campaigns.",
        "physics_teacher_ablation": SCHEMES,
        "pretraining_metadata": bundle_metadata,
        "zero_shot_grouped_bootstrap": comparisons,
        "decision_rule": "Primary support requires SPM-to-I2EM to beat source mean and source-only statistical baselines in zero-shot external data, then retain an advantage under the realizable field-group adaptation counts.",
        "scope_limit": "External generalization is not established until a fully normalized independent table passes audit_external_domain.py without using its targets for model selection.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(summary_metrics.to_string(index=False), flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
