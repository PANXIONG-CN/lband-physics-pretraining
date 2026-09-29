"""Pre-registered whole-field few-shot adaptation on SMEX02.

The zero-shot result must already exist.  This script does not retune model
families on SMEX02.  It uses 2, 3, and 6 whole adaptation fields (for the
nominal 5%, 10%, and 20% settings), estimates one common-component offset,
and fine-tunes the locked differential heads.  Risk shrinkage remains fixed by
SMAPVEX12-only grouped validation.
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
import evaluate_multifidelity_pretraining as multi  # noqa: E402
import evaluate_multifidelity_real_transfer as transfer  # noqa: E402


METHODS = [
    "zero_source_mean",
    "common_offset_source_mean",
    "common_offset_source_ridge",
    "scratch",
    "spm_only",
    "i2em_only",
    "spm_to_i2em",
    "risk_spm_to_i2em",
]
RESPONSES = ["HH", "VV", "common", "differential"]


def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = frame[list(base.FEATURE_NAMES)].to_numpy(dtype=float)
    channels = frame[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
    return features, channels, base.to_components(channels)


def regression(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
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


def response_dict(channels: np.ndarray) -> dict[str, np.ndarray]:
    components = base.to_components(channels)
    return {
        "HH": channels[:, 0],
        "VV": channels[:, 1],
        "common": components[:, 0],
        "differential": components[:, 1],
    }


def source_ensemble(
    source: pd.DataFrame,
    spm: pd.DataFrame,
    i2em: pd.DataFrame,
    repeats: int,
    weights: list[float],
    spm_epochs: int,
    i2em_epochs: int,
    source_epochs: int,
    inner_folds: int,
    seed: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    features, channels, components = arrays(source)
    common = components[:, 0]
    differential = components[:, 1]
    groups = source.field_id.astype(str).to_numpy()
    all_rows = np.arange(len(source), dtype=int)
    members: list[dict[str, object]] = []
    tuning_frames = []
    for repeat in range(1, repeats + 1):
        repeat_seed = seed + (repeat - 1) * 100_000
        bundles, metadata = transfer.build_bundles(
            multi.features_from(spm),
            spm[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float),
            multi.features_from(i2em),
            i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float),
            spm_epochs,
            i2em_epochs,
            repeat_seed,
        )
        scratch = base.scratch_single(
            bundles["spm_only"], features, differential, source_epochs, repeat_seed + 10_007
        )
        models = {
            scheme: base.fine_tune_single(
                bundles[scheme],
                features,
                differential,
                source_epochs,
                repeat_seed + 10_100 + index,
            )
            for index, scheme in enumerate(["spm_only", "i2em_only", "spm_to_i2em"])
        }
        weight, records = base.tune_differential_shrinkage(
            bundles["spm_to_i2em"],
            features,
            differential,
            groups,
            all_rows,
            weights,
            inner_folds,
            source_epochs,
            repeat_seed + 500_000,
        )
        tuning = pd.DataFrame(records)
        tuning.insert(0, "selected_weight", weight)
        tuning.insert(0, "repeat", repeat)
        tuning_frames.append(tuning)
        members.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "bundles": bundles,
                "scratch": scratch,
                "models": models,
                "channel_ridge": Ridge(alpha=1.0).fit(features, channels),
                "common_mean": float(np.mean(common)),
                "differential_mean": float(np.mean(differential)),
                "shrinkage_weight": weight,
                "pretraining": metadata,
            }
        )
        print(f"Locked source member {repeat}/{repeats}", flush=True)
    return members, pd.concat(tuning_frames, ignore_index=True)


def adapt_member(
    member: dict[str, object],
    adaptation: pd.DataFrame,
    test: pd.DataFrame,
    adaptation_epochs: int,
    split_seed: int,
) -> dict[str, np.ndarray]:
    adapt_features, adapt_channels, adapt_components = arrays(adaptation)
    test_features, _, _ = arrays(test)
    common_mean = float(member["common_mean"])
    differential_mean = float(member["differential_mean"])
    adapted_common = np.full(len(test), float(np.mean(adapt_components[:, 0])))
    source_common = np.full(len(test), common_mean)
    source_diff = np.full(len(test), differential_mean)
    predictions: dict[str, np.ndarray] = {
        "zero_source_mean": base.from_components(source_common, source_diff),
        "common_offset_source_mean": base.from_components(adapted_common, source_diff),
    }

    ridge = member["channel_ridge"]
    ridge_adapt_components = base.to_components(
        np.asarray(ridge.predict(adapt_features), dtype=float)
    )
    ridge_test_components = base.to_components(
        np.asarray(ridge.predict(test_features), dtype=float)
    )
    common_offset = float(np.mean(adapt_components[:, 0] - ridge_adapt_components[:, 0]))
    predictions["common_offset_source_ridge"] = base.from_components(
        ridge_test_components[:, 0] + common_offset,
        ridge_test_components[:, 1],
    )

    bundles = member["bundles"]
    source_models = {"scratch": member["scratch"], **member["models"]}
    raw_differentials: dict[str, np.ndarray] = {}
    for index, scheme in enumerate(["scratch", "spm_only", "i2em_only", "spm_to_i2em"]):
        bundle_name = "spm_only" if scheme == "scratch" else scheme
        bundle = bundles[bundle_name]
        source_bundle = base.SingleOutputBundle(
            model=source_models[scheme],
            feature_scaler=bundle.feature_scaler,
            target_scaler=bundle.target_scaler,
            metadata={"stage": "source_locked_then_external_few_shot"},
        )
        adapted_model = base.fine_tune_single(
            source_bundle,
            adapt_features,
            adapt_components[:, 1],
            adaptation_epochs,
            split_seed + index * 100,
        )
        predicted_diff = base.predict_single(source_bundle, adapted_model, test_features)
        raw_differentials[scheme] = predicted_diff
        predictions[scheme] = base.from_components(adapted_common, predicted_diff)

    risk_diff = base.shrink_prediction_to_mean(
        raw_differentials["spm_to_i2em"],
        differential_mean,
        float(member["shrinkage_weight"]),
    )
    predictions["risk_spm_to_i2em"] = base.from_components(adapted_common, risk_diff)
    return predictions


def ensemble(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        method: np.mean([member[method] for member in values], axis=0)
        for method in METHODS
    }


def metric_rows(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    split_repeat: int,
    requested_fraction: float,
    actual_fraction: float,
    adaptation_fields: int,
    test_fields: int,
) -> list[dict[str, object]]:
    refs = response_dict(observed)
    rows = []
    for method in METHODS:
        preds = response_dict(predictions[method])
        for response in RESPONSES:
            rows.append(
                {
                    "split_repeat": split_repeat,
                    "requested_fraction": requested_fraction,
                    "actual_fraction": actual_fraction,
                    "adaptation_fields": adaptation_fields,
                    "test_fields": test_fields,
                    "method": method,
                    "response": response,
                    "n": len(observed),
                    **regression(refs[response], preds[response]),
                }
            )
    return rows


def mean_channel_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((predicted - observed) ** 2, axis=0))))


def hierarchical_bootstrap(
    records: pd.DataFrame,
    first: str,
    second: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    split_ids = np.sort(records.split_repeat.unique())
    # Pre-aggregate counts and squared errors by held-out field.  Resampling
    # these sufficient statistics is exactly equivalent to concatenating all
    # rows for every bootstrap draw, but avoids millions of DataFrame slices.
    sufficient: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_id in split_ids:
        frame = records[records.split_repeat == split_id]
        rows = []
        for _, field in frame.groupby(frame.field_id.astype(str), sort=False):
            observed = field[["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(dtype=float)
            first_values = field[[f"hh_{first}", f"vv_{first}"]].to_numpy(dtype=float)
            second_values = field[[f"hh_{second}", f"vv_{second}"]].to_numpy(dtype=float)
            rows.append(
                (
                    len(field),
                    np.sum((first_values - observed) ** 2, axis=0),
                    np.sum((second_values - observed) ** 2, axis=0),
                )
            )
        sufficient[int(split_id)] = (
            np.asarray([row[0] for row in rows], dtype=float),
            np.vstack([row[1] for row in rows]),
            np.vstack([row[2] for row in rows]),
        )

    def sufficient_delta(
        item: tuple[np.ndarray, np.ndarray, np.ndarray],
        rng: np.random.Generator | None = None,
    ) -> float:
        counts, first_sse, second_sse = item
        indices = (
            np.arange(len(counts))
            if rng is None
            else rng.integers(0, len(counts), size=len(counts))
        )
        denominator = float(np.sum(counts[indices]))
        first_rmse = np.sqrt(np.sum(first_sse[indices], axis=0) / denominator)
        second_rmse = np.sqrt(np.sum(second_sse[indices], axis=0) / denominator)
        return float(np.mean(first_rmse) - np.mean(second_rmse))

    point = float(
        np.mean([sufficient_delta(sufficient[int(value)]) for value in split_ids])
    )
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations)
    for iteration in range(iterations):
        sampled_splits = rng.choice(split_ids, len(split_ids), replace=True)
        boot[iteration] = float(
            np.mean(
                [sufficient_delta(sufficient[int(value)], rng) for value in sampled_splits]
            )
        )
    return {
        "mean_split_rmse_delta_db": point,
        "ci_2_5_percent_db": float(np.quantile(boot, 0.025)),
        "ci_97_5_percent_db": float(np.quantile(boot, 0.975)),
        "probability_first_better": float(np.mean(boot < 0)),
    }


def save_plot(summary: pd.DataFrame, output: Path) -> None:
    part = summary[
        summary.response.isin(["HH", "VV"])
    ].groupby(["actual_fraction", "method"], as_index=False).rmse_mean_db.mean()
    fig, ax = plt.subplots(figsize=(10.5, 5.7), constrained_layout=True)
    for method in METHODS:
        curve = part[part.method == method].sort_values("actual_fraction")
        if not curve.empty:
            ax.plot(
                curve.actual_fraction * 100,
                curve.rmse_mean_db,
                marker="o",
                label=method,
            )
    ax.set(
        xlabel="Whole SMEX02 fields used for adaptation (%)",
        ylabel="Mean HH/VV RMSE on untouched fields (dB)",
        title="Pre-registered SMEX02 few-shot adaptation",
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(output / "01_few_shot_learning_curve.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-registered SMEX02 whole-field few-shot evaluation")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--spm", required=True, type=Path)
    parser.add_argument("--i2em-requests", required=True, type=Path)
    parser.add_argument("--i2em-results", required=True, type=Path)
    parser.add_argument("--zero-shot-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--split-repeats", type=int, default=20)
    parser.add_argument("--model-repeats", type=int, default=5)
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--source-epochs", type=int, default=120)
    parser.add_argument("--adaptation-epochs", type=int, default=30)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--shrinkage-weights", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    zero_summary = json.loads(args.zero_shot_summary.read_text(encoding="utf-8"))
    if zero_summary.get("status") != "PROSPECTIVE_ZERO_SHOT_COMPLETE":
        raise ValueError("The frozen prospective zero-shot stage is not complete")
    weights = sorted({float(value) for value in args.shrinkage_weights.split(",")})
    if weights != [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("Shrinkage candidates differ from the frozen zero-shot contract")

    source = audit.prepare_table(args.source.resolve(), "SMAPVEX12", "topp_both")
    external = audit.prepare_table(args.external.resolve(), "SMEX02", "topp_both")
    source = source.loc[source.finite_model_row].sort_values(["field_id", "acquisition_date"]).reset_index(drop=True)
    external = external.loc[external.finite_model_row].sort_values(["field_id", "acquisition_date"]).reset_index(drop=True)
    spm = pd.read_csv(args.spm.resolve())
    i2em = multi.load_paired(args.i2em_requests.resolve(), args.i2em_results.resolve())
    members, tuning = source_ensemble(
        source,
        spm,
        i2em,
        args.model_repeats,
        weights,
        args.spm_epochs,
        args.i2em_epochs,
        args.source_epochs,
        args.inner_folds,
        args.seed,
    )

    fields = np.sort(external.field_id.astype(str).unique())
    requested = sorted({float(value) for value in args.fractions.split(",")})
    counts = {fraction: max(1, min(len(fields) - 1, int(np.ceil(fraction * len(fields))))) for fraction in requested}
    expected = {0.05: 2, 0.10: 3, 0.20: 6}
    if len(fields) == 30 and counts != expected:
        raise AssertionError(f"Unexpected 30-field adaptation plan: {counts}")

    metric_output = []
    prediction_output = []
    assignment_output = []
    for split_repeat in range(1, args.split_repeats + 1):
        shuffled = np.random.default_rng(args.seed + split_repeat * 100_000).permutation(fields)
        for requested_fraction in requested:
            count = counts[requested_fraction]
            adaptation_fields = set(shuffled[:count])
            adapt_mask = external.field_id.astype(str).isin(adaptation_fields).to_numpy()
            adaptation = external.loc[adapt_mask].reset_index(drop=True)
            test = external.loc[~adapt_mask].reset_index(drop=True)
            per_member = []
            for member in members:
                per_member.append(
                    adapt_member(
                        member,
                        adaptation,
                        test,
                        args.adaptation_epochs,
                        args.seed + split_repeat * 1_000_000 + count * 10_000 + int(member["repeat"]) * 100,
                    )
                )
            predictions = ensemble(per_member)
            observed = test[base.OBSERVED_COLUMNS].to_numpy(dtype=float)
            for values in predictions.values():
                base.verify_error_identity(observed, values)
            actual_fraction = count / len(fields)
            metric_output.extend(
                metric_rows(
                    observed,
                    predictions,
                    split_repeat,
                    requested_fraction,
                    actual_fraction,
                    count,
                    int(test.field_id.nunique()),
                )
            )
            frame = test[["campaign_id", "acquisition_date", "field_id", *base.OBSERVED_COLUMNS]].copy()
            frame.insert(0, "adaptation_fields", ",".join(sorted(adaptation_fields)))
            frame.insert(0, "actual_fraction", actual_fraction)
            frame.insert(0, "requested_fraction", requested_fraction)
            frame.insert(0, "split_repeat", split_repeat)
            for method in METHODS:
                frame[f"hh_{method}"] = predictions[method][:, 0]
                frame[f"vv_{method}"] = predictions[method][:, 1]
            prediction_output.append(frame)
            for field in fields:
                assignment_output.append(
                    {
                        "split_repeat": split_repeat,
                        "requested_fraction": requested_fraction,
                        "actual_fraction": actual_fraction,
                        "field_id": field,
                        "role": "adaptation" if field in adaptation_fields else "test",
                    }
                )
        print(f"Completed field split {split_repeat}/{args.split_repeats}", flush=True)

    metrics = pd.DataFrame(metric_output)
    predictions = pd.concat(prediction_output, ignore_index=True)
    assignments = pd.DataFrame(assignment_output)
    summary_metrics = metrics.groupby(
        ["requested_fraction", "actual_fraction", "adaptation_fields", "test_fields", "method", "response"],
        as_index=False,
    ).agg(
        rmse_mean_db=("rmse_db", "mean"),
        rmse_std_db=("rmse_db", "std"),
        mae_mean_db=("mae_db", "mean"),
        bias_mean_db=("bias_db", "mean"),
        split_repeats=("split_repeat", "nunique"),
    )

    comparisons: dict[str, dict[str, object]] = {}
    for fraction_index, requested_fraction in enumerate(requested):
        subset = predictions[np.isclose(predictions.requested_fraction, requested_fraction)].copy()
        key = f"requested_{requested_fraction:.2f}_actual_{counts[requested_fraction] / len(fields):.6f}"
        comparisons[key] = {}
        for comparator_index, comparator in enumerate([
            "zero_source_mean",
            "common_offset_source_mean",
            "common_offset_source_ridge",
            "scratch",
            "spm_only",
            "i2em_only",
            "spm_to_i2em",
        ]):
            comparisons[key][f"risk_minus_{comparator}"] = hierarchical_bootstrap(
                subset,
                "risk_spm_to_i2em",
                comparator,
                args.bootstrap_iterations,
                args.seed + 2_000_000 + fraction_index * 10_000 + comparator_index * 100,
            )

    output.mkdir(parents=True)
    metrics.to_csv(output / "metrics_by_split.csv", index=False)
    summary_metrics.to_csv(output / "metrics_summary.csv", index=False)
    predictions.to_csv(output / "heldout_predictions.csv", index=False)
    assignments.to_csv(output / "field_split_assignments.csv", index=False)
    tuning.to_csv(output / "source_only_shrinkage_tuning.csv", index=False)
    save_plot(summary_metrics, output)

    curve = summary_metrics[
        summary_metrics.response.isin(["HH", "VV"])
    ].groupby(["requested_fraction", "actual_fraction", "method"], as_index=False).rmse_mean_db.mean()
    summary = {
        "status": "PREREGISTERED_FEW_SHOT_COMPLETE",
        "research_question": "Can a small number of complete SMEX02 fields calibrate campaign shift while preserving the frozen physics-transfer rule?",
        "protocol": {
            "adaptation_unit": "whole field_id",
            "split_repeats": args.split_repeats,
            "model_repeats": args.model_repeats,
            "common_adaptation": "one additive offset estimated from adaptation fields only",
            "differential_adaptation": f"30-epoch fine-tuning of locked source differential heads; risk shrinkage remains source-only",
            "test_unit": "all rows from untouched fields",
            "uncertainty": "hierarchical bootstrap over split repeats and held-out field blocks",
        },
        "adaptation_plan": [
            {
                "requested_fraction": fraction,
                "adaptation_fields": counts[fraction],
                "actual_fraction": counts[fraction] / len(fields),
                "test_fields": len(fields) - counts[fraction],
            }
            for fraction in requested
        ],
        "mean_hh_vv_rmse_db": [
            {
                "requested_fraction": float(row.requested_fraction),
                "actual_fraction": float(row.actual_fraction),
                "method": str(row.method),
                "rmse_db": float(row.rmse_mean_db),
            }
            for row in curve.sort_values(["requested_fraction", "rmse_mean_db"]).itertuples(index=False)
        ],
        "paired_hierarchical_bootstrap": comparisons,
        "guardrail": "This stage estimates adaptation performance only. It does not retroactively change the completed zero-shot verdict.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(curve.to_string(index=False), flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
