"""Evaluate direct physics plus adaptation-field channel offsets on SMEX02.

All target-domain offsets use labels from the original adaptation fields only.
Evaluation is restricted to the shared 138-row physics-valid cohort. Existing
neural predictions are reused without refitting. Physics baselines share only
common-valid adaptation rows; neural baselines share ALL original adaptation
rows. These are separate matched-information comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNELS = ["sigma0_hh_db", "sigma0_vv_db"]
METHODS = [
    "two_channel_mean",
    "two_channel_mean_all_adaptation",
    "i2em_actual_angle_plus_offset",
    "spm_actual_angle_plus_offset",
    "spm_only",
    "spm_to_i2em",
    "risk_spm_to_i2em",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    channel_rmse = np.sqrt(np.mean(error**2, axis=0))
    observed_diff = observed[:, 1] - observed[:, 0]
    predicted_diff = predicted[:, 1] - predicted[:, 0]
    observed_centered = observed_diff - observed_diff.mean()
    predicted_centered = predicted_diff - predicted_diff.mean()
    denominator = float(np.mean(observed_centered**2))
    centered_mse = float(np.mean((predicted_centered - observed_centered) ** 2))
    return {
        "n_test_rows": int(len(observed)),
        "hh_rmse_db": float(channel_rmse[0]),
        "vv_rmse_db": float(channel_rmse[1]),
        "mean_channel_rmse_db": float(channel_rmse.mean()),
        "pooled_channel_rmse_db": float(np.sqrt(np.mean(error**2))),
        "hh_bias_db": float(np.mean(error[:, 0])),
        "vv_bias_db": float(np.mean(error[:, 1])),
        "differential_centered_skill": float(1.0 - centered_mse / denominator),
    }


def conditional_field_reweighting(
    prediction_frame: pd.DataFrame,
    metrics_by_split: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fields = sorted(prediction_frame["field_id"].astype(str).unique())
    field_index = {field: index for index, field in enumerate(fields)}
    weights = rng.multinomial(
        len(fields), np.ones(len(fields)) / len(fields), size=iterations
    )
    interval_rows = []
    contrast_rows = []
    pairs = [
        ("i2em_actual_angle_plus_offset", "two_channel_mean"),
        ("spm_only", "two_channel_mean_all_adaptation"),
        ("spm_to_i2em", "spm_only"),
        ("risk_spm_to_i2em", "two_channel_mean_all_adaptation"),
        ("spm_actual_angle_plus_offset", "two_channel_mean"),
        ("spm_to_i2em", "two_channel_mean_all_adaptation"),
    ]

    for fraction, part in prediction_frame.groupby("requested_fraction", sort=True):
        split_ids = sorted(part["split_repeat"].unique())
        counts = []
        errors = {method: [] for method in METHODS}
        for split_id in split_ids:
            split = part.loc[part["split_repeat"].eq(split_id)]
            index = np.array([field_index[str(field)] for field in split["field_id"]])
            count = np.zeros(len(fields))
            np.add.at(count, index, 1)
            counts.append(count)
            observed = split[CHANNELS].to_numpy(float)
            for method in METHODS:
                predicted = split[[f"hh_{method}", f"vv_{method}"]].to_numpy(float)
                squared = (predicted - observed) ** 2
                accumulated = np.zeros((len(fields), 2))
                np.add.at(accumulated, index, squared)
                errors[method].append(accumulated)
        counts_array = np.asarray(counts)
        denominators = weights @ counts_array.T
        if (denominators <= 0).any():
            raise AssertionError("A field-weight draw produced an empty held-out cohort")
        bootstrap: dict[str, np.ndarray] = {}
        fraction_metrics = metrics_by_split.loc[
            metrics_by_split["requested_fraction"].eq(fraction)
        ]
        for method in METHODS:
            error_array = np.asarray(errors[method])
            bootstrap[method] = np.sqrt(
                np.einsum("bf,sfc->bsc", weights, error_array)
                / denominators[:, :, None]
            ).mean(axis=(1, 2))
            point = float(
                fraction_metrics.loc[
                    fraction_metrics["method"].eq(method), "mean_channel_rmse_db"
                ].mean()
            )
            low, high = np.quantile(bootstrap[method], [0.025, 0.975])
            interval_rows.append(
                {
                    "requested_fraction": fraction,
                    "method": method,
                    "mean_channel_rmse_db": point,
                    "conditional_ci_low_db": float(low),
                    "conditional_ci_high_db": float(high),
                    "valid_splits": len(split_ids),
                }
            )
        for first, second in pairs:
            delta = bootstrap[first] - bootstrap[second]
            first_point = float(
                fraction_metrics.loc[
                    fraction_metrics["method"].eq(first), "mean_channel_rmse_db"
                ].mean()
            )
            second_point = float(
                fraction_metrics.loc[
                    fraction_metrics["method"].eq(second), "mean_channel_rmse_db"
                ].mean()
            )
            low, high = np.quantile(delta, [0.025, 0.975])
            contrast_rows.append(
                {
                    "requested_fraction": fraction,
                    "first_method": first,
                    "second_method": second,
                    "delta_first_minus_second_db": first_point - second_point,
                    "conditional_ci_low_db": float(low),
                    "conditional_ci_high_db": float(high),
                    "probability_first_lower_rmse": float(np.mean(delta < 0)),
                }
            )
    return pd.DataFrame(interval_rows), pd.DataFrame(contrast_rows)


def save_figure(summary: pd.DataFrame, output: Path) -> None:
    """Two separate panels; never rank unequal adaptation information together."""
    groups = [
        ("physics", [("two_channel_mean", "Mean: common-valid adaptation rows"),
                     ("i2em_actual_angle_plus_offset", "I$^2$EM + channel offsets")]),
        ("neural", [("two_channel_mean_all_adaptation", "Mean: all original adaptation rows"),
                    ("spm_only", "SPM-pretrained"),
                    ("spm_to_i2em", "SPM to I$^2$EM"),
                    ("risk_spm_to_i2em", "Risk-shrunk")]),
    ]
    for name, methods in groups:
        fig, ax = plt.subplots(figsize=(5.3, 3.55), layout="constrained")
        for i,(method,label) in enumerate(methods):
            part=summary.loc[summary.method.eq(method)].sort_values("requested_fraction")
            values=part.mean_channel_rmse_db.to_numpy(float)
            low=part.conditional_ci_low_db.to_numpy(float)
            high=part.conditional_ci_high_db.to_numpy(float)
            ax.errorbar(np.arange(3),values,
                yerr=np.vstack([np.maximum(values-low,0),np.maximum(high-values,0)]),
                marker=["o","s","^","D"][i],linestyle=["--","-","-.",":"][i],
                capsize=2,label=label)
        ax.set_xticks(np.arange(3),["2 fields","3 fields","6 fields"])
        ax.set_xlabel("Planned adaptation fields")
        ax.set_ylabel("Mean HH/VV RMSE (dB)")
        ax.legend(fontsize=8,loc="upper right")
        ax.grid(axis="y",alpha=.25)
        fig.savefig(output/f"Fig_common_{name}.pdf",bbox_inches="tight")
        fig.savefig(output/f"Fig_common_{name}.png",dpi=400,bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    common_path = root / "reproducibility/results/common_cohort/predictions.csv"
    split_path = root / "reproducibility/results/cross_domain/field_split_assignments.csv"
    heldout_path = root / "reproducibility/results/cross_domain/heldout_predictions.csv"
    shrinkage_path = root / "reproducibility/results/advisor_final/physics_offset_control/risk_selected_weights.csv"
    target_path = root / "reproducibility/data/target/smex02_field_day_model_ready.csv"
    input_paths = [common_path, split_path, heldout_path, shrinkage_path, target_path]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    common = pd.read_csv(common_path, dtype={"field_id": str})
    assignments = pd.read_csv(split_path, dtype={"field_id": str})
    heldout = pd.read_csv(heldout_path, dtype={"field_id": str})
    target = pd.read_csv(target_path, dtype={"field_id": str})
    selected_weights = pd.read_csv(shrinkage_path).sort_values("repeat")
    keys = ["field_id", "acquisition_date"]

    metric_rows = []
    prediction_frames = []
    audit_rows = []
    for (split_repeat, fraction), assignment in assignments.groupby(
        ["split_repeat", "requested_fraction"], sort=True
    ):
        adaptation_fields = set(
            assignment.loc[assignment["role"].eq("adaptation"), "field_id"].astype(str)
        )
        test_fields = set(
            assignment.loc[assignment["role"].eq("test"), "field_id"].astype(str)
        )
        if adaptation_fields & test_fields:
            raise AssertionError("Adaptation/test field overlap")
        adaptation = common.loc[common["field_id"].isin(adaptation_fields)].copy()
        all_adaptation = target.loc[target["field_id"].isin(adaptation_fields)].copy()
        if all_adaptation.empty:
            raise AssertionError("Original adaptation set is empty")
        test_reference = common.loc[common["field_id"].isin(test_fields)].copy()
        if adaptation.empty:
            audit_rows.append(
                {
                    "split_repeat": split_repeat,
                    "requested_fraction": fraction,
                    "status": "excluded_no_common_valid_adaptation_rows",
                    "planned_adaptation_fields": len(adaptation_fields),
                    "usable_adaptation_fields": 0,
                    "usable_adaptation_rows": 0,
                    "test_rows": len(test_reference),
                }
            )
            continue
        split_predictions = heldout.loc[
            heldout["split_repeat"].eq(split_repeat)
            & np.isclose(heldout["requested_fraction"], fraction)
        ].copy()
        merged = test_reference.merge(
            split_predictions,
            on=keys,
            how="inner",
            suffixes=("_reference", ""),
            validate="one_to_one",
        )
        if len(merged) != len(test_reference):
            raise AssertionError("Held-out predictions do not cover the shared test cohort")
        for column in CHANNELS:
            if not np.allclose(merged[f"{column}_reference"], merged[column]):
                raise AssertionError(f"Observed values differ after merge: {column}")
        observed = merged[CHANNELS].to_numpy(float)
        adaptation_observed = adaptation[CHANNELS].to_numpy(float)
        channel_mean = adaptation_observed.mean(axis=0)
        channel_mean_all = all_adaptation[CHANNELS].to_numpy(float).mean(axis=0)
        i2em_offset = (
            adaptation_observed
            - adaptation[["i2em_hh_db", "i2em_vv_db"]].to_numpy(float)
        ).mean(axis=0)
        spm_offset = (
            adaptation_observed
            - adaptation[["spm_hh_db", "spm_vv_db"]].to_numpy(float)
        ).mean(axis=0)
        predictions = {
            "two_channel_mean": np.broadcast_to(channel_mean, observed.shape).copy(),
            "two_channel_mean_all_adaptation": np.broadcast_to(channel_mean_all, observed.shape).copy(),
            "i2em_actual_angle_plus_offset": (
                merged[["i2em_hh_db", "i2em_vv_db"]].to_numpy(float) + i2em_offset
            ),
            "spm_actual_angle_plus_offset": (
                merged[["spm_hh_db", "spm_vv_db"]].to_numpy(float) + spm_offset
            ),
            "spm_only": merged[["hh_spm_only", "vv_spm_only"]].to_numpy(float),
            "spm_to_i2em": merged[
                ["hh_spm_to_i2em", "vv_spm_to_i2em"]
            ].to_numpy(float),
            "risk_spm_to_i2em": merged[
                ["hh_risk_spm_to_i2em", "vv_risk_spm_to_i2em"]
            ].to_numpy(float),
        }
        actual_fraction = float(assignment["actual_fraction"].iloc[0])
        usable_fields = int(adaptation["field_id"].nunique())
        usable_rows = int(len(adaptation))
        audit_rows.append(
            {
                "split_repeat": split_repeat,
                "requested_fraction": fraction,
                "actual_fraction": actual_fraction,
                "status": "evaluated",
                "planned_adaptation_fields": len(adaptation_fields),
                "usable_adaptation_fields": usable_fields,
                "usable_adaptation_rows": usable_rows,
                "all_original_adaptation_rows": len(all_adaptation),
                "test_rows": len(merged),
            }
        )
        saved = merged[[*keys, *CHANNELS]].copy()
        saved.insert(0, "actual_fraction", actual_fraction)
        saved.insert(0, "requested_fraction", fraction)
        saved.insert(0, "split_repeat", split_repeat)
        saved["usable_adaptation_fields"] = usable_fields
        saved["usable_adaptation_rows"] = usable_rows
        saved["all_original_adaptation_rows"] = len(all_adaptation)
        # Check that neural common-response predictions used the full adaptation mean.
        neural_common=(predictions["spm_only"][:,0]+predictions["spm_only"][:,1])/2
        if not np.allclose(neural_common,channel_mean_all.mean(),atol=1e-10,rtol=0):
            raise AssertionError("Frozen neural adaptation information does not match")
        for method, predicted in predictions.items():
            values = score(observed, predicted)
            metric_rows.append(
                {
                    "split_repeat": split_repeat,
                    "requested_fraction": fraction,
                    "actual_fraction": actual_fraction,
                    "planned_adaptation_fields": len(adaptation_fields),
                    "usable_adaptation_fields": usable_fields,
                    "usable_adaptation_rows": usable_rows,
                "all_original_adaptation_rows": len(all_adaptation),
                    "method": method,
                    **values,
                }
            )
            saved[f"hh_{method}"] = predicted[:, 0]
            saved[f"vv_{method}"] = predicted[:, 1]
        prediction_frames.append(saved)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    audit = pd.DataFrame(audit_rows)
    summary, contrasts = conditional_field_reweighting(
        predictions,
        metrics,
        args.bootstrap_iterations,
        args.seed + 8_000_000,
    )
    adaptation_summary = (
        audit.loc[audit["status"].eq("evaluated")]
        .groupby(["requested_fraction", "actual_fraction"], as_index=False)
        .agg(
            valid_splits=("split_repeat", "nunique"),
            usable_adaptation_fields_mean=("usable_adaptation_fields", "mean"),
            usable_adaptation_fields_min=("usable_adaptation_fields", "min"),
            usable_adaptation_fields_max=("usable_adaptation_fields", "max"),
            usable_adaptation_rows_mean=("usable_adaptation_rows", "mean"),
            usable_adaptation_rows_min=("usable_adaptation_rows", "min"),
            usable_adaptation_rows_max=("usable_adaptation_rows", "max"),
            common_test_rows_mean=("test_rows", "mean"),
            all_original_adaptation_rows_mean=("all_original_adaptation_rows", "mean"),
        )
    )

    predictions.to_csv(output / "heldout_common_predictions.csv", index=False)
    metrics.to_csv(output / "metrics_by_split.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    audit.to_csv(output / "split_audit.csv", index=False)
    adaptation_summary.to_csv(output / "adaptation_sample_summary.csv", index=False)
    selected_weights.to_csv(output / "risk_selected_weights.csv", index=False)
    save_figure(summary, output)

    protocol = {
        "status": "COMPLETE",
        "analysis_role": "post-hoc matched-information comparisons on the shared physics-valid test cohort",
        "comparison_groups": {
            "physics": "physical offsets and common-valid-row mean use identical common-valid adaptation labels",
            "neural": "frozen neural predictions and all-original-row mean use identical original adaptation labels"
        },
        "training_performed": False,
        "model_selection_performed": False,
        "cohort": {
            "rows": int(common.shape[0]),
            "fields": int(common.field_id.nunique()),
        },
        "offset_definition": (
            "per-channel mean of observed minus physical prediction over common-valid "
            "rows in the original adaptation fields"
        ),
        "heldout_labels_used_for_calibration": False,
        "uncertainty": {
            "method": "shared field-identity conditional reweighting across fixed splits",
            "iterations": args.bootstrap_iterations,
            "seed": args.seed + 8_000_000,
        },
        "risk_selected_weights_by_repeat": selected_weights.to_dict("records"),
        "inputs": {str(path.relative_to(root)): sha256(path) for path in input_paths},
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(adaptation_summary.to_string(index=False))
    print(summary.to_string(index=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
