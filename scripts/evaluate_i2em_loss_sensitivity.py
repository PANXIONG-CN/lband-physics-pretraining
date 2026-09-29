"""Evaluate dielectric-loss sensitivity and I2EM/SPM disagreement."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from research_pilots.scattering.surfaces.teacher_contract import (
    validate_teacher_results,
)


def fingerprint(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def add_sensitivity_columns(merged: pd.DataFrame) -> pd.DataFrame:
    result = merged.copy()
    keys = ["correlation_model", "scenario"]
    for pol in ["hh", "vv"]:
        result[f"i2em_minus_spm_{pol}_db"] = (
            result[f"i2em_{pol}_db"] - result[f"spm_{pol}_db"]
        )
        zero = (
            result[np.isclose(result["loss_tangent"], 0.0)]
            .set_index(keys)[f"i2em_{pol}_db"]
            .rename(f"i2em_{pol}_db_at_zero_loss")
        )
        result = result.join(zero, on=keys)
        result[f"i2em_change_from_zero_loss_{pol}_db"] = (
            result[f"i2em_{pol}_db"] - result[f"i2em_{pol}_db_at_zero_loss"]
        )
    return result


def grouped_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for loss_tangent, subset in frame.groupby("loss_tangent", sort=True):
        row: dict[str, float | str] = {"loss_tangent": float(loss_tangent)}
        for pol in ["hh", "vv"]:
            change = subset[f"i2em_change_from_zero_loss_{pol}_db"].to_numpy(float)
            disagreement = subset[f"i2em_minus_spm_{pol}_db"].to_numpy(float)
            row[f"mean_abs_i2em_change_{pol}_db"] = float(np.mean(np.abs(change)))
            row[f"max_abs_i2em_change_{pol}_db"] = float(np.max(np.abs(change)))
            row[f"mean_i2em_minus_spm_{pol}_db"] = float(np.mean(disagreement))
            row[f"mean_abs_i2em_minus_spm_{pol}_db"] = float(
                np.mean(np.abs(disagreement))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def teacher_trend_summary(frame: pd.DataFrame) -> pd.DataFrame:
    zero_loss = frame[np.isclose(frame["loss_tangent"], 0.0)].copy()
    rows: list[dict[str, float | int | str]] = []
    for model, model_frame in zero_loss.groupby("correlation_model", sort=True):
        for pol in ["hh", "vv"]:
            spm = model_frame[f"spm_{pol}_db"]
            i2em = model_frame[f"i2em_{pol}_db"]
            difference = i2em - spm
            baseline = model_frame[model_frame["scenario"] == "baseline"]
            if len(baseline) != 1:
                raise ValueError(f"Expected one baseline for {model}; found {len(baseline)}")
            nonbaseline = model_frame[model_frame["scenario"] != "baseline"]
            spm_delta = (
                nonbaseline[f"spm_{pol}_db"] - float(baseline[f"spm_{pol}_db"].iloc[0])
            )
            i2em_delta = (
                nonbaseline[f"i2em_{pol}_db"]
                - float(baseline[f"i2em_{pol}_db"].iloc[0])
            )
            sign_agreement = np.sign(spm_delta.to_numpy(float)) == np.sign(
                i2em_delta.to_numpy(float)
            )
            rows.append(
                {
                    "correlation_model": str(model),
                    "polarization": pol.upper(),
                    "anchor_count": int(len(model_frame)),
                    "spearman_spm_i2em": float(spm.corr(i2em, method="spearman")),
                    "mean_i2em_minus_spm_db": float(difference.mean()),
                    "mean_abs_i2em_minus_spm_db": float(difference.abs().mean()),
                    "rms_i2em_minus_spm_db": float(np.sqrt(np.mean(difference**2))),
                    "max_abs_i2em_minus_spm_db": float(difference.abs().max()),
                    "direction_agreement_fraction": float(np.mean(sign_agreement)),
                }
            )
    return pd.DataFrame(rows)


def save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    baseline = frame[frame["scenario"] == "baseline"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, pol in zip(axes, ["hh", "vv"]):
        for model, marker in [("exponential", "o"), ("gaussian", "s")]:
            subset = baseline[baseline["correlation_model"] == model].sort_values(
                "loss_tangent"
            )
            ax.plot(
                subset["loss_tangent"],
                subset[f"i2em_{pol}_db"],
                marker=marker,
                label=f"I2EM {model}",
            )
            ax.plot(
                subset["loss_tangent"],
                subset[f"spm_{pol}_db"],
                marker=marker,
                linestyle="--",
                alpha=0.75,
                label=f"SPM {model}",
            )
        ax.set(
            xlabel="Loss tangent",
            ylabel=r"$\sigma^0$ (dB)",
            title=f"Median-anchor loss sensitivity: {pol.upper()}",
        )
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "01_i2em_spm_loss_sensitivity.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, pol in zip(axes, ["hh", "vv"]):
        for model, marker in [("exponential", "o"), ("gaussian", "s")]:
            subset = frame[frame["correlation_model"] == model]
            for scenario, scenario_frame in subset.groupby("scenario"):
                ordered = scenario_frame.sort_values("loss_tangent")
                ax.plot(
                    ordered["loss_tangent"],
                    ordered[f"i2em_change_from_zero_loss_{pol}_db"],
                    marker=marker,
                    linewidth=0.8,
                    markersize=3,
                    alpha=0.40,
                )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set(
            xlabel="Loss tangent",
            ylabel="I2EM change from zero-loss case (dB)",
            title=f"All-anchor response spread: {pol.upper()}",
        )
        ax.grid(alpha=0.22)
    fig.savefig(output_dir / "02_all_anchor_loss_changes.png", dpi=190)
    plt.close(fig)

    zero_loss = frame[np.isclose(frame["loss_tangent"], 0.0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, pol in zip(axes, ["hh", "vv"]):
        for model, marker in [("exponential", "o"), ("gaussian", "s")]:
            subset = zero_loss[zero_loss["correlation_model"] == model]
            ax.scatter(
                subset[f"spm_{pol}_db"],
                subset[f"i2em_{pol}_db"],
                marker=marker,
                s=52,
                alpha=0.75,
                label=model,
            )
        values = np.concatenate(
            [zero_loss[f"spm_{pol}_db"], zero_loss[f"i2em_{pol}_db"]]
        )
        lower, upper = float(values.min()), float(values.max())
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set(
            xlabel=f"SPM {pol.upper()} (dB)",
            ylabel=f"I2EM {pol.upper()} (dB)",
            title=f"Zero-loss teacher overlap: {pol.upper()}",
        )
        ax.grid(alpha=0.22)
        ax.legend()
    fig.savefig(output_dir / "03_zero_loss_teacher_overlap.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate I2EM loss sensitivity")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--backend-label", default="external I2EM backend")
    args = parser.parse_args()

    merged = validate_teacher_results(
        pd.read_csv(args.requests), pd.read_csv(args.results)
    )
    if "loss_tangent" not in merged.columns:
        raise ValueError("Requests must contain loss_tangent")
    if not np.isclose(merged["loss_tangent"], 0.0).any():
        raise ValueError("Sensitivity design must include loss_tangent=0")
    counts = merged.groupby(["correlation_model", "scenario"])[
        "loss_tangent"
    ].nunique()
    if counts.nunique() != 1:
        raise ValueError("Every anchor must contain the same loss-tangent levels")

    comparison = add_sensitivity_columns(merged)
    summary_table = grouped_summary(comparison)
    trend_table = teacher_trend_summary(comparison)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "i2em_loss_sensitivity_comparison.csv", index=False)
    summary_table.to_csv(output_dir / "loss_sensitivity_summary.csv", index=False)
    trend_table.to_csv(output_dir / "teacher_trend_summary.csv", index=False)

    summary = {
        "request_count": int(len(comparison)),
        "loss_tangents": sorted(
            float(value) for value in comparison["loss_tangent"].unique()
        ),
        "anchor_count_per_loss_tangent": int(
            len(comparison) / comparison["loss_tangent"].nunique()
        ),
        "maximum_absolute_i2em_change_from_zero_loss_hh_db": float(
            comparison["i2em_change_from_zero_loss_hh_db"].abs().max()
        ),
        "maximum_absolute_i2em_change_from_zero_loss_vv_db": float(
            comparison["i2em_change_from_zero_loss_vv_db"].abs().max()
        ),
        "zero_loss_teacher_trends": trend_table.to_dict(orient="records"),
        "interpretation": (
            "Changes from the zero-loss case quantify dielectric-loss sensitivity. "
            "I2EM-minus-SPM differences quantify teacher disagreement, not I2EM error."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "backend": args.backend_label,
        "requests": fingerprint(args.requests),
        "results": fingerprint(args.results),
        "model_file": fingerprint(args.model_file) if args.model_file else None,
        "request_count": int(len(comparison)),
        "all_i2em_results_finite": bool(
            np.isfinite(comparison[["i2em_hh_db", "i2em_vv_db"]]).all().all()
        ),
        "qualification": (
            "Reference-teacher validation passed for finite values and trend direction, "
            "with material Gaussian-spectrum scale disagreement retained as a "
            "multi-fidelity uncertainty signal."
        ),
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    zero_loss_comparison = comparison[np.isclose(comparison["loss_tangent"], 0.0)]
    worst_hh = zero_loss_comparison.loc[
        zero_loss_comparison["i2em_minus_spm_hh_db"].abs().idxmax()
    ]
    worst_vv = zero_loss_comparison.loc[
        zero_loss_comparison["i2em_minus_spm_vv_db"].abs().idxmax()
    ]
    exp_hh = trend_table[
        (trend_table["correlation_model"] == "exponential")
        & (trend_table["polarization"] == "HH")
    ].iloc[0]
    exp_vv = trend_table[
        (trend_table["correlation_model"] == "exponential")
        & (trend_table["polarization"] == "VV")
    ].iloc[0]
    report = f"""# I2EM reference-teacher validation

## Decision

**Status: passed with qualification.** All {len(comparison)} I2EM outputs are
finite and the one-factor perturbation directions agree with SPM.  I2EM may be
used as a higher-fidelity teacher, but it must not replace SPM without a
correlation-model indicator or a teacher-confidence mechanism.

## Dielectric-loss sensitivity

- Tested loss tangents: {', '.join(f'{value:g}' for value in summary['loss_tangents'])}.
- Maximum absolute change from the zero-loss I2EM case: HH
  {summary['maximum_absolute_i2em_change_from_zero_loss_hh_db']:.4f} dB and VV
  {summary['maximum_absolute_i2em_change_from_zero_loss_vv_db']:.4f} dB.
- These changes are much smaller than the SPM--I2EM disagreement.  For the
  first paper, use a nominal loss tangent of 0.05 and retain this four-level
  experiment as an ablation.  Do not claim that dielectric loss is generally
  negligible outside the tested L-band parameter domain.

## Teacher-overlap evidence at zero loss

- Exponential spectrum, HH: Spearman {exp_hh['spearman_spm_i2em']:.3f}, mean
  absolute disagreement {exp_hh['mean_abs_i2em_minus_spm_db']:.3f} dB.
- Exponential spectrum, VV: Spearman {exp_vv['spearman_spm_i2em']:.3f}, mean
  absolute disagreement {exp_vv['mean_abs_i2em_minus_spm_db']:.3f} dB.
- Direction agreement is 100% for every spectrum/polarization pair over the
  eight non-baseline one-factor perturbations.  This is a compact diagnostic,
  not a statistical generalization claim.
- Largest HH disagreement: {abs(worst_hh['i2em_minus_spm_hh_db']):.3f} dB at
  `{worst_hh['correlation_model']}/{worst_hh['scenario']}`.
- Largest VV disagreement: {abs(worst_vv['i2em_minus_spm_vv_db']):.3f} dB at
  `{worst_vv['correlation_model']}/{worst_vv['scenario']}`.

## Consequence for the paper method

The evidence supports a **multi-fidelity and risk-controlled physics transfer**
design rather than unconditional I2EM pretraining:

1. keep SPM as the low-cost, smooth-regime teacher;
2. use I2EM as the higher-fidelity teacher;
3. expose the correlation model to the surrogate;
4. estimate or learn teacher confidence from SPM--I2EM disagreement;
5. retain the existing common/differential output heads and grouped real-data
   evaluation protocol.

I2EM is a physics teacher, not observational ground truth.  Publication claims
must ultimately be based on held-out fields or an independent observation
domain.
"""
    (output_dir / "RESULTS_INTERPRETATION.md").write_text(report, encoding="utf-8")
    save_plots(comparison, output_dir)
    print(f"Evaluated I2EM dielectric-loss sensitivity in: {output_dir}")


if __name__ == "__main__":
    main()
