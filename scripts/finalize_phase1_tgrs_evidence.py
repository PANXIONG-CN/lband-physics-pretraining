"""
Complete Phase-1 evidence consolidation for the TGRS manuscript.

This program performs three tasks:

1. Reconstruct the 18 zero-loss I2EM anchor results from the completed
   loss-tangent sensitivity experiment.
2. Run the existing anchor evaluator and update the validation manifest.
3. Render one publication-oriented physics-teacher evidence figure.

No network is retrained and no model is reselected.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

VALIDATION_DIR = (
    ROOT
    / "outputs"
    / "scattering"
    / "rough_ground"
    / "i2em_validation_20260909"
)

LOSS_DIR = (
    ROOT
    / "outputs"
    / "scattering"
    / "rough_ground"
    / "i2em_loss_sensitivity_20260909_v1"
)

PAIRED_CSV = (
    ROOT
    / "outputs"
    / "scattering"
    / "rough_ground"
    / "multifidelity_pretraining_20260910_v1"
    / "evaluation"
    / "paired_teacher_disagreement.csv"
)

FIGURE_DIR = (
    ROOT
    / "paper"
    / "figures"
    / "tgrs_v3"
    / "main"
)

SUMMARY_DIR = (
    ROOT
    / "outputs"
    / "scattering"
    / "rough_ground"
    / "phase1_tgrs_evidence_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")


def complete_anchor_validation() -> dict:
    validation_requests_path = VALIDATION_DIR / "i2em_requests.csv"
    loss_requests_path = LOSS_DIR / "i2em_loss_sensitivity_requests.csv"
    loss_results_path = LOSS_DIR / "i2em_loss_sensitivity_results.csv"

    require(validation_requests_path)
    require(loss_requests_path)
    require(loss_results_path)

    validation = pd.read_csv(validation_requests_path)
    loss_requests = pd.read_csv(loss_requests_path)
    loss_results = pd.read_csv(loss_results_path)

    loss_requests["loss_tangent"] = pd.to_numeric(
        loss_requests["loss_tangent"],
        errors="raise",
    )

    zero_loss = loss_requests.loc[
        np.isclose(loss_requests["loss_tangent"], 0.0)
    ].copy()

    if len(zero_loss) != 18:
        raise ValueError(
            f"Expected 18 zero-loss anchors, but found {len(zero_loss)}"
        )

    zero_loss["source_request_id"] = zero_loss["request_id"].astype(str)
    zero_loss["request_id"] = zero_loss["request_id"].str.replace(
        r"^lt_0p000__",
        "",
        regex=True,
    )

    result_values = loss_results.rename(
        columns={"request_id": "source_request_id"}
    )

    zero_loss = zero_loss.merge(
        result_values,
        on="source_request_id",
        how="left",
        validate="one_to_one",
    )

    if zero_loss[["i2em_hh_db", "i2em_vv_db"]].isna().any().any():
        raise ValueError("Some zero-loss I2EM results are missing")

    parameter_columns = [
        "frequency_ghz",
        "dielectric_real",
        "dielectric_loss_positive",
        "rms_height_m",
        "correlation_length_m",
        "incidence_angle_deg",
    ]

    comparison = validation[
        ["request_id", "correlation_model", "scenario", *parameter_columns]
    ].merge(
        zero_loss[
            ["request_id", "correlation_model", "scenario", *parameter_columns]
        ],
        on=["request_id", "correlation_model", "scenario"],
        suffixes=("_validation", "_loss"),
        validate="one_to_one",
    )

    if len(comparison) != len(validation):
        raise ValueError("The validation and zero-loss anchor sets do not match")

    for column in parameter_columns:
        left = comparison[f"{column}_validation"].to_numpy(dtype=float)
        right = comparison[f"{column}_loss"].to_numpy(dtype=float)

        if not np.allclose(left, right, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Anchor inputs differ for parameter: {column}"
            )

    reconstructed_results = validation[["request_id"]].merge(
        zero_loss[["request_id", "i2em_hh_db", "i2em_vv_db"]],
        on="request_id",
        how="left",
        validate="one_to_one",
    )

    result_path = VALIDATION_DIR / "i2em_results.csv"
    reconstructed_results.to_csv(result_path, index=False)

    evaluation_dir = VALIDATION_DIR / "evaluation"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_i2em_reference.py"),
            "--requests",
            str(validation_requests_path),
            "--results",
            str(result_path),
            "--output",
            str(evaluation_dir),
        ],
        check=True,
    )

    manifest_path = VALIDATION_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["teacher_execution_status"] = (
        "completed_from_identical_zero_loss_sensitivity_anchors"
    )
    manifest["result_source"] = {
        "source_experiment": str(LOSS_DIR),
        "source_request_count": 18,
        "identity_check": (
            "frequency, dielectric, roughness, correlation length, "
            "incidence angle, correlation model, and scenario matched"
        ),
        "result_file": str(result_path),
        "result_sha256": sha256(result_path),
    }
    manifest.pop("external_blocker", None)

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = evaluation_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def teacher_agreement_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    polarization: str,
    color: str,
) -> dict:
    pol = polarization.lower()

    x = frame[f"spm_{pol}_db"].to_numpy(dtype=float)
    y = frame[f"i2em_{pol}_db"].to_numpy(dtype=float)

    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    margin = max(0.5, 0.04 * (upper - lower))

    ax.scatter(
        x,
        y,
        s=15,
        facecolors="none",
        edgecolors=color,
        linewidths=0.7,
        alpha=0.65,
    )
    ax.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        color="#333333",
        linestyle="--",
        linewidth=0.9,
    )

    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"SPM {polarization.upper()} backscatter (dB)")
    ax.set_ylabel(f"I$^2$EM {polarization.upper()} backscatter (dB)")
    ax.grid(alpha=0.18, linewidth=0.5)

    mae = float(np.mean(np.abs(y - x)))
    rho = float(
        pd.Series(x).corr(pd.Series(y), method="spearman")
    )

    ax.text(
        0.04,
        0.95,
        f"MAE = {mae:.3f} dB\n"
        f"Spearman $\\rho$ = {rho:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    return {
        "sample_count": int(len(frame)),
        "mean_absolute_disagreement_db": mae,
        "spearman_spm_i2em": rho,
    }


def render_main_figure() -> dict:
    require(PAIRED_CSV)

    loss_summary_path = LOSS_DIR / "evaluation" / "loss_sensitivity_summary.csv"
    require(loss_summary_path)

    paired = pd.read_csv(PAIRED_CSV)
    test = paired.loc[paired["split"].astype(str) == "test"].copy()

    if len(test) != 128:
        raise ValueError(
            f"Expected 128 frozen test samples, but found {len(test)}"
        )

    test["mean_absolute_teacher_disagreement_db"] = 0.5 * (
        pd.to_numeric(test["teacher_absolute_delta_hh_db"])
        + pd.to_numeric(test["teacher_absolute_delta_vv_db"])
    )

    loss_summary = pd.read_csv(loss_summary_path).sort_values(
        "loss_tangent"
    )

    configure_plot_style()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.16, 5.0),
        constrained_layout=True,
    )

    hh_summary = teacher_agreement_panel(
        axes[0, 0],
        test,
        "HH",
        "#0072B2",
    )
    vv_summary = teacher_agreement_panel(
        axes[0, 1],
        test,
        "VV",
        "#D55E00",
    )

    axes[0, 0].set_title("(a) HH teacher agreement", loc="left")
    axes[0, 1].set_title("(b) VV teacher agreement", loc="left")

    stratum_style = {
        "interior_le_0.4": ("Interior", "#0072B2", "o"),
        "middle_0.4_to_0.7": ("Middle", "#E69F00", "s"),
        "nearer_boundary_gt_0.7": ("Near boundary", "#D55E00", "^"),
    }

    for key, (label, color, marker) in stratum_style.items():
        subset = test.loc[test["validity_stratum"] == key]
        axes[1, 0].scatter(
            subset["i2em_validity_utilization"],
            subset["mean_absolute_teacher_disagreement_db"],
            s=18,
            color=color,
            marker=marker,
            alpha=0.65,
            label=f"{label} (n={len(subset)})",
        )

    bin_edges = np.linspace(0.0, 1.0, 6)
    test["utilization_bin"] = pd.cut(
        test["i2em_validity_utilization"],
        bins=bin_edges,
        include_lowest=True,
    )

    binned = (
        test.groupby("utilization_bin", observed=True)
        .agg(
            utilization=("i2em_validity_utilization", "median"),
            disagreement=(
                "mean_absolute_teacher_disagreement_db",
                "median",
            ),
        )
        .dropna()
    )

    axes[1, 0].plot(
        binned["utilization"],
        binned["disagreement"],
        color="#222222",
        marker="D",
        linewidth=1.2,
        markersize=4,
        label="Binned median",
    )
    axes[1, 0].axvline(
        0.4,
        color="#777777",
        linestyle=":",
        linewidth=0.8,
    )
    axes[1, 0].axvline(
        0.7,
        color="#777777",
        linestyle=":",
        linewidth=0.8,
    )
    axes[1, 0].set_xlabel("I$^2$EM validity-domain utilization")
    axes[1, 0].set_ylabel(
        "Mean absolute teacher disagreement (dB)"
    )
    axes[1, 0].set_title(
        "(c) Disagreement across the validity domain",
        loc="left",
    )
    axes[1, 0].grid(alpha=0.18, linewidth=0.5)
    axes[1, 0].legend(ncol=2, loc="upper left")

    axes[1, 1].plot(
        loss_summary["loss_tangent"],
        loss_summary["max_abs_i2em_change_hh_db"],
        color="#0072B2",
        marker="o",
        linestyle="-",
        label="HH",
    )
    axes[1, 1].plot(
        loss_summary["loss_tangent"],
        loss_summary["max_abs_i2em_change_vv_db"],
        color="#D55E00",
        marker="s",
        linestyle="--",
        label="VV",
    )
    axes[1, 1].set_xlabel("Dielectric loss tangent")
    axes[1, 1].set_ylabel(
        "Maximum absolute change from zero-loss case (dB)"
    )
    axes[1, 1].set_title(
        "(d) Dielectric-loss sensitivity",
        loc="left",
    )
    axes[1, 1].set_xticks([0.0, 0.02, 0.05, 0.10])
    axes[1, 1].grid(alpha=0.18, linewidth=0.5)
    axes[1, 1].legend()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_base = FIGURE_DIR / "fig02_teacher_comparison"

    fig.savefig(
        figure_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        figure_base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    stratum_summary = {}
    for key in stratum_style:
        subset = test.loc[test["validity_stratum"] == key]
        stratum_summary[key] = {
            "n": int(len(subset)),
            "mean_absolute_disagreement_db": float(
                subset[
                    "mean_absolute_teacher_disagreement_db"
                ].mean()
            ),
        }

    return {
        "test_sample_count": int(len(test)),
        "hh": hh_summary,
        "vv": vv_summary,
        "validity_strata": stratum_summary,
        "loss_tangent_values": (
            loss_summary["loss_tangent"].astype(float).tolist()
        ),
        "maximum_loss_change_hh_db": float(
            loss_summary["max_abs_i2em_change_hh_db"].max()
        ),
        "maximum_loss_change_vv_db": float(
            loss_summary["max_abs_i2em_change_vv_db"].max()
        ),
        "figure_pdf": str(figure_base.with_suffix(".pdf")),
        "figure_png": str(figure_base.with_suffix(".png")),
    }


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    anchor_summary = complete_anchor_validation()
    figure_summary = render_main_figure()

    final_summary = {
        "phase": "phase1_physics_teacher_evidence",
        "anchor_validation": anchor_summary,
        "formal_multifidelity_test": figure_summary,
        "interpretation_scope": {
            "teacher_comparison": (
                "SPM and I2EM are compared as two physics teachers."
            ),
            "loss_sensitivity": (
                "Loss-tangent results quantify parameter sensitivity."
            ),
            "model_selection": (
                "This consolidation does not retrain or select a model."
            ),
        },
        "input_provenance": {
            "paired_teacher_csv": {
                "path": str(PAIRED_CSV),
                "sha256": sha256(PAIRED_CSV),
            },
            "loss_requests": {
                "path": str(
                    LOSS_DIR / "i2em_loss_sensitivity_requests.csv"
                ),
                "sha256": sha256(
                    LOSS_DIR / "i2em_loss_sensitivity_requests.csv"
                ),
            },
            "loss_results": {
                "path": str(
                    LOSS_DIR / "i2em_loss_sensitivity_results.csv"
                ),
                "sha256": sha256(
                    LOSS_DIR / "i2em_loss_sensitivity_results.csv"
                ),
            },
        },
    }

    summary_path = SUMMARY_DIR / "phase1_summary.json"
    summary_path.write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Phase 1 completed.")
    print(f"Anchor validation: {VALIDATION_DIR / 'evaluation'}")
    print(f"Main figure: {figure_summary['figure_pdf']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
