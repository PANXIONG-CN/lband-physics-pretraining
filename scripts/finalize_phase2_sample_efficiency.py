"""
Consolidate the formal 10-repeat simulation sample-efficiency results.

The script:
1. checks consistency between neural and residual experiments;
2. produces one TGRS-oriented two-panel figure;
3. saves a compact numerical summary and input provenance.

It does not retrain or select any model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SAMPLE_DIR = (
    ROOT
    / "outputs"
    / "scattering"
    / "rough_ground"
    / "multifidelity_sample_efficiency_20260910_v1"
)

RESIDUAL_DIR = (
    ROOT
    / "outputs"
    / "scattering"
    / "rough_ground"
    / "residual_baselines_v1"
)

FIGURE_DIR = (
    ROOT
    / "paper"
    / "figures"
    / "tgrs_v3"
    / "main"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "scattering"
    / "rough_ground"
    / "phase2_sample_efficiency_v1"
)

EXPECTED_SIZES = [32, 64, 128, 256]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def configure_style() -> None:
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


def validate_inputs() -> tuple[pd.DataFrame, dict]:
    residual_path = RESIDUAL_DIR / "summary.csv"
    residual_manifest_path = RESIDUAL_DIR / "manifest.json"
    sample_summary_path = SAMPLE_DIR / "summary.json"

    require(residual_path)
    require(residual_manifest_path)
    require(sample_summary_path)

    residual = pd.read_csv(residual_path)
    residual_manifest = json.loads(
        residual_manifest_path.read_text(encoding="utf-8")
    )
    sample_summary = json.loads(
        sample_summary_path.read_text(encoding="utf-8")
    )

    if residual_manifest.get("status") != "COMPLETE":
        raise ValueError("Residual-baseline experiment is not complete")

    if int(residual_manifest.get("repeats", -1)) != 10:
        raise ValueError("Residual-baseline experiment must use 10 repeats")

    if int(sample_summary.get("repeats", -1)) != 10:
        raise ValueError("Sample-efficiency experiment must use 10 repeats")

    if int(sample_summary.get("independent_i2em_test_samples", -1)) != 128:
        raise ValueError("Expected 128 independent I2EM test samples")

    observed_sizes = sorted(
        residual["size"].astype(int).unique().tolist()
    )
    if observed_sizes != EXPECTED_SIZES:
        raise ValueError(
            f"Unexpected I2EM budgets: {observed_sizes}"
        )

    required_methods = {
        "i2em_only",
        "spm_to_i2em",
        "residual_ridge",
        "residual_rbf",
    }
    observed_methods = set(residual["method"].astype(str))

    missing_methods = required_methods.difference(observed_methods)
    if missing_methods:
        raise ValueError(
            f"Missing methods: {sorted(missing_methods)}"
        )

    ranking = pd.DataFrame(sample_summary["ranking"]).rename(
        columns={
            "i2em_train_samples": "size",
            "mean_hh_vv_rmse_db": "expected_rmse",
        }
    )

    comparison = ranking.merge(
        residual[["size", "method", "rmse_mean_db"]],
        on=["size", "method"],
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != 8:
        raise ValueError(
            "Neural sample-efficiency results do not match residual study"
        )

    if not np.allclose(
        comparison["expected_rmse"],
        comparison["rmse_mean_db"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Neural results differ between the two frozen experiments"
        )

    return residual, {
        "residual_path": residual_path,
        "residual_manifest_path": residual_manifest_path,
        "sample_summary_path": sample_summary_path,
    }


def method_rows(
    frame: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    result = frame.loc[
        frame["method"].astype(str) == method
    ].copy()
    result["size"] = result["size"].astype(int)
    result = result.sort_values("size")

    if result["size"].tolist() != EXPECTED_SIZES:
        raise ValueError(
            f"Incomplete sample budgets for method: {method}"
        )

    return result


def render_figure(frame: pd.DataFrame) -> Path:
    configure_style()

    colors = {
        "i2em_only": "#E69F00",
        "spm_to_i2em": "#D55E00",
        "residual_ridge": "#009E73",
        "residual_rbf": "#CC79A7",
    }

    markers = {
        "i2em_only": "s",
        "spm_to_i2em": "o",
        "residual_ridge": "^",
        "residual_rbf": "D",
    }

    linestyles = {
        "i2em_only": "--",
        "spm_to_i2em": "-",
        "residual_ridge": "-.",
        "residual_rbf": ":",
    }

    labels = {
        "i2em_only": "I$^2$EM only",
        "spm_to_i2em": "SPM $\\rightarrow$ I$^2$EM",
        "residual_ridge": "Residual Ridge",
        "residual_rbf": "Residual RBF",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.75),
        constrained_layout=True,
    )

    for method in ["i2em_only", "spm_to_i2em"]:
        data = method_rows(frame, method)
        axes[0].errorbar(
            data["size"],
            data["rmse_mean_db"],
            yerr=data["rmse_std_db"],
            color=colors[method],
            marker=markers[method],
            linestyle=linestyles[method],
            capsize=2.5,
            label=labels[method],
        )

    axes[0].set_xlabel("Number of I$^2$EM training samples")
    axes[0].set_ylabel("Mean HH/VV test RMSE (dB)")
    axes[0].set_title(
        "(a) Sequential pretraining",
        loc="left",
    )
    axes[0].set_xticks(EXPECTED_SIZES)
    axes[0].grid(alpha=0.18, linewidth=0.5)
    axes[0].legend()

    for method in [
        "i2em_only",
        "spm_to_i2em",
        "residual_ridge",
        "residual_rbf",
    ]:
        data = method_rows(frame, method)
        axes[1].errorbar(
            data["size"],
            data["rmse_mean_db"],
            yerr=data["rmse_std_db"],
            color=colors[method],
            marker=markers[method],
            linestyle=linestyles[method],
            capsize=2.5,
            label=labels[method],
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of I$^2$EM training samples")
    axes[1].set_ylabel("Mean HH/VV test RMSE (dB)")
    axes[1].set_title(
        "(b) Equal-I$^2$EM-label comparison",
        loc="left",
    )
    axes[1].set_xticks(EXPECTED_SIZES)
    axes[1].grid(
        alpha=0.18,
        linewidth=0.5,
        which="both",
    )
    axes[1].legend(ncol=2)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_base = FIGURE_DIR / "Fig_phase2_sample_efficiency"

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

    return figure_base


def build_summary(
    frame: pd.DataFrame,
    figure_base: Path,
    sources: dict,
) -> dict:
    pivot = frame.pivot(
        index="size",
        columns="method",
        values="rmse_mean_db",
    )

    budgets = {}

    for size in EXPECTED_SIZES:
        i2em = float(pivot.loc[size, "i2em_only"])
        sequential = float(pivot.loc[size, "spm_to_i2em"])
        residual_rbf = float(pivot.loc[size, "residual_rbf"])
        residual_ridge = float(pivot.loc[size, "residual_ridge"])

        budgets[str(size)] = {
            "i2em_only_rmse_db": i2em,
            "sequential_rmse_db": sequential,
            "residual_ridge_rmse_db": residual_ridge,
            "residual_rbf_rmse_db": residual_rbf,
            "sequential_reduction_vs_i2em_only_percent": (
                100.0 * (1.0 - sequential / i2em)
            ),
            "residual_rbf_reduction_vs_sequential_percent": (
                100.0 * (1.0 - residual_rbf / sequential)
            ),
        }

    return {
        "phase": "phase2_simulation_sample_efficiency",
        "repeats": 10,
        "i2em_training_budgets": EXPECTED_SIZES,
        "independent_i2em_test_samples": 128,
        "budget_definition": (
            "All compared methods use the same number of I2EM labels."
        ),
        "compute_note": (
            "Residual methods additionally require one SPM evaluation "
            "for each query."
        ),
        "results_by_budget": budgets,
        "figure_pdf": str(figure_base.with_suffix(".pdf")),
        "figure_png": str(figure_base.with_suffix(".png")),
        "sources": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in sources.items()
        },
    }


def main() -> None:
    frame, sources = validate_inputs()
    figure_base = render_figure(frame)
    summary = build_summary(frame, figure_base, sources)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "phase2_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Phase 2 completed.")
    print(f"Figure: {figure_base.with_suffix('.pdf')}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
