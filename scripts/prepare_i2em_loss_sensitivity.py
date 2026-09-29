"""Prepare a traceable dielectric-loss sensitivity grid for the I2EM teacher."""

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

from prepare_i2em_validation_grid import finite_cohort, make_grid
from research_pilots.scattering.surfaces.teacher_contract import (
    RESULT_COLUMNS,
    validate_teacher_requests,
)


def fingerprint(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def loss_label(value: float) -> str:
    return f"lt_{value:.3f}".replace(".", "p")


def make_sensitivity_grid(
    frame: pd.DataFrame,
    frequency_ghz: float,
    incidence_angle_deg: float,
    loss_tangents: list[float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    grids: list[pd.DataFrame] = []
    design: dict[str, object] | None = None
    for loss_tangent in loss_tangents:
        grid, current_design = make_grid(
            frame,
            frequency_ghz,
            incidence_angle_deg,
            loss_tangent,
        )
        prefix = loss_label(loss_tangent)
        grid["loss_tangent"] = float(loss_tangent)
        grid["request_id"] = prefix + "__" + grid["request_id"].astype(str)
        grids.append(grid)
        if design is None:
            design = current_design
    combined = pd.concat(grids, ignore_index=True)
    combined = validate_teacher_requests(combined)
    if combined["request_id"].duplicated().any():
        raise ValueError("Sensitivity request identifiers must be unique")
    assert design is not None
    return combined, design


def save_plot(grid: pd.DataFrame, path: Path) -> None:
    baseline = grid[grid["scenario"] == "baseline"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    styles = {
        ("exponential", "hh"): ("o", "tab:blue"),
        ("exponential", "vv"): ("s", "tab:orange"),
        ("gaussian", "hh"): ("^", "tab:green"),
        ("gaussian", "vv"): ("D", "tab:red"),
    }
    for (model, pol), (marker, color) in styles.items():
        subset = baseline[baseline["correlation_model"] == model].sort_values(
            "loss_tangent"
        )
        axes[0].plot(
            subset["loss_tangent"],
            subset[f"spm_{pol}_db"],
            marker=marker,
            color=color,
            label=f"SPM {model} {pol.upper()}",
        )
    axes[0].set(
        xlabel="Loss tangent",
        ylabel=r"$\sigma^0$ (dB)",
        title="SPM loss sensitivity at the cohort median anchor",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)

    for model, marker in [("exponential", "o"), ("gaussian", "s")]:
        subset = grid[grid["correlation_model"] == model]
        axes[1].scatter(
            subset["i2em_k_rms_height"],
            subset["i2em_rms_to_correlation_ratio"],
            marker=marker,
            s=38,
            alpha=0.55,
            label=model,
        )
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.25, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="k × RMS height",
        ylabel="RMS height / correlation length",
        title="Validity audit for all sensitivity requests",
    )
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare I2EM dielectric-loss sensitivity requests"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument(
        "--loss-tangents",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.10],
    )
    args = parser.parse_args()
    loss_tangents = sorted(set(float(value) for value in args.loss_tangents))
    if not loss_tangents or any(value < 0 for value in loss_tangents):
        raise ValueError("Loss tangents must be a non-empty set of non-negative values")

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    request_path = output_dir / "i2em_loss_sensitivity_requests.csv"
    if request_path.exists():
        raise FileExistsError("Output already exists; choose a new directory")

    frame = finite_cohort(input_path)
    grid, design = make_sensitivity_grid(
        frame,
        args.frequency_ghz,
        args.incidence_angle_deg,
        loss_tangents,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(request_path, index=False)
    template = grid[["request_id"]].copy()
    template[RESULT_COLUMNS[1]] = np.nan
    template[RESULT_COLUMNS[2]] = np.nan
    template.to_csv(output_dir / "i2em_loss_sensitivity_results_template.csv", index=False)

    manifest = {
        "input": fingerprint(input_path),
        "samples_used_to_design_grid": int(len(frame)),
        "frequency_ghz": args.frequency_ghz,
        "nominal_incidence_angle_deg": args.incidence_angle_deg,
        "loss_tangents": loss_tangents,
        "requests_per_loss_tangent": int(len(grid) / len(loss_tangents)),
        "request_count": int(len(grid)),
        "valid_request_count": int(grid["i2em_request_valid"].sum()),
        "design": design,
        "teacher_execution_status": "pending_matlab_backend",
        "interpretation_guardrails": [
            "I2EM is a higher-fidelity teacher, not observational ground truth.",
            "Loss sensitivity is varied one factor at a time over a fixed anchor design.",
            "Thirty- and fifty-degree cases are numerical trend checks; current observations are nominally 40 degrees.",
            "All requests must satisfy ks < 1 and RMS/correlation <= 0.25.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_plot(grid, output_dir / "01_loss_sensitivity_design.png")
    print(
        f"Prepared {len(grid)} valid requests across {len(loss_tangents)} "
        f"loss-tangent levels in: {output_dir}"
    )


if __name__ == "__main__":
    main()
