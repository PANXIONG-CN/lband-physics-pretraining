"""Prepare a compact, traceable I2EM validation grid from the real cohort."""

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

from research_pilots.scattering.surfaces.dielectric import (
    complex_relative_permittivity,
)
from research_pilots.scattering.surfaces.spm import spm_backscatter_db
from research_pilots.scattering.surfaces.teacher_contract import (
    RESULT_COLUMNS,
    validate_teacher_requests,
)


SOURCE_COLUMNS = {
    "dielectric_real": "soil_real_dielectric",
    "rms_height_m": "rms_height_m",
    "correlation_length_m": "correlation_length_m",
}


def fingerprint(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def finite_cohort(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(SOURCE_COLUMNS.values()).difference(frame.columns))
    if missing:
        raise ValueError(f"Input cohort is missing columns: {missing}")
    numeric = list(SOURCE_COLUMNS.values())
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    finite = np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    result = frame.loc[finite].copy()
    if result.empty:
        raise ValueError("No finite rows available for validation-grid design")
    return result


def make_grid(
    frame: pd.DataFrame,
    frequency_ghz: float,
    baseline_angle_deg: float,
    loss_tangent: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    quantiles = {
        target: {
            "q10": float(frame[source].quantile(0.10)),
            "q50": float(frame[source].quantile(0.50)),
            "q90": float(frame[source].quantile(0.90)),
        }
        for target, source in SOURCE_COLUMNS.items()
    }
    baseline = {target: values["q50"] for target, values in quantiles.items()}
    scenarios: list[dict[str, object]] = [
        {"scenario": "baseline", **baseline, "incidence_angle_deg": baseline_angle_deg}
    ]
    for target in SOURCE_COLUMNS:
        for label in ["q10", "q90"]:
            item = {"scenario": f"{target}_{label}", **baseline}
            item[target] = quantiles[target][label]
            item["incidence_angle_deg"] = baseline_angle_deg
            scenarios.append(item)
    for angle in [30.0, 50.0]:
        scenarios.append(
            {
                "scenario": f"incidence_angle_{angle:g}_deg",
                **baseline,
                "incidence_angle_deg": angle,
            }
        )

    rows = []
    for correlation_model in ["exponential", "gaussian"]:
        for scenario in scenarios:
            rows.append(
                {
                    "request_id": f"{correlation_model}_{scenario['scenario']}",
                    "correlation_model": correlation_model,
                    "frequency_ghz": frequency_ghz,
                    "dielectric_loss_positive": float(
                        scenario["dielectric_real"] * loss_tangent
                    ),
                    **scenario,
                }
            )
    grid = validate_teacher_requests(pd.DataFrame(rows))
    epsilon = complex_relative_permittivity(
        grid["dielectric_real"], loss_tangent=loss_tangent
    )
    for correlation_model in ["exponential", "gaussian"]:
        mask = grid["correlation_model"] == correlation_model
        predicted = spm_backscatter_db(
            epsilon[mask],
            grid.loc[mask, "rms_height_m"],
            grid.loc[mask, "correlation_length_m"],
            frequency_ghz * 1e9,
            baseline_angle_deg,
            spectrum_model=correlation_model,
        )
        # Angle sensitivity is computed separately because SPM accepts one angle per call.
        for row_index in grid.index[mask]:
            angle = float(grid.loc[row_index, "incidence_angle_deg"])
            point = spm_backscatter_db(
                complex(
                    float(grid.loc[row_index, "dielectric_real"]),
                    -float(grid.loc[row_index, "dielectric_loss_positive"]),
                ),
                float(grid.loc[row_index, "rms_height_m"]),
                float(grid.loc[row_index, "correlation_length_m"]),
                frequency_ghz * 1e9,
                angle,
                spectrum_model=correlation_model,
            )
            grid.loc[row_index, "spm_hh_db"] = float(point["hh_db"])
            grid.loc[row_index, "spm_vv_db"] = float(point["vv_db"])
    design = {
        "quantiles": quantiles,
        "baseline": baseline,
        "angles_deg": [30.0, baseline_angle_deg, 50.0],
        "design": "one-factor-at-a-time at cohort q10/q50/q90 anchors",
        "observed_geometry_note": (
            f"Only {baseline_angle_deg:g} deg is the current nominal observation geometry; "
            "30 and 50 deg are numerical diagnostics, not observed-domain validation."
        ),
    }
    return grid, design


def save_plot(grid: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for model, marker in [("exponential", "o"), ("gaussian", "s")]:
        subset = grid[grid.correlation_model == model]
        axes[0].scatter(
            subset.i2em_k_rms_height,
            subset.i2em_rms_to_correlation_ratio,
            label=model,
            marker=marker,
            s=48,
            alpha=0.75,
        )
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(0.25, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="k × RMS height",
        ylabel="RMS height / correlation length",
        title="I2EM reference implementation domain",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend()

    baseline = grid[grid.scenario == "baseline"]
    positions = np.arange(len(baseline))
    width = 0.35
    axes[1].bar(positions - width / 2, baseline.spm_hh_db, width, label="SPM HH")
    axes[1].bar(positions + width / 2, baseline.spm_vv_db, width, label="SPM VV")
    axes[1].set_xticks(positions, baseline.correlation_model)
    axes[1].set(ylabel="sigma0 (dB)", title="SPM anchors awaiting I2EM reference")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an auditable I2EM validation grid")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument("--loss-tangent", type=float, default=0.0)
    args = parser.parse_args()
    if args.loss_tangent < 0:
        raise ValueError("loss tangent must be non-negative")
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    expected = output_dir / "i2em_requests.csv"
    if expected.exists():
        raise FileExistsError("Output already exists; choose a new directory")
    frame = finite_cohort(input_path)
    grid, design = make_grid(
        frame,
        args.frequency_ghz,
        args.incidence_angle_deg,
        args.loss_tangent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(expected, index=False)
    template = grid[["request_id"]].copy()
    template[RESULT_COLUMNS[1]] = np.nan
    template[RESULT_COLUMNS[2]] = np.nan
    template.to_csv(output_dir / "i2em_results_template.csv", index=False)
    manifest = {
        "input": fingerprint(input_path),
        "samples_used_to_design_grid": int(len(frame)),
        "request_count": int(len(grid)),
        "valid_request_count": int(grid.i2em_request_valid.sum()),
        "frequency_ghz": args.frequency_ghz,
        "nominal_incidence_angle_deg": args.incidence_angle_deg,
        "loss_tangent": args.loss_tangent,
        "design": design,
        "teacher_execution_status": "pending_external_backend",
        "external_blocker": "Local MATLAB license expired; no native-Windows pyi2em wheel is published.",
        "guardrails": [
            "I2EM is a reference teacher, not ground truth.",
            "The request grid satisfies ks < 1 and RMS/correlation <= 0.25.",
            "The dielectric-loss assumption must be included in sensitivity analysis.",
            "Thirty- and fifty-degree requests test numerical trends outside the current nominal 40-degree observation geometry.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_plot(grid, output_dir / "01_validation_grid_and_spm_anchors.png")
    print(f"Prepared {len(grid)} valid I2EM reference requests in: {output_dir}")


if __name__ == "__main__":
    main()
