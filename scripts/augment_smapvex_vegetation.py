"""Add SV12VA/SV12VWC vegetation features to the current SMAPVEX12 table."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

try:
    from research_pilots.scattering.data.vegetation import (
        calculate_physics_confidence,
        collocate_nearest_in_situ,
        collocate_vwc_maps,
        discover_vwc_maps,
        read_in_situ_vegetation,
    )
except ImportError:  # standalone validation before copying into repository
    MODULE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(MODULE_DIR))
    from vegetation import (  # type: ignore[no-redef]
        calculate_physics_confidence,
        collocate_nearest_in_situ,
        collocate_vwc_maps,
        discover_vwc_maps,
        read_in_situ_vegetation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collocate SV12VA and SV12VWC with the rough-ground field-day table."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--vegetation-root", required=True, type=Path)
    parser.add_argument("--vwc-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-days", type=int, default=8)
    parser.add_argument(
        "--gate-alpha",
        type=float,
        default=0.8,
        help="Initial diagnostic value only; tune inside nested CV during modelling.",
    )
    return parser.parse_args()


def _as_percent(series: pd.Series) -> float:
    return 100.0 * float(series.mean()) if len(series) else 0.0


def build_summary(
    enriched: pd.DataFrame,
    vegetation_field_day: pd.DataFrame,
    vegetation_coordinates: pd.DataFrame,
    map_count: int,
    max_days: int,
) -> dict[str, object]:
    observed_fields = set(enriched["field_id"].dropna().astype(str))
    vegetation_fields = set(vegetation_coordinates["field_id"].dropna().astype(str))
    common_fields = observed_fields & vegetation_fields
    source_counts = (
        enriched["vwc_map_source"].fillna("missing").value_counts(dropna=False).to_dict()
    )
    summary: dict[str, object] = {
        "input_rows": int(len(enriched)),
        "input_fields": int(len(observed_fields)),
        "vegetation_field_day_rows": int(len(vegetation_field_day)),
        "vegetation_coordinate_sites": int(len(vegetation_coordinates)),
        "downloaded_vwc_map_dates": int(map_count),
        "field_id_overlap": {
            "common_count": int(len(common_fields)),
            "only_in_radar_table": sorted(observed_fields - vegetation_fields),
            "only_in_vegetation_files": sorted(vegetation_fields - observed_fields),
        },
        "max_in_situ_time_offset_days": int(max_days),
        "in_situ_coverage_percent": _as_percent(enriched["vegetation_in_situ_available"]),
        "vwc_map_coverage_percent": _as_percent(enriched["vwc_map_available"]),
        "vwc_map_source_rows": {str(key): int(value) for key, value in source_counts.items()},
        "physics_confidence": {
            "mean": float(enriched["physics_confidence_initial"].mean()),
            "median": float(enriched["physics_confidence_initial"].median()),
            "fraction_below_0_1": float((enriched["physics_confidence_initial"] < 0.1).mean()),
        },
    }

    overlap = enriched.dropna(
        subset=[
            "vegetation_water_content_map_kg_m2",
            "vegetation_water_content_in_situ_kg_m2",
        ]
    )
    if len(overlap) >= 3:
        difference = (
            overlap["vegetation_water_content_map_kg_m2"]
            - overlap["vegetation_water_content_in_situ_kg_m2"]
        )
        correlation = overlap[
            [
                "vegetation_water_content_map_kg_m2",
                "vegetation_water_content_in_situ_kg_m2",
            ]
        ].corr().iloc[0, 1]
        summary["map_vs_in_situ"] = {
            "paired_rows": int(len(overlap)),
            "bias_kg_m2": float(difference.mean()),
            "rmse_kg_m2": float(np.sqrt(np.mean(np.square(difference)))),
            "correlation": float(correlation),
        }
    else:
        summary["map_vs_in_situ"] = {"paired_rows": int(len(overlap))}
    return summary


def plot_diagnostics(enriched: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    coverage = pd.Series(
        {
            "In-situ vegetation": enriched["vegetation_in_situ_available"].mean() * 100.0,
            "Daily VWC map": enriched["vwc_map_available"].mean() * 100.0,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(coverage.index, coverage.values, color=["#2a9d8f", "#e9c46a"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Row coverage (%)")
    ax.set_title("Vegetation feature coverage")
    ax.bar_label(bars, fmt="%.1f%%")
    fig.tight_layout()
    fig.savefig(output_dir / "01_vegetation_coverage.png", dpi=220)
    plt.close(fig)

    vwc_column = "vegetation_water_content_map_kg_m2"
    panels: list[tuple[str, str, str]] = []
    for polarization in ("hh", "vv"):
        observed = f"sigma0_{polarization}_db"
        physical = f"exponential_spm_{polarization}_raw_db"
        if observed in enriched and physical in enriched:
            panels.append((polarization.upper(), observed, physical))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.6), squeeze=False)
        for axis, (label, observed, physical) in zip(axes[0], panels):
            valid = enriched.dropna(subset=[vwc_column, observed, physical])
            residual = valid[observed] - valid[physical]
            axis.scatter(valid[vwc_column], residual, s=28, alpha=0.75, color="#264653")
            axis.axhline(0.0, color="black", linewidth=1.0)
            axis.set_xlabel("SV12VWC (kg m$^{-2}$)")
            axis.set_ylabel("Observed - bare-soil SPM (dB)")
            axis.set_title(f"{label}: vegetation-linked model discrepancy")
        fig.tight_layout()
        fig.savefig(output_dir / "02_spm_residual_vs_vwc.png", dpi=220)
        plt.close(fig)

    paired = enriched.dropna(
        subset=[
            "vegetation_water_content_map_kg_m2",
            "vegetation_water_content_in_situ_kg_m2",
        ]
    )
    if not paired.empty:
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.scatter(
            paired["vegetation_water_content_in_situ_kg_m2"],
            paired["vegetation_water_content_map_kg_m2"],
            s=28,
            alpha=0.75,
            color="#e76f51",
        )
        limits = [
            0.0,
            float(
                np.nanmax(
                    paired[
                        [
                            "vegetation_water_content_in_situ_kg_m2",
                            "vegetation_water_content_map_kg_m2",
                        ]
                    ].to_numpy()
                )
            ),
        ]
        ax.plot(limits, limits, "k--", linewidth=1.0)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("In-situ VWC (kg m$^{-2}$)")
        ax.set_ylabel("Mapped VWC (kg m$^{-2}$)")
        ax.set_title("Independent vegetation-source agreement")
        fig.tight_layout()
        fig.savefig(output_dir / "03_vwc_map_vs_in_situ.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.input, dtype={"field_id": "string"})
    required = {
        "acquisition_date",
        "field_id",
        "field_utm_x_min",
        "field_utm_x_max",
        "field_utm_y_min",
        "field_utm_y_max",
        "spm_k_rms_height",
        "spm_rms_slope_proxy",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    vegetation_field_day, vegetation_coordinates = read_in_situ_vegetation(
        args.vegetation_root
    )
    enriched = collocate_nearest_in_situ(
        table,
        vegetation_field_day,
        tolerance_days=args.max_days,
    )
    enriched = collocate_vwc_maps(enriched, args.vwc_root)
    enriched = calculate_physics_confidence(enriched, alpha=args.gate_alpha)
    summary = build_summary(
        enriched,
        vegetation_field_day,
        vegetation_coordinates,
        map_count=len(discover_vwc_maps(args.vwc_root)),
        max_days=args.max_days,
    )

    enriched.to_csv(args.output_dir / "vegetation_enriched_predictions.csv", index=False)
    vegetation_field_day.to_csv(args.output_dir / "vegetation_field_day.csv", index=False)
    vegetation_coordinates.to_csv(args.output_dir / "vegetation_site_coordinates.csv", index=False)
    (args.output_dir / "vegetation_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_diagnostics(enriched, args.output_dir)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved vegetation-enriched data to: {args.output_dir}")


if __name__ == "__main__":
    main()
