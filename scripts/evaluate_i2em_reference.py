"""Validate and compare externally computed I2EM results with SPM anchors."""

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

from research_pilots.scattering.surfaces.teacher_contract import (
    validate_teacher_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate I2EM reference anchors")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merged = validate_teacher_results(
        pd.read_csv(args.requests), pd.read_csv(args.results)
    )
    for pol in ["hh", "vv"]:
        merged[f"i2em_minus_spm_{pol}_db"] = (
            merged[f"i2em_{pol}_db"] - merged[f"spm_{pol}_db"]
        )
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "i2em_spm_anchor_comparison.csv", index=False)

    summary = {
        "request_count": int(len(merged)),
        "models": sorted(merged.correlation_model.unique().tolist()),
        "mean_i2em_minus_spm_hh_db": float(merged.i2em_minus_spm_hh_db.mean()),
        "mean_i2em_minus_spm_vv_db": float(merged.i2em_minus_spm_vv_db.mean()),
        "maximum_absolute_difference_hh_db": float(merged.i2em_minus_spm_hh_db.abs().max()),
        "maximum_absolute_difference_vv_db": float(merged.i2em_minus_spm_vv_db.abs().max()),
        "interpretation": (
            "Differences quantify teacher disagreement, not I2EM error. Agreement in the "
            "smooth overlap regime and physically plausible trends are required before pretraining."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, pol in zip(axes, ["hh", "vv"]):
        for model, marker in [("exponential", "o"), ("gaussian", "s")]:
            subset = merged[merged.correlation_model == model]
            ax.scatter(
                subset[f"spm_{pol}_db"],
                subset[f"i2em_{pol}_db"],
                marker=marker,
                s=48,
                alpha=0.75,
                label=model,
            )
        values = np.concatenate([merged[f"spm_{pol}_db"], merged[f"i2em_{pol}_db"]])
        lower, upper = float(values.min()), float(values.max())
        ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        ax.set(
            xlabel=f"SPM {pol.upper()} (dB)",
            ylabel=f"I2EM {pol.upper()} (dB)",
            title=f"Teacher overlap: {pol.upper()}",
        )
        ax.grid(alpha=0.22)
        ax.legend()
    fig.savefig(output_dir / "01_i2em_vs_spm_anchors.png", dpi=190)
    plt.close(fig)
    print(f"Validated I2EM reference results in: {output_dir}")


if __name__ == "__main__":
    main()
