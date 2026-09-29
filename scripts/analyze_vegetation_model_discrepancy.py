"""Test whether vegetation explains bare-soil SPM model discrepancy.

The unit of resampling is the agricultural field, not an individual row.
This preserves within-field dependence and avoids claiming significance from
pseudo-replicated acquisition dates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Field-block analysis of vegetation-linked SPM discrepancy."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def choose_vwc(frame: pd.DataFrame) -> pd.Series:
    mapped = pd.to_numeric(
        frame.get("vegetation_water_content_map_kg_m2"), errors="coerce"
    )
    in_situ = pd.to_numeric(
        frame.get("vegetation_water_content_in_situ_kg_m2"), errors="coerce"
    )
    return mapped.fillna(in_situ)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def field_block_bootstrap(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    repeats: int,
    seed: int,
) -> tuple[float, float, int]:
    groups = [group for _, group in frame.groupby("field_id", observed=True)]
    if len(groups) < 3:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        bootstrap = pd.concat(sampled, ignore_index=True)
        statistic = safe_spearman(
            bootstrap[x_column].to_numpy(dtype=float),
            bootstrap[y_column].to_numpy(dtype=float),
        )
        if np.isfinite(statistic):
            values.append(statistic)
    if not values:
        return np.nan, np.nan, 0
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper), len(values)


def _association_record(
    frame: pd.DataFrame,
    polarization: str,
    outcome: str,
    outcome_column: str,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    valid = frame.dropna(subset=["field_id", "vwc_for_analysis", outcome_column]).copy()
    rho = safe_spearman(
        valid["vwc_for_analysis"].to_numpy(dtype=float),
        valid[outcome_column].to_numpy(dtype=float),
    )
    lower, upper, completed = field_block_bootstrap(
        valid,
        "vwc_for_analysis",
        outcome_column,
        repeats,
        seed,
    )
    return {
        "polarization": polarization.upper(),
        "association": outcome,
        "row_count": int(len(valid)),
        "field_count": int(valid["field_id"].nunique()),
        "spearman_rho": rho,
        "field_bootstrap_ci_lower": lower,
        "field_bootstrap_ci_upper": upper,
        "bootstrap_completed": int(completed),
        "ci_excludes_zero": bool(np.isfinite(lower) and np.isfinite(upper) and (lower > 0 or upper < 0)),
    }


def analyse(frame: pd.DataFrame, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["field_id"] = working["field_id"].astype("string")
    working["vwc_for_analysis"] = choose_vwc(working)
    records: list[dict[str, object]] = []

    for offset, polarization in enumerate(("hh", "vv")):
        observed = f"sigma0_{polarization}_db"
        physical = f"exponential_spm_{polarization}_raw_db"
        if observed not in working or physical not in working:
            continue
        working[f"{polarization}_spm_residual_db"] = (
            pd.to_numeric(working[observed], errors="coerce")
            - pd.to_numeric(working[physical], errors="coerce")
        )
        working[f"{polarization}_spm_absolute_error_db"] = working[
            f"{polarization}_spm_residual_db"
        ].abs()
        working[f"{polarization}_within_field_residual_db"] = working[
            f"{polarization}_spm_residual_db"
        ] - working.groupby("field_id", observed=True)[f"{polarization}_spm_residual_db"].transform("mean")
        working["vwc_within_field"] = working["vwc_for_analysis"] - working.groupby(
            "field_id", observed=True
        )["vwc_for_analysis"].transform("mean")

        records.append(
            _association_record(
                working,
                polarization,
                "signed_spm_residual_vs_vwc",
                f"{polarization}_spm_residual_db",
                repeats,
                seed + 10 * offset,
            )
        )
        records.append(
            _association_record(
                working,
                polarization,
                "absolute_spm_error_vs_vwc",
                f"{polarization}_spm_absolute_error_db",
                repeats,
                seed + 10 * offset + 1,
            )
        )
        within = working.rename(columns={"vwc_within_field": "vwc_for_analysis_original"}).copy()
        within["vwc_for_analysis"] = working["vwc_within_field"]
        records.append(
            _association_record(
                within,
                polarization,
                "within_field_signed_residual_vs_vwc",
                f"{polarization}_within_field_residual_db",
                repeats,
                seed + 10 * offset + 2,
            )
        )
    return pd.DataFrame.from_records(records), working


def plot_results(frame: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for column, polarization in enumerate(("hh", "vv")):
        residual = f"{polarization}_spm_residual_db"
        absolute = f"{polarization}_spm_absolute_error_db"
        if residual not in frame:
            continue
        valid = frame.dropna(subset=["vwc_for_analysis", residual, absolute])
        axes[0, column].scatter(
            valid["vwc_for_analysis"], valid[residual], s=26, alpha=0.72, color="#264653"
        )
        axes[0, column].axhline(0.0, color="black", linewidth=1.0)
        axes[0, column].set_title(f"{polarization.upper()} signed discrepancy")
        axes[1, column].scatter(
            valid["vwc_for_analysis"], valid[absolute], s=26, alpha=0.72, color="#e76f51"
        )
        axes[1, column].set_title(f"{polarization.upper()} discrepancy magnitude")
        axes[1, column].set_xlabel("VWC (kg m$^{-2}$)")
        axes[0, column].set_ylabel("Observed - SPM (dB)")
        axes[1, column].set_ylabel("Absolute SPM error (dB)")
    fig.tight_layout()
    fig.savefig(output_dir / "01_vegetation_spm_discrepancy.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, dtype={"field_id": "string"})
    statistics, augmented = analyse(frame, args.bootstrap, args.seed)
    statistics.to_csv(args.output_dir / "vegetation_discrepancy_statistics.csv", index=False)
    augmented.to_csv(args.output_dir / "vegetation_discrepancy_rows.csv", index=False)
    plot_results(augmented, args.output_dir)

    mechanism_rows = statistics[
        statistics["association"].isin(
            ["absolute_spm_error_vs_vwc", "within_field_signed_residual_vs_vwc"]
        )
    ]
    summary = {
        "bootstrap_repeats_requested": int(args.bootstrap),
        "mechanism_supported_for_any_polarization": bool(
            mechanism_rows["ci_excludes_zero"].any()
        ),
        "interpretation": (
            "A non-zero field-bootstrap interval supports vegetation-aware physics weighting. "
            "If all intervals include zero, keep VWC as a predictive covariate but do not claim "
            "that it validates the proposed constraint gate."
        ),
    }
    (args.output_dir / "vegetation_discrepancy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(statistics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
