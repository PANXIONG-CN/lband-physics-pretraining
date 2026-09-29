"""Run a first-order SPM baseline on the collocated SMAPVEX12 field-day table.

The script deliberately separates three questions:

1. Does the Topp soil-moisture polynomial reproduce the measured dielectric data?
2. Which samples are inside the conservative first-order SPM validity domain?
3. Can SPM reproduce the level and variation of measured PALS HH/VV sigma0?

The bias-corrected result estimates only one additive dB offset inside each
training fold.  Fields, rather than individual rows, are held out so that the
calibration does not leak information between dates from the same field.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

try:
    from research_pilots.scattering.surfaces.dielectric import (
        complex_relative_permittivity,
        fresnel_reflection_coefficients,
        topp_real_permittivity,
    )
    from research_pilots.scattering.surfaces.metrics import (
        grouped_oof_bias_calibration,
        regression_metrics,
    )
    from research_pilots.scattering.surfaces.rough_surface import (
        spm_validity,
        wavelength_m,
    )
    from research_pilots.scattering.surfaces.spm import spm_backscatter_db
except ImportError:  # permits a standalone smoke test before copying to src/
    MODULE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(MODULE_DIR))
    from dielectric import (  # type: ignore[no-redef]
        complex_relative_permittivity,
        fresnel_reflection_coefficients,
        topp_real_permittivity,
    )
    from metrics import (  # type: ignore[no-redef]
        grouped_oof_bias_calibration,
        regression_metrics,
    )
    from rough_surface import spm_validity, wavelength_m  # type: ignore[no-redef]
    from spm import spm_backscatter_db  # type: ignore[no-redef]


REQUIRED_COLUMNS = {
    "acquisition_date",
    "field_id",
    "soil_moisture_m3_m3",
    "soil_real_dielectric",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
    "sigma0_hh_db",
    "sigma0_vv_db",
}
OBSERVED_COLUMNS = {"hh": "sigma0_hh_db", "vv": "sigma0_vv_db"}
SOURCE_LABELS = {"measured": "Measured dielectric", "topp": "Topp dielectric"}
POLARIZATION_LABELS = {"hh": "HH", "vv": "VV"}


def load_input_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": "string"})
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")
    frame["field_id"] = frame["field_id"].str.strip()
    frame["acquisition_date"] = pd.to_datetime(
        frame["acquisition_date"], errors="raise"
    )
    if frame["field_id"].isna().any():
        raise ValueError("field_id contains missing values")
    return frame


def finite_summary(observed, predicted) -> dict[str, float | int | None]:
    """Unit-neutral comparison statistics used for dielectric permittivity."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    y = observed[mask]
    p = predicted[mask]
    residual = p - y
    correlation = (
        float(np.corrcoef(y, p)[0, 1])
        if len(y) >= 3 and np.ptp(y) > 0 and np.ptp(p) > 0
        else None
    )
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_obs": float(np.mean(residual)),
        "pearson_r": correlation,
    }


def invalid_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["spm_positive_correlation_length"]):
        reasons.append("nonpositive_correlation_length")
    if not bool(row["spm_ks_within_limit"]):
        reasons.append("k_rms_height_above_limit")
    if not bool(row["spm_slope_within_limit"]):
        reasons.append("rms_height_to_correlation_length_above_limit")
    return "valid" if not reasons else ";".join(reasons)


def add_physics_columns(
    frame: pd.DataFrame,
    frequency_hz: float,
    incidence_angle_deg: float,
    loss_tangent: float,
    ks_limit: float,
    slope_limit: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    result["dielectric_measured"] = result["soil_real_dielectric"].astype(float)
    result["dielectric_topp"] = topp_real_permittivity(
        result["soil_moisture_m3_m3"]
    )
    result["rms_height_m"] = result["pals_rms_height_cm"] / 100.0
    result["correlation_length_m"] = (
        result["pals_correlation_length_cm"] / 100.0
    )

    validity = spm_validity(
        result["rms_height_m"],
        result["correlation_length_m"],
        frequency_hz,
        ks_limit=ks_limit,
        slope_limit=slope_limit,
    )
    result["spm_valid"] = validity["valid"]
    result["spm_positive_correlation_length"] = validity[
        "positive_correlation_length"
    ]
    result["spm_ks_within_limit"] = validity["ks_within_limit"]
    result["spm_slope_within_limit"] = validity["slope_within_limit"]
    result["spm_k_rms_height"] = validity["k_rms_height"]
    result["spm_rms_slope_proxy"] = validity["rms_slope_proxy"]
    result["spm_invalid_reason"] = result.apply(invalid_reason, axis=1)

    source_diagnostics: dict[str, object] = {}
    for source in SOURCE_LABELS:
        epsilon = complex_relative_permittivity(
            result[f"dielectric_{source}"], loss_tangent=loss_tangent
        )
        fresnel_h, fresnel_v = fresnel_reflection_coefficients(
            epsilon, incidence_angle_deg
        )
        result[f"{source}_fresnel_h_power"] = np.abs(fresnel_h) ** 2
        result[f"{source}_fresnel_v_power"] = np.abs(fresnel_v) ** 2

        prediction = spm_backscatter_db(
            epsilon,
            result["rms_height_m"],
            result["correlation_length_m"],
            frequency_hz,
            incidence_angle_deg,
        )
        result[f"{source}_spm_roughness_spectrum_m2"] = prediction[
            "roughness_spectrum_m2"
        ]
        result[f"{source}_spm_alpha_hh_magnitude"] = np.abs(
            prediction["alpha_hh"]
        )
        result[f"{source}_spm_alpha_vv_magnitude"] = np.abs(
            prediction["alpha_vv"]
        )
        positive_length = result["spm_positive_correlation_length"]
        for pol in OBSERVED_COLUMNS:
            values = np.asarray(prediction[f"{pol}_db"], dtype=float)
            result[f"{source}_spm_{pol}_raw_db"] = np.where(
                positive_length, values, np.nan
            )

        source_diagnostics[source] = {
            "mean_fresnel_h_power": float(
                result[f"{source}_fresnel_h_power"].mean()
            ),
            "mean_fresnel_v_power": float(
                result[f"{source}_fresnel_v_power"].mean()
            ),
        }

    metadata = {
        "frequency_hz": float(frequency_hz),
        "wavelength_m": float(wavelength_m(frequency_hz)),
        "incidence_angle_deg": float(incidence_angle_deg),
        "loss_tangent": float(loss_tangent),
        "ks_limit": float(ks_limit),
        "slope_limit": float(slope_limit),
        "source_diagnostics": source_diagnostics,
    }
    return result, metadata


def evaluate_predictions(
    frame: pd.DataFrame, n_splits: int
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    result = frame.copy()
    valid = result["spm_valid"].to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []
    calibration_summary: dict[str, object] = {}

    for source in SOURCE_LABELS:
        for pol, observed_column in OBSERVED_COLUMNS.items():
            raw_column = f"{source}_spm_{pol}_raw_db"
            calibrated_column = f"{source}_spm_{pol}_group_oof_db"
            result[calibrated_column] = np.nan
            subset = valid & np.isfinite(result[observed_column]) & np.isfinite(
                result[raw_column]
            )
            indices = np.flatnonzero(subset)
            observed = result.loc[subset, observed_column].to_numpy(dtype=float)
            predicted = result.loc[subset, raw_column].to_numpy(dtype=float)
            groups = result.loc[subset, "field_id"].astype(str).to_numpy()

            raw_metrics = regression_metrics(observed, predicted)
            rows.append(
                {
                    "dielectric_source": source,
                    "polarization": pol.upper(),
                    "prediction": "raw_spm",
                    **raw_metrics,
                }
            )

            calibrated, offsets = grouped_oof_bias_calibration(
                observed, predicted, groups, n_splits=n_splits
            )
            result.iloc[
                indices, result.columns.get_loc(calibrated_column)
            ] = calibrated
            calibrated_metrics = regression_metrics(observed, calibrated)
            rows.append(
                {
                    "dielectric_source": source,
                    "polarization": pol.upper(),
                    "prediction": "group_oof_bias_calibrated",
                    **calibrated_metrics,
                }
            )
            calibration_summary[f"{source}_{pol}"] = {
                "fold_offsets_db": [float(value) for value in offsets],
                "mean_offset_db": float(np.mean(offsets)),
                "std_offset_db": float(np.std(offsets, ddof=0)),
                "minimum_offset_db": float(np.min(offsets)),
                "maximum_offset_db": float(np.max(offsets)),
            }

    return result, rows, calibration_summary


def scatter_identity(ax, observed, predicted, title: str) -> None:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    ax.scatter(observed[mask], predicted[mask], s=21, alpha=0.58, edgecolors="none")
    lower = float(min(np.min(observed[mask]), np.min(predicted[mask])))
    upper = float(max(np.max(observed[mask]), np.max(predicted[mask])))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed sigma0 (dB)")
    ax.set_ylabel("Predicted sigma0 (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.22)


def save_plots(frame: pd.DataFrame, output_dir: Path, ks_limit: float, slope_limit: float) -> None:
    valid = frame["spm_valid"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(
        frame["soil_real_dielectric"], frame["dielectric_topp"], s=20, alpha=0.6
    )
    extrema = [
        float(min(frame["soil_real_dielectric"].min(), frame["dielectric_topp"].min())),
        float(max(frame["soil_real_dielectric"].max(), frame["dielectric_topp"].max())),
    ]
    axes[0].plot(extrema, extrema, "k--", linewidth=1)
    axes[0].set(xlabel="Measured real permittivity", ylabel="Topp estimate", title="Dielectric comparison")
    axes[0].grid(alpha=0.22)
    dielectric_residual = frame["dielectric_topp"] - frame["soil_real_dielectric"]
    axes[1].scatter(frame["soil_moisture_m3_m3"], dielectric_residual, s=20, alpha=0.6)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Volumetric soil moisture (m3/m3)", ylabel="Topp - measured permittivity", title="Dielectric residual")
    axes[1].grid(alpha=0.22)
    fig.savefig(output_dir / "01_dielectric_comparison.png", dpi=180)
    plt.close(fig)

    for prediction_name, suffix, filename in [
        ("raw", "raw_db", "02_spm_raw_observed_vs_predicted.png"),
        ("OOF offset", "group_oof_db", "03_spm_oof_bias_corrected.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        for row, source in enumerate(SOURCE_LABELS):
            for column, pol in enumerate(OBSERVED_COLUMNS):
                subset = frame.loc[valid]
                scatter_identity(
                    axes[row, column],
                    subset[OBSERVED_COLUMNS[pol]],
                    subset[f"{source}_spm_{pol}_{suffix}"],
                    f"{SOURCE_LABELS[source]} | {POLARIZATION_LABELS[pol]} | {prediction_name}",
                )
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    valid_frame = frame.loc[valid]
    for ax, pol in zip(axes, ["hh", "vv"]):
        raw_residual = valid_frame[f"measured_spm_{pol}_raw_db"] - valid_frame[OBSERVED_COLUMNS[pol]]
        corrected_residual = valid_frame[f"measured_spm_{pol}_group_oof_db"] - valid_frame[OBSERVED_COLUMNS[pol]]
        ax.scatter(valid_frame["soil_moisture_m3_m3"], raw_residual, s=20, alpha=0.48, label="Raw SPM")
        ax.scatter(valid_frame["soil_moisture_m3_m3"], corrected_residual, s=20, alpha=0.48, label="OOF offset")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Volumetric soil moisture (m3/m3)", ylabel="Prediction - observation (dB)", title=f"Residual structure: {pol.upper()}")
        ax.grid(alpha=0.22)
        ax.legend()
    fig.savefig(output_dir / "04_spm_residuals_vs_moisture.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.scatter(frame.loc[~valid, "spm_k_rms_height"], frame.loc[~valid, "spm_rms_slope_proxy"], s=30, alpha=0.62, color="#d95f02", label="Outside conservative domain")
    ax.scatter(frame.loc[valid, "spm_k_rms_height"], frame.loc[valid, "spm_rms_slope_proxy"], s=30, alpha=0.62, color="#1b9e77", label="SPM-valid")
    ax.axvline(ks_limit, color="black", linestyle="--", linewidth=1, label=f"k*s = {ks_limit:g}")
    ax.axhline(slope_limit, color="gray", linestyle="--", linewidth=1, label=f"s/L = {slope_limit:g}")
    ax.set(xlabel="k times RMS height", ylabel="RMS height / correlation length", title="First-order SPM validity diagnostics")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend()
    fig.savefig(output_dir / "05_spm_validity_domain.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    observed_ratio = valid_frame["sigma0_hh_db"] - valid_frame["sigma0_vv_db"]
    measured_ratio = valid_frame["measured_spm_hh_raw_db"] - valid_frame["measured_spm_vv_raw_db"]
    topp_ratio = valid_frame["topp_spm_hh_raw_db"] - valid_frame["topp_spm_vv_raw_db"]
    ax.scatter(valid_frame["soil_real_dielectric"], observed_ratio, s=24, alpha=0.5, label="Observed HH - VV")
    order = np.argsort(valid_frame["soil_real_dielectric"].to_numpy())
    x = valid_frame["soil_real_dielectric"].to_numpy()[order]
    ax.plot(x, measured_ratio.to_numpy()[order], linewidth=1.8, label="SPM, measured dielectric")
    ax.scatter(valid_frame["soil_real_dielectric"], topp_ratio, s=15, alpha=0.45, label="SPM, Topp dielectric")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Measured real permittivity", ylabel="HH - VV (dB)", title="Copolarization-ratio diagnostic")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.savefig(output_dir / "06_copolarization_ratio.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a first-order SPM rough-soil baseline on SMAPVEX12"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument("--loss-tangent", type=float, default=0.0)
    parser.add_argument("--ks-limit", type=float, default=0.3)
    parser.add_argument("--slope-limit", type=float, default=0.21)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_hz = args.frequency_ghz * 1e9

    frame = load_input_table(input_path)
    physics, physics_metadata = add_physics_columns(
        frame,
        frequency_hz=frequency_hz,
        incidence_angle_deg=args.incidence_angle_deg,
        loss_tangent=args.loss_tangent,
        ks_limit=args.ks_limit,
        slope_limit=args.slope_limit,
    )
    evaluated, metric_rows, calibration = evaluate_predictions(
        physics, n_splits=args.folds
    )
    metrics = pd.DataFrame(metric_rows)
    dielectric_metrics = finite_summary(
        evaluated["soil_real_dielectric"], evaluated["dielectric_topp"]
    )

    valid_count = int(evaluated["spm_valid"].sum())
    reason_counts = {
        str(key): int(value)
        for key, value in evaluated["spm_invalid_reason"].value_counts().items()
    }
    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "sample_count": int(len(evaluated)),
        "date_count": int(evaluated["acquisition_date"].nunique()),
        "field_count": int(evaluated["field_id"].nunique()),
        "spm_valid_count": valid_count,
        "spm_invalid_count": int(len(evaluated) - valid_count),
        "spm_valid_fraction": float(valid_count / len(evaluated)),
        "spm_validity_reason_counts": reason_counts,
        "dielectric_topp_vs_measured": dielectric_metrics,
        "physics": physics_metadata,
        "group_oof_bias_calibration": calibration,
        "interpretation_guardrails": [
            "Metrics use only rows inside the configured conservative SPM domain.",
            "The calibrated result fits only an additive dB offset with field-held-out folds.",
            "First-order SPM predicts zero cross-polarized return, so HV/VH are not evaluated.",
            "Measured PALS return may include vegetation, heterogeneity, footprints, and roughness scales absent from this bare-soil first-order model.",
        ],
    }

    csv_frame = evaluated.copy()
    csv_frame["acquisition_date"] = csv_frame["acquisition_date"].dt.strftime(
        "%Y-%m-%d"
    )
    csv_frame.to_csv(output_dir / "spm_predictions.csv", index=False)
    metrics.to_csv(output_dir / "spm_metrics.csv", index=False)
    with (output_dir / "spm_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_plots(evaluated, output_dir, args.ks_limit, args.slope_limit)

    print("First-order SPM baseline completed.")
    print(f"Input samples: {len(evaluated)}")
    print(f"SPM-valid samples: {valid_count} ({valid_count / len(evaluated):.1%})")
    print(f"Fields: {evaluated['field_id'].nunique()}; dates: {evaluated['acquisition_date'].nunique()}")
    print("\nEvaluation metrics (valid domain only):")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
