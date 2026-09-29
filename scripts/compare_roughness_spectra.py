"""Compare Gaussian and exponential roughness spectra in first-order SPM.

Only the roughness correlation model changes.  The collocated observations,
measured dielectric input, incidence angle, validity mask, and field-held-out
evaluation protocol are identical.  This isolates spectrum sensitivity from
all other modelling choices.
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
    )
    from research_pilots.scattering.surfaces.metrics import (
        grouped_oof_bias_calibration,
        regression_metrics,
    )
    from research_pilots.scattering.surfaces.rough_surface import (
        backscatter_spectrum,
        spm_validity,
        wavenumber_rad_m,
    )
    from research_pilots.scattering.surfaces.spm import spm_backscatter_db
except ImportError:  # standalone validation before copying into the repository
    MODULE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(MODULE_DIR))
    from dielectric import complex_relative_permittivity  # type: ignore[no-redef]
    from metrics import (  # type: ignore[no-redef]
        grouped_oof_bias_calibration,
        regression_metrics,
    )
    from rough_surface import (  # type: ignore[no-redef]
        backscatter_spectrum,
        spm_validity,
        wavenumber_rad_m,
    )
    from spm import spm_backscatter_db  # type: ignore[no-redef]


SPECTRA = ("gaussian", "exponential")
OBSERVED = {"hh": "sigma0_hh_db", "vv": "sigma0_vv_db"}
REQUIRED_COLUMNS = {
    "acquisition_date",
    "field_id",
    "soil_real_dielectric",
    "soil_moisture_m3_m3",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
    *OBSERVED.values(),
}


def load_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": "string"})
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")
    frame["field_id"] = frame["field_id"].str.strip()
    frame["acquisition_date"] = pd.to_datetime(
        frame["acquisition_date"], errors="raise"
    )
    return frame


def add_predictions(
    frame: pd.DataFrame,
    frequency_hz: float,
    incidence_angle_deg: float,
    loss_tangent: float,
    ks_limit: float,
    slope_limit: float,
) -> pd.DataFrame:
    result = frame.copy()
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
    result["spm_k_rms_height"] = validity["k_rms_height"]
    result["spm_rms_slope_proxy"] = validity["rms_slope_proxy"]

    epsilon = complex_relative_permittivity(
        result["soil_real_dielectric"], loss_tangent=loss_tangent
    )
    for spectrum in SPECTRA:
        prediction = spm_backscatter_db(
            epsilon,
            result["rms_height_m"],
            result["correlation_length_m"],
            frequency_hz,
            incidence_angle_deg,
            spectrum_model=spectrum,
        )
        result[f"{spectrum}_spectrum_m2"] = prediction[
            "roughness_spectrum_m2"
        ]
        for polarization in OBSERVED:
            result[f"{spectrum}_spm_{polarization}_raw_db"] = np.where(
                result["correlation_length_m"] > 0,
                prediction[f"{polarization}_db"],
                np.nan,
            )
    result["exponential_minus_gaussian_spectrum_db"] = 10.0 * np.log10(
        result["exponential_spectrum_m2"] / result["gaussian_spectrum_m2"]
    )
    return result


def evaluate(
    frame: pd.DataFrame, folds: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    metric_rows: list[dict[str, object]] = []
    offsets_summary: dict[str, object] = {}
    base_mask = result["spm_valid"].to_numpy(dtype=bool)

    for spectrum in SPECTRA:
        for polarization, observed_column in OBSERVED.items():
            raw_column = f"{spectrum}_spm_{polarization}_raw_db"
            calibrated_column = f"{spectrum}_spm_{polarization}_group_oof_db"
            result[calibrated_column] = np.nan
            mask = (
                base_mask
                & np.isfinite(result[observed_column])
                & np.isfinite(result[raw_column])
            )
            indices = np.flatnonzero(mask)
            y = result.loc[mask, observed_column].to_numpy(dtype=float)
            raw = result.loc[mask, raw_column].to_numpy(dtype=float)
            groups = result.loc[mask, "field_id"].astype(str).to_numpy()

            metric_rows.append(
                {
                    "spectrum": spectrum,
                    "polarization": polarization.upper(),
                    "prediction": "raw_spm",
                    **regression_metrics(y, raw),
                }
            )
            calibrated, offsets = grouped_oof_bias_calibration(
                y, raw, groups, n_splits=folds
            )
            result.iloc[
                indices, result.columns.get_loc(calibrated_column)
            ] = calibrated
            metric_rows.append(
                {
                    "spectrum": spectrum,
                    "polarization": polarization.upper(),
                    "prediction": "group_oof_bias_calibrated",
                    **regression_metrics(y, calibrated),
                }
            )
            offsets_summary[f"{spectrum}_{polarization}"] = {
                "fold_offsets_db": [float(value) for value in offsets],
                "mean_offset_db": float(np.mean(offsets)),
                "std_offset_db": float(np.std(offsets, ddof=0)),
            }

    return result, pd.DataFrame(metric_rows), offsets_summary


def scatter_identity(ax, observed, predicted, title: str) -> None:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    ax.scatter(y[mask], p[mask], s=22, alpha=0.55, edgecolors="none")
    lower = float(min(y[mask].min(), p[mask].min()))
    upper = float(max(y[mask].max(), p[mask].max()))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    ax.set(xlabel="Observed sigma0 (dB)", ylabel="Predicted sigma0 (dB)", title=title)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.grid(alpha=0.22)


def save_plots(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    frequency_hz: float,
    incidence_angle_deg: float,
) -> None:
    valid = frame["spm_valid"]
    valid_frame = frame.loc[valid]

    lengths = np.linspace(
        max(0.001, float(valid_frame["correlation_length_m"].min()) * 0.7),
        float(valid_frame["correlation_length_m"].max()) * 1.15,
        300,
    )
    k_value = wavenumber_rad_m(frequency_hz)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for spectrum in SPECTRA:
        values = backscatter_spectrum(
            lengths, k_value, incidence_angle_deg, model=spectrum
        )
        ax.plot(lengths * 100.0, 10.0 * np.log10(values), linewidth=2, label=spectrum.capitalize())
    ax.scatter(
        valid_frame["pals_correlation_length_cm"],
        10.0 * np.log10(valid_frame["gaussian_spectrum_m2"]),
        s=22,
        alpha=0.42,
        label="Observed L values",
    )
    ax.set(xlabel="Correlation length L (cm)", ylabel="Roughness spectrum W (dB m2)", title="Spectrum assumption at the PALS Bragg wavenumber")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.savefig(output_dir / "01_gaussian_vs_exponential_spectrum.png", dpi=180)
    plt.close(fig)

    for suffix, label, filename in [
        ("raw_db", "raw", "02_raw_predictions.png"),
        ("group_oof_db", "field-held-out offset", "03_oof_bias_corrected_predictions.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
        for row, spectrum in enumerate(SPECTRA):
            for column, polarization in enumerate(OBSERVED):
                scatter_identity(
                    axes[row, column],
                    valid_frame[OBSERVED[polarization]],
                    valid_frame[f"{spectrum}_spm_{polarization}_{suffix}"],
                    f"{spectrum.capitalize()} | {polarization.upper()} | {label}",
                )
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    axes[0].scatter(
        valid_frame["pals_correlation_length_cm"],
        valid_frame["exponential_minus_gaussian_spectrum_db"],
        s=25,
        alpha=0.6,
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Correlation length L (cm)", ylabel="Exponential - Gaussian W (dB)", title="Spectrum-induced prediction shift")
    axes[0].grid(alpha=0.22)

    corrected = metrics[metrics["prediction"] == "group_oof_bias_calibrated"].copy()
    labels = corrected["spectrum"].str.capitalize() + " " + corrected["polarization"]
    axes[1].bar(labels, corrected["rmse_db"], color=["#4c78a8", "#4c78a8", "#f58518", "#f58518"])
    axes[1].set(ylabel="Field-held-out RMSE (dB)", title="Fair comparison after offset calibration")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.22)
    fig.savefig(output_dir / "04_spectrum_sensitivity_and_ranking.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, polarization in zip(axes, OBSERVED):
        for spectrum in SPECTRA:
            residual = (
                valid_frame[f"{spectrum}_spm_{polarization}_group_oof_db"]
                - valid_frame[OBSERVED[polarization]]
            )
            ax.scatter(
                valid_frame["soil_moisture_m3_m3"],
                residual,
                s=22,
                alpha=0.48,
                label=spectrum.capitalize(),
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Volumetric soil moisture (m3/m3)", ylabel="Prediction - observation (dB)", title=f"Residual structure: {polarization.upper()}")
        ax.grid(alpha=0.22)
        ax.legend()
    fig.savefig(output_dir / "05_residuals_vs_soil_moisture.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Gaussian and exponential SPM roughness spectra"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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

    frame = load_table(input_path)
    predictions = add_predictions(
        frame,
        frequency_hz=frequency_hz,
        incidence_angle_deg=args.incidence_angle_deg,
        loss_tangent=args.loss_tangent,
        ks_limit=args.ks_limit,
        slope_limit=args.slope_limit,
    )
    evaluated, metrics, offsets = evaluate(predictions, folds=args.folds)
    valid = evaluated["spm_valid"]
    raw = metrics[metrics["prediction"] == "raw_spm"]
    corrected = metrics[metrics["prediction"] == "group_oof_bias_calibrated"]

    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "samples": int(len(evaluated)),
        "valid_samples": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "fields": int(evaluated.loc[valid, "field_id"].nunique()),
        "dates": int(evaluated.loc[valid, "acquisition_date"].nunique()),
        "frequency_ghz": float(args.frequency_ghz),
        "incidence_angle_deg": float(args.incidence_angle_deg),
        "roughness_spectrum_shift_db": {
            "mean_exponential_minus_gaussian": float(
                evaluated.loc[valid, "exponential_minus_gaussian_spectrum_db"].mean()
            ),
            "minimum": float(
                evaluated.loc[valid, "exponential_minus_gaussian_spectrum_db"].min()
            ),
            "maximum": float(
                evaluated.loc[valid, "exponential_minus_gaussian_spectrum_db"].max()
            ),
        },
        "raw_metric_records": raw.to_dict(orient="records"),
        "group_oof_metric_records": corrected.to_dict(orient="records"),
        "calibration_offsets": offsets,
        "decision_rule": (
            "Prefer the spectrum with lower field-held-out RMSE only if the "
            "difference is practically meaningful and residual structure also improves."
        ),
    }

    output_frame = evaluated.copy()
    output_frame["acquisition_date"] = output_frame[
        "acquisition_date"
    ].dt.strftime("%Y-%m-%d")
    output_frame.to_csv(output_dir / "spectrum_predictions.csv", index=False)
    metrics.to_csv(output_dir / "spectrum_metrics.csv", index=False)
    with (output_dir / "spectrum_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_plots(
        evaluated,
        metrics,
        output_dir,
        frequency_hz=frequency_hz,
        incidence_angle_deg=args.incidence_angle_deg,
    )

    print("Roughness-spectrum comparison completed.")
    print(f"Samples: {len(evaluated)}; SPM-valid: {int(valid.sum())} ({valid.mean():.1%})")
    print("\nMetrics on the common valid domain:")
    print(metrics.to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
