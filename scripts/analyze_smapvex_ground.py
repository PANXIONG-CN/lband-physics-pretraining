"""Exploratory physics analysis of SMAPVEX12 field-day collocations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


POLARIZATIONS = {
    "VV": "sigma0_vv_db",
    "HH": "sigma0_hh_db",
    "HV": "sigma0_hv_db",
    "VH": "sigma0_vh_db",
}


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input table not found: {path}")
    frame = pd.read_csv(path, dtype={"field_id": "string"})
    frame["field_id"] = frame["field_id"].str.strip()
    frame["acquisition_date"] = pd.to_datetime(
        frame["acquisition_date"], errors="raise"
    )
    required = {
        "acquisition_date",
        "field_id",
        "soil_moisture_m3_m3",
        "pals_rms_height_cm",
        "pals_correlation_length_cm",
        "pals_sample_count",
        "match_distance_m_median",
        *POLARIZATIONS.values(),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Table {path.name} lacks columns: {sorted(missing)}")
    return frame


def field_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    paired = paired[np.isfinite(paired["x"]) & np.isfinite(paired["y"])]
    return paired["x"].to_numpy(), paired["y"].to_numpy()


def association_statistics(
    x: pd.Series,
    y: pd.Series,
) -> dict[str, float | int | None]:
    x_values, y_values = finite_pair(x, y)
    if len(x_values) < 3 or np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        return {
            "n": int(len(x_values)),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "slope": None,
            "intercept": None,
            "r_squared": None,
        }
    pearson = pearsonr(x_values, y_values)
    spearman = spearmanr(x_values, y_values)
    regression = linregress(x_values, y_values)
    return {
        "n": int(len(x_values)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "slope": float(regression.slope),
        "intercept": float(regression.intercept),
        "r_squared": float(regression.rvalue**2),
    }


def within_field_statistics(
    frame: pd.DataFrame,
    predictor: str,
    response: str,
) -> dict[str, float | int | None]:
    working = frame[["field_id", predictor, response]].dropna().copy()
    working["x_within"] = working[predictor] - working.groupby("field_id")[
        predictor
    ].transform("mean")
    working["y_within"] = working[response] - working.groupby("field_id")[
        response
    ].transform("mean")
    return association_statistics(working["x_within"], working["y_within"])


def partial_correlation(
    x: pd.Series,
    y: pd.Series,
    control: pd.Series,
) -> dict[str, float | int | None]:
    working = pd.DataFrame({"x": x, "y": y, "control": control}).dropna()
    if len(working) < 4 or np.ptp(working["control"]) == 0:
        return {"n": int(len(working)), "pearson_r": None, "pearson_p": None}
    x_fit = np.polyfit(working["control"], working["x"], 1)
    y_fit = np.polyfit(working["control"], working["y"], 1)
    x_residual = working["x"] - np.polyval(x_fit, working["control"])
    y_residual = working["y"] - np.polyval(y_fit, working["control"])
    result = pearsonr(x_residual, y_residual)
    return {
        "n": int(len(working)),
        "pearson_r": float(result.statistic),
        "pearson_p": float(result.pvalue),
    }


def build_correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    field_mean = (
        frame.groupby("field_id", as_index=False)
        .agg(
            soil_moisture_m3_m3=("soil_moisture_m3_m3", "mean"),
            pals_rms_height_cm=("pals_rms_height_cm", "first"),
            pals_correlation_length_cm=("pals_correlation_length_cm", "first"),
            **{
                column: (column, "mean")
                for column in POLARIZATIONS.values()
            },
        )
    )

    for polarization, response in POLARIZATIONS.items():
        for scale, result in [
            (
                "pooled_field_day",
                association_statistics(frame["soil_moisture_m3_m3"], frame[response]),
            ),
            (
                "within_field",
                within_field_statistics(frame, "soil_moisture_m3_m3", response),
            ),
        ]:
            rows.append(
                {
                    "polarization": polarization,
                    "predictor": "soil_moisture_m3_m3",
                    "analysis_scale": scale,
                    **result,
                }
            )

        for predictor in ["pals_rms_height_cm", "pals_correlation_length_cm"]:
            result = association_statistics(field_mean[predictor], field_mean[response])
            partial = partial_correlation(
                field_mean[predictor],
                field_mean[response],
                field_mean["soil_moisture_m3_m3"],
            )
            rows.append(
                {
                    "polarization": polarization,
                    "predictor": predictor,
                    "analysis_scale": "between_field",
                    **result,
                    "partial_r_controlling_mean_moisture": partial["pearson_r"],
                    "partial_p_controlling_mean_moisture": partial["pearson_p"],
                }
            )
    return pd.DataFrame(rows)


def compare_quality_policies(
    strict: pd.DataFrame,
    all_flags: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["acquisition_date", "field_id", *POLARIZATIONS.values()]
    shared = strict[columns].merge(
        all_flags[columns],
        on=["acquisition_date", "field_id"],
        how="inner",
        suffixes=("_strict", "_all"),
        validate="one_to_one",
    )
    rows: list[dict] = []
    for polarization, column in POLARIZATIONS.items():
        strict_column = f"{column}_strict"
        all_column = f"{column}_all"
        delta = shared[all_column] - shared[strict_column]
        stats = association_statistics(shared[strict_column], shared[all_column])
        rows.append(
            {
                "polarization": polarization,
                "shared_rows": int(len(shared)),
                "mean_all_minus_strict_db": float(delta.mean()),
                "median_all_minus_strict_db": float(delta.median()),
                "mae_db": float(delta.abs().mean()),
                "max_abs_difference_db": float(delta.abs().max()),
                "pearson_r": stats["pearson_r"],
            }
        )
    return pd.DataFrame(rows), shared


def analyze_radius_sensitivity(
    radius_tables: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    reference_radius = max(radius_tables)
    reference = radius_tables[reference_radius]
    keys = ["acquisition_date", "field_id"]
    rows: list[dict] = []
    for radius, frame in sorted(radius_tables.items()):
        row: dict[str, float | int] = {
            "radius_m": radius,
            "rows": int(len(frame)),
            "fields": int(frame["field_id"].nunique()),
            "dates": int(frame["acquisition_date"].nunique()),
            "median_pals_sample_count": float(frame["pals_sample_count"].median()),
            "median_match_distance_m": float(frame["match_distance_m_median"].median()),
        }
        shared = frame[keys + list(POLARIZATIONS.values())].merge(
            reference[keys + list(POLARIZATIONS.values())],
            on=keys,
            how="inner",
            suffixes=("_radius", "_reference"),
            validate="one_to_one",
        )
        row["rows_shared_with_600m"] = int(len(shared))
        for polarization, column in POLARIZATIONS.items():
            difference = shared[f"{column}_radius"] - shared[f"{column}_reference"]
            row[f"{polarization.lower()}_mae_vs_600m_db"] = float(
                difference.abs().mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_coverage(frame: pd.DataFrame, output: Path) -> None:
    fields = sorted(frame["field_id"].unique(), key=field_sort_key)
    dates = sorted(frame["acquisition_date"].unique())
    matrix = (
        frame.pivot(index="acquisition_date", columns="field_id", values="pals_sample_count")
        .reindex(index=dates, columns=fields)
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(np.log10(matrix.to_numpy() + 1), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(fields)), fields, rotation=90, fontsize=8)
    ax.set_yticks(
        np.arange(len(dates)),
        [pd.Timestamp(value).strftime("%m-%d") for value in dates],
        fontsize=8,
    )
    ax.set_xlabel("Field ID")
    ax.set_ylabel("Acquisition date (2012)")
    ax.set_title("Strict-QC LoAlt PALS coverage (color = log10(sample count + 1))")
    fig.colorbar(image, ax=ax, label="log10(sample count + 1)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_distributions(frame: pd.DataFrame, output: Path) -> None:
    variables = [
        ("soil_moisture_m3_m3", "Soil moisture (m3/m3)"),
        ("pals_rms_height_cm", "RMS height (cm)"),
        ("pals_correlation_length_cm", "Correlation length (cm)"),
        ("pals_sample_count", "PALS samples per field-day"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, label) in zip(axes.flat, variables):
        ax.hist(frame[column].dropna(), bins=18, color="#4472C4", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
    fig.suptitle("Model-ready input distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_moisture_relations(
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    x = frame["soil_moisture_m3_m3"]
    for ax, (polarization, response) in zip(axes.flat, POLARIZATIONS.items()):
        ax.scatter(x, frame[response], s=16, alpha=0.55, color="#4472C4")
        valid_x, valid_y = finite_pair(x, frame[response])
        if len(valid_x) >= 3:
            fit = np.polyfit(valid_x, valid_y, 1)
            x_line = np.linspace(valid_x.min(), valid_x.max(), 100)
            ax.plot(x_line, np.polyval(fit, x_line), color="#C00000", lw=2)
        pooled = correlations[
            (correlations["polarization"] == polarization)
            & (correlations["predictor"] == "soil_moisture_m3_m3")
            & (correlations["analysis_scale"] == "pooled_field_day")
        ].iloc[0]
        within = correlations[
            (correlations["polarization"] == polarization)
            & (correlations["predictor"] == "soil_moisture_m3_m3")
            & (correlations["analysis_scale"] == "within_field")
        ].iloc[0]
        ax.set_title(
            f"{polarization}: pooled r={pooled['pearson_r']:.2f}, "
            f"within-field r={within['pearson_r']:.2f}"
        )
        ax.set_ylabel("Backscatter sigma0 (dB)")
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Soil moisture (m3/m3)")
    axes[1, 1].set_xlabel("Soil moisture (m3/m3)")
    fig.suptitle("Backscatter response to soil moisture")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_roughness_relations(frame: pd.DataFrame, output: Path) -> None:
    field_mean = frame.groupby("field_id", as_index=False).agg(
        pals_rms_height_cm=("pals_rms_height_cm", "first"),
        **{column: (column, "mean") for column in POLARIZATIONS.values()},
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (polarization, response) in zip(axes.flat, POLARIZATIONS.items()):
        ax.scatter(
            field_mean["pals_rms_height_cm"],
            field_mean[response],
            s=34,
            alpha=0.75,
            color="#70AD47",
        )
        stats = association_statistics(
            field_mean["pals_rms_height_cm"], field_mean[response]
        )
        ax.set_title(f"{polarization}: between-field r={stats['pearson_r']:.2f}")
        ax.set_ylabel("Mean backscatter sigma0 (dB)")
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("PALS-direction RMS height (cm)")
    axes[1, 1].set_xlabel("PALS-direction RMS height (cm)")
    fig.suptitle("Field-mean backscatter versus surface roughness")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_polarization_diagnostics(frame: pd.DataFrame, output: Path) -> None:
    hh_minus_vv = frame["sigma0_hh_db"] - frame["sigma0_vv_db"]
    hv_minus_vh = frame["sigma0_hv_db"] - frame["sigma0_vh_db"]
    co_linear = 0.5 * (
        np.power(10.0, frame["sigma0_hh_db"] / 10.0)
        + np.power(10.0, frame["sigma0_vv_db"] / 10.0)
    )
    cross_linear = 0.5 * (
        np.power(10.0, frame["sigma0_hv_db"] / 10.0)
        + np.power(10.0, frame["sigma0_vh_db"] / 10.0)
    )
    co_cross_ratio_db = 10.0 * np.log10(co_linear / cross_linear)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(frame["soil_moisture_m3_m3"], hh_minus_vv, s=16, alpha=0.55)
    axes[0, 0].set(xlabel="Soil moisture (m3/m3)", ylabel="HH - VV (dB)")
    axes[0, 1].scatter(frame["soil_moisture_m3_m3"], hv_minus_vh, s=16, alpha=0.55)
    axes[0, 1].axhline(0.0, color="black", lw=1)
    axes[0, 1].set(xlabel="Soil moisture (m3/m3)", ylabel="HV - VH (dB)")
    axes[1, 0].scatter(frame["sigma0_hv_db"], frame["sigma0_vh_db"], s=16, alpha=0.55)
    limits = [
        min(frame["sigma0_hv_db"].min(), frame["sigma0_vh_db"].min()),
        max(frame["sigma0_hv_db"].max(), frame["sigma0_vh_db"].max()),
    ]
    axes[1, 0].plot(limits, limits, color="black", lw=1)
    axes[1, 0].set(xlabel="HV sigma0 (dB)", ylabel="VH sigma0 (dB)")
    axes[1, 1].scatter(
        frame["soil_moisture_m3_m3"], co_cross_ratio_db, s=16, alpha=0.55
    )
    axes[1, 1].set(
        xlabel="Soil moisture (m3/m3)", ylabel="Co-pol / cross-pol ratio (dB)"
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Polarimetric consistency and contrast diagnostics")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_quality_comparison(shared: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (polarization, column) in zip(axes.flat, POLARIZATIONS.items()):
        x = shared[f"{column}_strict"]
        y = shared[f"{column}_all"]
        difference = (y - x).abs().mean()
        ax.scatter(x, y, s=16, alpha=0.55)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(limits, limits, color="black", lw=1)
        ax.set_title(f"{polarization}: MAE={difference:.3f} dB")
        ax.set_xlabel("Strict QC (dB)")
        ax.set_ylabel("All flags (dB)")
        ax.grid(alpha=0.2)
    fig.suptitle("Heading-quality policy sensitivity on shared field-days")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_radius_sensitivity(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(table["radius_m"], table["rows"], marker="o", label="Rows")
    axes[0].plot(table["radius_m"], table["fields"], marker="s", label="Fields")
    axes[0].set(xlabel="Maximum match distance (m)", ylabel="Count")
    axes[0].legend()
    for polarization in POLARIZATIONS:
        axes[1].plot(
            table["radius_m"],
            table[f"{polarization.lower()}_mae_vs_600m_db"],
            marker="o",
            label=polarization,
        )
    axes[1].set(
        xlabel="Maximum match distance (m)",
        ylabel="MAE relative to 600 m result (dB)",
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Spatial matching radius sensitivity")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SMAPVEX12 field-day collocation products."
    )
    parser.add_argument("--strict-input", required=True)
    parser.add_argument("--all-flags-input", required=True)
    parser.add_argument("--radius-400-input", required=True)
    parser.add_argument("--radius-500-input", required=True)
    parser.add_argument("--radius-600-input", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/scattering/rough_ground/analysis",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strict = load_table(args.strict_input)
    all_flags = load_table(args.all_flags_input)
    radius_tables = {
        400: load_table(args.radius_400_input),
        500: load_table(args.radius_500_input),
        600: load_table(args.radius_600_input),
    }

    correlations = build_correlation_table(strict)
    quality_comparison, quality_shared = compare_quality_policies(strict, all_flags)
    radius_sensitivity = analyze_radius_sensitivity(radius_tables)

    correlations.to_csv(output_dir / "correlation_statistics.csv", index=False)
    quality_comparison.to_csv(
        output_dir / "quality_policy_comparison.csv", index=False
    )
    radius_sensitivity.to_csv(output_dir / "radius_sensitivity.csv", index=False)

    plot_coverage(strict, output_dir / "01_coverage_matrix.png")
    plot_distributions(strict, output_dir / "02_input_distributions.png")
    plot_moisture_relations(
        strict, correlations, output_dir / "03_sigma0_vs_soil_moisture.png"
    )
    plot_roughness_relations(strict, output_dir / "04_sigma0_vs_roughness.png")
    plot_polarization_diagnostics(strict, output_dir / "05_polarization_diagnostics.png")
    plot_quality_comparison(
        quality_shared, output_dir / "06_heading_quality_sensitivity.png"
    )
    plot_radius_sensitivity(
        radius_sensitivity, output_dir / "07_radius_sensitivity.png"
    )

    dates = int(strict["acquisition_date"].nunique())
    fields = int(strict["field_id"].nunique())
    summary = {
        "strict_qc": {
            "rows": int(len(strict)),
            "dates": dates,
            "fields": fields,
            "field_date_coverage_fraction": float(len(strict) / (dates * fields)),
            "soil_moisture_range_m3_m3": [
                float(strict["soil_moisture_m3_m3"].min()),
                float(strict["soil_moisture_m3_m3"].max()),
            ],
            "pals_sample_count_range": [
                int(strict["pals_sample_count"].min()),
                int(strict["pals_sample_count"].max()),
            ],
        },
        "all_flags": {
            "rows": int(len(all_flags)),
            "dates": int(all_flags["acquisition_date"].nunique()),
            "fields": int(all_flags["field_id"].nunique()),
        },
        "quality_policy_shared_rows": int(len(quality_shared)),
        "radius_rows": {
            str(int(row.radius_m)): int(row.rows)
            for row in radius_sensitivity.itertuples(index=False)
        },
        "interpretation_rules": [
            "Treat pooled correlations as descriptive because field-days are repeated observations.",
            "Use within-field moisture correlations for temporal response evidence.",
            "Use one-row-per-field means for roughness relationships because roughness is static.",
            "Do not select a radius only because it gives more rows; prefer stable sigma0 and adequate spatial coverage.",
        ],
    }
    with (output_dir / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print("\nSMAPVEX12 exploratory analysis complete")
    print(f"Strict-QC rows: {len(strict):,}")
    print(f"Strict-QC dates: {dates}")
    print(f"Strict-QC fields: {fields}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
