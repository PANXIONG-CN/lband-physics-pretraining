"""Build field-day SMAPVEX12 tables for rough-ground scattering analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from research_pilots.scattering.data.read_smapvex import (  # noqa: E402
        SIGMA0_COLUMNS,
        discover_smapvex_files,
        parse_pals_filename,
        read_pals_file,
        read_soil_moisture,
        read_surface_roughness,
    )
except ModuleNotFoundError:  # standalone smoke test before copying into repository
    from read_smapvex import (  # type: ignore[no-redef]  # noqa: E402
        SIGMA0_COLUMNS,
        discover_smapvex_files,
        parse_pals_filename,
        read_pals_file,
        read_soil_moisture,
        read_surface_roughness,
    )


def add_field_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the experimental field from Site_ID, e.g. 11-3 -> 11."""
    result = frame.copy()
    result["field_id"] = (
        result["site_id"].astype("string").str.strip().str.split("-", n=1).str[0]
    )
    if result["field_id"].isna().any():
        raise ValueError("Some Site_ID values could not be converted to field_id")
    return result


def build_field_geometry(soil: pd.DataFrame) -> pd.DataFrame:
    """Represent each agricultural field by its 16-site grid and centroid."""
    sites = (
        soil[["field_id", "site_id", "utm_x", "utm_y"]]
        .dropna(subset=["utm_x", "utm_y"])
        .drop_duplicates(subset=["field_id", "site_id"])
    )
    return (
        sites.groupby("field_id", as_index=False)
        .agg(
            field_site_count=("site_id", "nunique"),
            field_utm_x=("utm_x", "mean"),
            field_utm_y=("utm_y", "mean"),
            field_utm_x_min=("utm_x", "min"),
            field_utm_x_max=("utm_x", "max"),
            field_utm_y_min=("utm_y", "min"),
            field_utm_y_max=("utm_y", "max"),
        )
        .sort_values("field_id")
        .reset_index(drop=True)
    )


def build_soil_field_day(soil: pd.DataFrame) -> pd.DataFrame:
    """Average probes equally by site first, then by field and date."""
    working = soil.copy()
    working["acquisition_date"] = working["sample_date"].dt.normalize()
    working = working.dropna(
        subset=["acquisition_date", "field_id", "site_id", "soil_moisture_m3_m3"]
    )

    site_day = (
        working.groupby(
            ["acquisition_date", "field_id", "site_id"], as_index=False
        )
        .agg(
            soil_moisture_site_m3_m3=("soil_moisture_m3_m3", "median"),
            soil_real_dielectric_site=("soil_real_dielectric", "median"),
            soil_observation_count=("soil_moisture_m3_m3", "count"),
        )
    )

    field_day = (
        site_day.groupby(["acquisition_date", "field_id"], as_index=False)
        .agg(
            soil_moisture_m3_m3=("soil_moisture_site_m3_m3", "mean"),
            soil_moisture_median_m3_m3=("soil_moisture_site_m3_m3", "median"),
            soil_moisture_std_m3_m3=("soil_moisture_site_m3_m3", "std"),
            soil_site_count=("site_id", "nunique"),
            soil_observation_count=("soil_observation_count", "sum"),
            soil_real_dielectric=("soil_real_dielectric_site", "mean"),
            soil_real_dielectric_median=("soil_real_dielectric_site", "median"),
            soil_real_dielectric_std=("soil_real_dielectric_site", "std"),
        )
        .sort_values(["acquisition_date", "field_id"])
        .reset_index(drop=True)
    )
    return field_day


def build_field_roughness(roughness: pd.DataFrame) -> pd.DataFrame:
    """Create one PALS-direction roughness estimate per field."""
    return (
        roughness.groupby("field_id", as_index=False)
        .agg(
            pals_rms_height_cm=("pals_rms_height_cm", "mean"),
            pals_rms_height_cm_median=("pals_rms_height_cm", "median"),
            pals_rms_height_cm_std=("pals_rms_height_cm", "std"),
            pals_correlation_length_cm=("pals_correlation_length_cm", "mean"),
            pals_correlation_length_cm_median=(
                "pals_correlation_length_cm",
                "median",
            ),
            pals_correlation_length_cm_std=("pals_correlation_length_cm", "std"),
            roughness_site_count=("site_id", "nunique"),
        )
        .sort_values("field_id")
        .reset_index(drop=True)
    )


def aggregate_pals_by_field(
    frame: pd.DataFrame,
    field_geometry: pd.DataFrame,
    max_distance_m: float,
) -> tuple[pd.DataFrame, int]:
    """Assign clean PALS samples to their nearest field centroid and aggregate."""
    tree = cKDTree(field_geometry[["field_utm_x", "field_utm_y"]].to_numpy())
    distance, index = tree.query(frame[["utm_x", "utm_y"]].to_numpy(), k=1)
    matched = frame.loc[distance <= max_distance_m].copy()
    matched_distance = distance[distance <= max_distance_m]
    matched_index = index[distance <= max_distance_m]

    if matched.empty:
        return pd.DataFrame(), 0

    matched["field_id"] = field_geometry.iloc[matched_index]["field_id"].to_numpy()
    matched["match_distance_m"] = matched_distance
    for column in SIGMA0_COLUMNS:
        matched[f"{column}_linear"] = np.power(10.0, matched[column] / 10.0)

    named_aggregations: dict[str, tuple[str, str]] = {
        "pals_sample_count": ("field_id", "size"),
        "match_distance_m_mean": ("match_distance_m", "mean"),
        "match_distance_m_median": ("match_distance_m", "median"),
        "match_distance_m_max": ("match_distance_m", "max"),
        "pals_utc_seconds_min": ("utc_seconds", "min"),
        "pals_utc_seconds_max": ("utc_seconds", "max"),
    }
    for column in SIGMA0_COLUMNS:
        named_aggregations[f"{column}_median"] = (column, "median")
        named_aggregations[f"{column}_std"] = (column, "std")
        named_aggregations[f"{column}_linear_mean"] = (
            f"{column}_linear",
            "mean",
        )

    result = matched.groupby("field_id", as_index=False).agg(**named_aggregations)
    for column in SIGMA0_COLUMNS:
        linear_column = f"{column}_linear_mean"
        # This is the physically meaningful spatial mean: average power first,
        # then convert the mean back to dB.
        result[column] = 10.0 * np.log10(result[linear_column])

    return result, len(matched)


def collocate_pals(
    pals_files: list[Path],
    field_geometry: pd.DataFrame,
    soil_field_day: pd.DataFrame,
    max_distance_m: float,
    sigma_min_db: float,
    sigma_max_db: float,
    heading_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    inventory: list[dict] = []
    low_altitude_files = [
        path
        for path in pals_files
        if parse_pals_filename(path)["altitude_mode"] == "LoAlt"
    ]

    for number, path in enumerate(low_altitude_files, start=1):
        metadata = parse_pals_filename(path)
        date = pd.Timestamp(metadata["acquisition_date"]).normalize()
        daily_fields = soil_field_day.loc[
            soil_field_day["acquisition_date"] == date, "field_id"
        ].unique()
        geometry = field_geometry[field_geometry["field_id"].isin(daily_fields)].copy()

        print(f"[{number:02d}/{len(low_altitude_files):02d}] collocating {path.name}")
        pals = read_pals_file(path)
        raw_rows = len(pals)
        actual_flag_zero_rows = int(pals["heading_uncertainty_flag"].eq(0).sum())
        flag_clean = (
            pals["heading_uncertainty_flag"].eq(0)
            if heading_policy == "zero"
            else pd.Series(True, index=pals.index)
        )
        retained_heading_rows = int(flag_clean.sum())
        pals = pals.loc[flag_clean].copy()

        range_clean = pd.Series(True, index=pals.index)
        for column in SIGMA0_COLUMNS:
            range_clean &= pals[column].between(sigma_min_db, sigma_max_db)
        range_rows = int(range_clean.sum())
        pals = pals.loc[range_clean].copy()

        aggregated, matched_rows = aggregate_pals_by_field(
            pals,
            geometry,
            max_distance_m=max_distance_m,
        )
        if not aggregated.empty:
            aggregated.insert(0, "acquisition_date", date)
            outputs.append(aggregated)

        inventory.append(
            {
                "file_name": path.name,
                "acquisition_date": date.date().isoformat(),
                "raw_rows": raw_rows,
                "heading_flag_zero_rows": actual_flag_zero_rows,
                "heading_policy_retained_rows": retained_heading_rows,
                "sigma_range_clean_rows": range_rows,
                "spatially_matched_rows": matched_rows,
                "matched_field_count": int(aggregated["field_id"].nunique())
                if not aggregated.empty
                else 0,
            }
        )

    if not outputs:
        raise RuntimeError("No LoAlt PALS records matched the ground fields")
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(inventory)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collocate SMAPVEX12 LoAlt PALS and ground observations by field-day."
    )
    parser.add_argument(
        "--input-root",
        default="data/scattering/raw/SMAPVEX12",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/scattering/rough_ground/collocation",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=600.0,
        help="Maximum distance from a PALS geolocation to the nearest field centroid",
    )
    parser.add_argument("--sigma-min-db", type=float, default=-40.0)
    parser.add_argument("--sigma-max-db", type=float, default=0.0)
    parser.add_argument("--min-pals-samples", type=int, default=10)
    parser.add_argument(
        "--heading-policy",
        choices=["zero", "all"],
        default="zero",
        help="Use only heading flag 0 (main analysis) or all flags (sensitivity run)",
    )
    args = parser.parse_args()

    if args.max_distance_m <= 0 or args.min_pals_samples <= 0:
        raise ValueError("Distance and sample-count thresholds must be positive")
    if args.sigma_min_db >= args.sigma_max_db:
        raise ValueError("sigma-min-db must be smaller than sigma-max-db")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_smapvex_files(args.input_root)

    soil = add_field_id(
        read_soil_moisture(
            files["soil_moisture_data"],
            files["soil_moisture_coordinates"],
        )
    )
    roughness = add_field_id(
        read_surface_roughness(
            files["surface_roughness_data"],
            files["surface_roughness_coordinates"],
        )
    )
    field_geometry = build_field_geometry(soil)
    soil_field_day = build_soil_field_day(soil)
    field_roughness = build_field_roughness(roughness)

    pals_field_day, inventory = collocate_pals(
        files["pals_files"],
        field_geometry,
        soil_field_day,
        max_distance_m=args.max_distance_m,
        sigma_min_db=args.sigma_min_db,
        sigma_max_db=args.sigma_max_db,
        heading_policy=args.heading_policy,
    )

    all_rows = (
        soil_field_day.merge(
            field_geometry,
            on="field_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            pals_field_day,
            on=["acquisition_date", "field_id"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            field_roughness,
            on="field_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["acquisition_date", "field_id"])
        .reset_index(drop=True)
    )

    all_rows["roughness_available"] = all_rows[
        ["pals_rms_height_cm", "pals_correlation_length_cm"]
    ].notna().all(axis=1)
    all_rows["roughness_expected_range"] = (
        all_rows["pals_rms_height_cm"].between(0.0, 3.0)
        & all_rows["pals_correlation_length_cm"].between(0.0, 30.0)
    )
    all_rows["enough_pals_samples"] = (
        all_rows["pals_sample_count"] >= args.min_pals_samples
    )
    all_rows["model_ready"] = (
        all_rows["roughness_available"]
        & all_rows["enough_pals_samples"]
        & all_rows["soil_moisture_m3_m3"].notna()
        & all_rows[SIGMA0_COLUMNS].notna().all(axis=1)
    )

    wavelength_m = 299_792_458.0 / 1.26e9
    wavenumber_rad_m = 2.0 * np.pi / wavelength_m
    all_rows["frequency_ghz"] = 1.26
    all_rows["nominal_incidence_angle_deg"] = 40.0
    all_rows["wavelength_m"] = wavelength_m
    all_rows["k_rms_height"] = (
        wavenumber_rad_m * all_rows["pals_rms_height_cm"] / 100.0
    )
    all_rows["correlation_length_wavelengths"] = (
        all_rows["pals_correlation_length_cm"] / 100.0 / wavelength_m
    )

    model_ready = all_rows.loc[all_rows["model_ready"]].copy()
    all_rows["acquisition_date"] = all_rows["acquisition_date"].dt.date.astype(str)
    model_ready["acquisition_date"] = model_ready[
        "acquisition_date"
    ].dt.date.astype(str)

    all_rows.to_csv(output_dir / "smapvex_field_day_collocation_all.csv", index=False)
    model_ready.to_csv(
        output_dir / "smapvex_field_day_model_ready.csv",
        index=False,
    )
    inventory.to_csv(output_dir / "collocation_inventory.csv", index=False)
    field_geometry.to_csv(output_dir / "field_geometry.csv", index=False)
    field_roughness.to_csv(output_dir / "field_roughness.csv", index=False)

    summary = {
        "dataset": "SMAPVEX12",
        "method": "LoAlt PALS nearest-field-centroid field-day aggregation",
        "parameters": {
            "max_distance_m": args.max_distance_m,
            "sigma0_valid_range_db": [args.sigma_min_db, args.sigma_max_db],
            "heading_policy": args.heading_policy,
            "min_pals_samples": args.min_pals_samples,
            "pals_altitude_mode": "LoAlt",
        },
        "ground_field_count": int(field_geometry["field_id"].nunique()),
        "soil_field_day_rows": int(len(soil_field_day)),
        "pals_field_day_rows": int(len(pals_field_day)),
        "collocated_rows": int(len(all_rows)),
        "model_ready_rows": int(len(model_ready)),
        "collocated_field_count": int(all_rows["field_id"].nunique()),
        "collocated_date_count": int(all_rows["acquisition_date"].nunique()),
        "model_ready_field_count": int(model_ready["field_id"].nunique()),
        "model_ready_date_count": int(model_ready["acquisition_date"].nunique()),
        "dates_without_retained_quality_rows": inventory.loc[
            inventory["heading_policy_retained_rows"] == 0,
            "acquisition_date",
        ].astype(str).tolist()
        if args.heading_policy == "zero"
        else [],
        "fields_without_roughness": sorted(
            all_rows.loc[~all_rows["roughness_available"], "field_id"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "rows_outside_expected_roughness_range": int(
            (~all_rows["roughness_expected_range"] & all_rows["roughness_available"]).sum()
        ),
        "acceptance_checks": {
            "no_duplicate_field_date": not all_rows.duplicated(
                ["acquisition_date", "field_id"]
            ).any(),
            "retains_at_least_10_dates": int(
                all_rows["acquisition_date"].nunique()
            )
            >= 10,
            "model_ready_table_nonempty": not model_ready.empty,
            "all_model_rows_have_roughness": bool(
                model_ready["roughness_available"].all()
            ),
            "all_model_rows_meet_min_pals_samples": bool(
                model_ready["enough_pals_samples"].all()
            ),
        },
        "scientific_notes": [
            "Soil probes are averaged by site before field aggregation, preventing unequal probe counts from weighting a site more heavily.",
            "PALS power is averaged in linear units and only then converted back to dB.",
            "Field and acquisition_date remain explicit grouping variables for leakage-safe train/test splits.",
            "The 600 m centroid threshold is an initial approximation and should be tested in a later 400/500/600 m sensitivity analysis.",
            "With heading_policy=zero, the first four flight dates are expected to be excluded because every LoAlt record has flag 1.",
        ],
    }
    with (output_dir / "collocation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print("\nSMAPVEX12 field-day collocation complete")
    print(json.dumps(summary["acceptance_checks"], indent=2, ensure_ascii=False))
    print(f"Collocated rows: {len(all_rows):,}")
    print(f"Model-ready rows: {len(model_ready):,}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
