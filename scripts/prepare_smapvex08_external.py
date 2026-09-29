"""Build the locked SMAPVEX08 field-day external-validation table."""

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
import rasterio
from rasterio.warp import transform
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from research_pilots.scattering.data.read_smapvex08 import (
        SIGMA0_COLUMNS,
        discover_files,
        iter_pals_chunks,
        linear_power_mean_db,
        parse_pals_filename,
        read_matchup,
        read_soil_moisture,
        read_surface_roughness,
        read_vegetation,
    )
except ModuleNotFoundError:
    from read_smapvex08 import (  # type: ignore[no-redef]
        SIGMA0_COLUMNS,
        discover_files,
        iter_pals_chunks,
        linear_power_mean_db,
        parse_pals_filename,
        read_matchup,
        read_soil_moisture,
        read_surface_roughness,
        read_vegetation,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_geometry(soil: pd.DataFrame) -> pd.DataFrame:
    sites = (
        soil[["field_id", "site_id", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates()
    )
    geometry = (
        sites.groupby("field_id", as_index=False)
        .agg(
            field_site_count=("site_id", "nunique"),
            field_latitude=("latitude", "mean"),
            field_longitude=("longitude", "mean"),
        )
        .sort_values("field_id")
        .reset_index(drop=True)
    )
    x, y = transform(
        "EPSG:4326",
        "EPSG:32618",
        geometry["field_longitude"].tolist(),
        geometry["field_latitude"].tolist(),
    )
    geometry["field_utm_x"] = x
    geometry["field_utm_y"] = y
    return geometry


def aggregate_soil(soil: pd.DataFrame) -> pd.DataFrame:
    site_day = (
        soil.dropna(subset=["soil_moisture_m3_m3"])
        .groupby(["acquisition_date", "field_id", "site_id"], as_index=False)
        .agg(
            soil_moisture_site_m3_m3=("soil_moisture_m3_m3", "median"),
            gravimetric_vsm_site_m3_m3=("gravimetric_vsm_m3_m3", "median"),
            probe_position_count=("probe_position_count", "sum"),
        )
    )
    return (
        site_day.groupby(["acquisition_date", "field_id"], as_index=False)
        .agg(
            soil_moisture_m3_m3=("soil_moisture_site_m3_m3", "mean"),
            soil_moisture_median_m3_m3=("soil_moisture_site_m3_m3", "median"),
            soil_moisture_std_m3_m3=("soil_moisture_site_m3_m3", "std"),
            gravimetric_vsm_m3_m3=("gravimetric_vsm_site_m3_m3", "mean"),
            soil_site_count=("site_id", "nunique"),
            probe_position_count=("probe_position_count", "sum"),
        )
        .sort_values(["acquisition_date", "field_id"])
        .reset_index(drop=True)
    )


def aggregate_roughness(roughness: pd.DataFrame) -> pd.DataFrame:
    return (
        roughness.dropna(subset=["field_id"])
        .groupby("field_id", as_index=False)
        .agg(
            pals_rms_height_cm=("pals_rms_height_cm", "mean"),
            pals_rms_height_cm_std=("pals_rms_height_cm", "std"),
            pals_correlation_length_cm=("pals_correlation_length_cm", "mean"),
            pals_correlation_length_cm_std=("pals_correlation_length_cm", "std"),
            roughness_profile_count=("Name", "size"),
        )
        .sort_values("field_id")
        .reset_index(drop=True)
    )


def aggregate_vegetation(vegetation: pd.DataFrame) -> pd.DataFrame:
    return (
        vegetation.groupby(["acquisition_date", "field_id"], as_index=False)
        .agg(
            vwc_kg_m2=("vwc_kg_m2", "mean"),
            lai=("lai", "mean"),
            crop=("crop", "first"),
        )
        .sort_values(["acquisition_date", "field_id"])
    )


def inspect_vwc_raster(path: Path) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        # Read only a small window.  The 270 MB raster must never be loaded in full
        # just to inspect its contract.
        window = rasterio.windows.Window(
            0, 0, min(512, dataset.width), min(512, dataset.height)
        )
        sample = dataset.read(1, window=window, masked=True)
        return {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "shape": [dataset.height, dataset.width],
            "bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
            "pixel_size_m": [abs(dataset.transform.a), abs(dataset.transform.e)],
            "bounds": list(dataset.bounds),
            "nodata": dataset.nodata,
            "inspection_window_valid_count": int(sample.count()),
        }


def aggregate_pals(
    pals_files: list[Path],
    geometry: pd.DataFrame,
    valid_dates: set[pd.Timestamp],
    radius_m: float,
    sigma_min_db: float,
    sigma_max_db: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = geometry.reset_index(drop=True)
    tree = cKDTree(active[["field_utm_x", "field_utm_y"]].to_numpy())
    pieces: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []

    for number, path in enumerate(pals_files, start=1):
        metadata = parse_pals_filename(path)
        date = pd.Timestamp(metadata["acquisition_date"])
        raw_rows = qc_rows = matched_rows = 0
        matched_fields: set[str] = set()
        if date in valid_dates:
            print(f"[{number:02d}/{len(pals_files):02d}] {path.name}")
            for chunk in iter_pals_chunks(path):
                raw_rows += len(chunk)
                clean = (
                    chunk["incidence_angle_deg"].between(30.0, 50.0)
                    & chunk["sigma0_hh_db"].between(sigma_min_db, sigma_max_db)
                    & chunk["sigma0_vv_db"].between(sigma_min_db, sigma_max_db)
                )
                chunk = chunk.loc[clean].copy()
                qc_rows += len(chunk)
                if chunk.empty:
                    continue
                x, y = transform(
                    "EPSG:4326",
                    "EPSG:32618",
                    chunk["longitude"].tolist(),
                    chunk["latitude"].tolist(),
                )
                distance, index = tree.query(np.column_stack([x, y]), k=1)
                keep = distance <= radius_m
                if not np.any(keep):
                    continue
                matched = chunk.loc[keep, [
                    "source_file",
                    "acquisition_date",
                    "incidence_angle_deg",
                    "range_m",
                    *SIGMA0_COLUMNS,
                ]].copy()
                matched["field_id"] = active.iloc[index[keep]]["field_id"].to_numpy()
                matched["match_distance_m"] = distance[keep]
                matched_rows += len(matched)
                matched_fields.update(matched["field_id"].astype(str))
                pieces.append(matched)
        inventory.append(
            {
                "source_file": path.name,
                "acquisition_date": date.date().isoformat(),
                "used_for_ground_match": date in valid_dates,
                "raw_rows": raw_rows,
                "qc_rows": qc_rows,
                "matched_rows": matched_rows,
                "matched_field_count": len(matched_fields),
            }
        )

    if not pieces:
        raise RuntimeError("No quality-controlled PALS samples matched the roughness fields")
    matched = pd.concat(pieces, ignore_index=True)

    rows: list[dict[str, object]] = []
    for (date, field), group in matched.groupby(["acquisition_date", "field_id"]):
        row: dict[str, object] = {
            "acquisition_date": pd.Timestamp(date),
            "field_id": str(field),
            "pals_sample_count": len(group),
            "pals_flight_file_count": group["source_file"].nunique(),
            "incidence_angle_deg": group["incidence_angle_deg"].mean(),
            "range_m_mean": group["range_m"].mean(),
            "match_distance_m_mean": group["match_distance_m"].mean(),
            "match_distance_m_max": group["match_distance_m"].max(),
        }
        for column in SIGMA0_COLUMNS:
            row[column] = linear_power_mean_db(group[column])
            row[f"{column}_median"] = group[column].median()
            row[f"{column}_std"] = group[column].std()
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(inventory)


def radius_sensitivity(
    pals_files: list[Path],
    geometry: pd.DataFrame,
    valid_dates: set[pd.Timestamp],
    radii: list[float],
) -> pd.DataFrame:
    """Count support at several radii without using target errors to select one."""
    active = geometry.reset_index(drop=True)
    tree = cKDTree(active[["field_utm_x", "field_utm_y"]].to_numpy())
    counters = {radius: {"rows": 0, "pairs": set()} for radius in radii}
    for path in pals_files:
        date = pd.Timestamp(parse_pals_filename(path)["acquisition_date"])
        if date not in valid_dates:
            continue
        for chunk in iter_pals_chunks(path):
            clean = (
                chunk["incidence_angle_deg"].between(30.0, 50.0)
                & chunk["sigma0_hh_db"].between(-40.0, 0.0)
                & chunk["sigma0_vv_db"].between(-40.0, 0.0)
            )
            chunk = chunk.loc[clean]
            if chunk.empty:
                continue
            x, y = transform(
                "EPSG:4326", "EPSG:32618", chunk.longitude.tolist(), chunk.latitude.tolist()
            )
            distance, index = tree.query(np.column_stack([x, y]), k=1)
            for radius in radii:
                keep = distance <= radius
                counters[radius]["rows"] += int(np.sum(keep))
                fields = active.iloc[index[keep]]["field_id"].astype(str).tolist()
                counters[radius]["pairs"].update((date.date().isoformat(), f) for f in fields)
    return pd.DataFrame(
        [
            {
                "radius_m": radius,
                "matched_pals_rows": values["rows"],
                "matched_field_days": len(values["pairs"]),
                "matched_fields": len({field for _, field in values["pairs"]}),
                "matched_dates": len({date for date, _ in values["pairs"]}),
            }
            for radius, values in counters.items()
        ]
    )


def save_figures(table: pd.DataFrame, sensitivity: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].scatter(
        table["soil_moisture_m3_m3"], table["sigma0_hh_db"], label="HH", alpha=0.75
    )
    axes[0].scatter(
        table["soil_moisture_m3_m3"], table["sigma0_vv_db"], label="VV", alpha=0.75
    )
    axes[0].set(
        xlabel="Site-calibrated volumetric soil moisture (m3/m3)",
        ylabel="PALS sigma0 (dB)",
        title="SMAPVEX08 external observations",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(
        sensitivity["radius_m"], sensitivity["matched_field_days"], marker="o"
    )
    axes[1].set(
        xlabel="Nearest-field radius (m)",
        ylabel="Matched roughness field-days",
        title="Spatial-support sensitivity (count only)",
    )
    axes[1].grid(alpha=0.2)
    fig.savefig(output / "01_external_data_and_radius_support.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the SMAPVEX08 independent field-day table"
    )
    parser.add_argument(
        "--input-root", default="data/scattering/raw/SMAPVEX08", type=Path
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/scattering/rough_ground/smapvex08_external_v1",
        type=Path,
    )
    parser.add_argument("--match-radius-m", type=float, default=500.0)
    parser.add_argument("--min-pals-samples", type=int, default=50)
    parser.add_argument("--sigma-min-db", type=float, default=-40.0)
    parser.add_argument("--sigma-max-db", type=float, default=0.0)
    parser.add_argument(
        "--radius-sensitivity", default="300,400,500,600,800"
    )
    args = parser.parse_args()
    if args.match_radius_m <= 0 or args.min_pals_samples <= 0:
        raise ValueError("Radius and minimum sample count must be positive")

    output = args.output_dir.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; use a new versioned directory")
    output.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.input_root.resolve())
    soil = read_soil_moisture(files["soil_moisture"], files["soil_coordinates"])
    roughness = read_surface_roughness(files["surface_roughness"])
    vegetation = read_vegetation(files["vegetation_in_situ"])
    matchup = read_matchup(files["matchup"])

    geometry = field_geometry(soil)
    soil_field_day = aggregate_soil(soil)
    roughness_field = aggregate_roughness(roughness)
    vegetation_field_day = aggregate_vegetation(vegetation)
    rough_fields = set(roughness_field["field_id"].astype(str))
    active_geometry = geometry.loc[geometry["field_id"].isin(rough_fields)].copy()
    valid_dates = set(pd.to_datetime(soil_field_day["acquisition_date"]).dt.normalize())

    pals_field_day, inventory = aggregate_pals(
        files["pals_files"],
        active_geometry,
        valid_dates,
        args.match_radius_m,
        args.sigma_min_db,
        args.sigma_max_db,
    )
    radii = [float(value) for value in args.radius_sensitivity.split(",")]
    sensitivity = radius_sensitivity(
        files["pals_files"], active_geometry, valid_dates, radii
    )

    all_rows = (
        soil_field_day.merge(geometry, on="field_id", how="left", validate="many_to_one")
        .merge(pals_field_day, on=["acquisition_date", "field_id"], how="inner", validate="one_to_one")
        .merge(roughness_field, on="field_id", how="left", validate="many_to_one")
        .merge(vegetation_field_day, on=["acquisition_date", "field_id"], how="left", validate="one_to_one")
        .sort_values(["acquisition_date", "field_id"])
        .reset_index(drop=True)
    )
    all_rows.insert(0, "campaign_id", "SMAPVEX08")
    all_rows["frequency_ghz"] = 1.26
    all_rows["nominal_incidence_angle_deg"] = 40.0
    all_rows["roughness_available"] = all_rows[
        ["pals_rms_height_cm", "pals_correlation_length_cm"]
    ].notna().all(axis=1)
    all_rows["model_ready"] = (
        all_rows["roughness_available"]
        & all_rows["soil_moisture_m3_m3"].between(0.0, 0.60)
        & all_rows["sigma0_hh_db"].between(args.sigma_min_db, args.sigma_max_db)
        & all_rows["sigma0_vv_db"].between(args.sigma_min_db, args.sigma_max_db)
        & (all_rows["pals_sample_count"] >= args.min_pals_samples)
        & (all_rows["pals_rms_height_cm"] > 0)
        & (all_rows["pals_correlation_length_cm"] > 0)
    )
    model_ready = all_rows.loc[all_rows["model_ready"]].copy()

    for frame in (all_rows, model_ready, soil_field_day, pals_field_day, vegetation_field_day):
        if "acquisition_date" in frame:
            frame["acquisition_date"] = pd.to_datetime(frame["acquisition_date"]).dt.date.astype(str)

    all_rows.to_csv(output / "smapvex08_field_day_collocation_all.csv", index=False)
    model_ready.to_csv(output / "smapvex08_field_day_model_ready.csv", index=False)
    inventory.to_csv(output / "pals_file_inventory.csv", index=False)
    sensitivity.to_csv(output / "match_radius_sensitivity.csv", index=False)
    geometry.to_csv(output / "field_geometry.csv", index=False)
    soil_field_day.to_csv(output / "soil_field_day.csv", index=False)
    roughness_field.to_csv(output / "field_roughness.csv", index=False)
    vegetation_field_day.to_csv(output / "vegetation_field_day.csv", index=False)
    save_figures(model_ready, sensitivity, output)

    matchup_sm_available = int(pd.to_numeric(matchup["SM"], errors="coerce").notna().sum())
    summary = {
        "dataset": "SMAPVEX08",
        "campaign_role": "independent external domain",
        "portable_forward_task": "L-band HH/VV sigma0 from soil moisture and surface roughness",
        "parameters": {
            "frequency_ghz": 1.26,
            "match_radius_m": args.match_radius_m,
            "minimum_pals_samples": args.min_pals_samples,
            "valid_sigma0_db": [args.sigma_min_db, args.sigma_max_db],
            "valid_incidence_angle_deg": [30.0, 50.0],
            "soil_moisture_policy": "mean of A/B/C field-specific-calibration VSM",
            "roughness_policy": "mean of directional profiles; official mm converted to cm",
            "backscatter_policy": "mean in linear power, then convert to dB",
        },
        "source_inventory": {
            "pals_files": len(files["pals_files"]),
            "pals_dates": len({parse_pals_filename(p)["acquisition_date"] for p in files["pals_files"]}),
            "soil_dates": int(soil_field_day["acquisition_date"].nunique()),
            "soil_fields": int(soil_field_day["field_id"].nunique()),
            "soil_sites_without_distributed_gps_coordinates": sorted(
                soil.loc[~soil["coordinate_available"], "site_id"]
                .drop_duplicates()
                .astype(str)
                .tolist()
            ),
            "roughness_fields": int(roughness_field["field_id"].nunique()),
            "in_situ_vegetation_field_days": int(len(vegetation_field_day)),
            "nsidc0666_rows": int(len(matchup)),
            "nsidc0666_rows_with_soil_moisture": matchup_sm_available,
            "vwc_raster": inspect_vwc_raster(Path(files["vwc_raster"])),
        },
        "outputs": {
            "collocated_rows": len(all_rows),
            "model_ready_rows": len(model_ready),
            "model_ready_fields": int(model_ready["field_id"].nunique()),
            "model_ready_dates": int(model_ready["acquisition_date"].nunique()),
            "rows_with_in_situ_vwc": int(model_ready["vwc_kg_m2"].notna().sum()),
        },
        "acceptance_checks": {
            "all_67_pals_files_present": len(files["pals_files"]) == 67,
            "seven_ground_dates_present": soil_field_day["acquisition_date"].nunique() == 7,
            "at_least_five_external_roughness_fields": roughness_field["field_id"].nunique() >= 5,
            "at_least_three_model_ready_dates": model_ready["acquisition_date"].nunique() >= 3,
            "no_duplicate_model_field_dates": not model_ready.duplicated(["campaign_id", "field_id", "acquisition_date"]).any(),
            "all_model_rows_finite": bool(
                np.isfinite(
                    model_ready[[
                        "soil_moisture_m3_m3",
                        "pals_rms_height_cm",
                        "pals_correlation_length_cm",
                        "sigma0_hh_db",
                        "sigma0_vv_db",
                    ]].to_numpy(dtype=float)
                ).all()
            ),
        },
        "scientific_guardrails": [
            "The external targets must not be used to tune radius, architecture, epochs, gate thresholds, or feature sets.",
            "The 500 m radius is a geometry-based initial choice; radius sensitivity is reported as support counts, not selected by prediction error.",
            f"NSIDC-0666 has soil moisture in only {matchup_sm_available}/{len(matchup)} SMAPVEX08 rows; it is retained as a mapping diagnostic rather than replacing the reconstructed core table.",
            "In-situ VWC is too sparse to be a mandatory portable input and is retained only for auxiliary discrepancy analysis.",
            "Roughness is available for nine fields and is the limiting factor in external sample size.",
        ],
    }
    summary["status"] = (
        "READY_FOR_EXTERNAL_AUDIT"
        if all(summary["acceptance_checks"].values())
        else "NOT_READY"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
