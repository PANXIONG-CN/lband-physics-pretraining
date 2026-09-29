"""Prepare a target-error-blind SMEX02 campaign-field-date contract."""

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
from rasterio.warp import transform
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_pilots.scattering.data.read_smex02 import (  # noqa: E402
    discover_files,
    iter_pals_chunks,
    linear_power_mean_db,
    parse_pals_filename,
    read_soil_moisture_summary,
    read_surface_roughness,
)
from research_pilots.scattering.surfaces.dielectric import (  # noqa: E402
    topp_real_permittivity,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_soil(soil: pd.DataFrame) -> pd.DataFrame:
    valid = soil.loc[
        soil["soil_moisture_in_physical_range"] & soil["coordinate_available"]
    ].copy()
    return (
        valid.groupby(["acquisition_date", "field_id"], as_index=False)
        .agg(
            soil_moisture_m3_m3=("soil_moisture_m3_m3", "mean"),
            soil_moisture_median_m3_m3=("soil_moisture_m3_m3", "median"),
            soil_moisture_std_m3_m3=("soil_moisture_m3_m3", "std"),
            soil_moisture_general_cal_m3_m3=(
                "soil_moisture_general_cal_m3_m3", "mean"
            ),
            soil_sample_count=("soil_sample_count", "sum"),
            soil_location_count=("latitude", "size"),
            field_latitude=("latitude", "mean"),
            field_longitude=("longitude", "mean"),
        )
        .sort_values(["acquisition_date", "field_id"])
        .reset_index(drop=True)
    )


def aggregate_roughness(roughness: pd.DataFrame) -> pd.DataFrame:
    valid = roughness.loc[
        roughness["field_id"].notna() & roughness["roughness_positive"]
    ].copy()
    return (
        valid.groupby("field_id", as_index=False)
        .agg(
            pals_rms_height_cm=("rms_height_cm", "mean"),
            pals_rms_height_cm_std=("rms_height_cm", "std"),
            pals_correlation_length_cm=("correlation_length_cm", "mean"),
            pals_correlation_length_cm_std=("correlation_length_cm", "std"),
            adjusted_rms_height_cm=("adjusted_rms_height_cm", "mean"),
            roughness_profile_count=("profile_id", "size"),
            roughness_scan_mode_count=("scan_mode", "nunique"),
        )
        .sort_values("field_id")
        .reset_index(drop=True)
    )


def sampling_nodes(soil: pd.DataFrame) -> pd.DataFrame:
    nodes = (
        soil.loc[soil["coordinate_available"], ["field_id", "latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    x, y = transform(
        "EPSG:4326",
        "EPSG:32615",
        nodes["longitude"].tolist(),
        nodes["latitude"].tolist(),
    )
    nodes["utm_x"] = x
    nodes["utm_y"] = y
    return nodes


def match_pals_to_nodes(
    pals_files: list[Path],
    nodes: pd.DataFrame,
    valid_dates: set[pd.Timestamp],
    maximum_radius_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree = cKDTree(nodes[["utm_x", "utm_y"]].to_numpy(dtype=float))
    pieces: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    selected = [
        "source_file",
        "acquisition_date",
        "incidence_angle_deg",
        "range_m",
        "sigma0_hh_db",
        "sigma0_vv_db",
        "sigma0_vh_db",
        "sigma0_hv_db",
    ]

    for number, path in enumerate(pals_files, start=1):
        metadata = parse_pals_filename(path)
        date = pd.Timestamp(metadata["acquisition_date"])
        raw_rows = qc_rows = candidate_rows = 0
        if date in valid_dates:
            print(f"[{number:02d}/{len(pals_files):02d}] {path.name}", flush=True)
            for chunk in iter_pals_chunks(path):
                raw_rows += len(chunk)
                numeric = [
                    "latitude", "longitude", "incidence_angle_deg", "range_m",
                    "sigma0_hh_db", "sigma0_vv_db",
                ]
                finite = np.isfinite(chunk[numeric].to_numpy(dtype=float)).all(axis=1)
                qc = (
                    finite
                    & chunk["latitude"].between(41.0, 43.0)
                    & chunk["longitude"].between(-95.0, -92.0)
                    & chunk["incidence_angle_deg"].between(25.0, 60.0)
                    & chunk["sigma0_hh_db"].between(-50.0, 10.0)
                    & chunk["sigma0_vv_db"].between(-50.0, 10.0)
                )
                chunk = chunk.loc[qc].copy()
                qc_rows += len(chunk)
                if chunk.empty:
                    continue
                x, y = transform(
                    "EPSG:4326", "EPSG:32615",
                    chunk["longitude"].tolist(), chunk["latitude"].tolist(),
                )
                distance, index = tree.query(np.column_stack([x, y]), k=1)
                keep = distance <= maximum_radius_m
                if not np.any(keep):
                    continue
                matched = chunk.loc[keep, selected].copy()
                matched["field_id"] = nodes.iloc[index[keep]]["field_id"].to_numpy()
                matched["match_distance_m"] = distance[keep]
                candidate_rows += len(matched)
                pieces.append(matched)
        inventory.append(
            {
                "source_file": path.name,
                "acquisition_date": date.date().isoformat(),
                "used_for_overlap": date in valid_dates,
                "raw_rows": raw_rows,
                "qc_rows": qc_rows,
                "nearest_candidate_rows": candidate_rows,
            }
        )

    if not pieces:
        raise RuntimeError("No quality-controlled PALS rows reached the maximum radius")
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(inventory)


def support_sensitivity(nearest: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for radius in radii:
        selected = nearest.loc[nearest["match_distance_m"] <= radius]
        pairs = selected[["acquisition_date", "field_id"]].drop_duplicates()
        rows.append(
            {
                "radius_m": radius,
                "matched_pals_rows": len(selected),
                "matched_field_days": len(pairs),
                "matched_fields": pairs["field_id"].nunique(),
                "matched_dates": pairs["acquisition_date"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def aggregate_pals(nearest: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    matched = nearest.loc[nearest["match_distance_m"] <= radius_m].copy()
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
        for column in [
            "sigma0_hh_db", "sigma0_vv_db", "sigma0_vh_db", "sigma0_hv_db"
        ]:
            row[column] = linear_power_mean_db(group[column])
            row[f"{column}_median"] = pd.to_numeric(
                group[column], errors="coerce"
            ).median()
            row[f"{column}_std"] = pd.to_numeric(
                group[column], errors="coerce"
            ).std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["acquisition_date", "field_id"]
    ).reset_index(drop=True)


def save_support_plot(
    sensitivity: pd.DataFrame,
    model_ready: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(
        sensitivity["radius_m"], sensitivity["matched_field_days"], marker="o"
    )
    axes[0].set(
        xlabel="Nearest sampling-location radius (m)",
        ylabel="Matched field-days",
        title="Target-error-blind spatial support",
    )
    counts = model_ready.groupby("acquisition_date").size()
    axes[1].bar(counts.index.strftime("%m-%d"), counts.values)
    axes[1].set(
        xlabel="Date", ylabel="Model-ready fields", title="Development subset support"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / "01_smex02_intake_support.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SMEX02 without fitting or selecting a predictive model"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--match-radius-m", type=float, default=500.0)
    parser.add_argument(
        "--sensitivity-radii-m", nargs="+", type=float,
        default=[250.0, 400.0, 500.0, 600.0, 750.0],
    )
    parser.add_argument(
        "--mode", choices=["development", "final"], default="development"
    )
    parser.add_argument(
        "--confirm-complete-pals-download",
        action="store_true",
        help="Required in final mode after all official PALS radar dates are present.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; use a new versioned directory")
    if args.match_radius_m <= 0:
        raise ValueError("match-radius-m must be positive")
    if args.mode == "final" and not args.confirm_complete_pals_download:
        raise ValueError(
            "Final mode requires --confirm-complete-pals-download after the "
            "complete official PALS radar product has been inventoried"
        )

    files = discover_files(args.input.resolve())
    soil_raw = read_soil_moisture_summary(files["soil_moisture_summary"])
    rough_raw = read_surface_roughness(
        files["roughness_grid"], files["roughness_slope"]
    )
    soil = aggregate_soil(soil_raw)
    roughness = aggregate_roughness(rough_raw)
    nodes = sampling_nodes(soil_raw)

    pals_dates = {
        pd.Timestamp(parse_pals_filename(path)["acquisition_date"])
        for path in files["pals_files"]
    }
    soil_dates = set(soil["acquisition_date"])
    overlap_dates = pals_dates.intersection(soil_dates)
    if not overlap_dates:
        raise RuntimeError("PALS and ThetaProbe products have no overlapping dates")

    maximum_radius = max([args.match_radius_m, *args.sensitivity_radii_m])
    nearest, inventory = match_pals_to_nodes(
        files["pals_files"], nodes, overlap_dates, maximum_radius
    )
    sensitivity = support_sensitivity(nearest, args.sensitivity_radii_m)
    pals = aggregate_pals(nearest, args.match_radius_m)

    collocated = (
        soil.merge(roughness, on="field_id", how="left", validate="many_to_one")
        .merge(
            pals,
            on=["acquisition_date", "field_id"],
            how="outer",
            validate="one_to_one",
        )
        .sort_values(["acquisition_date", "field_id"])
        .reset_index(drop=True)
    )
    collocated.insert(0, "campaign_id", "SMEX02")
    collocated["soil_real_dielectric"] = topp_real_permittivity(
        collocated["soil_moisture_m3_m3"]
    )
    required = [
        "soil_moisture_m3_m3",
        "soil_real_dielectric",
        "pals_rms_height_cm",
        "pals_correlation_length_cm",
        "incidence_angle_deg",
        "sigma0_hh_db",
        "sigma0_vv_db",
    ]
    finite = np.isfinite(collocated[required].to_numpy(dtype=float)).all(axis=1)
    collocated["model_ready"] = (
        finite
        & collocated["pals_rms_height_cm"].gt(0)
        & collocated["pals_correlation_length_cm"].gt(0)
    )
    model_ready = collocated.loc[collocated["model_ready"]].copy()
    if model_ready.empty:
        raise RuntimeError("No model-ready SMEX02 field-date rows were produced")
    if model_ready.duplicated(["campaign_id", "field_id", "acquisition_date"]).any():
        raise ValueError("Duplicate campaign-field-date rows in SMEX02 contract")

    output.mkdir(parents=True, exist_ok=True)
    soil.to_csv(output / "soil_field_day.csv", index=False)
    roughness.to_csv(output / "field_roughness.csv", index=False)
    pals.to_csv(output / "pals_field_day.csv", index=False)
    inventory.to_csv(output / "pals_file_inventory.csv", index=False)
    sensitivity.to_csv(output / "match_radius_sensitivity.csv", index=False)
    collocated.to_csv(output / "smex02_field_day_collocation_all.csv", index=False)
    model_ready.to_csv(output / "smex02_field_day_model_ready.csv", index=False)
    save_support_plot(sensitivity, model_ready, output)

    summary = {
        "status": "DEVELOPMENT_READY" if args.mode == "development" else "FINAL_INPUT_READY",
        "mode": args.mode,
        "complete_pals_download_user_confirmed": bool(
            args.confirm_complete_pals_download
        ),
        "scientific_role": "prospective third-domain intake; no model fitting performed",
        "input_root": str(Path(files["root"]).resolve()),
        "input_files": {
            "pals_count": len(files["pals_files"]),
            "pals_dates": sorted(date.date().isoformat() for date in pals_dates),
            "soil_summary": str(Path(files["soil_moisture_summary"]).resolve()),
            "soil_summary_sha256": sha256(Path(files["soil_moisture_summary"])),
            "roughness_grid_sha256": sha256(Path(files["roughness_grid"])),
            "roughness_slope_sha256": sha256(Path(files["roughness_slope"])),
        },
        "overlap_dates": sorted(date.date().isoformat() for date in overlap_dates),
        "match_radius_m": args.match_radius_m,
        "radius_selection_rule": "pre-specified geometry/support rule; target errors not used",
        "soil": {
            "raw_rows": len(soil_raw),
            "field_day_rows": len(soil),
            "fields": int(soil["field_id"].nunique()),
            "dates": int(soil["acquisition_date"].nunique()),
            "primary_measurement": "VSM_ssc site-specific calibrated ThetaProbe moisture",
        },
        "roughness": {
            "raw_profiles": len(rough_raw),
            "fields": int(roughness["field_id"].nunique()),
            "units": "rms height and correlation length in cm",
        },
        "contract": {
            "all_rows": len(collocated),
            "model_ready_rows": len(model_ready),
            "model_ready_fields": int(model_ready["field_id"].nunique()),
            "model_ready_dates": int(model_ready["acquisition_date"].nunique()),
            "unique_campaign_field_date": True,
        },
        "guardrails": [
            "PALS dB observations are averaged in linear power.",
            "Only L-band HH/VV are primary targets; S-band is not substituted.",
            "Spatial radius sensitivity reports support counts only.",
            "No predictive model or target-error comparison is run here.",
            "Development output cannot be presented as final prospective validation.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"SMEX02 intake outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
