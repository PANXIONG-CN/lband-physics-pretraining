"""Quality audit for SMAPVEX12 PALS, soil moisture, and roughness data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from research_pilots.scattering.data.read_smapvex import (  # noqa: E402
        PALS_COLUMNS,
        SIGMA0_COLUMNS,
        discover_smapvex_files,
        iter_pals_chunks,
        parse_pals_filename,
        read_soil_moisture,
        read_surface_roughness,
    )
except ModuleNotFoundError:  # permits a standalone pre-installation smoke test
    from read_smapvex import (  # type: ignore[no-redef]  # noqa: E402
        PALS_COLUMNS,
        SIGMA0_COLUMNS,
        discover_smapvex_files,
        iter_pals_chunks,
        parse_pals_filename,
        read_soil_moisture,
        read_surface_roughness,
    )


def _update_range(ranges: dict, name: str, series: pd.Series) -> None:
    valid = series.dropna()
    if valid.empty:
        return
    low = float(valid.min())
    high = float(valid.max())
    current = ranges.setdefault(name, {"min": None, "max": None})
    current["min"] = low if current["min"] is None else min(current["min"], low)
    current["max"] = high if current["max"] is None else max(current["max"], high)


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame.isna().sum().items()}


def audit_pals(files: list[Path], chunksize: int) -> tuple[dict, pd.DataFrame]:
    inventory: list[dict] = []
    total_rows = 0
    global_missing = Counter()
    global_flags = Counter({"flag_0": 0, "flag_1": 0, "missing": 0, "other": 0})
    global_outside = Counter({column: 0 for column in SIGMA0_COLUMNS})
    global_ranges: dict[str, dict[str, float | None]] = {}

    for index, path in enumerate(files, start=1):
        metadata = parse_pals_filename(path)
        file_rows = 0
        file_flags = Counter({"flag_0": 0, "flag_1": 0, "missing": 0, "other": 0})
        file_outside = Counter({column: 0 for column in SIGMA0_COLUMNS})
        file_ranges: dict[str, dict[str, float | None]] = {}

        print(f"[{index:02d}/{len(files):02d}] auditing {path.name}")
        for chunk in iter_pals_chunks(path, chunksize=chunksize):
            rows = len(chunk)
            file_rows += rows
            total_rows += rows
            global_missing.update(_missing_counts(chunk[PALS_COLUMNS]))

            flags = chunk["heading_uncertainty_flag"]
            counts = {
                "flag_0": int((flags == 0).sum()),
                "flag_1": int((flags == 1).sum()),
                "missing": int(flags.isna().sum()),
                "other": int((flags.notna() & ~flags.isin([0, 1])).sum()),
            }
            file_flags.update(counts)
            global_flags.update(counts)

            for column in SIGMA0_COLUMNS:
                values = chunk[column]
                outside = int((values.notna() & ~values.between(-40.0, 0.0)).sum())
                file_outside[column] += outside
                global_outside[column] += outside
                _update_range(file_ranges, column, values)
                _update_range(global_ranges, column, values)

            for column in ["utc_seconds", "latitude", "longitude", "utm_x", "utm_y"]:
                _update_range(file_ranges, column, chunk[column])
                _update_range(global_ranges, column, chunk[column])

        row = {
            "file_name": path.name,
            "acquisition_date": pd.Timestamp(metadata["acquisition_date"]).date().isoformat(),
            "altitude_mode": metadata["altitude_mode"],
            "footprint_m": int(metadata["footprint_m"]),
            "rows": file_rows,
            **file_flags,
        }
        for column in SIGMA0_COLUMNS:
            row[f"{column}_min"] = file_ranges[column]["min"]
            row[f"{column}_max"] = file_ranges[column]["max"]
            row[f"{column}_outside_minus40_to_0"] = file_outside[column]
        inventory.append(row)

    dates = sorted(
        {pd.Timestamp(parse_pals_filename(path)["acquisition_date"]).date().isoformat() for path in files}
    )
    altitude_counts = Counter(
        str(parse_pals_filename(path)["altitude_mode"]) for path in files
    )
    summary = {
        "file_count": len(files),
        "date_count": len(dates),
        "dates": dates,
        "files_by_altitude": dict(sorted(altitude_counts.items())),
        "total_rows": total_rows,
        "heading_uncertainty_flag_counts": dict(global_flags),
        "recommended_flag_zero_rows": int(global_flags["flag_0"]),
        "missing_values": dict(global_missing),
        "value_ranges": global_ranges,
        "sigma0_outside_minus40_to_0_db": dict(global_outside),
        "notes": [
            "PALS sigma0 values are already in dB; do not apply 10*log10 again.",
            "Keep raw rows. For later collocation, create a separate view with heading_uncertainty_flag == 0.",
            "HiAlt and LoAlt have different footprints and must not be mixed without an explicit scale strategy.",
        ],
    }
    return summary, pd.DataFrame(inventory)


def audit_soil_moisture(frame: pd.DataFrame) -> dict:
    moisture = frame["soil_moisture_m3_m3"]
    dielectric = frame["soil_real_dielectric"]
    dates = frame["sample_date"].dropna()
    return {
        "rows": int(len(frame)),
        "site_count": int(frame["site_id"].nunique(dropna=True)),
        "date_count": int(dates.dt.normalize().nunique()),
        "date_min": dates.min().date().isoformat() if not dates.empty else None,
        "date_max": dates.max().date().isoformat() if not dates.empty else None,
        "missing_values": _missing_counts(frame),
        "rows_without_coordinates": int(frame[["utm_x", "utm_y"]].isna().any(axis=1).sum()),
        "soil_moisture_range_m3_m3": {
            "min": float(moisture.min()),
            "max": float(moisture.max()),
        },
        "soil_real_dielectric_range": {
            "min": float(dielectric.min()),
            "max": float(dielectric.max()),
        },
        "soil_moisture_outside_0_to_1": int(
            (moisture.notna() & ~moisture.between(0.0, 1.0)).sum()
        ),
        "nonpositive_dielectric": int((dielectric.notna() & (dielectric <= 0.0)).sum()),
    }


def audit_surface_roughness(frame: pd.DataFrame) -> dict:
    height = frame["pals_rms_height_cm"]
    correlation = frame["pals_correlation_length_cm"]
    missing_coordinates = frame[["utm_x", "utm_y"]].isna().any(axis=1)
    sites_without_coordinates = sorted(
        frame.loc[missing_coordinates, "site_id"].dropna().astype(str).unique().tolist()
    )
    return {
        "rows": int(len(frame)),
        "site_count": int(frame["site_id"].nunique(dropna=True)),
        "missing_values": _missing_counts(frame),
        "rows_without_coordinates": int(missing_coordinates.sum()),
        "sites_without_coordinates": sites_without_coordinates,
        "nonforest_rows_without_coordinates": int(
            (
                missing_coordinates
                & ~frame["site_id"].astype("string").str.startswith("F", na=False)
            ).sum()
        ),
        "pals_rms_height_cm_range": {
            "min": float(height.min()),
            "max": float(height.max()),
        },
        "pals_correlation_length_cm_range": {
            "min": float(correlation.min()),
            "max": float(correlation.max()),
        },
        "pals_height_outside_0_to_3_cm": int(
            (height.notna() & ~height.between(0.0, 3.0)).sum()
        ),
        "pals_correlation_length_outside_0_to_30_cm": int(
            (correlation.notna() & ~correlation.between(0.0, 30.0)).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the three SMAPVEX12 products before collocation."
    )
    parser.add_argument(
        "--input-root",
        default="data/scattering/raw/SMAPVEX12",
        help="SMAPVEX12 root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/scattering/rough_ground/inspection",
        help="Directory for audit products",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_smapvex_files(args.input_root)

    soil = read_soil_moisture(
        files["soil_moisture_data"], files["soil_moisture_coordinates"]
    )
    roughness = read_surface_roughness(
        files["surface_roughness_data"], files["surface_roughness_coordinates"]
    )
    pals_summary, inventory = audit_pals(files["pals_files"], args.chunksize)

    report = {
        "dataset": "SMAPVEX12",
        "input_root": str(Path(args.input_root).resolve()),
        "coordinate_reference": "UTM Zone 14N, WGS84",
        "pals": pals_summary,
        "soil_moisture": audit_soil_moisture(soil),
        "surface_roughness": audit_surface_roughness(roughness),
    }
    pals_dates = set(report["pals"]["dates"])
    soil_dates = {
        value.date().isoformat()
        for value in soil["sample_date"].dropna().dt.normalize().unique()
    }
    report["date_alignment"] = {
        "common_date_count": len(pals_dates & soil_dates),
        "common_dates": sorted(pals_dates & soil_dates),
        "pals_only_dates": sorted(pals_dates - soil_dates),
        "soil_only_dates": sorted(soil_dates - pals_dates),
    }
    report["acceptance_checks"] = {
        "pals_has_32_files": report["pals"]["file_count"] == 32,
        "pals_has_16_dates": report["pals"]["date_count"] == 16,
        "pals_has_16_files_per_altitude": report["pals"]["files_by_altitude"]
        == {"HiAlt": 16, "LoAlt": 16},
        "soil_coordinates_complete": report["soil_moisture"]["rows_without_coordinates"] == 0,
        "all_pals_dates_have_soil_measurements": not report["date_alignment"]["pals_only_dates"],
        "roughness_nonforest_coordinates_complete": report["surface_roughness"][
            "nonforest_rows_without_coordinates"
        ]
        == 0,
    }

    inventory.to_csv(output_dir / "pals_file_inventory.csv", index=False)
    soil.to_csv(output_dir / "soil_moisture_normalized.csv", index=False)
    roughness.to_csv(output_dir / "surface_roughness_normalized.csv", index=False)
    with (output_dir / "smapvex_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print("\nSMAPVEX12 audit complete")
    print(json.dumps(report["acceptance_checks"], indent=2, ensure_ascii=False))
    print(f"PALS rows audited: {report['pals']['total_rows']:,}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
