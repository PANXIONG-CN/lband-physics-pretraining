from pathlib import Path
import argparse
import json
import hashlib
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_pilots.scattering.data.vegetation import read_in_situ_vegetation

DATA = ROOT / "reproducibility/data/source"
VWC = "vegetation_water_content_in_situ_kg_m2"


def fingerprint(path):
    path = Path(path)
    return {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def match_source(source, vegetation, max_gap_days):
    if max_gap_days < 0:
        raise ValueError("max-gap-days must be nonnegative.")
    source, vegetation = source.copy(), vegetation.copy()
    source["field_id"] = source.field_id.str.strip()
    vegetation["field_id"] = vegetation.field_id.str.strip()
    source["acquisition_date"] = pd.to_datetime(
        source.acquisition_date, errors="raise"
    )
    vegetation["sample_date"] = pd.to_datetime(
        vegetation.sample_date, errors="raise"
    )
    vegetation[VWC] = pd.to_numeric(vegetation[VWC], errors="raise")
    vegetation = vegetation.loc[
        np.isfinite(vegetation[VWC]) & (vegetation[VWC] >= 0)
    ].copy()

    if vegetation.duplicated(["field_id", "sample_date"]).any():
        raise ValueError("Vegetation input must have one row per field-date.")

    if source.duplicated(["field_id", "acquisition_date"]).any():
        raise ValueError("Source input must have one row per field-date.")

    matched = []
    for _, row in source.iterrows():
        candidates = vegetation.loc[
            vegetation.field_id == row.field_id
        ].copy()
        if candidates.empty:
            continue
        candidates["gap"] = (
            candidates.sample_date - row.acquisition_date
        ).dt.days.abs()
        # Equal-distance ties use the earlier sampling date.
        selected = candidates.sort_values(["gap", "sample_date"]).iloc[0]
        if selected["gap"] > max_gap_days:
            continue

        record = row.to_dict()
        record[VWC] = float(selected[VWC])
        record["vegetation_sample_date"] = selected.sample_date
        record["vegetation_gap_days"] = int(selected["gap"])
        if "crop_type" in selected.index:
            record["crop_type"] = selected.crop_type
        matched.append(record)

    result = pd.DataFrame(matched)
    if result.empty:
        raise RuntimeError("No matched vegetation records.")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=DATA / "smapvex12_portable_source.csv",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--vegetation", type=Path, help="Existing field-day vegetation table.")
    inputs.add_argument("--vegetation-root", type=Path, help="Directory with the three official SV12VA files.")
    parser.add_argument("--max-gap-days", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("Choose a new output directory.")
    if args.max_gap_days < 0:
        raise ValueError("max-gap-days must be nonnegative.")

    source = pd.read_csv(args.input, dtype={"field_id": str})
    if args.vegetation_root is not None:
        vegetation, coordinates = read_in_situ_vegetation(args.vegetation_root)
        vegetation_inputs = [fingerprint(p) for p in sorted(args.vegetation_root.rglob("SV12VA*"))
                             if p.is_file() and p.suffix != ".json"]
    else:
        path = args.vegetation or DATA / "vegetation_field_day.csv"
        vegetation = pd.read_csv(path, dtype={"field_id": str})
        vegetation_inputs = [fingerprint(path)]
        coordinates = None
    result = match_source(source, vegetation, args.max_gap_days)

    args.output.mkdir(parents=True)
    result.to_csv(args.output / "matched_source.csv", index=False)
    vegetation.to_csv(args.output / "vegetation_field_day.csv", index=False)
    if coordinates is not None:
        coordinates.to_csv(args.output / "vegetation_site_coordinates.csv", index=False)
    report = {
        "source_input": fingerprint(args.input), "vegetation_inputs": vegetation_inputs,
        "script": fingerprint(__file__),
        "reader": fingerprint(ROOT / "src/research_pilots/scattering/data/vegetation.py"),
        "product_doi": "10.5067/X2EF9ZKL0DGC",
        "source_rows": len(source),
        "matched_rows": len(result),
        "matched_fields": int(result.field_id.nunique()),
        "matched_dates": int(result.acquisition_date.nunique()),
        "max_gap_days": args.max_gap_days,
        "matched_fraction": len(result) / len(source),
        "vwc_min_kg_m2": float(result[VWC].min()),
        "vwc_max_kg_m2": float(result[VWC].max()),
        "selection": "same field, nearest date, earlier date breaks ties",
    }
    (args.output / "coverage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()