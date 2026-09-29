from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/scattering/rough_ground"
VWC = "vegetation_water_content_in_situ_kg_m2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=BASE / "portable_source_v1/smapvex12_portable_source.csv",
    )
    parser.add_argument(
        "--vegetation", type=Path,
        default=BASE / "vegetation_audit/vegetation_field_day.csv",
    )
    parser.add_argument("--max-gap-days", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("Choose a new output directory.")
    if args.max_gap_days < 0:
        raise ValueError("max-gap-days must be nonnegative.")

    source = pd.read_csv(args.input, dtype={"field_id": str})
    vegetation = pd.read_csv(args.vegetation, dtype={"field_id": str})
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
        if selected["gap"] > args.max_gap_days:
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

    args.output.mkdir(parents=True)
    result.to_csv(args.output / "matched_source.csv", index=False)
    report = {
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