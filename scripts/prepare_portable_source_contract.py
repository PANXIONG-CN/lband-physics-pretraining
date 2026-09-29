"""Create a physically valid, versioned SMAPVEX12 source-domain contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_AND_TARGETS = [
    "soil_moisture_m3_m3",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
    "sigma0_hh_db",
    "sigma0_vv_db",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter the source field-day table to the portable physical contract"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--campaign-id", default="SMAPVEX12")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output_dir.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; use a new versioned directory")
    frame = pd.read_csv(source, dtype={"field_id": "string"})
    missing = sorted(set(FEATURES_AND_TARGETS + ["acquisition_date", "field_id"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Source table is missing columns: {missing}")

    numeric = frame[FEATURES_AND_TARGETS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    positive_roughness = (
        numeric["pals_rms_height_cm"].gt(0)
        & numeric["pals_correlation_length_cm"].gt(0)
    )
    valid = finite & positive_roughness
    rejection_reason = np.select(
        [
            ~finite,
            finite & ~positive_roughness,
        ],
        [
            "non_finite_portable_variable",
            "nonpositive_roughness_or_correlation_length",
        ],
        default="unknown",
    )
    rejected = frame.loc[~valid].copy()
    rejected["rejection_reason"] = rejection_reason[~valid]
    portable = frame.loc[valid].copy()
    portable.insert(0, "campaign_id", args.campaign_id)
    portable["acquisition_date"] = pd.to_datetime(
        portable["acquisition_date"], errors="raise"
    ).dt.date.astype(str)
    if portable.duplicated(["campaign_id", "field_id", "acquisition_date"]).any():
        raise ValueError("Portable source table contains duplicate campaign-field-date rows")

    output.mkdir(parents=True, exist_ok=True)
    portable.to_csv(output / "smapvex12_portable_source.csv", index=False)
    rejected.to_csv(output / "rejected_source_rows.csv", index=False)
    summary = {
        "input": str(source),
        "input_sha256": sha256(source),
        "campaign_id": args.campaign_id,
        "input_rows": len(frame),
        "retained_rows": len(portable),
        "rejected_rows": len(rejected),
        "retained_fields": int(portable["field_id"].nunique()),
        "retained_dates": int(portable["acquisition_date"].nunique()),
        "rejection_counts": rejected["rejection_reason"].value_counts().to_dict(),
        "contract": {
            "all_portable_variables_finite": True,
            "rms_height_strictly_positive": True,
            "correlation_length_strictly_positive": True,
            "unique_unit": "campaign-field-date",
        },
        "scientific_note": "Rows are removed only by input-domain validity, never by target error or external-domain performance.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
