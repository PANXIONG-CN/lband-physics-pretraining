"""Prepare I2EM requests exactly at the real observation cohort points."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_decoupled_physics_heads as base  # noqa: E402
import prepare_multifidelity_pretraining as multi  # noqa: E402


def fingerprint(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare observation-aligned I2EM teacher requests")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--incidence-angle-deg", type=float, default=40.0)
    parser.add_argument("--loss-tangent", type=float, default=0.05)
    parser.add_argument("--correlation-model", choices=["exponential", "gaussian"], default="exponential")
    args = parser.parse_args()
    input_path = args.input.resolve()
    output = args.output.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Output already exists; choose a new directory")
    frame = base.load_table(input_path)
    frame = multi.add_spm_targets(
        frame, args.frequency_ghz * 1e9, args.incidence_angle_deg,
        args.correlation_model, args.loss_tangent,
    )
    requests = multi.to_i2em_requests(
        frame, "observation", args.frequency_ghz, args.incidence_angle_deg,
        args.correlation_model, args.loss_tangent,
    )
    output.mkdir(parents=True, exist_ok=True)
    requests.to_csv(output / "observation_i2em_requests.csv", index=False)
    manifest = {
        "purpose": "Compare SPM and I2EM directly at every SPM-valid field-date observation without fitting to observations.",
        "input": fingerprint(input_path),
        "rows": len(requests),
        "fields": int(requests.field_id.nunique()),
        "dates": int(requests.acquisition_date.nunique()),
        "settings": {"frequency_ghz": args.frequency_ghz, "incidence_angle_deg": args.incidence_angle_deg, "loss_tangent": args.loss_tangent, "correlation_model": args.correlation_model},
        "guardrails": ["Direct teacher error includes vegetation, aggregation, measurement, and structural model mismatch.", "I2EM is not calibrated to these observations and is not ground truth.", "The audit does not use observation values to tune either teacher."],
        "code": fingerprint(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(requests)} observation-aligned I2EM requests in {output}")


if __name__ == "__main__":
    main()
