"""Execute an I2EM request table using the optional pyi2em backend."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from research_pilots.scattering.surfaces.teacher_contract import (
    validate_teacher_requests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optional pyi2em physics teacher")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        from pyi2em import sigma0_backscatter
    except ImportError as exc:
        raise RuntimeError(
            "pyi2em is not installed. Version 0.1.5 publishes Linux wheels but no "
            "native-Windows wheel. Run this adapter in a pinned Linux/WSL environment, "
            "or restore MATLAB and use run_i2em_requests_matlab.m. Do not replace the "
            "teacher with fabricated values."
        ) from exc

    requests = validate_teacher_requests(pd.read_csv(args.input))
    rows = []
    for row in requests.itertuples(index=False):
        response = sigma0_backscatter(
            freq_ghz=float(row.frequency_ghz),
            rms_height_m=float(row.rms_height_m),
            corr_length_m=float(row.correlation_length_m),
            theta_deg=float(row.incidence_angle_deg),
            er_complex=complex(
                float(row.dielectric_real), float(row.dielectric_loss_positive)
            ),
            correl=str(row.correlation_model),
            include_hv=False,
            return_db=True,
        )
        rows.append(
            {
                "request_id": row.request_id,
                "i2em_hh_db": float(np.asarray(response["hh"]).squeeze()),
                "i2em_vv_db": float(np.asarray(response["vv"]).squeeze()),
            }
        )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    metadata = {
        "backend": "pyi2em",
        "version": importlib.metadata.version("pyi2em"),
        "input": str(args.input.resolve()),
        "output": str(output_path),
        "request_count": len(rows),
        "complex_permittivity_convention": "pyi2em API receives epsilon_real + j*positive_loss",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"I2EM results saved to: {output_path}")


if __name__ == "__main__":
    main()
