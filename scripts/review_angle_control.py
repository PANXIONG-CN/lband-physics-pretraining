from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_pilots.scattering.surfaces.spm import spm_backscatter_db
from research_pilots.scattering.surfaces.teacher_contract import (
    validate_teacher_requests,
    validate_teacher_results,
)

BASE = ROOT / "outputs/scattering/rough_ground"


def topp(m):
    return 3.03 + 9.3*m + 146*m*m - 76.7*m*m*m


def metrics(frame, hh, vv):
    y = frame[["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(float)
    p = frame[[hh, vv]].to_numpy(float)
    obs = [y[:, 0], y[:, 1], y.mean(1), y[:, 1] - y[:, 0]]
    pred = [p[:, 0], p[:, 1], p.mean(1), p[:, 1] - p[:, 0]]
    rows = []
    for name, a, b in zip(["HH", "VV", "common", "differential"], obs, pred):
        variance = np.var(a)
        centered = (b - b.mean()) - (a - a.mean())
        rows.append({
            "response": name,
            "n": len(a),
            "rmse_db": float(np.sqrt(np.mean((b-a)**2))),
            "bias_db": float(np.mean(b-a)),
            "centered_rmse_db": float(np.sqrt(np.mean(centered**2))),
            "centered_skill": (
                float(1-np.mean(centered**2)/variance)
                if variance > 1e-12 else np.nan
            ),
        })
    return rows


def prepare(output):
    if output.exists():
        raise FileExistsError("Choose a new output directory.")

    source = pd.read_csv(
        BASE / "portable_source_v1/smapvex12_portable_source.csv",
        dtype={"field_id": str},
    )
    target = pd.read_csv(
        BASE / "smex02_external_final_20260911_v1/"
        "smex02_field_day_model_ready.csv",
        dtype={"field_id": str},
    )
    source["campaign"] = "SMAPVEX12"
    source["angle_used"] = source["nominal_incidence_angle_deg"]
    target["campaign"] = "SMEX02"
    target["angle_used"] = target["incidence_angle_deg"]
    data = pd.concat([source, target], ignore_index=True)
    data["unit_id"] = (
        data.campaign + "_" + data.field_id.astype(str)
        + "_" + data.acquisition_date.astype(str)
    )

    m = data.soil_moisture_m3_m3.to_numpy(float)
    data["epsilon"] = topp(m)
    data["height_m"] = data.pals_rms_height_cm / 100.0
    data["length_m"] = data.pals_correlation_length_cm / 100.0

    k = 2*np.pi*1.26e9/299792458.0
    numeric = [
        "epsilon", "height_m", "length_m", "angle_used",
        "sigma0_hh_db", "sigma0_vv_db",
    ]
    keep = (
        np.isfinite(data[numeric].to_numpy(float)).all(axis=1)
        & (data.height_m > 0)
        & (data.length_m > 0)
        & (k*data.height_m <= 0.3)
        & (data.height_m/data.length_m <= 0.21)
        & data.angle_used.between(0, 90, inclusive="left")
    )

    output.mkdir(parents=True)
    data.assign(in_comparison=keep).to_csv(
        output / "coverage.csv", index=False
    )
    data = data.loc[keep].copy()
    if data.empty:
        raise RuntimeError("No samples in the common comparison domain.")

    records = []
    for _, r in data.iterrows():
        for scenario, angle in [
            ("fixed40", 40.0),
            ("field_day_angle", float(r.angle_used)),
        ]:
            epsilon = complex(r.epsilon, -0.05*r.epsilon)
            prediction = spm_backscatter_db(
                relative_permittivity=epsilon,
                rms_height_m=float(r.height_m),
                correlation_length_m=float(r.length_m),
                frequency_hz=1.26e9,
                incidence_angle_deg=angle,
                spectrum_model="exponential",
            )
            records.append({
                "request_id": r.unit_id + "_" + scenario,
                "unit_id": r.unit_id,
                "campaign": r.campaign,
                "field_id": r.field_id,
                "acquisition_date": r.acquisition_date,
                "scenario": scenario,
                "correlation_model": "exponential",
                "frequency_ghz": 1.26,
                "incidence_angle_deg": angle,
                "rms_height_m": float(r.height_m),
                "correlation_length_m": float(r.length_m),
                "dielectric_real": float(r.epsilon),
                "dielectric_loss_positive": float(0.05*r.epsilon),
                "sigma0_hh_db": float(r.sigma0_hh_db),
                "sigma0_vv_db": float(r.sigma0_vv_db),
                "spm_hh_db": float(prediction["hh_db"]),
                "spm_vv_db": float(prediction["vv_db"]),
            })

    requests = validate_teacher_requests(pd.DataFrame(records))
    requests.to_csv(output / "requests.csv", index=False)

    report = []
    for (campaign, scenario), part in requests.groupby(["campaign", "scenario"]):
        for record in metrics(part, "spm_hh_db", "spm_vv_db"):
            report.append({
                "campaign": campaign, "scenario": scenario,
                "teacher": "SPM", **record,
            })
    pd.DataFrame(report).to_csv(output / "spm_metrics.csv", index=False)

    (output / "protocol.json").write_text(json.dumps({
        "angle_support": "field-day mean, not per-radar-record integration",
        "loss_tangent": 0.05,
        "frequency_ghz": 1.26,
        "correlation": "exponential",
        "cohort": "common SPM/I2EM input-validity subset",
        "dielectric": "topp_both",
    }, indent=2), encoding="utf-8")
    print(requests.groupby(["campaign", "scenario"]).size())


def evaluate(output, results_path):
    requests = pd.read_csv(output / "requests.csv")
    results = pd.read_csv(results_path)
    paired = validate_teacher_results(requests, results)

    destination = output / "i2em_evaluation"
    if destination.exists():
        raise FileExistsError("I2EM evaluation already exists.")
    destination.mkdir()

    report = []
    for (campaign, scenario), part in paired.groupby(["campaign", "scenario"]):
        for record in metrics(part, "i2em_hh_db", "i2em_vv_db"):
            report.append({
                "campaign": campaign, "scenario": scenario,
                "teacher": "I2EM", **record,
            })
    pd.DataFrame(report).to_csv(destination / "metrics.csv", index=False)
    paired.to_csv(destination / "paired_predictions.csv", index=False)
    print(pd.DataFrame(report).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.results is None:
        prepare(args.output)
    else:
        evaluate(args.output, args.results)