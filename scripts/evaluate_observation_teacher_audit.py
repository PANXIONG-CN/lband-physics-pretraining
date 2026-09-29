"""Audit SPM and I2EM teachers against observation-aligned field data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_decoupled_physics_heads as base  # noqa: E402
from research_pilots.scattering.surfaces.teacher_contract import validate_teacher_results  # noqa: E402


def metrics(frame: pd.DataFrame) -> pd.DataFrame:
    observed = frame[["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(dtype=float)
    predictions = {
        "spm": frame[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float),
        "i2em": frame[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float),
    }
    observed_components = base.to_components(observed)
    rows = []
    for method, channels in predictions.items():
        components = base.to_components(channels)
        for name, reference, predicted in [
            ("HH", observed[:, 0], channels[:, 0]),
            ("VV", observed[:, 1], channels[:, 1]),
            ("common", observed_components[:, 0], components[:, 0]),
            ("differential", observed_components[:, 1], components[:, 1]),
        ]:
            error = predicted - reference
            rows.append({"teacher": method, "response": name, "n": len(frame), "rmse_db": float(np.sqrt(np.mean(error**2))), "mae_db": float(np.mean(np.abs(error))), "bias_db": float(np.mean(error))})
    return pd.DataFrame(rows)


def save_plots(frame: pd.DataFrame, metric_table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        part = metric_table[metric_table.response == response]
        ax.bar(part.teacher, part.rmse_db, color=["#4C78A8", "#F58518"])
        ax.set(ylabel="Direct teacher RMSE against observations (dB)", title=response)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "01_direct_teacher_observation_rmse.png", dpi=190)
    plt.close(fig)

    observed = frame[["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(dtype=float)
    spm = frame[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float)
    i2em = frame[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    improvement = np.sqrt(np.mean((spm - observed) ** 2, axis=1)) - np.sqrt(np.mean((i2em - observed) ** 2, axis=1))
    disagreement = np.sqrt(np.mean((i2em - spm) ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    ax.scatter(disagreement, improvement, s=20, alpha=0.55)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set(xlabel="SPM-I2EM disagreement (joint dB)", ylabel="SPM error - I2EM error (dB)", title="Does higher fidelity improve individual observations?")
    ax.grid(alpha=0.2)
    fig.savefig(output / "02_disagreement_vs_observation_improvement.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Observation-aligned physics-teacher audit")
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; choose a new directory")
    frame = validate_teacher_results(pd.read_csv(args.requests.resolve(), dtype={"field_id": "string"}), pd.read_csv(args.results.resolve()))
    observed = frame[["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(dtype=float)
    spm = frame[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float)
    i2em = frame[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    groups = frame.field_id.astype(str).to_numpy()
    comparison = {
        "joint_channels": base.grouped_bootstrap_joint_delta(observed, i2em, spm, groups, args.bootstrap_iterations, args.seed),
        "components": base.grouped_bootstrap_delta(base.to_components(observed), base.to_components(i2em), base.to_components(spm), groups, ["common", "differential"], args.bootstrap_iterations, args.seed + 1),
    }
    metric_table = metrics(frame)
    spm_error = np.sqrt(np.mean((spm - observed) ** 2, axis=1))
    i2em_error = np.sqrt(np.mean((i2em - observed) ** 2, axis=1))
    disagreement = np.sqrt(np.mean((i2em - spm) ** 2, axis=1))
    improvement = spm_error - i2em_error
    rho, pvalue = spearmanr(disagreement, improvement)
    frame["spm_joint_point_error_db"] = spm_error
    frame["i2em_joint_point_error_db"] = i2em_error
    frame["teacher_joint_disagreement_db"] = disagreement
    frame["i2em_improvement_over_spm_db"] = improvement
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "observation_aligned_teacher_predictions.csv", index=False)
    metric_table.to_csv(output / "teacher_metrics.csv", index=False)
    save_plots(frame, metric_table, output)
    summary = {
        "research_question": "Is the higher-fidelity I2EM teacher closer than SPM to the current field observations before any statistical calibration?",
        "samples": len(frame), "fields": int(frame.field_id.nunique()), "dates": int(frame.acquisition_date.nunique()),
        "teacher_metrics": metric_table.to_dict(orient="records"),
        "i2em_minus_spm_grouped_bootstrap": comparison,
        "disagreement_vs_i2em_improvement": {"spearman_rho": float(rho), "p_value_unadjusted": float(pvalue)},
        "fraction_rows_i2em_better": float(np.mean(i2em_error < spm_error)),
        "decision_rule": "Higher physics fidelity is observation-beneficial only if I2EM-minus-SPM error intervals exclude zero below zero. Otherwise use teacher disagreement as a risk signal, not as proof of truth.",
        "scope_limit": "Direct discrepancies include vegetation, footprint aggregation, roughness uncertainty, and measurement mismatch; they do not isolate the bare-soil scattering formula alone.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(metric_table.to_string(index=False), flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
