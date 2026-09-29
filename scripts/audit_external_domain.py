"""Audit an independent-domain table before any external validation.

The audit deliberately does not fit a predictive model.  It verifies that an
external campaign represents the same HH/VV forward task, applies a portable
dielectric-feature contract, and quantifies covariate support relative to the
source campaign without using external targets for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = [
    "soil_moisture_m3_m3",
    "soil_real_dielectric",
    "pals_rms_height_cm",
    "pals_correlation_length_cm",
]
TARGETS = ["sigma0_hh_db", "sigma0_vv_db"]
IDENTIFIERS = ["campaign_id", "acquisition_date", "field_id"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topp_real_permittivity(moisture: np.ndarray) -> np.ndarray:
    """Topp et al. cubic conversion used only for a portable feature contract."""
    mv = np.asarray(moisture, dtype=float)
    return 3.03 + 9.3 * mv + 146.0 * mv**2 - 76.7 * mv**3


def prepare_table(path: Path, role: str, dielectric_policy: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"field_id": "string", "campaign_id": "string"})
    required = {"acquisition_date", "field_id", "soil_moisture_m3_m3", *TARGETS}
    if dielectric_policy == "measured":
        required.add("soil_real_dielectric")
    required.update({"pals_rms_height_cm", "pals_correlation_length_cm"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{role} table is missing columns: {missing}")
    if "campaign_id" not in frame:
        frame["campaign_id"] = role
    frame["campaign_id"] = frame["campaign_id"].fillna(role).astype("string").str.strip()
    frame["field_id"] = frame["field_id"].astype("string").str.strip()
    frame["acquisition_date"] = pd.to_datetime(frame["acquisition_date"], errors="raise")
    numeric = ["soil_moisture_m3_m3", "pals_rms_height_cm", "pals_correlation_length_cm", *TARGETS]
    if "soil_real_dielectric" in frame:
        numeric.append("soil_real_dielectric")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if dielectric_policy == "topp_both":
        if "soil_real_dielectric" in frame:
            frame["soil_real_dielectric_measured"] = frame["soil_real_dielectric"]
        frame["soil_real_dielectric"] = topp_real_permittivity(frame["soil_moisture_m3_m3"])
    duplicate = frame.duplicated(["campaign_id", "field_id", "acquisition_date"])
    if duplicate.any():
        raise ValueError(f"{role} has {int(duplicate.sum())} duplicate campaign-field-date rows")
    finite = np.isfinite(frame[FEATURES + TARGETS].to_numpy(dtype=float)).all(axis=1)
    frame["finite_model_row"] = finite
    return frame


def feature_shift(source: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    source = source.loc[source.finite_model_row]
    external = external.loc[external.finite_model_row]
    rows: list[dict[str, float | str]] = []
    for name in FEATURES:
        src = source[name].to_numpy(dtype=float)
        ext = external[name].to_numpy(dtype=float)
        low, high = np.quantile(src, [0.01, 0.99])
        pooled = np.sqrt((np.var(src, ddof=1) + np.var(ext, ddof=1)) / 2.0)
        rows.append(
            {
                "feature": name,
                "source_mean": float(np.mean(src)),
                "external_mean": float(np.mean(ext)),
                "source_q01": float(low),
                "source_q99": float(high),
                "external_min": float(np.min(ext)),
                "external_max": float(np.max(ext)),
                "external_fraction_inside_source_q01_q99": float(np.mean((ext >= low) & (ext <= high))),
                "standardized_mean_difference": float((np.mean(ext) - np.mean(src)) / pooled) if pooled > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def validity_summary(frame: pd.DataFrame, frequency_ghz: float) -> dict[str, float]:
    wavelength = 299_792_458.0 / (frequency_ghz * 1e9)
    k_sigma = 2.0 * np.pi / wavelength * frame.pals_rms_height_cm.to_numpy(dtype=float) / 100.0
    slope = frame.pals_rms_height_cm.to_numpy(dtype=float) / frame.pals_correlation_length_cm.to_numpy(dtype=float)
    valid = np.isfinite(k_sigma) & np.isfinite(slope) & (k_sigma < 0.30) & (slope < 0.21)
    return {
        "frequency_ghz": float(frequency_ghz),
        "fraction_spm_valid_ks_lt_0_30_and_sigma_over_L_lt_0_21": float(np.mean(valid)),
        "k_sigma_max": float(np.nanmax(k_sigma)),
        "sigma_over_L_max": float(np.nanmax(slope)),
    }


def save_plot(shift: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].barh(shift.feature, shift.external_fraction_inside_source_q01_q99, color="#4C78A8")
    axes[0].axvline(0.8, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, 1), xlabel="External fraction inside source 1%-99% range", title="Feature support overlap")
    axes[1].barh(shift.feature, shift.standardized_mean_difference, color="#F58518")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(xlabel="Standardized mean difference", title="External-domain shift")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(output / "01_external_feature_support.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent-domain contract and shift audit")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dielectric-policy", choices=["topp_both", "measured"], default="topp_both")
    parser.add_argument("--frequency-ghz", type=float, default=1.26)
    parser.add_argument("--minimum-external-fields", type=int, default=5)
    parser.add_argument("--minimum-external-dates", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; use a new versioned directory")
    source = prepare_table(args.source.resolve(), "source", args.dielectric_policy)
    external = prepare_table(args.external.resolve(), "external", args.dielectric_policy)
    source_campaigns = set(source.campaign_id.dropna().astype(str))
    external_campaigns = set(external.campaign_id.dropna().astype(str))
    shift = feature_shift(source, external)
    ready_checks = {
        "same_forward_task_has_HH_and_VV": all(name in external for name in TARGETS),
        "campaign_ids_disjoint": source_campaigns.isdisjoint(external_campaigns),
        "enough_external_fields": external.field_id.nunique() >= args.minimum_external_fields,
        "enough_external_dates": external.acquisition_date.nunique() >= args.minimum_external_dates,
        "all_model_rows_finite": bool(external.finite_model_row.all()),
        "nonzero_roughness_and_correlation_length": bool((external.pals_rms_height_cm > 0).all() and (external.pals_correlation_length_cm > 0).all()),
    }
    source_years = sorted(source.acquisition_date.dt.year.unique().astype(int).tolist())
    external_years = sorted(external.acquisition_date.dt.year.unique().astype(int).tolist())
    summary = {
        "research_question": "Is the external campaign technically compatible and genuinely independent before zero-shot evaluation?",
        "status": "READY" if all(ready_checks.values()) else "NOT_READY",
        "ready_checks": ready_checks,
        "dielectric_policy": args.dielectric_policy,
        "dielectric_policy_rationale": "topp_both recomputes dielectric from soil moisture in both domains because measured dielectric is not consistently available across campaigns.",
        "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source.resolve()), "rows": len(source), "finite_rows": int(source.finite_model_row.sum()), "fields": int(source.field_id.nunique()), "dates": int(source.acquisition_date.nunique()), "campaigns": sorted(source_campaigns), "years": source_years, "spm_validity": validity_summary(source.loc[source.finite_model_row], args.frequency_ghz)},
        "external": {"path": str(args.external.resolve()), "sha256": sha256(args.external.resolve()), "rows": len(external), "finite_rows": int(external.finite_model_row.sum()), "fields": int(external.field_id.nunique()), "dates": int(external.acquisition_date.nunique()), "campaigns": sorted(external_campaigns), "years": external_years, "spm_validity": validity_summary(external.loc[external.finite_model_row], args.frequency_ghz)},
        "independence_evidence": {"different_campaign": source_campaigns.isdisjoint(external_campaigns), "different_year": set(source_years).isdisjoint(external_years)},
        "external_targets_are_for_final_scoring_only": True,
        "guardrails": ["Do not select architecture, gate thresholds, epochs, or features using external target performance.", "Report zero-shot results before any few-shot adaptation.", "Select adaptation fields by group and score only on untouched external fields.", "Treat low source-support overlap as extrapolation evidence, not as rows to silently discard."],
    }
    output.mkdir(parents=True, exist_ok=True)
    source.to_csv(output / "source_portable_contract.csv", index=False)
    external.to_csv(output / "external_portable_contract.csv", index=False)
    shift.to_csv(output / "feature_shift_audit.csv", index=False)
    save_plot(shift, output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Audit outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
