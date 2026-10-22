#!/usr/bin/env python3
"""Read-only E2 evaluation of frozen scattering checkpoints on source observations.

This script performs forward evaluation only. It never fits a model, changes
checkpoint arrays, or reads SMEX02 labels. Outputs go to a new directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SOURCE = REPRO / "data/source/smapvex12_portable_source.csv"
CHECKPOINT_DIR = REPRO / "results/trajectory_reconstruction"
EXPECTED_SHA256 = {
    "source": "7b52a128dfbdd49c6270bc80203227cb63daa7f7e312a1f40a50c6ecbe15e785",
    "member_1": "b062d816011b0ec903f8f3827240113cefefac487b27a147855a9f17abefd72b",
    "member_2": "384b5339d75bb38d05bd4b8345f0ecfe8932da369a3aa668c6ee0776c2934f47",
    "member_3": "c31c8b76dac4e23606c870086f8d319991760259749495b7e31177cfde7b2a56",
    "member_4": "0b538fc6bd9c524da2b499902b5d580adae84425ac31b871e809ea81db3fa51e",
    "member_5": "afc40dd4ee0c4d9e4b6b3626eccaa65eb1794e2c6a439c6638584fcc335e17ca",
}
STAGES = {
    "spm_pretraining": "spm",
    "sequential_pretraining": "pretraining",
    "source_inherited_in_sample": "source",
    "source_reset_in_sample": "source_reset_adam_003",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source() -> list[dict[str, str]]:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 240:
        raise ValueError(f"Expected 240 source records, found {len(records)}")
    keys = [(item["field_id"], item["acquisition_date"]) for item in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicated source field-date key")
    if any(item["model_ready"].lower() != "true" for item in records):
        raise ValueError("Source table contains non-model-ready records")
    return sorted(records, key=lambda item: (item["field_id"], item["acquisition_date"]))


def source_arrays(records: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moisture = np.array([float(item["soil_moisture_m3_m3"]) for item in records])
    dielectric = 3.03 + 9.3 * moisture + 146.0 * moisture**2 - 76.7 * moisture**3
    height = np.array([float(item["pals_rms_height_cm"]) for item in records])
    length = np.array([float(item["pals_correlation_length_cm"]) for item in records])
    ks = np.array([float(item["k_rms_height"]) for item in records])
    x = np.column_stack([moisture, dielectric, height, length])
    y = np.array([float(item["sigma0_vv_db"]) - float(item["sigma0_hh_db"]) for item in records])
    valid = (ks < 0.3) & (height / length <= 0.21)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (height <= 0).any() or (length <= 0).any():
        raise ValueError("Invalid source input or differential observation")
    if valid.sum() != 186 or len({records[i]["field_id"] for i in np.flatnonzero(valid)}) != 19:
        raise ValueError("SPM-valid source cohort differs from the recorded 186 rows and 19 fields")
    return x, y, valid


def forward(checkpoint: np.lib.npyio.NpzFile, prefix: str, x: np.ndarray) -> np.ndarray:
    hidden = (x - checkpoint["x_mean"]) / checkpoint["x_scale"]
    for layer in (0, 1):
        hidden = np.tanh(hidden @ checkpoint[f"{prefix}_coefs_{layer}"] + checkpoint[f"{prefix}_intercepts_{layer}"])
    standardized = hidden @ checkpoint[f"{prefix}_coefs_2"] + checkpoint[f"{prefix}_intercepts_2"]
    return (standardized * checkpoint["y_scale"] + checkpoint["y_mean"]).ravel()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    residual = p - y
    yc = y - np.mean(y)
    pc = p - np.mean(p)
    obs_variance = float(np.mean(yc**2))
    pred_variance = float(np.mean(pc**2))
    centered_mse = float(np.mean((pc - yc)**2))
    corr = float(np.mean(yc * pc) / np.sqrt(obs_variance * pred_variance)) if pred_variance > 1e-20 else None
    answer = {
        "bias_db": float(np.mean(residual)),
        "rmse_db": float(np.sqrt(np.mean(residual**2))),
        "centered_rmse_db": float(np.sqrt(centered_mse)),
        "centered_skill": float(1.0 - centered_mse / obs_variance),
        "correlation": corr,
        "variance_ratio": float(pred_variance / obs_variance),
        "observed_variance_db2": obs_variance,
        "predicted_variance_db2": pred_variance,
    }
    if not np.isclose(answer["rmse_db"]**2, answer["bias_db"]**2 + centered_mse, atol=1e-11):
        raise AssertionError("RMSE decomposition failed")
    return answer


def variation_metrics(y: np.ndarray, p: np.ndarray, fields: np.ndarray) -> dict[str, float]:
    yc = y - y.mean()
    pc = p - p.mean()
    within_y = np.empty_like(y)
    within_p = np.empty_like(p)
    for field in np.unique(fields):
        mask = fields == field
        within_y[mask] = y[mask] - y[mask].mean()
        within_p[mask] = p[mask] - p[mask].mean()
    between_y = yc - within_y
    between_p = pc - within_p
    total_variance = float(np.mean(yc**2))
    within_variance = float(np.mean(within_y**2))
    between_variance = float(np.mean(between_y**2))
    return {
        "total_skill": float(1.0 - np.mean((pc - yc)**2) / total_variance),
        "within_field_skill": float(1.0 - np.mean((within_p - within_y)**2) / within_variance),
        "between_field_skill": float(1.0 - np.mean((between_p - between_y)**2) / between_variance),
        "within_observed_variance_fraction": float(within_variance / total_variance),
        "between_observed_variance_fraction": float(between_variance / total_variance),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_intervals(
    records: list[dict[str, str]], y: np.ndarray, predictions: dict[str, np.ndarray],
    mask: np.ndarray, cohort: str, iterations: int, seed: int,
) -> tuple[list[dict], list[dict]]:
    indices = np.flatnonzero(mask)
    fields = sorted({records[i]["field_id"] for i in indices})
    groups = [np.array([i for i in indices if records[i]["field_id"] == field], dtype=int) for field in fields]
    rng = np.random.default_rng(seed)
    samples = {stage: np.empty(iterations) for stage in predictions}
    for iteration in range(iterations):
        drawn = rng.integers(0, len(groups), size=len(groups))
        selection = np.concatenate([groups[j] for j in drawn])
        for stage, pred in predictions.items():
            samples[stage][iteration] = metrics(y[selection], pred[selection])["centered_skill"]
    rows = []
    for stage, pred in predictions.items():
        lower, upper = np.quantile(samples[stage], [0.025, 0.975])
        rows.append({
            "cohort": cohort, "stage": stage, "n_records": len(indices), "n_fields": len(fields),
            "centered_skill": metrics(y[indices], pred[indices])["centered_skill"],
            "ci_low": float(lower), "ci_high": float(upper),
            "bootstrap_iterations": iterations, "bootstrap_seed": seed,
        })
    contrasts = []
    for first, second in [
        ("spm_pretraining", "sequential_pretraining"),
        ("sequential_pretraining", "source_inherited_in_sample"),
        ("sequential_pretraining", "source_reset_in_sample"),
    ]:
        difference = samples[second] - samples[first]
        lower, upper = np.quantile(difference, [0.025, 0.975])
        point = metrics(y[indices], predictions[second][indices])["centered_skill"] - metrics(y[indices], predictions[first][indices])["centered_skill"]
        contrasts.append({
            "cohort": cohort, "first": first, "second": second,
            "delta_skill_second_minus_first": float(point),
            "ci_low": float(lower), "ci_high": float(upper),
            "bootstrap_iterations": iterations, "bootstrap_seed": seed,
        })
    return rows, contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=27260910)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1000:
        raise ValueError("At least 1000 grouped bootstrap draws are required")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    input_paths = {"source": SOURCE}
    input_paths.update({f"member_{j}": CHECKPOINT_DIR / f"rebuilt_member_{j}.npz" for j in range(1, 6)})
    input_hashes = {name: sha256(path) for name, path in input_paths.items()}
    for name, digest in input_hashes.items():
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Input hash mismatch: {name}")

    records = read_source()
    x, y, valid = source_arrays(records)
    member_predictions: list[dict] = []
    stage_members: dict[str, list[np.ndarray]] = {name: [] for name in STAGES}
    for member in range(1, 6):
        with np.load(input_paths[f"member_{member}"], allow_pickle=False) as checkpoint:
            current = {stage: forward(checkpoint, prefix, x) for stage, prefix in STAGES.items()}
        for stage, values in current.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite prediction in {stage}, member {member}")
            stage_members[stage].append(values)
        for i, item in enumerate(records):
            member_predictions.append({
                "field_id": item["field_id"],
                "acquisition_date": item["acquisition_date"],
                "member": member,
                "d_observed_db": float(y[i]),
                "spm_valid": bool(valid[i]),
                **{f"d_{stage}_db": float(values[i]) for stage, values in current.items()},
            })
    ensemble = {stage: np.mean(np.stack(values), axis=0) for stage, values in stage_members.items()}
    source_intercept = float(np.mean(y - ensemble["sequential_pretraining"]))
    if not np.isclose(source_intercept, -3.8079110886071925, atol=1e-10):
        raise AssertionError(f"Source intercept fails archived reference check: {source_intercept}")

    cohorts = {"all_source_240": np.ones(len(records), dtype=bool), "spm_valid_186": valid}
    ensemble_metrics: list[dict] = []
    member_metrics: list[dict] = []
    variation: list[dict] = []
    intervals: list[dict] = []
    contrasts: list[dict] = []
    field_ids = np.array([item["field_id"] for item in records])
    for cohort, mask in cohorts.items():
        for stage, pred in ensemble.items():
            ensemble_metrics.append({
                "cohort": cohort, "stage": stage, "n_records": int(mask.sum()),
                "n_fields": len(set(field_ids[mask])), "evaluation_role": "in_sample_source_training" if "in_sample" in stage else "no_source_label_training",
                **metrics(y[mask], pred[mask]),
            })
            variation.append({
                "cohort": cohort, "stage": stage, "n_records": int(mask.sum()),
                "n_fields": len(set(field_ids[mask])),
                **variation_metrics(y[mask], pred[mask], field_ids[mask]),
            })
            for member, member_pred in enumerate(stage_members[stage], start=1):
                member_metrics.append({
                    "cohort": cohort, "stage": stage, "member": member,
                    "n_records": int(mask.sum()), "n_fields": len(set(field_ids[mask])),
                    **metrics(y[mask], member_pred[mask]),
                })
        interval_rows, contrast_rows = bootstrap_intervals(
            records, y, ensemble, mask, cohort, args.bootstrap_iterations, args.bootstrap_seed
        )
        intervals.extend(interval_rows)
        contrasts.extend(contrast_rows)

    ensemble_predictions = [
        {
            "field_id": item["field_id"], "acquisition_date": item["acquisition_date"],
            "d_observed_db": float(y[i]), "spm_valid": bool(valid[i]),
            **{f"d_{stage}_db": float(pred[i]) for stage, pred in ensemble.items()},
        }
        for i, item in enumerate(records)
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "predictions_by_member.csv", member_predictions)
    write_rows(output / "ensemble_predictions.csv", ensemble_predictions)
    write_rows(output / "metrics_by_member.csv", member_metrics)
    write_rows(output / "metrics_ensemble.csv", ensemble_metrics)
    write_rows(output / "variation_level.csv", variation)
    write_rows(output / "field_bootstrap_intervals.csv", intervals)
    write_rows(output / "paired_skill_contrasts.csv", contrasts)
    manifest = {
        "status": "COMPLETE",
        "analysis_role": "Approved E2 source-domain forward evaluation; no training or selection",
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_sha256": input_hashes,
        "script_sha256": sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "source_records": len(records), "source_fields": len(set(field_ids)),
        "spm_valid_records": int(valid.sum()), "spm_valid_fields": len(set(field_ids[valid])),
        "feature_rule": "Topp dielectric recomputed from soil moisture; original NPZ scales fixed",
        "output_rule": "Differential VV minus HH in dB",
        "stages": STAGES,
        "source_intercept_reference_check_db": source_intercept,
        "selection": "None; all model weights and evaluation cohorts fixed",
        "source_trained_stage_caveat": "Source-trained endpoints are scored in sample; they are not source held-out validation estimates",
        "bootstrap": {"unit": "field_id", "iterations": args.bootstrap_iterations, "seed": args.bootstrap_seed, "conditional_on": "saved weights and observed source dates"},
        "target_labels_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "source_records": len(records),
        "source_fields": len(set(field_ids)), "spm_valid_records": int(valid.sum()),
        "source_intercept_reference_check_db": source_intercept,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()