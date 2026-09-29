"""Freeze the exact inputs and configuration for prospective SMEX02 scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the prospective SMEX02 zero-shot evaluation contract"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--final-intake-summary", required=True, type=Path)
    parser.add_argument("--final-audit-summary", required=True, type=Path)
    parser.add_argument("--stage5-manifest", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--spm", required=True, type=Path)
    parser.add_argument("--i2em-requests", required=True, type=Path)
    parser.add_argument("--i2em-results", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Freeze directory already exists: {output}")

    audit = json.loads(args.final_audit_summary.read_text(encoding="utf-8"))
    intake = json.loads(args.final_intake_summary.read_text(encoding="utf-8"))
    if audit.get("status") != "READY" or not all(audit.get("ready_checks", {}).values()):
        raise ValueError("Final external audit is not READY")
    if intake.get("status") != "FINAL_INPUT_READY":
        raise ValueError("Final intake is not marked FINAL_INPUT_READY")

    files = {
        "source_table": args.source,
        "external_table": args.external,
        "final_intake_summary": args.final_intake_summary,
        "final_audit_summary": args.final_audit_summary,
        "stage5_manifest": args.stage5_manifest,
        "preregistration": args.preregistration,
        "spm_teacher_samples": args.spm,
        "i2em_requests": args.i2em_requests,
        "i2em_results": args.i2em_results,
        "evaluator": args.evaluator,
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_TARGET_SCORING",
        "scientific_role": "immutable contract for one-time prospective SMEX02 zero-shot evaluation",
        "files": {name: fingerprint(path) for name, path in files.items()},
        "locked_configuration": {
            "dielectric_policy": "topp_both",
            "feature_names": [
                "soil_moisture_m3_m3",
                "soil_real_dielectric",
                "pals_rms_height_cm",
                "pals_correlation_length_cm",
            ],
            "common_head_primary": "source-training mean",
            "differential_head_primary": "SPM-to-I2EM sequential pretraining, source-real fitting, source-only grouped-CV shrinkage",
            "spm_epochs": 200,
            "i2em_epochs": 100,
            "source_epochs": 120,
            "source_inner_folds": 4,
            "repeats": 5,
            "shrinkage_candidates": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
            "bootstrap_field_blocks": 4000,
            "seed": 20260910,
            "minimum_field_subgroup_rows": 3,
        },
        "guardrails": [
            "SMEX02 targets were not used to choose architecture, features, epochs, seeds, or shrinkage candidates.",
            "Zero-shot scoring is run before any SMEX02 few-shot adaptation.",
            "Any later method revision requires a new independent prospective domain.",
        ],
    }
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
