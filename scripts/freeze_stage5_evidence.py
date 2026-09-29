"""Create an integrity manifest for the locked Stage-5 evidence.

The script is intentionally read-only with respect to experiment artifacts. It
hashes every selected file and writes one new JSON manifest. If the requested
manifest already exists, the script stops instead of overwriting it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(inputs: list[Path], output: Path) -> list[Path]:
    files: set[Path] = set()
    output_resolved = output.resolve()
    for item in inputs:
        resolved = item.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Evidence input does not exist: {resolved}")
        candidates = [resolved] if resolved.is_file() else resolved.rglob("*")
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve() != output_resolved:
                files.add(candidate.resolve())
    return sorted(files, key=lambda p: str(p).lower())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze Stage-5 evidence by recording sizes and SHA-256 hashes."
    )
    parser.add_argument(
        "--input", action="append", required=True, help="File or directory to hash."
    )
    parser.add_argument("--output", required=True, help="New JSON manifest path.")
    parser.add_argument(
        "--label", default="stage5_smapvex08_locked_evidence_v2", help="Snapshot label."
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen manifest: {output.resolve()}"
        )

    inputs = [Path(value) for value in args.input]
    files = iter_files(inputs, output)
    if not files:
        raise RuntimeError("No evidence files were found.")

    records = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "smapvex08_role": "locked_external_test_only",
            "model_selection_allowed": False,
            "final_result_directories": [
                "smapvex08_external_v2",
                "smapvex08_external_audit_v2",
                "cross_domain_transfer_20260910_v2",
            ],
            "exploratory_v1_directories_excluded_from_claims": True,
        },
        "file_count": len(records),
        "total_size_bytes": sum(item["size_bytes"] for item in records),
        "files": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen {len(records)} files")
    print(f"Manifest: {output.resolve()}")


if __name__ == "__main__":
    main()
