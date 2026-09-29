"""Check a repository before creating a public release."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".py", ".ps1", ".m", ".md", ".tex", ".bib", ".json",
    ".yaml", ".yml", ".toml", ".cff", ".txt",
}
SKIP_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "data", "outputs",
    "paper", "external",
}
REQUIRED = {
    "README.md", "LICENSE", ".gitignore", "pyproject.toml",
    "CITATION.cff", "DATA.md", "REPRODUCIBILITY.md",
    "THIRD_PARTY_NOTICES.md",
}
ABSOLUTE_WINDOWS = re.compile(r"[A-Za-z]:\\(?:[^\\\r\n]+\\)+")
SENSITIVE = re.compile(
    r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"authorization\s*[:=]|bearer\s+[A-Za-z0-9._-]+|\.netrc)"
)


def iter_public_text(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if relative.as_posix() == "scripts/open_source_audit.py":
            continue
        yield path, relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    args = parser.parse_args()
    root = args.project_root.resolve()

    problems: list[str] = []
    for name in sorted(REQUIRED):
        if not (root / name).exists():
            problems.append(f"missing required file: {name}")

    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {"data", "outputs", "paper"}:
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > args.max_file_mb:
            problems.append(f"large public candidate: {relative} ({size_mb:.1f} MB)")

    for path, relative in iter_public_text(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if ABSOLUTE_WINDOWS.search(text):
            problems.append(f"absolute Windows path: {relative}")
        if SENSITIVE.search(text):
            problems.append(f"possible credential text: {relative}")

    if problems:
        print("Open-source audit found issues:")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        return 1

    print("Open-source audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
