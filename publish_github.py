"""Upload a verified code/data companion to the existing public GitHub repo.

Default: local verification only. --publish uses an authenticated GitHub CLI,
preserves default-branch history and never force-pushes. Credentials are not
read from files, embedded in this script, or printed.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = "PANXIONG-CN/lband-physics-pretraining"
VERSION = "0.2.1"


def run(args: list[str], *, cwd: Path = ROOT, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, check=True, capture_output=capture)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Push to the existing repository.")
    args = parser.parse_args()
    run([sys.executable, "scripts/verify_release.py", "--public-only"])
    with (ROOT / "release_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        paths = [row["path"] for row in csv.DictReader(handle)]
    paths.append("release_manifest.csv")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths in manifest.")
    print(f"Destination: https://github.com/{REPO}; version {VERSION}; {len(paths)} files.")
    if not args.publish:
        print("Local checks complete; no remote changes. Use --publish to upload.")
        return
    for tool in ("git", "gh"):
        if not shutil.which(tool):
            raise RuntimeError(f"Install {tool} and authenticate GitHub CLI before publishing.")
    profile = json.loads(run(["gh", "api", "user"], capture=True).stdout)
    if profile["login"].lower() != "panxiong-cn":
        raise RuntimeError("Authenticate GitHub CLI as PANXIONG-CN.")
    metadata = json.loads(run(["gh", "api", f"repos/{REPO}"], capture=True).stdout)
    if metadata.get("private") or metadata.get("archived") or not metadata.get("permissions", {}).get("push"):
        raise RuntimeError("An active public destination and push permission are required.")
    branch = metadata["default_branch"]
    # Use the existing gh credential provider for these Git commands only.
    git = ["git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]
    with tempfile.TemporaryDirectory(prefix="lband-publish-") as temp:
        checkout = Path(temp) / "repository"
        run(git + ["clone", "--single-branch", "--branch", branch,
                   f"https://github.com/{REPO}.git", str(checkout)])
        existing = set(run(["git", "ls-files"], cwd=checkout, capture=True).stdout.splitlines())
        unexpected = existing - set(paths)
        if unexpected:
            raise RuntimeError("Reconcile remote-only files before publication: " + ", ".join(sorted(unexpected)))
        for relative in paths:
            destination = checkout / relative
            if not destination.resolve().is_relative_to(checkout.resolve()):
                raise ValueError(f"Unsafe destination: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        run([sys.executable, "scripts/verify_release.py", "--public-only"], cwd=checkout)
        # The verified manifest includes archived validation logs ignored by *.log.
        # Force-add only these explicit release paths, never the whole checkout.
        run(["git", "add", "--all", "--force", "--", *paths], cwd=checkout)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=checkout)
        if diff.returncode == 0:
            print("The remote already contains this exact snapshot.")
            return
        if diff.returncode != 1:
            raise RuntimeError("Could not inspect the staged diff.")
        identity = ["git", "-c", f'user.name={profile["login"]}', "-c",
                    f'user.email={profile["id"]}+{profile["login"]}@users.noreply.github.com']
        run(identity + ["commit", "-m", f"Publish v{VERSION}: code and processed-data companion"], cwd=checkout)
        run(git + ["push", "origin", f"HEAD:refs/heads/{branch}"], cwd=checkout)
        commit = run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).stdout.strip()
        remote = json.loads(run(["gh", "api", f"repos/{REPO}/git/ref/heads/{branch}"], capture=True).stdout)
        if remote["object"]["sha"] != commit:
            raise RuntimeError("Remote branch changed; inspect its final commit before reporting completion.")
        print(f"Uploaded and verified: https://github.com/{REPO}/commit/{commit}")
        print("Update the README publication status and the private manuscript repository flag after verification.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Publication stopped: {error}", file=sys.stderr)
        sys.exit(1)
