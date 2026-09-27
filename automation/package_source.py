#!/usr/bin/env python3
"""Stage the Hydra source as a Kaggle dataset dir (`kragglenote2forwork/arc3-hydra-src`, versioned). No upload.

    python automation/package_source.py [--out DIR] [--note TEXT]

Stages (layout kept, so relative paths inside the code still resolve on Kaggle):
    agent/**                         harness: Duck (vendored + patches), scheduler, hydra_run, perception, ...
    serving/**                       launch.py + profiles (B03)
    automation/serving_bench.py      the mock / EXP-001 bench launch.py calls for `mock` profiles and suites
    automation/serving_pack/**       recorded Duck prompts for the bench
    HYDRA_SOURCE.json                marker + manifest: dataset id, version, per-file sha256, tree sha256
    dataset-metadata.json            Kaggle dataset metadata (title/id/licenses)
    README.md                        dataset card (no secrets)
Skips __pycache__, *.pyc, .pytest_cache. Every staged file is secret-scanned (automation/preflight.py) and the
staging aborts on a hit. Version = v<UTC yyyymmdd-HHMMSS>-<tree sha256[:10]>; pin it in a candidate
(`source_version`) so the notebook refuses a stale attachment in commit mode.

Upload (the orchestrator / user does this; this script never talks to Kaggle):
    first time : kaggle datasets create  -p <DIR> --dir-mode zip
    afterwards : kaggle datasets version -p <DIR> --dir-mode zip -m "<version>"
then wait until `kaggle datasets status kragglenote2forwork/arc3-hydra-src` is ready before pushing a notebook.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "automation"))
from preflight import scan_secrets  # noqa: E402

DATASET_ID = "kragglenote2forwork/arc3-hydra-src"
DATASET_TITLE = "arc3-hydra-src"
DEFAULT_OUT_ROOT = Path("/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad/kaggle_datasets")
INCLUDE = ("agent", "serving", "automation/serving_bench.py", "automation/serving_pack")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
MARKER = "HYDRA_SOURCE.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(repo: Path = REPO) -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE:
        path = repo / rel
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files += [p for p in sorted(path.rglob("*")) if p.is_file() and not (set(p.relative_to(repo).parts) & SKIP_PARTS)
                      and p.suffix not in SKIP_SUFFIXES]
    return files


def tree_sha(entries: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(entries):
        h.update(f"{rel}\0{entries[rel]}\n".encode())
    return h.hexdigest()


def stage(out: Path | None = None, *, note: str = "", repo: Path = REPO, now: dt.datetime | None = None) -> dict:
    files = source_files(repo)
    missing = [rel for rel in ("agent/hydra_run.py", "agent/scheduler.py", "agent/duck_offline.py", "serving/launch.py")
               if not (repo / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot stage: {missing} missing")
    entries = {str(p.relative_to(repo)): sha256_file(p) for p in files}
    tree = tree_sha(entries)
    now = now or dt.datetime.now(dt.timezone.utc)
    version = f"v{now:%Y%m%d-%H%M%S}-{tree[:10]}"
    out = Path(out) if out else DEFAULT_OUT_ROOT / DATASET_TITLE / version
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty (never overwritten; pick another --out)")
    out.mkdir(parents=True, exist_ok=True)
    for p in files:
        dst = out / p.relative_to(repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    manifest = {
        "schema": 1,
        "dataset": DATASET_ID,
        "version": version,
        "created_utc": now.isoformat(timespec="seconds"),
        "note": note,
        "tree_sha256": tree,
        "n_files": len(entries),
        "bytes": sum((repo / rel).stat().st_size for rel in entries),
        "files": entries,
    }
    (out / MARKER).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET_TITLE, "id": DATASET_ID, "licenses": [{"name": "other"}]}, indent=1) + "\n")
    (out / "README.md").write_text(
        f"# arc3-hydra-src {version}\n\nSource snapshot of the ARC-AGI-3 Hydra harness (vendored Duck + campaign "
        "patches, scheduler, notebook runtime, serving profiles). Private; mounted by notebooks built with "
        "`automation/build_notebook.py`, located via the `HYDRA_SOURCE.json` marker. Contains no credentials.\n")
    hits = scan_secrets([out], root=out)
    if hits:
        raise RuntimeError("secret scan failed, not staging: " + "; ".join(str(h) for h in hits[:10]))
    return {"dir": str(out), "version": version, "tree_sha256": tree, "n_files": len(entries),
            "bytes": manifest["bytes"], "dataset": DATASET_ID}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", default=None, help=f"staging dir (default {DEFAULT_OUT_ROOT}/{DATASET_TITLE}/<version>)")
    ap.add_argument("--note", default="", help="free text stored in the manifest")
    a = ap.parse_args(argv)
    info = stage(Path(a.out) if a.out else None, note=a.note)
    print(json.dumps(info, indent=1))
    print(f"\nstaged {info['n_files']} files ({info['bytes'] / 1e6:.1f} MB) -> {info['dir']}")
    print("NOT uploaded. To upload (first time: `create` instead of `version`):")
    print(f"  kaggle datasets version -p {info['dir']} --dir-mode zip -m \"{info['version']}\"")
    print(f"then build the notebook with --source-version {info['version']} (or --source-dir {info['dir']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
