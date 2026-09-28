#!/usr/bin/env python3
"""Stage `agent/duck/` (the vendored Duck bundle with our [DUCK-PATCH] source edits) as the private Kaggle dataset
`kragglenote2forwork/arc3-duck-bundle` that the Hydra-1 notebooks attach instead of keithtyser's smoke bundle.

The bundle layout is identical to keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1 (same SOURCE_IDENTITY.json,
byte-identical serving_setup.py: its identity check hashes itself, the model and the runtime, not src/). Only the
Duck source under src/ differs (switchable patches, all documented in PATCHES.md).

Refuses to stage when: serving_setup.py / SOURCE_IDENTITY.json differ from the pinned hashes, a credential is found,
or a .gz file is present (Kaggle auto-decompresses them, MISTAKES #13). Writes DUCK_BUNDLE_VERSION.json, which the
Hydra-1 notebook asserts, so a mis-attached bundle fails loudly instead of silently running stock code.

    python automation/hydra1/stage_bundle.py --out $SCRATCH/kaggle_datasets/arc3-duck-bundle
    # then (private; additive — new versions never delete old ones):
    #   kaggle datasets create  -p <out> --dir-mode zip                 (first time)
    #   kaggle datasets version -p <out> --dir-mode zip -m "<version>"  (updates)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "agent" / "duck"
DATASET_ID = "kragglenote2forwork/arc3-duck-bundle"
# serving_setup.py verifies its own sha256 against SOURCE_IDENTITY.json at runtime; pin both here too.
SERVING_SETUP_SHA256 = "037c041c9bd9dcffa9084b32f47af9cf1bf35849d5eaf2e1a3422daac098b2e2"
SECRET_PATTERNS = [re.compile(p) for p in (r"KGAT_[A-Za-z0-9]{16,}", r"hf_[A-Za-z0-9]{30,}", r"ghp_[A-Za-z0-9]{30,}",
                                          r"github_pat_[A-Za-z0-9_]{30,}", r"sk-[A-Za-z0-9]{30,}")]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _secret_values() -> list[str]:
    return [v for k in ("KAGGLE_API_TOKEN", "HF_TOKEN", "ARC_API_KEY", "GITHUB_TOKEN") if len(v := os.environ.get(k, "")) >= 12]


def stage(out: Path) -> dict:
    if _sha256(SRC / "serving_setup.py") != SERVING_SETUP_SHA256:
        raise SystemExit("serving_setup.py changed: its runtime identity check would fail on Kaggle")
    identity = json.loads((SRC / "SOURCE_IDENTITY.json").read_text())
    if identity.get("serving_setup_sha256") != SERVING_SETUP_SHA256:
        raise SystemExit("SOURCE_IDENTITY.json no longer pins serving_setup.py")
    if not (SRC / "taaf-kaggle-bundle.json").is_file():
        raise SystemExit("bundle marker taaf-kaggle-bundle.json missing")

    if out.exists():  # local scratch staging dir only (never a repo or hub)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    secrets = _secret_values()
    files: dict[str, dict] = {}
    tree = hashlib.sha256()
    for path in sorted(SRC.rglob("*")):
        rel = path.relative_to(SRC)
        if path.is_dir() or any(part in SKIP_DIRS for part in rel.parts) or path.suffix in SKIP_SUFFIXES:
            continue
        if path.suffix == ".gz":
            raise SystemExit(f"refusing .gz file {rel}: Kaggle auto-decompresses it (MISTAKES #13)")
        data = path.read_bytes()
        if path.suffix not in {".pkl", ".png"}:
            text = data.decode("utf-8", errors="ignore")
            if any(p.search(text) for p in SECRET_PATTERNS) or any(s in text for s in secrets):
                raise SystemExit(f"refusing to stage {rel}: looks like it contains a credential")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        files[str(rel)] = {"bytes": len(data), "sha256": digest}
        if rel.parts[0] == "src":
            tree.update(f"{rel}\0{digest}\n".encode())
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    version = {
        "dataset": DATASET_ID,
        "git_commit": commit,
        "src_tree_sha256": tree.hexdigest(),
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": len(files),
        "patches_doc": "PATCHES.md (all [DUCK-PATCH] edits are switchable via DUCK_PATCHES / DUCK_PATCH_<ID>)",
    }
    (out / "DUCK_BUNDLE_VERSION.json").write_text(json.dumps(version, indent=1) + "\n")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "arc3-duck-bundle", "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]}, indent=1) + "\n")
    return {"out": str(out), **version}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    info = stage(Path(ap.parse_args().out))
    print(json.dumps(info, indent=1))
    print(f"Upload (private): kaggle datasets create -p {info['out']} --dir-mode zip   (or `version -m ...` later)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
