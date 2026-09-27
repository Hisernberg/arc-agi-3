#!/usr/bin/env python3
"""Assemble the private Kaggle dataset `kragglenote2forwork/arc3-serving-kit` that the EXP-001 notebook attaches.

The dataset carries our code + the replay pack with the repo layout preserved, so the notebook runs
`<kit>/serving/launch.py --suite exp001`. This script only builds the directory; it never uploads.

  python automation/exp001_notebook/build_kit.py --out /tmp/arc3-serving-kit
  # then, after review (private by default, GPU-free):
  #   kaggle datasets create -p /tmp/arc3-serving-kit --dir-mode zip          (first time)
  #   kaggle datasets version -p /tmp/arc3-serving-kit -m "..." --dir-mode zip (updates)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET_ID = "kragglenote2forwork/arc3-serving-kit"
FILES = [
    "automation/serving_bench.py",
    "automation/serving_pack/duck_0922.jsonl.gz",
    "serving/launch.py",
    "serving/README.md",
]
SECRET_PATTERNS = [re.compile(p) for p in (r"KGAT_[A-Za-z0-9]{16,}", r"hf_[A-Za-z0-9]{30,}", r"ghp_[A-Za-z0-9]{30,}",
                                          r"github_pat_[A-Za-z0-9_]{30,}", r"sk-[A-Za-z0-9]{30,}")]


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = list(FILES) + [str(p.relative_to(REPO)) for p in sorted((REPO / "serving" / "profiles").glob("*.json"))]
    manifest = {}
    for rel in files:
        src = REPO / rel
        data = src.read_bytes()
        if not rel.endswith(".gz"):
            text = data.decode("utf-8")
            for pat in SECRET_PATTERNS:
                if pat.search(text):
                    raise SystemExit(f"refusing to package {rel}: looks like it contains a credential")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        manifest[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (out / "arc3-serving-kit.json").write_text(json.dumps({"built_epoch": time.time(), "files": manifest}, indent=1) + "\n")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "arc3-serving-kit", "id": DATASET_ID, "licenses": [{"name": "CC0-1.0"}]}, indent=1) + "\n")
    return {"out": str(out), "files": len(manifest), "bytes": sum(v["bytes"] for v in manifest.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(build(Path(a.out)), indent=1))
    print("Upload (private, manual step): kaggle datasets create -p", a.out, "--dir-mode zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
