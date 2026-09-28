#!/usr/bin/env python3
"""Build the Hydra-1 Kaggle notebooks (EXP-004 = 1a, EXP-005 = 1b) on top of the Hydra-0 build.

Hydra-1 = Hydra-0 (same serving profile, AGENTFIX + anim frames + frame sheet + ARM P, wave-fit, smoke commits) but the
Duck source bundle is OUR vendored copy `kragglenote2forwork/arc3-duck-bundle` (automation/hydra1/stage_bundle.py)
instead of keithtyser's smoke bundle, so our switchable [DUCK-PATCH] source edits run on Kaggle:

  1a  S1 state-key isolation + v9 prompt (P1 true scoring + game clock, P2 UNDO, P3 play discipline)
  1b  1a + M2 verified-facts memory (median prompt 20.0k -> 9.5k tokens/request with AGENTFIX layered, CPU mock)

AGENTFIX keeps supplying F1/F3 (+F6/F9/F13/F19/ARM P) exactly as in Hydra-0, so our duplicate source fixes F1/F2/F3
are switched off. The Hydra-0 scoring/pacing cell is switched off (HYDRA_SCORING=0): v9 P1 supersedes it.
The switches are identical to agent/duck_offline.py PROFILES["hydra1a"/"hydra1b"] (asserted by tests), which is what
the CPU rehearsal runs.

Usage:  python automation/hydra1/build_hydra1.py --variant 1a|1b [--kv-gib 10] [--out notebooks/hydra1a]
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE_DATASET = "kragglenote2forwork/arc3-duck-bundle"
STOCK_BUNDLE = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"

VARIANTS = {
    "1a": {"DUCK_PATCHES": "1", "DUCK_PATCH_F1_IMAGES": "0", "DUCK_PATCH_F2_RESULT": "0", "DUCK_PATCH_F3_MEMORY": "0"},
    "1b": {"DUCK_PATCHES": "1", "DUCK_PATCH_F1_IMAGES": "0", "DUCK_PATCH_F2_RESULT": "0", "DUCK_PATCH_F3_MEMORY": "0",
           "DUCK_PATCH_M2_FACTS": "1"},
}
EXP_ID = {"1a": "EXP-004", "1b": "EXP-005"}


def _load_hydra0():
    spec = importlib.util.spec_from_file_location("build_hydra0", REPO / "automation" / "hydra0" / "build_hydra0.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _replace_once(src: str, old: str, new: str, where: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[{where}] expected exactly 1 occurrence, found {n}: {old[:80]!r}")
    return src.replace(old, new)


def build(variant: str, kv_gib: float, out_dir: Path) -> Path:
    h0 = _load_hydra0()
    with tempfile.TemporaryDirectory() as tmp:
        h0.build(kv_gib, Path(tmp))
        nb = json.loads((Path(tmp) / "arc3-hydra0.ipynb").read_text())
    cells = nb["cells"]
    src = lambda i: "".join(cells[i]["source"])  # noqa: E731

    def set_src(i: int, text: str) -> None:
        cells[i]["source"] = text.splitlines(keepends=True)

    # ---- cell 3: Duck patch switches (explicit assignments, after Hydra-0's setdefaults)
    env = VARIANTS[variant]
    block = (f"# HYDRA-1{variant} ({EXP_ID[variant]}): our Duck bundle's [DUCK-PATCH] switches. AGENTFIX supplies F1/F3, so\n"
             "# our duplicate F1/F2/F3 stay off; S1 + v9 prompt (P1-P3) on" + (" + M2 facts memory" if variant == "1b" else "") + ".\n"
             + "".join(f'os.environ["{k}"] = "{v}"\n' for k, v in env.items())
             + 'os.environ["HYDRA_SCORING"] = "0"   # v9 P1 (true scoring rule + game clock) supersedes the Hydra-0 lines\n')
    anchor = 'os.environ.setdefault("AGENTFIX_VALIDATION_RUNTIME_S", "600")   # commit runs are ~10 min smoke runs\n'
    set_src(3, _replace_once(src(3), anchor, anchor + block, "C3"))

    # ---- cell 7: attach our bundle; assert it is ours
    c7 = _replace_once(src(7), f'"{STOCK_BUNDLE}"', f'"{BUNDLE_DATASET}"', "C7-sources")
    c7 = _replace_once(
        c7, 'BUNDLE_DIR = _find_bundle_dir()\n',
        'BUNDLE_DIR = _find_bundle_dir()\n'
        '# HYDRA-1: the attached bundle must be ours (a stale/stock bundle would silently drop every DUCK-PATCH).\n'
        '_DUCK_BUNDLE_VERSION = BUNDLE_DIR / "DUCK_BUNDLE_VERSION.json"\n'
        'if not _DUCK_BUNDLE_VERSION.is_file():\n'
        f'    raise RuntimeError("expected {BUNDLE_DATASET} (DUCK_BUNDLE_VERSION.json missing) at " + str(BUNDLE_DIR))\n'
        'print("HYDRA1_BUNDLE", _DUCK_BUNDLE_VERSION.read_text().replace(chr(10), " "), flush=True)\n',
        "C7-assert")
    set_src(7, c7)

    # ---- cell 0: header
    md0 = "".join(cells[0]["source"])
    cells[0]["source"] = (f"## Hydra-1{variant} ({EXP_ID[variant]})\n\nBuilt by `automation/hydra1/build_hydra1.py`: Hydra-0 "
                          f"(`automation/hydra0/`) running our Duck bundle `{BUNDLE_DATASET}` with S1 + v9 prompt"
                          + (" + M2 facts memory" if variant == "1b" else "") + ".\n\n" + md0).splitlines(keepends=True)

    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            code = "\n".join(l for l in "".join(cell["source"]).splitlines() if not l.startswith(("!", "%")))
            compile(code, f"cell{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"arc3-hydra{variant}"
    code_file = f"{slug}.ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": f"kragglenote2forwork/{slug}",
        "title": f"ARC3 Hydra{variant}",  # slug(title) == slug (MISTAKES #1)
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [BUNDLE_DATASET, "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
        "docker_image": h0.DOCKER_IMAGE,
        "machine_shape": "NvidiaRtxPro6000",
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--kv-gib", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = build(a.variant, a.kv_gib, Path(a.out) if a.out else REPO / "notebooks" / f"hydra{a.variant}")
    print("built", out)


if __name__ == "__main__":
    main()
