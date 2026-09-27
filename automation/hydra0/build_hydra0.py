#!/usr/bin/env python3
"""Build the Hydra-0 Kaggle notebook (EXP-002) from the user's scored base notebook.

Base: taaf-flashnext-sheetu12b-0922 (code-identical to scottlegrand's 5.19 public notebook; research/10).
Every edit is an exact, asserted string replacement (fails loudly if the base changes). Changes:
  C3   serving profile: MTP off (frees ~7.5 GiB), KV 10 GiB (was 5), 28 seqs (= game concurrency);
       drop the ineffective LOCAL_ANALYZER_CONTEXT_WINDOW=16384 override (MISTAKES #10); smoke-mode defaults.
  C10  AGENTFIX_ACTION7 on: ACTION7 executable as UNDO (MISTAKES #11).
  NEW  Hydra patch cell: scoring rule + per-game pacing line in every user prompt (hydra_patch.py, fail-open).
  C14  analyzer_timeout 900 -> 1200.
  C16  wave-fit per-game budget from the measured remaining time (fixes the cut-off 4th wave);
       commit runs are SMOKE runs (3 public games, ~10 min) instead of a 2 h public-25 validation.

Usage:  python automation/hydra0/build_hydra0.py [--kv-gib 10] [--out notebooks/hydra0]
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = HERE / "base_taaf-flashnext-sheetu12b-0922.ipynb"
PATCH_SRC = HERE / "hydra_patch.py"

KERNEL_ID = "kragglenote2forwork/arc3-hydra0"
TITLE = "ARC3 Hydra0"  # slug(title) == "arc3-hydra0" (MISTAKES #1)
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")  # last green ARC run (09-26 v7)


def _replace_once(src: str, old: str, new: str, where: str) -> str:
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[{where}] expected exactly 1 occurrence, found {n}: {old[:80]!r}")
    return src.replace(old, new)


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def build(kv_gib: float, out_dir: Path) -> Path:
    nb = json.loads(BASE.read_text())
    cells = nb["cells"]
    src = lambda i: "".join(cells[i]["source"])  # noqa: E731

    # ---- C3: serving profile + env defaults
    c3 = src(3)
    old_profile = c3[c3.index("PUBLIC25_VLLM_PROFILE_NAME = "):c3.index("for key, value in PUBLIC25_VLLM_PROFILE_ENV.items():")]
    new_profile = f'''PUBLIC25_VLLM_PROFILE_NAME = 'hydra0-kv{kv_gib:g}-bf16-nomtp-s28-cg32-ctx32k'
PUBLIC25_VLLM_PROFILE_ENV = {{
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "{int(kv_gib * 1024**3)}",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "8192",
    "TAAF_VLLM_MAX_NUM_SEQS": "28",
    "TAAF_VLLM_MTP_TOKENS": "0",
    "TAAF_VLLM_OMP_THREADS": "1"
}}
# HYDRA-0 (EXP-002): MTP off frees ~7.5 GiB of weights (81.8 -> 74.3 GiB); spend it on KV ({kv_gib:g} GiB vs 5).
# Thuitanium measured MTP off + 7 GiB KV + 28 seqs = 722 vs 359 tok/s. Total footprint stays <= the proven
# 81.8 + 5 GiB. The analyzer context stays 32k (serving_setup.py pins it; the old 16k override never applied).
os.environ.setdefault("HYDRA_SCORING", "1")
os.environ.setdefault("HYDRA_WAVEFIT", "1")
os.environ.setdefault("AGENTFIX_VALIDATION_RUNTIME_S", "600")   # commit runs are ~10 min smoke runs
'''
    c3 = c3.replace(old_profile, new_profile)
    old_ctx = c3[c3.index("# Honest serving-only test (09-06)"):c3.index('os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "16384"\n') + len('os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "16384"\n')]
    c3 = c3.replace(old_ctx, "")
    assert "16384" not in c3 and '"TAAF_VLLM_MTP_TOKENS": "0"' in c3
    _set_source(cells[3], c3)

    # ---- C10: ACTION7 executable
    c10 = _replace_once(src(10), "_os.environ.setdefault('AGENTFIX_ACTION7', '0')",
                        "_os.environ.setdefault('AGENTFIX_ACTION7', '1')   # HYDRA-0: UNDO executable", "C10")
    _set_source(cells[10], c10)

    # ---- NEW cell after 10: Hydra patch (scoring + pacing), source embedded
    patch_code = PATCH_SRC.read_text()
    hydra_cell = ("# HYDRA-0 patch (automation/hydra0/hydra_patch.py, embedded verbatim and executed in its own\n"
                  "# module namespace so it cannot clobber AGENTFIX globals such as `install` / `_ON`).\n"
                  "import os, types as _hy_types\n"
                  f"_HYDRA_SRC = {patch_code!r}\n"
                  "_hy_mod = _hy_types.ModuleType('hydra_patch')\n"
                  "exec(compile(_HYDRA_SRC, 'hydra_patch.py', 'exec'), _hy_mod.__dict__)\n"
                  "import inference.agent.tool_agent as _hy_ta, inference.framework.solver as _hy_solv\n"
                  "_hy_report = _hy_mod.install(_hy_ta, _hy_solv)\n"
                  "print('HYDRA0_PATCH', _hy_report, flush=True)\n"
                  "if os.environ.get('HYDRA_SCORING', '1') != '0':\n"
                  "    assert _hy_report.get('installed'), _hy_report\n")
    cells.insert(11, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": hydra_cell.splitlines(keepends=True)})
    # indices after 11 shift by +1
    i14, i16 = 15, 17

    # ---- C14: analyzer timeout
    c14 = _replace_once(src(i14), "bm.solver.analyzer_timeout = 900.0", "bm.solver.analyzer_timeout = 1200.0", "C14")
    _set_source(cells[i14], c14)

    # ---- C16: smoke selection, wave-fit, relaxed coverage audit
    c16 = src(i16)
    c16 = _replace_once(
        c16,
        "    print(f'PUBLIC25_SELECTION games={len(bm.games)} passes=1', flush=True)\n",
        "    print(f'PUBLIC25_SELECTION games={len(bm.games)} passes=1', flush=True)\n"
        "    # HYDRA-0: commit runs are smoke runs (pipeline sanity; local scores barely predict the LB, r=0.16).\n"
        "    _smoke = [g.strip() for g in os.environ.get('HYDRA_SMOKE_GAMES',\n"
        "              'ft09-0d8bbf25,ls20-9607627b,vc33-5430563c').split(',') if g.strip()]\n"
        "    if _smoke:\n"
        "        bm.games = [offline_by_id[g] for g in _smoke]\n"
        "        print(f'HYDRA0_SMOKE games={_smoke}', flush=True)\n",
        "C16-smoke")
    c16 = _replace_once(
        c16,
        "# Start recovery only after setup readiness and all run gates pass.\n",
        "# HYDRA-0 wave-fit: size the per-game budget so every wave finishes before soft_end (stock 7920 s x 4 waves\n"
        "# of 110 games overruns the remaining time and cuts the last wave short).\n"
        "import math as _math\n"
        "_hy_n = len(bm.games)\n"
        "_hy_slots = max(1, int(bm.solver.concurrency or 1))\n"
        "_hy_waves = max(1, _math.ceil(_hy_n / _hy_slots))\n"
        "_hy_remaining = (soft_end - datetime.now()).total_seconds() - float(os.environ.get('HYDRA_WAVEFIT_MARGIN_S', '240'))\n"
        "if os.environ.get('HYDRA_WAVEFIT', '1') != '0':\n"
        "    bm.solver.max_runtime_s_per_game = float(max(600.0, _hy_remaining / _hy_waves))\n"
        "print(f'HYDRA0_WAVEFIT games={_hy_n} slots={_hy_slots} waves={_hy_waves} remaining_s={_hy_remaining:.0f} '\n"
        "      f'per_game_s={bm.solver.max_runtime_s_per_game:.0f}', flush=True)\n\n"
        "# Start recovery only after setup readiness and all run gates pass.\n",
        "C16-wavefit")
    c16 = _replace_once(
        c16,
        "        if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):\n",
        "        if public_run_ids != [g.env_name for g in bm.games]:\n",
        "C16-audit")
    _set_source(cells[i16], c16)

    # ---- sanity: every code cell parses (top-level await allowed)
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            code = "".join(cell["source"])
            code = "\n".join(line for line in code.splitlines() if not line.startswith(("!", "%")))  # col-0 magics only
            compile(code, f"cell{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    # ---- write notebook + metadata
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    md0 = "".join(cells[0]["source"])
    cells[0]["source"] = ("## Hydra-0 (EXP-002)\n\nBuilt by `automation/hydra0/build_hydra0.py` from "
                          "taaf-flashnext-sheetu12b-0922. MTP off + 10 GiB KV + 28 seqs, UNDO executable, scoring + "
                          "pacing prompt lines, analyzer timeout 1200 s, wave-fit per-game budget, smoke-mode commits.\n\n"
                          + md0).splitlines(keepends=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = "arc3-hydra0.ipynb"
    (out_dir / code_file).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = {
        "id": KERNEL_ID,
        "title": TITLE,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
                            "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
        "docker_image": DOCKER_IMAGE,
        "machine_shape": "NvidiaRtxPro6000",
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-gib", type=float, default=10.0)
    ap.add_argument("--out", default=str(REPO / "notebooks" / "hydra0"))
    a = ap.parse_args()
    out = build(a.kv_gib, Path(a.out))
    print("built", out)


if __name__ == "__main__":
    main()
