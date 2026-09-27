"""The built Hydra-0 notebook (notebooks/hydra0/arc3-hydra0.ipynb): run its real cells where CPU allows.

    python automation/hydra0/build_hydra0.py && $SCRATCH/venv/bin/python -m pytest tests/test_hydra0_notebook.py -q
"""
from __future__ import annotations

import json
import math
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "hydra0" / "arc3-hydra0.ipynb"
sys.path.insert(0, str(REPO / "agent"))

if not NB.exists():
    pytest.skip("build the notebook first: python automation/hydra0/build_hydra0.py", allow_module_level=True)

CELLS = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"]]


def _cell_with(marker: str) -> str:
    hits = [c for c in CELLS if marker in c]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


def test_metadata_title_matches_slug():
    meta = json.loads((NB.parent / "kernel-metadata.json").read_text())
    slug = meta["id"].split("/", 1)[1]
    assert slug == meta["title"].lower().replace(" ", "-")          # MISTAKES #1
    assert meta["machine_shape"] == "NvidiaRtxPro6000" and meta["enable_internet"] is False
    assert "arc-prize-2026-arc-agi-3" in meta["competition_sources"]


def test_profile_and_switches():
    c3 = _cell_with("PUBLIC25_VLLM_PROFILE_ENV")
    assert '"TAAF_VLLM_MTP_TOKENS": "0"' in c3 and "16384" not in c3
    assert '"TAAF_VLLM_MAX_NUM_SEQS": "28"' in c3
    assert "AGENTFIX_ACTION7', '1'" in _cell_with("AGENTFIX_ACTION7")
    assert "bm.solver.analyzer_timeout = 1200.0" in _cell_with("analyzer_timeout")


def test_wavefit_block_fits_all_waves():
    c = _cell_with("HYDRA0_WAVEFIT")
    block = c[c.index("# HYDRA-0 wave-fit"):c.index("# Start recovery only after")]
    for n_games, remaining in [(110, 28_000), (110, 31_000), (55, 30_000), (3, 540)]:
        solver = types.SimpleNamespace(concurrency=28, max_runtime_s_per_game=7920.0)
        ns = {"bm": types.SimpleNamespace(games=list(range(n_games)), solver=solver), "datetime": datetime,
              "os": __import__("os"), "soft_end": datetime.now() + timedelta(seconds=remaining)}
        exec(block, ns)
        per_game = solver.max_runtime_s_per_game
        waves = math.ceil(n_games / 28)
        if remaining > 2000:
            assert waves * per_game <= remaining, (n_games, remaining, per_game)
            assert per_game >= 0.9 * (remaining - 240) / waves
    # the stock 7920 s cap would NOT fit 110 games in 28k s:
    assert 4 * 7920 > 28_000


def test_patch_cell_runs_against_vendored_duck(tmp_path):
    pytest.importorskip("arcengine")
    import duck_offline
    if not Path(duck_offline.default_env_dir()).is_dir():
        pytest.skip("offline env files missing")
    from inference.agent import tool_agent as ta
    from inference.framework import solver as solv
    orig = (ta.ToolAgent._build_user_prompt, solv._HarnessGameSession.request_timeout_seconds)
    try:
        ns: dict = {}
        exec(_cell_with("HYDRA0_PATCH"), ns)
        assert ns["_hy_report"]["installed"]
        assert "install" not in ns or ns["install"] is not None  # patch lives in its own module namespace
        assert "_ON" not in ns                                     # AGENTFIX's global _ON untouched
        res = duck_offline.run_offline(("ls20", "vc33"), steps=5, out_dir=tmp_path / "run", quiet=True,
                                       max_runtime_s_per_game=600.0)
        text = "\n".join(p.read_text(errors="ignore") for p in res.job_dir.rglob("*.log"))
        assert "Scoring: a level you finish" in text and "Time left for this game:" in text
    finally:
        ta.ToolAgent._build_user_prompt, solv._HarnessGameSession.request_timeout_seconds = orig
