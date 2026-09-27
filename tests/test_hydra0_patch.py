"""Hydra-0 runtime patch (automation/hydra0/hydra_patch.py) on the vendored Duck loop with the mock LLM.

    $SCRATCH/venv/bin/python -m pytest tests/test_hydra0_patch.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent"))

pytest.importorskip("arcengine")
pytest.importorskip("arc_agi")

import duck_offline  # noqa: E402

if not Path(duck_offline.default_env_dir()).is_dir():
    pytest.skip("offline environment_files not found", allow_module_level=True)

from inference.agent import tool_agent as ta  # noqa: E402
from inference.framework import solver as solver_mod  # noqa: E402


def _load_patch():
    spec = importlib.util.spec_from_file_location("hydra_patch", REPO / "automation" / "hydra0" / "hydra_patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prompt_text(job_dir: Path) -> str:
    texts = [p.read_text(errors="ignore") for p in job_dir.rglob("*.log")]
    texts += [p.read_text(errors="ignore") for p in job_dir.rglob("*.txt")]
    return "\n".join(texts)


def test_scoring_and_pacing_lines_reach_the_prompt(tmp_path, monkeypatch):
    patch = _load_patch()
    orig_bup = ta.ToolAgent._build_user_prompt
    orig_rts = solver_mod._HarnessGameSession.request_timeout_seconds
    try:
        report = patch.install(ta, solver_mod)
        assert report.get("installed"), report
        res = duck_offline.run_offline(("ls20", "ft09"), steps=6, out_dir=tmp_path / "run", quiet=True,
                                       max_runtime_s_per_game=900.0)
        text = _prompt_text(res.job_dir)
        assert "Scoring: a level you finish scores min(1.15" in text
        assert "Time left for this game:" in text and "of 15 min" in text
        assert all(g["state"] in ("gave_up", "won", "cancelled") for g in res.games.values()), res.games
    finally:
        ta.ToolAgent._build_user_prompt = orig_bup
        solver_mod._HarnessGameSession.request_timeout_seconds = orig_rts


def test_switch_off_is_a_noop(monkeypatch):
    patch = _load_patch()
    monkeypatch.setenv("HYDRA_SCORING", "0")
    orig = ta.ToolAgent._build_user_prompt
    report = patch.install(ta, solver_mod)
    assert not report.get("installed")
    assert ta.ToolAgent._build_user_prompt is orig
