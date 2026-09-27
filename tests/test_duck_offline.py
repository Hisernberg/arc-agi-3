"""B01: the vendored Duck harness runs end-to-end offline on CPU against a scripted mock LLM.

Run with the campaign venv (python 3.12, arc_agi 0.9.8, arcengine 0.9.3, scipy, imageio, pytest):
    $SCRATCH/venv/bin/python -m pytest tests/test_duck_offline.py -q

Covers: 3 games x 20 actions through the real TAAF Benchmark -> HarnessSolver -> ToolAgent loop,
per-game state isolation (plans/MISTAKES.md #2), score.json via inference.tools.eval, no network, the
source patches F1/F2/F3/S1 and their kill switches, and in-code config == the shipped pickles.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent"))

pytest.importorskip("arcengine", reason="needs the competition runtime (arcengine 0.9.3)")
pytest.importorskip("arc_agi", reason="needs the competition runtime (arc_agi 0.9.8)")

import duck_offline  # noqa: E402  (sets the shipped analyzer env before the Duck modules load)

ENV_DIR = duck_offline.default_env_dir()
if not Path(ENV_DIR).is_dir():
    pytest.skip(f"offline environment_files not found at {ENV_DIR} (set ARC_ENV_DIR)", allow_module_level=True)

from inference.agent import tool_agent as ta  # noqa: E402
from inference.agent.runtime_state import game_key_from_state_path, parse_state_path  # noqa: E402

GAMES = ("ls20", "vc33", "ar25")
STEPS = 20


def _fp(text: str) -> int:  # same fingerprint the scripted sandbox code prints
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 2305843009213693951
    return value


def _initial_frame_fp(game_id: str) -> int:
    from arcenv import ArcEnv
    from inference.utils.grid_utils import format_grid_ascii

    return _fp(format_grid_ascii(ArcEnv(game_id.split("-")[0], env_dir=ENV_DIR).frame.tolist()))


# ------------------------------------------------------------------------------------ unit tests


def test_state_key_parses_the_production_state_path():
    # the exact path HarnessSolver._play_one writes (MISTAKES #2: v4-v7 resolved this to "unknown")
    path = Path("/kaggle/working/artifacts/ls20-9607627b_p0_tool_runtime_state.json")
    key = parse_state_path(path)
    assert key is not None
    assert (key.game_id, key.pass_index, key.run_stem) == ("ls20-9607627b", 0, "ls20-9607627b_p0")
    assert game_key_from_state_path("/x/artifacts/my_game-1_p12_tool_runtime_state.json") == "my_game-1"
    ids = {game_key_from_state_path(f"/w/artifacts/{g}_p0_tool_runtime_state.json") for g in duck_offline.PUBLIC_GAME_IDS}
    assert ids == set(duck_offline.PUBLIC_GAME_IDS)
    assert parse_state_path("/w/artifacts/tool_runtime_state.json") is None
    assert parse_state_path("/w/artifacts/ls20_p0_tool_runtime_state.json.tmp") is None


def _agent():
    return ta.ToolAgent(model="local", base_url="http://127.0.0.1:9/v1")


def test_session_is_isolated_per_state_file(tmp_path, monkeypatch):
    art = tmp_path / "artifacts"
    art.mkdir()
    a, b = art / "ls20-9607627b_p0_tool_runtime_state.json", art / "vc33-5430563c_p0_tool_runtime_state.json"
    agent = _agent()
    agent._ensure_session(a)
    assert agent.game_key == "ls20-9607627b"
    agent._history_messages = [{"role": "user", "content": "ls20 history"}]
    agent._summarized_knowledge["world_model"] = "ls20 facts"
    agent._ensure_session(b)
    assert agent.game_key == "vc33-5430563c"
    assert agent._history_messages == [] and agent._summarized_knowledge["world_model"] == ""

    # kill switch restores the stock behaviour: the session is the shared artifacts/ directory
    monkeypatch.setenv("DUCK_PATCH_S1_STATE_KEY", "0")
    stock = _agent()
    stock._ensure_session(a)
    stock._history_messages = [{"role": "user", "content": "ls20 history"}]
    stock._ensure_session(b)
    assert stock._history_messages == [{"role": "user", "content": "ls20 history"}]  # the leak S1 removes


def test_prompt_log_is_per_game_even_when_alone(tmp_path, monkeypatch):
    art = tmp_path / "artifacts"
    art.mkdir()
    state = art / "ls20-9607627b_p0_tool_runtime_state.json"
    state.write_text("{}")
    assert ta._resolve_prompt_log_path(state) == tmp_path / "prompts" / "ls20-9607627b_p0.log"
    monkeypatch.setenv("DUCK_PATCHES", "0")
    assert ta._resolve_prompt_log_path(state) == tmp_path / "prompts" / "prompt.log"  # stock: shared file


def test_f3_tolerant_labels_and_switch(monkeypatch):
    text = "World model (revised): walls block\n- Plan (next): go left\nGoal model: reach door\n"
    labels = ["World model", "Goal model", "Plan"]
    assert ta._extract_labeled_blocks(text, labels) == {
        "World model": "walls block", "Plan": "go left", "Goal model": "reach door"}
    assert ta._extract_labeled_blocks("Plan for the next move: left", ["Plan"]) == {}
    monkeypatch.setenv("DUCK_PATCH_F3_MEMORY", "0")
    assert ta._extract_labeled_blocks(text, labels) == {"Goal model": "reach door"}


def test_f3_game_over_keeps_world_model(monkeypatch):
    agent = _agent()
    agent._summarized_knowledge["world_model"] = "keep me"
    agent._last_step_summary = {"game_over": True}
    agent._update_summarized_knowledge_from_step_summary()
    assert agent._summarized_knowledge["world_model"] == "keep me"
    agent._last_step_summary = {"level_transition": True}
    agent._update_summarized_knowledge_from_step_summary()
    assert agent._summarized_knowledge["world_model"] == ""


def test_f1_strips_all_but_newest_history_images(monkeypatch):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 3000}}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": f"turn {i}\n\nCurrent grid image:"}, image]}
        for i in range(4)
    ]
    kept = ta._strip_history_images(messages, keep=2)
    assert [isinstance(m["content"], list) for m in kept] == [False, False, True, True]
    assert kept[0]["content"] == "turn 0"
    flat = ta._estimate_tokens(messages)
    monkeypatch.setenv("DUCK_PATCH_F1_IMAGES", "0")
    assert ta._estimate_tokens(messages) > 2 * flat  # stock charges len(base64)//3 per image


def test_in_code_config_matches_shipped_pickles():
    duck = REPO / "agent" / "duck"
    bm = pickle.loads((duck / "benchmark_initial.pkl").read_bytes())
    for key, value in duck_offline.SHIPPED_SOLVER_FIELDS.items():
        assert getattr(bm.solver, key) == value, key
    for key, value in duck_offline.SHIPPED_BENCHMARK_FIELDS.items():
        assert getattr(bm, key) == value, key
    assert [g.env_name for g in bm.games] == list(duck_offline.PUBLIC_GAME_IDS)
    assert type(duck_offline.shipped_solver()) is type(bm.solver)
    target = pickle.loads((duck / "deploy_target.pkl").read_bytes())
    for key, value in duck_offline.SHIPPED_TARGET_FIELDS.items():
        assert getattr(target, key) == value, key


# ------------------------------------------------------------------------------ end-to-end runs


@pytest.fixture(scope="module")
def patched_run(tmp_path_factory):
    return duck_offline.run_offline(
        GAMES, steps=STEPS, out_dir=tmp_path_factory.mktemp("duck_patched"), profile="patched", quiet=True)


def test_three_games_run_twenty_steps_through_the_real_loop(patched_run):
    r = patched_run
    assert len(r.game_ids) == 3 and not r.mock_errors
    for game_id, info in r.games.items():
        assert info["state"] == "gave_up", (game_id, info)  # cap reached cleanly, not crashed
        assert info["actions"] >= STEPS, (game_id, info)
        assert str(info["solver_note"]).startswith("tokens="), info
    per_game = {g: [rec for rec in r.records if rec.game == g] for g in r.game_ids}
    for game_id, records in per_game.items():
        assert {rec.scenario for rec in records} >= set(duck_offline.OPENING_SCRIPT), game_id
    bench = json.loads((r.job_dir / "benchmark.json").read_text())
    assert sorted(run["game_id"] for run in bench["game_runs"]) == sorted(r.game_ids)
    assert all(len(run["history"]) >= STEPS for run in bench["game_runs"])


def test_score_file_is_produced(patched_run):
    score = json.loads(patched_run.score_path.read_text())
    assert patched_run.score_path == patched_run.job_dir / "score.json"
    assert set(score["games"]) == set(patched_run.game_ids)
    assert score["metadata"]["game_count"] == 3
    assert isinstance(score["score"], float)


def test_no_network(patched_run):
    attempts = patched_run.network_attempts
    assert attempts, "the mock server was never contacted"
    assert all(ok for _, _, ok in attempts), [a for a in attempts if not a[2]]
    assert len({port for _, port, _ in attempts}) == 1  # only the mock's port


def test_per_game_state_is_isolated(patched_run):
    r = patched_run
    # one analyzer per game, each bound to its own game id (the v7.1 bug: all shared "unknown")
    assert {g: a.game_key for g, a in r.agents.items()} == {g: g for g in r.game_ids}
    assert len({id(a) for a in r.agents.values()}) == 3
    # each game's sandbox saw that game's own board: the step-0 fingerprint printed inside the python
    # tool equals an independent arcenv reset of the same game, and the three differ
    seen = {}
    for game_id in r.game_ids:
        fps = {int(x) for v in r.tool_results.values() if v["game"] == game_id
               for x in re.findall(r"FP (\d+) step 0 level", v["content"])}
        assert fps == {_initial_frame_fp(game_id)}, game_id
        seen[game_id] = fps.pop()
    assert len(set(seen.values())) == 3
    # per-game artifacts, never a shared prompt log
    prompts = sorted(p.name for p in (r.job_dir / "prompts").iterdir())
    assert prompts == sorted(f"{g}_p0.log" for g in r.game_ids)
    for game_id in r.game_ids:
        transcript = (r.job_dir / "transcripts" / f"{game_id}_p0.txt").read_text()
        urls = set(re.findall(r"base_url: \S+/g/([^/]+)/v1", transcript))
        assert urls == {game_id}
        assert f"WM-MARKER-{game_id}" in transcript
        for other in set(r.game_ids) - {game_id}:
            assert f"WM-MARKER-{other}" not in transcript


def test_patches_visible_to_the_model(patched_run):
    r = patched_run
    acting = [v for v in r.tool_results.values() if "ACTED True" in v["content"]]
    assert acting and all('"board_changed"' in v["content"] for v in acting)  # F2
    for game_id in r.game_ids:  # F3: "World model (revised):" was parsed and carried into later prompts
        transcript = (r.job_dir / "transcripts" / f"{game_id}_p0.txt").read_text()
        assert f"- World model: WM-MARKER-{game_id}" in transcript
    assert max(rec.n_images for rec in r.records) <= 3  # F1: 2 history images + the current board
    ts = r.token_summary
    assert ts["requests"] == len(r.records) > 0
    assert ts["prompt_tokens_per_request"]["median"] > ts["section_tokens_mean"]["system"]


def test_stock_profile_restores_shipped_bundle_behaviour(tmp_path):
    r = duck_offline.run_offline(GAMES[:2], steps=8, out_dir=tmp_path / "stock", profile="stock", quiet=True)
    assert all(info["actions"] >= 8 and info["state"] == "gave_up" for info in r.games.values())
    acting = [v for v in r.tool_results.values() if "ACTED True" in v["content"]]
    assert acting and not any('"board_changed"' in v["content"] for v in acting)  # no F2
    transcript = (r.job_dir / "transcripts" / f"{r.game_ids[0]}_p0.txt").read_text()
    assert f"- World model: WM-MARKER-{r.game_ids[0]}" not in transcript  # no F3
    assert max(rec.n_images for rec in r.records) > 3  # no F1: every history board is resent
