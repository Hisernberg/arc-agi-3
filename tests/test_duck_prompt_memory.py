"""B06 (memory/compaction) + B07 (prompt v9): the [DUCK-PATCH P1/P2/P3/M1/M2] source patches.

Run with the campaign venv:
    $SCRATCH/venv/bin/python -m pytest tests/test_duck_prompt_memory.py -q

Everything runs through the real TAAF -> HarnessSolver -> ToolAgent loop against the scripted mock
LLM of agent/duck_offline.py, whose replies carry `reasoning_content` (4,000 chars). Covers:
- P2: ACTION7 is executable (as UNDO) on every public game that advertises it (MISTAKES #11);
- P1: the scoring rule is in the system prompt and a game-clock line is in every turn prompt;
- P1+P3: system prompt growth <= +300 tokens;
- M2: the verified-facts block is in every turn prompt, capped, never drops harness-verified
  facts, and the lowered trimming budget holds for every request;
- M1: carried history keeps the reasoning of exactly the newest K assistant messages;
- per-game isolation of the new state (S1 key);
- every v9 switch off == the pre-v9 requests byte-for-byte (tests/golden, captured before B06/B07).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
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

from inference.agent import action_names, patches  # noqa: E402
from inference.agent import tool_agent as ta  # noqa: E402
from inference.agent.facts_memory import FactsLedger  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402

GOLDEN = json.loads((REPO / "tests" / "golden" / "duck_requests_pre_v9.json").read_text())
PRE_V9_SYSTEM_PROMPT_SHA = "d091a36db1ccf79e5caa35b2cd31949370ef95810d00d713cc98eb27a3cd60a6"  # 12,681 chars
UNDO_GAMES = ("ar25", "bp35", "lf52", "sb26", "sk48", "su15")  # the public games advertising ACTION7
GAMES = ("ls20", "vc33", "ar25")
FOLLOWUPS = ("You have not acted yet", "You did not call a tool")
V9_ENV = ("DUCK_PATCHES_V9", "DUCK_PATCH_P1_SCORING", "DUCK_PATCH_P2_UNDO", "DUCK_PATCH_P3_DISCIPLINE",
          "DUCK_PATCH_M1_THINK", "DUCK_PATCH_M2_FACTS", "DUCK_PATCH_M1_THINK_KEEP", "DUCK_PATCH_M2_BUDGET",
          "DUCK_PATCH_M2_FACTS_TOKENS")
_TIMING_RE = re.compile(r'(\\?"(?:run_elapsed_seconds|time_remaining_seconds)\\?"\s*:\s*)[-0-9.eE+]+')


# ------------------------------------------------------------------------------------ helpers


@contextlib.contextmanager
def _env(**overrides: str):
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run(games, steps, out_dir, profile, **env):
    """run_offline + every request payload the mock received, per game (deep copies)."""
    payloads: dict[str, list[dict]] = {}
    original = duck_offline.ScriptedPolicy.respond

    def respond(self, game, payload):
        payloads.setdefault(game, []).append(copy.deepcopy(payload))
        return original(self, game, payload)

    duck_offline.ScriptedPolicy.respond = respond
    try:
        with _env(**env):
            result = duck_offline.run_offline(games, steps=steps, out_dir=out_dir, profile=profile, quiet=True)
    finally:
        duck_offline.ScriptedPolicy.respond = original
    return result, payloads


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def _turn_prompt_index(messages: list[dict]) -> int:
    """Index of the current turn's prompt (the newest user message that is not a follow-up)."""
    return max(i for i, m in enumerate(messages) if m["role"] == "user" and not _text(m).startswith(FOLLOWUPS))


def _turn_starts(payloads: list[dict]) -> list[dict]:
    return [p for p in payloads if p["messages"][-1]["role"] == "user" and not _text(p["messages"][-1]).startswith(FOLLOWUPS)]


def _facts_block(text: str) -> str | None:
    match = re.search(r"Verified facts .*?(?=\nend of world model)", text, re.S)
    return match.group(0) if match else None


def _digest(payload: dict) -> str:
    normalized = _TIMING_RE.sub(r"\1<t>", json.dumps(payload, sort_keys=True, ensure_ascii=True))
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _harness_estimate(payload: dict) -> int:
    """The harness's own trimming estimate of a request (M1 mirrors `reasoning` into
    `reasoning_content` only on the wire, after trimming, so the mirror is not counted)."""
    messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in payload["messages"]]
    return ta._estimate_tokens({"messages": messages, "tools": payload["tools"], "tool_choice": payload.get("tool_choice")})


# ------------------------------------------------------------------------------------ unit tests


def test_switch_defaults_and_group_switch(monkeypatch):
    for key in V9_ENV + ("DUCK_PATCHES",):
        monkeypatch.delenv(key, raising=False)
    assert patches.report() == {
        "F1_IMAGES": True, "F2_RESULT": True, "F3_MEMORY": True, "S1_STATE_KEY": True,
        "P1_SCORING": True, "P2_UNDO": True, "P3_DISCIPLINE": True, "M1_THINK": False, "M2_FACTS": False,
    }
    monkeypatch.setenv("DUCK_PATCH_M2_FACTS", "1")
    assert patches.enabled("M2_FACTS")
    monkeypatch.setenv("DUCK_PATCHES_V9", "0")  # the whole v9 group off, B01 fixes untouched
    assert not any(patches.enabled(p) for p in patches.V9_PATCH_IDS)
    assert all(patches.enabled(p) for p in ("F1_IMAGES", "F2_RESULT", "F3_MEMORY", "S1_STATE_KEY"))
    monkeypatch.setenv("DUCK_PATCHES_V9", "1")
    monkeypatch.setenv("DUCK_PATCHES", "0")
    assert not any(patches.report().values())


def test_every_advertised_action_is_executable(monkeypatch):
    """MISTAKES #11: valid_actions must be a subset of what the harness can execute."""
    import arcengine
    from arcenv import ArcEnv, list_games

    advertised = {game: ArcEnv(game, env_dir=ENV_DIR).available_actions for game in list_games(ENV_DIR)}
    assert {g for g, acts in advertised.items() if 7 in acts} == set(UNDO_GAMES)
    for game, action_ids in advertised.items():
        for action_id in action_ids:
            engine = arcengine.GameAction.from_id(int(action_id)).name
            model = action_names.to_model_action(engine)
            assert action_names.to_engine_action(model) == engine, (game, engine, model)
    assert action_names.to_model_action("ACTION7") == "UNDO"
    assert action_names.to_engine_action("undo") == action_names.to_engine_action("ACTION7") == "ACTION7"
    monkeypatch.setenv("DUCK_PATCH_P2_UNDO", "0")  # stock: advertised as ACTION7 but not executable
    assert action_names.to_model_action("ACTION7") == "ACTION7"
    assert action_names.to_engine_action("ACTION7") is None


def test_system_prompt_v9_within_300_tokens(monkeypatch):
    for key in V9_ENV:
        monkeypatch.delenv(key, raising=False)
    new = ta._build_system_prompt(tool_output_tokens=1024)
    monkeypatch.setenv("DUCK_PATCHES_V9", "0")
    old = ta._build_system_prompt(tool_output_tokens=1024)
    assert hashlib.sha256(old.encode()).hexdigest() == PRE_V9_SYSTEM_PROMPT_SHA  # untouched when off
    assert len(new) - len(old) <= 300 * 3.5, (len(old), len(new))
    for needle in ("min(1.15, (human_actions / your_actions)^2)", "level k has weight k", "cost nothing",
                   "Depth beats efficiency", "game clock", "never RESET at a level's start",
                   "which action and effect cleared it", "known video games", "visible reply"):
        assert needle in new, needle
    assert "Optimize for as few in-game actions" in old and "Optimize for as few in-game actions" not in new
    monkeypatch.setenv("DUCK_PATCHES_V9", "1")
    monkeypatch.setenv("DUCK_PATCH_P1_SCORING", "0")
    assert "Scoring:" not in ta._build_system_prompt(tool_output_tokens=1024)
    monkeypatch.setenv("DUCK_PATCH_P3_DISCIPLINE", "0")
    assert ta._build_system_prompt(tool_output_tokens=1024) == old


def _history(levels: list[tuple[int, list[tuple[str, list[list[int]]]]]]):
    """Build runtime-state history: [(level, [(action, grid_after), ...]), ...]."""
    blank = [[0] * 64 for _ in range(64)]
    entries = [HistoryEntry(action="", frame=Frame(grid=tuple(map(tuple, blank)), step=0, level=1))]
    step = 0
    for level, actions in levels:
        for action, grid in actions:
            step += 1
            entries.append(HistoryEntry(action=action, frame=Frame(grid=tuple(map(tuple, grid)), step=step, level=level)))
    return entries


def _grid(**cells):
    grid = [[0] * 64 for _ in range(64)]
    for key, value in cells.items():
        r, c = map(int, key[1:].split("_"))
        grid[r][c] = value
    return grid


def test_facts_ledger_classifies_caps_and_never_drops_verified_facts():
    moved = _grid(r20_20=9, r20_21=9)
    hud = _grid(r20_20=9, r20_21=9, r63_5=3)
    colours = [[(r * 7 + c) % 16 for c in range(64)] for r in range(64)]  # every colour clickable
    level1 = [("UP", _grid(r20_20=9)), ("UP", moved), ("LEFT", moved), ("SPACE", hud)]
    level2 = [(f"MOUSE(row={r}, col={c})", colours if i % 2 else moved) for i, (r, c) in
              enumerate((r, c) for r in range(10, 30, 2) for c in range(10, 30, 3))]
    level2 += [("DOWN", colours), ("UNDO", moved), ("RESET", moved)]
    history = _history([(1, level1[:3]), (2, level1[3:]), (2, level2), (3, [("RIGHT", moved)])])
    ledger = FactsLedger("toy-game")
    ledger.note_action_results([{"executed": True, "action_num": len(history) - 3, "game_over": True}])
    assert ledger.ingest(history) == len(history) - 1
    assert ledger.ingest(history) == 0  # incremental: nothing new
    t1 = ledger.levels[1].tallies
    assert (t1["UP"].n, t1["UP"].changed, t1["LEFT"].no_change) == (2, 2, 1)
    assert ledger.levels[1].cleared_at == 4 and ledger.levels[1].tallies["SPACE"].level_up == 1
    status = {"levels_completed": 2, "number_of_levels": 5, "actions_per_level": [4, len(level2), 1],
              "baseline_actions": [3, 30, 9, 9, 9]}
    keys = set(t1) | set(ledger.levels[2].tallies) | set(ledger.levels[3].tallies)
    assert sum(k.startswith("MOUSE@") for k in keys) > 6  # forces MOUSE grouping in the rendering
    cap = 700 * 3
    lean = ledger.render(status=status, notes={}, max_chars=cap)
    huge = {"world_model": "W " * 3000, "current_plan": "P " * 3000, "goal_model": "G " * 3000,
            "cross_level_notes": "C " * 3000}
    ledger.archive_notes(1, {"world_model": "level one rule " * 50})
    full = ledger.render(status=status, notes=huge, max_chars=cap)
    for lines in (lean, full):
        text = "\n".join(lines)
        assert len(text) <= cap, len(text)
        for key in keys:  # every verified (level, action) key survives, alone or in a MOUSE@{..} group
            if key.startswith("MOUSE@"):
                assert re.search(r"MOUSE@(\{[^}]*" + re.escape(key[6:]) + r"|" + re.escape(key[6:]) + r" )", text), key
            else:
                assert re.search(r"(?<![A-Z])" + key + r" \d+x: ", text), key
        assert "Levels cleared: 2 of 5 (actions/human baseline: L1 4/3, L2 " in text
        assert "GAME_OVER 1" in text and "Cleared by: L1 at action 4" in text
    verified = [line for line in lean if not line.startswith("- Revise")]
    assert all(line in full for line in verified)  # long model notes never displace verified facts
    assert "Model notes" in "\n".join(full) and "L1 final notes" in "\n".join(full)


def test_session_switch_resets_status_and_facts(tmp_path):
    art = tmp_path / "artifacts"
    art.mkdir()
    a, b = art / "ls20-9607627b_p0_tool_runtime_state.json", art / "vc33-5430563c_p0_tool_runtime_state.json"
    agent = ta.ToolAgent(model="local", base_url="http://127.0.0.1:9/v1")
    agent.set_game_status(a, {"levels_completed": 3, "number_of_levels": 7})
    ledger = agent._facts_ledger()
    ledger.ingest(_history([(1, [("UP", _grid(r9_9=2))])]))
    assert agent.game_key == ledger.game_key == "ls20-9607627b" and ledger.total_actions == 1
    agent._ensure_session(b)  # next game: nothing carried over
    assert agent.game_key == "vc33-5430563c" and agent.facts is None and agent._game_status is None
    assert agent._facts_ledger().game_key == "vc33-5430563c" and agent._facts_ledger().total_actions == 0


def test_history_compaction_and_reasoning_strip_units():
    prompt = ("The code executed 1 action in the previous sequence.\nExecuted actions: UP.\nYou are still on the same level.\n"
              "Current state: step 5, level 1.\nGame clock: 1.0 min used.\nValid actions right now: UP.\n"
              "Only tool: `python`.\nVerified facts (harness-checked ...):\n- Levels cleared: 0.\nend of world model.")
    assert ta._compact_history_user_text(prompt) == (
        "The code executed 1 action in the previous sequence.\nExecuted actions: UP.\n"
        "You are still on the same level.\nCurrent state: step 5, level 1.")
    assert ta._compact_history_user_text("You have not acted yet. Investigate first. Then ...") == "You have not acted yet."
    msgs = [{"role": "assistant", "reasoning": f"r{i}", "content": None} for i in range(5)]
    kept = ta._strip_history_reasoning(msgs, keep=2)
    assert [bool(m.get("reasoning")) for m in kept] == [False, False, False, True, True]
    assert msgs[0]["reasoning"] == "r0"  # input not mutated


# ------------------------------------------------------------------------------ end-to-end runs


@pytest.fixture(scope="module")
def v9_run(tmp_path_factory):
    return _run(GAMES, 24, tmp_path_factory.mktemp("duck_v9"), "v9")


@pytest.fixture(scope="module")
def undo_run(tmp_path_factory):
    return _run(UNDO_GAMES, 14, tmp_path_factory.mktemp("duck_undo"), "patched")


def test_undo_executes_on_every_action7_game(undo_run, tmp_path):
    result, payloads = undo_run
    assert not result.mock_errors
    for game_id, info in result.games.items():
        assert info["state"] == "gave_up" and info["actions"] >= 14, (game_id, info)
        assert "ACTION7" in info["action_ids"], (game_id, info["action_ids"])  # UNDO reached the engine
        undo_results = [v["content"] for v in result.tool_results.values()
                        if v["game"] == game_id and v["scenario"] == "advertised_unmappable"]
        assert undo_results and all("ACTED True" in c for c in undo_results), (game_id, undo_results)
        assert not any(re.search(r"Unknown action[^\n]*(ACTION7|UNDO)", v["content"])  # (JUMP/ACTION9 probes are
                       for v in result.tool_results.values() if v["game"] == game_id)  # rejected on purpose)
        prompts = [_text(p["messages"][-1]) for p in _turn_starts(payloads[game_id])]
        assert prompts and all(re.search(r"^Valid actions right now: .*\bUNDO\b", t, re.M) for t in prompts)
        assert all("UNDO reverts your previous action and costs one action" in t for t in prompts)
    # before P2 the same step was rejected: ACTION7 advertised but "Unknown action"
    stock, _ = _run(("ar25",), 14, tmp_path / "pre", "pre_v9")
    rejected = [v["content"] for v in stock.tool_results.values() if v["scenario"] == "advertised_unmappable"]
    assert rejected and all("Unknown action" in c for c in rejected)
    assert "ACTION7" not in stock.games[stock.game_ids[0]]["action_ids"]


def test_scoring_rule_and_game_clock_every_turn(v9_run):
    result, payloads = v9_run
    clock_re = re.compile(r"^Game clock: ([\d.]+) min used, ([\d.]+) min left \((\d+)% of this game's time\)\. "
                          r"Levels cleared: \d+ of \d+\.$", re.M)
    for game_id in result.game_ids:
        used = []
        for payload in payloads[game_id]:
            assert "min(1.15, (human_actions / your_actions)^2)" in payload["messages"][0]["content"]
            match = clock_re.search(_text(payload["messages"][_turn_prompt_index(payload["messages"])]))
            assert match, game_id
            used.append(float(match.group(1)))
        assert used == sorted(used) and len(used) >= 20


def test_facts_block_every_turn_and_capped(v9_run):
    result, payloads = v9_run
    cap = 700 * 3  # DUCK_PATCH_M2_FACTS_TOKENS default, enforced as chars = tokens * 3
    for game_id in result.game_ids:
        blocks = 0
        for payload in _turn_starts(payloads[game_id]):
            text = _text(payload["messages"][-1])
            block = _facts_block(text)
            assert block is not None, game_id
            assert len(block) <= cap
            step = int(re.search(r"Current state: step (\d+)", text).group(1))
            total = int(re.search(r"(\d+) in total\.", block).group(1))
            assert total == step - 1  # the ledger has every executed action, even after history trimming
            blocks += 1
        assert blocks >= 10
        assert f"WM-MARKER-{game_id}" in _text(_turn_starts(payloads[game_id])[-1]["messages"][-1])


def test_trimming_budget_respected_and_prompts_shrink(v9_run):
    result, payloads = v9_run
    budget = ta._M2_BUDGET_DEFAULT
    for game_id in result.game_ids:
        estimates = [_harness_estimate(p) for p in payloads[game_id]]
        assert max(estimates) <= budget, (game_id, max(estimates))
        assert max(estimates) > 0.8 * budget  # the budget is actually binding in steady state
        transcript = (result.job_dir / "transcripts" / f"{game_id}_p0.txt").read_text()
        assert set(re.findall(r"context_budget_tokens: (\d+)", transcript)) == {str(budget)}
        # old turn prompts are compacted: static instructions and old facts blocks are not carried
        for payload in payloads[game_id]:
            messages = payload["messages"]
            for message in messages[1:_turn_prompt_index(messages)]:
                if message["role"] == "user":
                    assert "Verified facts" not in _text(message) and "Only tool: `python`" not in _text(message)
    steady = [r.prompt_tokens for r in result.records if r.index >= 16]
    assert sorted(steady)[len(steady) // 2] <= 12_000  # chars/3.5 + images by pixels (duck_offline)


def test_reasoning_retained_for_last_k_turns(v9_run, tmp_path):
    def check(payloads, keep):
        seen_full = False
        for game_payloads in payloads.values():
            for payload in game_payloads:
                messages = payload["messages"]
                t = _turn_prompt_index(messages)
                carried = [bool(m.get("reasoning")) for m in messages[1:t] if m["role"] == "assistant"]
                in_flight = [m for m in messages[t:] if m["role"] == "assistant"]
                assert sum(carried) == min(keep, len(carried)), carried
                assert all(carried[len(carried) - sum(carried):])  # the retained ones are the newest
                seen_full = seen_full or sum(carried) == keep
                for message in messages:
                    if message["role"] == "assistant" and message.get("reasoning"):
                        assert message.get("reasoning_content") == message["reasoning"]  # template field
                assert all(m.get("reasoning") for m in in_flight)  # the in-flight turn keeps its own
                assert payload["chat_template_kwargs"]["preserve_thinking"] is True
        assert seen_full

    check(v9_run[1], ta._M1_THINK_KEEP_DEFAULT)
    _, payloads = _run(("ls20",), 12, tmp_path / "k1", "v9", DUCK_PATCH_M1_THINK_KEEP="1")
    check(payloads, 1)
    # M1 off (default profile): every carried turn keeps its reasoning, template default untouched
    _, payloads = _run(("ls20",), 12, tmp_path / "off", "patched")
    for payload in payloads["ls20-9607627b"]:
        carried = [m for m in payload["messages"][1:] if m["role"] == "assistant"]
        assert all(m.get("reasoning") and "reasoning_content" not in m for m in carried)
        assert "preserve_thinking" not in payload["chat_template_kwargs"]


def test_per_game_isolation_of_facts(v9_run):
    result, payloads = v9_run
    assert {g: a.game_key for g, a in result.agents.items()} == {g: g for g in result.game_ids}
    for game_id, agent in result.agents.items():
        assert agent.facts is not None and agent.facts.game_key == game_id
        # the ledger saw exactly this game's executed actions (read back from its own state file)
        assert agent.facts.total_actions <= result.games[game_id]["actions"]
        assert agent.facts.total_actions >= result.games[game_id]["actions"] - 3  # last batch not re-ingested
        for payload in payloads[game_id]:
            text = _text(payload["messages"][_turn_prompt_index(payload["messages"])])
            for other in set(result.game_ids) - {game_id}:
                assert f"WM-MARKER-{other}" not in text


def test_all_v9_switches_off_reproduce_pre_v9_requests(tmp_path):
    off = {"DUCK_PATCH_P1_SCORING": "0", "DUCK_PATCH_P2_UNDO": "0", "DUCK_PATCH_P3_DISCIPLINE": "0",
           "DUCK_PATCH_M1_THINK": "0", "DUCK_PATCH_M2_FACTS": "0"}
    runs = {
        "each switch off": ("patched", off, "patched"),
        "group switch off": ("pre_v9", {}, "patched"),
        "stock (DUCK_PATCHES=0)": ("stock", {}, "stock"),
    }
    for label, (profile, env, golden_key) in runs.items():
        spec = GOLDEN[golden_key]
        profile_env = dict(duck_offline.PROFILES[profile])
        original = duck_offline.PROFILES[profile]
        duck_offline.PROFILES[profile] = {**profile_env, **env}  # patch_profile applies these keys
        try:
            _, payloads = _run(tuple(spec["games"]), spec["steps"], tmp_path / profile / str(len(env)), profile)
        finally:
            duck_offline.PROFILES[profile] = original
        got = {game: [_digest(p) for p in reqs] for game, reqs in payloads.items()}
        assert got == spec["digests"], label
