#!/usr/bin/env python3
"""oracle_search.py - "oracle study" of the public ARC-AGI-3 levels (backlog B11, CPU only).

Question: where is cheap, *systematic* probing/search safe and valuable for an LLM agent?  Not a
per-game solver: everything the search sees is what an agent sees through the Kaggle gateway (64x64
frames + available_actions).  Only the *cloning* is an offline privilege; the report converts every
search into what it would cost with real actions (no clones on Kaggle).

Pipeline (all results go to $SCRATCH/oracle/*.jsonl):
  starts   replay the GPT-6 Astra recordings (current game versions, all 25 games are full WINs) in
           agent/arcenv.py, verify every recorded frame and every level completion, and store the action
           prefix that reaches each of the 183 level starts (+ Astra's per-level action counts).
  search   from each level start: breadth-first search with exact state cloning.  Actions = keyboard
           actions from available_actions (no RESET/undo) + ACTION6 restricted to one click per distinct
           object/cell (agent/perception.rank_click_candidates on a connected-component segmentation).
           Dedup key = frame hash with HUD/budget bars masked (agent/perception.HudTracker, calibrated on
           short random walks from the level start).  Time cap per level; multiprocessing over cores.
           Records: solved, min actions found, branching, depth, distinct states, no-op rate, and the
           real-action cost an online (clone-free) explorer would pay for the same search.
  random   structured random play (uniform over the same candidate actions, level RESET after
           GAME_OVER charged +1) - would random play plausibly solve the level?
  noop     fraction of actions in human / Astra traces that change nothing but the HUD.
  report   merge everything into tables + the probe-then-LLM score model (markdown on stdout).

Usage (python 3.12 venv with the competition wheels):
  python tools/oracle_search.py starts
  python tools/oracle_search.py search  [--cap 90] [--workers 4] [--games ls20,vc33] [--levels 1,2]
  python tools/oracle_search.py random  [--cap 20] [--workers 4]
  python tools/oracle_search.py noop
  python tools/oracle_search.py report  > table.md

Library use (e.g. a harness-side probe tool): ``bfs_level(env, cap_s)`` works on any ArcEnv positioned at
a level start; ``Keyer`` / ``candidate_actions`` are the frame-only state key and action set.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "agent"))

from arcenv import ArcEnv, _SCRATCH, level_score, list_games, score_from_level_actions  # noqa: E402
import perception as P  # noqa: E402

SCRATCH = os.environ.get("SCRATCH", _SCRATCH)
OUT = os.path.join(SCRATCH, "oracle")
ASTRA_DIR = os.path.join(SCRATCH, "web/replays/recordings_astra_pa_max")
HUMAN_GLOB = os.path.join(REPO, "*-*-*-*-*.json")  # arcprize.org replays saved at the repo root (jsonl)

AID = {"RESET": 0, "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6,
       "ACTION7": 7}


# ------------------------------------------------------------------------------------------------
# recordings
# ------------------------------------------------------------------------------------------------
def read_recording(path: str, frames: bool = True) -> list[dict]:
    """arcprize.org recording (jsonl, one env response per line) -> [{aid, x, y, state, lv, frame}]."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d = r.get("data", r)
            ai = d.get("action_input") or {}
            a = ai.get("id")
            aid = AID.get(a, a if isinstance(a, int) else None)
            dd = ai.get("data") or {}
            fr = None
            if frames and d.get("frame"):
                fr = np.asarray(d["frame"][-1], dtype=np.int8)
            out.append({"aid": aid, "x": dd.get("x"), "y": dd.get("y"), "state": d.get("state"),
                        "lv": d.get("levels_completed"), "game_id": d.get("game_id"), "frame": fr,
                        "nf": len(d.get("frame") or [])})
    return out


def astra_path(game: str) -> Optional[str]:
    ps = sorted(glob.glob(os.path.join(ASTRA_DIR, f"{game}-*.jsonl")))
    return ps[0] if ps else None


def act_tuple(r: dict) -> list:
    return [r["aid"], r["x"], r["y"]] if r["aid"] == 6 else [r["aid"]]


def step_env(env: ArcEnv, a) -> Any:
    return env.step(a[0], a[1], a[2]) if a[0] == 6 else env.step(a[0])


def replay_level_starts(game: str, path: str) -> dict:
    """Replay a recording in arcenv (plain online semantics: competition=False, like the recording) and
    return the action prefix reaching every level start, per-level action counts and a verification."""
    rec = read_recording(path)
    env = ArcEnv(game, competition=False)
    nlev = len(env.baseline)
    if rec and rec[0]["aid"] == 0:
        first, rest = rec[0], rec[1:]
    else:
        first, rest = None, rec
    mism = 0
    if first is not None and first["frame"] is not None and not np.array_equal(first["frame"], env.frame):
        mism += 1
    prefix: list = []
    starts = {0: []}
    lvl_actions = [0] * nlev  # actions charged to each level (RESETs included), scorecard style
    cur = 0
    for i, r in enumerate(rest):
        a = act_tuple(r)
        o = step_env(env, a)
        prefix.append(a)
        if o.ignored:
            continue
        if r["frame"] is not None and o.frame is not None and not np.array_equal(r["frame"], o.frame):
            mism += 1
        lc = o.levels_completed
        if a[0] == 0 and lc < cur:  # full reset back to level 1: a new "play" (the scorecard keeps the best
            cur = lc                  # play); keep the level starts found so far, restart the per-level counts
            lvl_actions = [0] * nlev
            continue
        if cur < nlev:
            lvl_actions[cur] += 1
        if lc > cur:
            for k in range(cur + 1, lc + 1):
                if k < nlev and k not in starts:
                    starts[k] = list(prefix)
            cur = lc
    # verify: from each start, the recorded continuation completes that level
    verified = {}
    for k, pre in starts.items():
        e = ArcEnv(game, competition=False)
        for a in pre:
            step_env(e, a)
        verified[k] = e.levels_completed == k
    return {"game": game, "source": os.path.basename(path), "n_levels": nlev, "final_levels": env.levels_completed,
            "final_state": env.state, "frame_mismatches": mism, "n_steps": len(rest),
            "starts": {str(k): v for k, v in starts.items()}, "start_ok": {str(k): v for k, v in verified.items()},
            "level_actions": lvl_actions, "baseline": env.baseline}


def cmd_starts(args) -> None:
    os.makedirs(OUT, exist_ok=True)
    games = args.games.split(",") if args.games else list_games()
    out_p = os.path.join(OUT, "level_starts.jsonl")
    rows = []
    for g in games:
        p = astra_path(g)
        if p is None:
            print(f"{g}: no Astra recording", flush=True)
            continue
        t0 = time.time()
        r = replay_level_starts(g, p)
        rows.append(r)
        ok = sum(r["start_ok"].values())
        print(f"{g}: levels {r['final_levels']}/{r['n_levels']} state={r['final_state']} starts_ok={ok}/{r['n_levels']} "
              f"frame_mismatches={r['frame_mismatches']}/{r['n_steps']} astra_actions={r['level_actions']} "
              f"({time.time() - t0:.1f}s)", flush=True)
    with open(out_p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", out_p)


def load_starts() -> dict:
    p = os.path.join(OUT, "level_starts.jsonl")
    return {r["game"]: r for r in map(json.loads, open(p))}


def env_at_level(game: str, level: int, starts: dict, competition: bool = True) -> ArcEnv:
    """Fresh env positioned at the start of `level` (0-based) by replaying the stored prefix.  The
    replay runs in plain mode (as recorded); the returned env is then switched to competition-mode
    RESET semantics (level RESET, never a full restart)."""
    env = ArcEnv(game, competition=False)
    for a in starts[game]["starts"][str(level)]:
        step_env(env, a)
    assert env.levels_completed == level, (game, level, env.levels_completed)
    env.competition = competition
    return env
