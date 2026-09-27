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


# ------------------------------------------------------------------------------------------------
# agent-realistic state key + action set (frame-only; uses the harness perception pack)
# ------------------------------------------------------------------------------------------------
def rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except Exception:  # pragma: no cover
        return 0.0


class Keyer:
    """State key = blake2b of the settled frame with the HUD mask blanked (what an agent can hash)."""

    def __init__(self, mask: Optional[np.ndarray] = None):
        self.mask = mask if mask is not None and mask.any() else None

    def key(self, frame: np.ndarray) -> bytes:
        g = np.asarray(frame, dtype=np.int8)
        if self.mask is not None:
            g = g.copy()
            g[self.mask] = -1
        return hashlib.blake2b(g.tobytes(), digest_size=12).digest()


class ActionSet:
    """Keyboard actions (available_actions minus RESET/undo) + one ACTION6 per distinct object/cell
    (perception.rank_click_candidates on a 4-connected same-colour segmentation, plus one empty-space
    click).  Cached per masked frame."""

    def __init__(self, hud_mask: Optional[np.ndarray], scale: tuple, max_clicks: int = 128):
        self.hud_mask = hud_mask
        self.scale = scale
        self.max_clicks = max_clicks
        self.cache: dict[bytes, list] = {}
        self.n_click_cands: list[int] = []

    def __call__(self, env: ArcEnv, key: bytes) -> list:
        acts = self.cache.get(key)
        if acts is not None:
            return acts
        avail = env.available_actions
        acts = [[a] for a in avail if a in (1, 2, 3, 4, 5)]
        if 6 in avail and env.frame is not None:
            seg = P.segment(env.frame, hud_mask=self.hud_mask, scale=self.scale)
            cands = P.rank_click_candidates(seg, self.hud_mask, max_n=self.max_clicks, include_bg=True)
            acts += [[6, int(c.x), int(c.y)] for c in cands]
            self.n_click_cands.append(len(cands))
        if len(self.cache) < 20000:
            self.cache[key] = acts
        return acts


def calibrate_hud(env: ArcEnv, n_walks: int = 6, walk_len: int = 30, seed: int = 0) -> tuple[np.ndarray, str]:
    """Run short random walks from the level start (on clones) through perception.HudTracker and return
    the union of the detected bar regions (the frame-only HUD/budget mask) + a description."""
    rng = random.Random(seed)
    mask = np.zeros((64, 64), bool)
    descs = []
    scale = P.detect_scale(env.frame)
    for w in range(n_walks):
        hud = P.HudTracker()
        e = env.clone()
        lv0 = e.levels_completed
        acts_of = ActionSet(None, scale)
        prev = e.frame.copy()
        for t in range(walk_len):
            acts = acts_of(e, prev.tobytes())
            if not acts:
                break
            a = rng.choice(acts)
            o = step_env(e, a)
            if o.ignored or o.frame is None or o.levels_completed != lv0 or o.state != "NOT_FINISHED":
                break
            hud.update(prev, o.frame, action_key=tuple(a))
            prev = o.frame.copy()
        mask |= hud.mask()
        if hud.bars:
            descs.append(hud.describe())
    return mask, " | ".join(sorted(set(descs)))


# ------------------------------------------------------------------------------------------------
# breadth-first search with exact clones
# ------------------------------------------------------------------------------------------------
def bfs_level(env: ArcEnv, cap_s: float = 90.0, keyer: Optional[Keyer] = None, actions: Optional[ActionSet] = None,
              max_depth: int = 400, rss_limit_mb: float = 2500.0) -> dict:
    """BFS from `env` (a level start) until the level is completed, the graph is exhausted or `cap_s` runs out.

    Also accounts what the *same* search would cost an online explorer without clones (every tried action
    is a real action):  lo = 1 per tried action + 1 to come back after every state-changing one (perfect
    undo);  hi = 1 per tried action + (RESET + replay of the d-step path) after every state-changing one."""
    t0 = time.time()
    keyer = keyer or Keyer()
    actions = actions or ActionSet(keyer.mask, P.detect_scale(env.frame))
    lv0 = env.levels_completed
    rkey = keyer.key(env.frame)
    seen = {rkey}
    frontier = [(env, [], rkey)]
    st = Counter()
    per_depth = []  # (depth, frontier size expanded, new states)
    cost_lo = cost_hi = 0
    beam_cut = False
    noop_root_kept = False
    depth_done = 0
    for depth in range(1, max_depth + 1):
        nxt = []
        n_new = 0
        for ni, (node, path, nkey) in enumerate(frontier):
            acts = actions(node, nkey)
            st["expanded"] += 1
            st["cands"] += len(acts)
            for i, a in enumerate(acts):
                if time.time() - t0 > cap_s:
                    per_depth.append((depth, ni, n_new))
                    return _bfs_result("timeout", st, seen, per_depth, depth_done, cost_lo, cost_hi, beam_cut, t0,
                                       actions)
                c = node.clone() if i < len(acts) - 1 else node
                o = step_env(c, a)
                st["tried"] += 1
                d = len(path)
                if o.levels_completed > lv0 or o.state == "WIN":
                    cost_lo += 1
                    cost_hi += 1
                    per_depth.append((depth, ni + 1, n_new))
                    r = _bfs_result("solved", st, seen, per_depth, depth_done, cost_lo, cost_hi, beam_cut, t0, actions)
                    r["path"] = path + [a]
                    r["length"] = len(path) + 1
                    return r
                if o.ignored or o.frame is None:
                    st["ignored"] += 1
                    cost_lo += 1
                    cost_hi += 1
                    continue
                if o.state == "GAME_OVER":
                    st["game_over"] += 1
                    cost_lo += 2
                    cost_hi += 2 + d
                    continue
                k = keyer.key(o.frame)
                if k == nkey:
                    st["noop"] += 1
                    cost_lo += 1
                    cost_hi += 1
                    # a first action may only flip hidden state (e.g. a tutorial demo): keep one such child
                    if depth == 1 and not noop_root_kept:
                        noop_root_kept = True
                        nxt.append((c, path + [a], k))
                    continue
                cost_lo += 2
                cost_hi += 2 + d
                if k in seen:
                    st["dup"] += 1
                    continue
                seen.add(k)
                n_new += 1
                if beam_cut:
                    continue
                nxt.append((c, path + [a], k))
                if len(nxt) % 256 == 0 and rss_mb() > rss_limit_mb:
                    beam_cut = True  # memory cap: keep what we have (result becomes an upper bound)
        per_depth.append((depth, len(frontier), n_new))
        depth_done = depth
        if not nxt:
            return _bfs_result("exhausted", st, seen, per_depth, depth_done, cost_lo, cost_hi, beam_cut, t0, actions)
        frontier = nxt
    return _bfs_result("max_depth", st, seen, per_depth, depth_done, cost_lo, cost_hi, beam_cut, t0, actions)


def _bfs_result(status, st, seen, per_depth, depth_done, cost_lo, cost_hi, beam_cut, t0, actions) -> dict:
    exp = max(1, st["expanded"])
    tried = max(1, st["tried"])
    new_states = len(seen) - 1
    return {"status": status, "depth_done": depth_done, "states": len(seen), "expanded": st["expanded"],
            "tried": st["tried"], "branch_cands": round(st["cands"] / exp, 2),
            "branch_eff": round(new_states / exp, 2),
            "noop_frac": round(st["noop"] / tried, 3), "dup_frac": round(st["dup"] / tried, 3),
            "gameover_frac": round(st["game_over"] / tried, 3),
            "online_cost_lo": cost_lo, "online_cost_hi": cost_hi, "beam_cut": beam_cut,
            "per_depth": per_depth[-12:], "secs": round(time.time() - t0, 1),
            "click_cands_mean": round(float(np.mean(actions.n_click_cands)), 1) if actions.n_click_cands else 0}


# ------------------------------------------------------------------------------------------------
# structured random play
# ------------------------------------------------------------------------------------------------
def random_play(env: ArcEnv, keyer: Keyer, actions: ActionSet, max_actions: int, cap_s: float, seed: int) -> dict:
    """Uniform random choice over the candidate actions; a GAME_OVER costs a level RESET (+1) and restarts
    from the level start.  Returns the actions spent until the level was completed (or None)."""
    rng = random.Random(seed)
    t0 = time.time()
    lv0 = env.levels_completed
    e = env.clone()
    n = resets = 0
    while n < max_actions and time.time() - t0 < cap_s:
        k = keyer.key(e.frame)
        acts = actions(e, k)
        if not acts:
            break
        o = step_env(e, rng.choice(acts))
        n += 1
        if o.levels_completed > lv0 or o.state == "WIN":
            return {"solved": True, "actions": n, "resets": resets}
        if o.state == "GAME_OVER" or o.ignored:
            e = env.clone()
            n += 1
            resets += 1
    return {"solved": False, "actions": n, "resets": resets, "timed_out": time.time() - t0 >= cap_s}


# ------------------------------------------------------------------------------------------------
# one level = calibrate HUD mask, BFS, random play
# ------------------------------------------------------------------------------------------------
def run_level(task: tuple) -> dict:
    game, level, cap_s, rand_cap_s, rand_runs, rand_max = task
    t0 = time.time()
    rec: dict = {"game": game, "level": level + 1}
    try:
        starts = load_starts()
        env = env_at_level(game, level, starts)
        rec["baseline"] = env.baseline[level]
        rec["astra_actions"] = starts[game]["level_actions"][level]
        rec["available"] = env.available_actions
        mask, desc = calibrate_hud(env)
        rec["hud_px"] = int(mask.sum())
        rec["hud"] = desc
        keyer = Keyer(mask)
        scale = P.detect_scale(env.frame, ignore=mask if mask.any() else None)
        rec["scale"] = list(scale)
        acts = ActionSet(keyer.mask, scale)
        rec["root_cands"] = len(acts(env, keyer.key(env.frame)))
        r = bfs_level(env.clone(), cap_s=cap_s, keyer=keyer, actions=acts)
        if r["status"] == "solved":  # verify by replay on a fresh env
            v = env_at_level(game, level, starts)
            for a in r["path"]:
                o = step_env(v, a)
            r["verified"] = v.levels_completed > level
        rec["bfs"] = r
        rr = []
        for s in range(rand_runs):
            rr.append(random_play(env, keyer, acts, rand_max, rand_cap_s / max(1, rand_runs), seed=1000 + s))
        rec["random"] = rr
    except Exception as ex:  # noqa: BLE001
        import traceback
        rec["error"] = repr(ex)[:300]
        rec["trace"] = traceback.format_exc()[-1500:]
    rec["secs"] = round(time.time() - t0, 1)
    rec["rss_mb"] = round(rss_mb())
    return rec


def cmd_search(args) -> None:
    os.makedirs(OUT, exist_ok=True)
    starts = load_starts()
    games = args.games.split(",") if args.games else sorted(starts)
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    out_p = os.path.join(OUT, f"search_{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_p):
        for r in map(json.loads, open(out_p)):
            if "error" not in r:
                done.add((r["game"], r["level"]))
    tasks = []
    for g in games:
        for lv in range(starts[g]["n_levels"]):
            if levels and lv + 1 not in levels:
                continue
            if (g, lv + 1) in done or str(lv) not in starts[g]["starts"]:
                continue
            tasks.append((g, lv, args.cap, args.rand_cap, args.rand_runs, args.rand_max))
    # interleave games so that slow games spread over the workers
    tasks.sort(key=lambda t: (t[1], t[0]))
    print(f"{len(tasks)} level tasks -> {out_p}", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool, open(out_p, "a") as f:
        for rec in pool.imap_unordered(run_level, tasks):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            b = rec.get("bfs", {})
            rs = rec.get("random", [])
            print(f"{rec['game']} L{rec['level']}: {b.get('status', rec.get('error'))} len={b.get('length')} "
                  f"base={rec.get('baseline')} depth={b.get('depth_done')} states={b.get('states')} "
                  f"br={b.get('branch_cands')}/{b.get('branch_eff')} cost={b.get('online_cost_lo')}-{b.get('online_cost_hi')} "
                  f"hud={rec.get('hud_px')} rand={sum(x['solved'] for x in rs)}/{len(rs)} ({rec['secs']}s, {rec.get('rss_mb')}MB)",
                  flush=True)


# ------------------------------------------------------------------------------------------------
# no-op share of recorded traces (frame-only, as an agent would see it)
# ------------------------------------------------------------------------------------------------
def trace_noops(rec: list[dict]) -> dict:
    """Walk a recording with an online perception.HudTracker.  An action is a no-op if its settled frame
    equals the previous one outside the HUD mask (or the only change is one edge tick).  'repeat' no-ops
    are those where the same action was already seen to do nothing in the same masked state: a memo-style
    no-op guard could have blocked them without any prediction."""
    hud = P.HudTracker()
    prev = None
    prev_lv = None
    st = Counter()
    per_level: dict = defaultdict(Counter)
    known_noop: set = set()
    for r in rec:
        fr = r["frame"]
        if r["aid"] == 0 or r["aid"] is None:
            st["reset"] += 1
            prev, prev_lv = fr, r["lv"]
            hud.reset_pending()
            continue
        if fr is None or prev is None:
            prev, prev_lv = fr, r["lv"]
            continue
        lv = prev_lv or 0
        st["actions"] += 1
        per_level[lv]["actions"] += 1
        kind = "click" if r["aid"] == 6 else "key"
        st[f"{kind}_actions"] += 1
        if r["lv"] != prev_lv or r["state"] in ("GAME_OVER", "WIN"):
            prev, prev_lv = fr, r["lv"]
            continue
        hud.update(prev, fr, action_key=(r["aid"], r["x"], r["y"]))
        m = hud.mask()
        ch = (prev != fr) & ~m
        noop = (not ch.any()) or hud.is_tick_only(prev, fr)
        if noop:
            skey = (P.masked_hash(prev, m), r["aid"], r["x"], r["y"])
            st["noop"] += 1
            st[f"{kind}_noop"] += 1
            per_level[lv]["noop"] += 1
            if skey in known_noop:
                st["repeat_noop"] += 1
                per_level[lv]["repeat_noop"] += 1
            known_noop.add(skey)
        prev, prev_lv = fr, r["lv"]
    out = dict(st)
    out["noop_frac"] = round(st["noop"] / max(1, st["actions"]), 3)
    out["repeat_noop_frac"] = round(st["repeat_noop"] / max(1, st["actions"]), 3)
    out["click_noop_frac"] = round(st["click_noop"] / max(1, st["click_actions"]), 3)
    out["key_noop_frac"] = round(st["key_noop"] / max(1, st["key_actions"]), 3)
    out["per_level"] = {int(k): dict(v) for k, v in sorted(per_level.items())}
    return out


def cmd_noop(args) -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(HUMAN_GLOB)):
        rec = read_recording(p)
        gid = next((r["game_id"] for r in rec if r["game_id"]), "?")
        # the 5 fully reasoned wins at the repo root are frontier-model replays (reasoning field present)
        first = json.loads(open(p).readline())
        is_model = bool((first.get("data", first).get("action_input") or {}).get("reasoning"))
        if not is_model:
            with open(p) as f:
                for i, line in enumerate(f):
                    if i > 5:
                        break
                    d = json.loads(line).get("data", {})
                    if (d.get("action_input") or {}).get("reasoning"):
                        is_model = True
        r = trace_noops(rec)
        r.update({"source": "arcprize_model" if is_model else "human", "file": os.path.basename(p), "game_id": gid})
        rows.append(r)
        print(f"{r['source']:14s} {gid:14s} actions={r['actions']:5d} noop={r['noop_frac']:.3f} "
              f"repeat={r['repeat_noop_frac']:.3f} click_noop={r['click_noop_frac']:.3f} key_noop={r['key_noop_frac']:.3f}",
              flush=True)
    for g in list_games():
        p = astra_path(g)
        if not p:
            continue
        rec = read_recording(p)
        r = trace_noops(rec)
        r.update({"source": "astra", "file": os.path.basename(p), "game_id": rec[0]["game_id"]})
        rows.append(r)
        print(f"{'astra':14s} {r['game_id']:14s} actions={r['actions']:5d} noop={r['noop_frac']:.3f} "
              f"repeat={r['repeat_noop_frac']:.3f} click_noop={r['click_noop_frac']:.3f} key_noop={r['key_noop_frac']:.3f}",
              flush=True)
    with open(os.path.join(OUT, "noop_traces.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("starts")
    s.add_argument("--games", default="")
    s = sub.add_parser("search")
    s.add_argument("--games", default="")
    s.add_argument("--levels", default="")
    s.add_argument("--cap", type=float, default=90.0, help="BFS seconds per level")
    s.add_argument("--rand-cap", type=float, default=20.0, help="random-play seconds per level (all runs)")
    s.add_argument("--rand-runs", type=int, default=5)
    s.add_argument("--rand-max", type=int, default=1500, help="max actions per random run")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    s.add_argument("--tag", default="main")
    s = sub.add_parser("noop")
    s = sub.add_parser("report")
    s.add_argument("--tag", default="main")
    args = ap.parse_args(argv)
    if args.cmd == "starts":
        cmd_starts(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "noop":
        cmd_noop(args)
    elif args.cmd == "report":
        cmd_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
