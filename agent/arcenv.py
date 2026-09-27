#!/usr/bin/env python3
"""arcenv.py - minimal, reusable OFFLINE harness for ARC-AGI-3 games (CPU only).

Runs the 25 public games directly from the competition's `environment_files/<game>/<ver>/<game>.py`
with the official `arcengine` runtime, reproduces the official scorecard bookkeeping
(`arc_agi.scorecard.Scorecard` + `EnvironmentScorecard.from_scorecard`, i.e. the exact code the
Kaggle / three.arcprize.org server runs) and supports exact state cloning for search/lookahead.

Requires the competition wheels (arcengine 0.9.3, arc_agi 0.9.8), e.g.
    SCRATCH/venv/bin/python agent/arcenv.py test

Environment variables
    ARC_ENV_DIR   directory containing <game>/<ver>/{<game>.py,metadata.json}
                  (default: $SCRATCH/kaggle/comp/environment_files, or ./environment_files)

Quick API
    env = ArcEnv("ls20")                  # loads offline, performs the initial RESET
    obs = env.step("ACTION1")             # or env.step(1) / env.step("UP") / env.step(6, x=10, y=20)
    obs.frame                             # last 64x64 int8 frame (colors 0..15); obs.frames = all anim frames
    obs.state, obs.levels_completed, obs.win_levels, obs.available_actions
    child = env.clone()                   # independent exact copy (game + scorecard + counters)
    env.score()                           # official per-game score dict (0..100)
    score_from_level_actions([...], baseline, levels_completed)  # closed-form scorer

CLI
    python arcenv.py list                       # games, tags, levels, baselines
    python arcenv.py show  <game> [actions...]  # e.g. show ls20 1 1 4 "6:32,20"   (prints ASCII frames)
    python arcenv.py test  [games...]           # smoke test: load/reset/random steps/clone fidelity/scorer
    python arcenv.py online                     # GET three.arcprize.org/api/games (needs ARC_API_KEY)
"""

from __future__ import annotations

import copy
import enum
import glob
import json
import os
import re
import sys
import types
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
from arcengine import ActionInput, ARCBaseGame, FrameDataRaw, GameAction, GameState

try:  # official scorer (arc_agi 0.9.8); optional so that the harness also works with arcengine only
    from arc_agi.models import EnvironmentInfo
    from arc_agi.scorecard import EnvironmentScorecard, Scorecard

    HAVE_ARC_AGI = True
except Exception:  # pragma: no cover
    HAVE_ARC_AGI = False

_SCRATCH = "/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad"
DEFAULT_ENV_DIR = os.environ.get("ARC_ENV_DIR") or next(
    (p for p in (os.path.join(_SCRATCH, "kaggle/comp/environment_files"), "environment_files",
                 "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files") if os.path.isdir(p)),
    "environment_files",
)

# Friendly aliases. ACTION1-4 are "up/down/left/right" by convention in every keyboard game,
# ACTION5 = "space/interact/confirm", ACTION6 = click(x, y), ACTION7 = undo (where supported).
ACTION_ALIASES = {
    "RESET": 0, "R": 0,
    "UP": 1, "U": 1, "W": 1, "DOWN": 2, "D": 2, "S": 2, "LEFT": 3, "L": 3, "A": 3, "RIGHT": 4, "RT": 4,
    "SPACE": 5, "INTERACT": 5, "CLICK": 6, "UNDO": 7, "Z": 7,
}

HEX = "0123456789abcdef"
COLOR_NAMES = {0: "white", 1: "off-white", 2: "light-grey", 3: "grey", 4: "off-black", 5: "black",
               6: "magenta", 7: "pink", 8: "red", 9: "blue", 10: "light-blue", 11: "yellow",
               12: "orange", 13: "maroon", 14: "green", 15: "purple"}


# --------------------------------------------------------------------------------------------
# Game discovery / loading
# --------------------------------------------------------------------------------------------
_CLASS_CACHE: dict[str, tuple[type, dict, bool]] = {}


def list_games(env_dir: str | None = None) -> list[str]:
    env_dir = env_dir or DEFAULT_ENV_DIR
    return sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                  for p in glob.glob(os.path.join(env_dir, "*", "*", "metadata.json")))


def load_game_class(game_id: str, env_dir: str | None = None) -> tuple[type, dict, bool]:
    """Return (GameClass, metadata dict, source_uses_builtin_id). Mirrors LocalEnvironmentWrapper."""
    env_dir = env_dir or DEFAULT_ENV_DIR
    base = game_id.split("-", 1)[0]
    key = f"{env_dir}::{game_id}"
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]
    metas = sorted(glob.glob(os.path.join(env_dir, base, "*", "metadata.json")))
    if "-" in game_id:
        metas = [m for m in metas if os.path.basename(os.path.dirname(m)) == game_id.split("-", 1)[1]] or metas
    if not metas:
        raise FileNotFoundError(f"game {game_id} not found under {env_dir}")
    meta_path = metas[-1]
    meta = json.load(open(meta_path))
    d = os.path.dirname(meta_path)
    class_name = meta.get("class_name") or (base[0].upper() + base[1:4])
    src_path = next(p for p in (os.path.join(d, f"{class_name.lower()}.py"), os.path.join(d, f"{class_name}.py"))
                    if os.path.exists(p))
    src = open(src_path, encoding="utf-8").read()
    mod = types.ModuleType(f"arc_agi_3.{base}")
    exec(compile(src, src_path, "exec"), mod.__dict__)
    cls = getattr(mod, class_name)
    assert issubclass(cls, ARCBaseGame)
    uses_id = re.search(r"(?<![A-Za-z0-9_.])id\(", src) is not None
    _CLASS_CACHE[key] = (cls, meta, uses_id)
    return _CLASS_CACHE[key]


# --------------------------------------------------------------------------------------------
# Exact cloning. Plain copy.deepcopy works for 23/25 games. bp35 and lf52 keep undo/snapshot
# tables keyed by the builtin id() of live objects; after a deepcopy those int keys point at the
# ORIGINAL objects, so the clone diverges (bp35) or leaks/replays long animations (lf52).
# deepcopy_remap_ids() rewrites every int equal to id(old_object) into id(new_object).
# --------------------------------------------------------------------------------------------
_ATOMIC = (str, bytes, float, complex, bool, type(None), range, types.FunctionType, types.BuiltinFunctionType,
           types.ModuleType, type, enum.Enum, np.ndarray, np.generic)


def deepcopy_remap_ids(obj: Any) -> Any:
    memo: dict[int, Any] = {}
    new = copy.deepcopy(obj, memo)
    idmap = {k: id(v) for k, v in memo.items() if k != id(memo) and v is not None}
    seen: set[int] = set()
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 20000))

    def fix(x: Any) -> tuple[Any, bool]:
        t = type(x)
        if t is int:
            nx = idmap.get(x)
            return (nx, True) if nx is not None else (x, False)
        if isinstance(x, _ATOMIC) or t is types.MethodType:
            return x, False
        if isinstance(x, tuple):
            vals, ch = [], False
            for e in x:
                ne, c = fix(e)
                vals.append(ne)
                ch |= c
            if not ch:
                return x, False
            return (t(*vals) if hasattr(x, "_fields") else t(vals)), True
        if isinstance(x, frozenset):
            vals = [fix(e) for e in x]
            return (frozenset(v for v, _ in vals), True) if any(c for _, c in vals) else (x, False)
        oid = id(x)
        if oid in seen:
            return x, False
        seen.add(oid)
        if isinstance(x, dict):
            items, ch = [], False
            for k, v in x.items():
                nk, ck = fix(k)
                nv, cv = fix(v)
                ch |= ck or cv
                items.append((nk, nv))
            if ch:
                x.clear()
                x.update(items)
            return x, False
        if isinstance(x, (list, deque)):
            for i, e in enumerate(x):
                ne, c = fix(e)
                if c:
                    x[i] = ne
            return x, False
        if isinstance(x, set):
            vals = [fix(e) for e in x]
            if any(c for _, c in vals):
                x.clear()
                x.update(v for v, _ in vals)
            return x, False
        d = getattr(x, "__dict__", None)
        if isinstance(d, dict):
            for k, v in list(d.items()):
                nv, c = fix(v)
                if c:
                    d[k] = nv
        for klass in t.__mro__:
            for s in getattr(klass, "__slots__", ()) or ():
                if s in ("__dict__", "__weakref__"):
                    continue
                try:
                    v = getattr(x, s)
                except AttributeError:
                    continue
                nv, c = fix(v)
                if c:
                    try:
                        setattr(x, s, nv)
                    except Exception:
                        pass
        return x, False

    try:
        fix(new)
    finally:
        sys.setrecursionlimit(old_limit)
    return new


# --------------------------------------------------------------------------------------------
# Observation + scoring helpers
# --------------------------------------------------------------------------------------------
@dataclass
class Obs:
    frames: list  # list[np.ndarray (64,64) int8]; >1 when the action plays an animation
    state: str  # NOT_FINISHED | WIN | GAME_OVER
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    level_index: int
    ignored: bool = False  # True if the action was swallowed (e.g. step after GAME_OVER, comp-mode RESET no-op)

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self.frames[-1] if self.frames else None


def level_score(baseline: int, actions: int) -> float:
    """Official per-level score in percent: min(115, 100*(baseline/actions)^2)."""
    return 0.0 if actions <= 0 else min(115.0, 100.0 * (baseline / actions) ** 2)


def score_from_level_actions(level_actions: list[int], baseline: list[int], levels_completed: int) -> float:
    """Closed-form re-implementation of EnvironmentScoreCalculator.to_score() (per game, 0..100).

    level_actions[i] = actions (incl. level RESETs) spent from the start of level i+1 until it was completed.
    weights w_i = i (1-indexed level number); uncompleted levels score 0 but keep their weight;
    result is capped at 100 * sum(w of levels with score>0) / sum(all w).
    """
    n = len(baseline)
    tot = sum(range(1, n + 1))
    s = sum((i + 1) * level_score(baseline[i], level_actions[i]) for i in range(min(levels_completed, n)))
    cap = 100.0 * sum(i + 1 for i in range(min(levels_completed, n))
                      if level_score(baseline[i], level_actions[i]) > 0) / tot
    return min(s / tot, cap)


def max_score_if_levels(baseline: list[int], k: int) -> float:
    """Upper bound on a game's score if only the first k levels are completed (at >= baseline efficiency)."""
    n = len(baseline)
    return 100.0 * sum(range(1, k + 1)) / sum(range(1, n + 1))


def ascii_frame(frame: np.ndarray, crop: bool = False) -> str:
    f = np.asarray(frame)
    if crop:
        bg = np.bincount(f.ravel().astype(np.int64) % 16).argmax()
        ys, xs = np.nonzero(f != bg)
        if len(ys):
            f = f[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return "\n".join("".join(HEX[int(v) % 16] for v in row) for row in f)


def parse_action(a: Any) -> tuple[int, dict]:
    """Accepts 0-7, 'ACTION3', 'RESET', 'UP', '6:x,y', ('ACTION6', x, y), GameAction."""
    data: dict = {}
    if isinstance(a, GameAction):
        return a.value, data
    if isinstance(a, (tuple, list)):
        aid, _ = parse_action(a[0])
        if len(a) >= 3:
            data = {"x": int(a[1]), "y": int(a[2])}
        return aid, data
    if isinstance(a, (int, np.integer)):
        return int(a), data
    s = str(a).strip().upper()
    if ":" in s:
        head, tail = s.split(":", 1)
        x, y = tail.split(",")
        return parse_action(head)[0], {"x": int(x), "y": int(y)}
    if s.startswith("ACTION"):
        return int(s[6:]), data
    if s.isdigit():
        return int(s), data
    return ACTION_ALIASES[s], data


# --------------------------------------------------------------------------------------------
# The environment
# --------------------------------------------------------------------------------------------
class ArcEnv:
    """One game, played offline, with official scorecard bookkeeping.

    competition=True reproduces the Kaggle gateway (RestAPI competition_mode): a RESET issued while
    the engine's per-level action counter is 0 (right after a level transition or a level reset)
    is NOT executed (no full restart), but is still charged 1 action on the scorecard.
    competition=False reproduces plain local/online play: such a RESET is a FULL reset back to
    level 1 and starts a new "play" on the scorecard (the game score is the max over plays).
    only_reset_levels=True emulates env ONLY_RESET_LEVELS=true (RESET never goes back to level 1
    unless the game is WON).
    """

    def __init__(self, game_id: str, env_dir: str | None = None, competition: bool = True,
                 only_reset_levels: bool = False, seed: int = 0, clone_mode: str = "auto"):
        self.cls, self.meta, self._uses_id = load_game_class(game_id, env_dir)
        self.game_id = self.meta["game_id"]
        self.baseline: list[int] = list(self.meta.get("baseline_actions") or [])
        self.tags = self.meta.get("tags") or []
        self.competition = competition
        self.only_reset_levels = only_reset_levels
        self.clone_mode = clone_mode  # "auto" | "deepcopy" | "remap"
        try:
            self.game: ARCBaseGame = self.cls(seed=seed)
        except TypeError:
            self.game = self.cls()
        self.guid = str(uuid.uuid4())
        self.card = Scorecard(card_id=str(uuid.uuid4()), competition_mode=competition) if HAVE_ARC_AGI else None
        self.history: list[tuple[int, dict]] = []  # every action sent (incl. RESETs)
        self.last: Optional[FrameDataRaw] = None
        self.n_actions = 0  # scorecard-style count of the current play (level RESETs count as 1)
        self.level_actions: list[int] = []  # per-completed-level action counts (current play)
        self._level_start = 0
        self._started = False
        self.reset()

    # ---- core ----------------------------------------------------------------------------
    def _perform(self, aid: int, data: dict) -> FrameDataRaw:
        prev = os.environ.get("ONLY_RESET_LEVELS")
        if aid == 0 and self.only_reset_levels:
            os.environ["ONLY_RESET_LEVELS"] = "true"
        try:
            fr = self.game.perform_action(ActionInput(id=GameAction.from_id(aid), data=data), raw=True)
        finally:
            if aid == 0 and self.only_reset_levels:
                if prev is None:
                    os.environ.pop("ONLY_RESET_LEVELS", None)
                else:
                    os.environ["ONLY_RESET_LEVELS"] = prev
        fr.guid = self.guid
        fr.game_id = self.game_id
        return fr

    def _account(self, fr: FrameDataRaw, full_reset: bool) -> None:
        if self.card is not None and fr.guid and len(fr.frame) > 0:
            self.card.update_scorecard(fr.guid, fr, full_reset)
        aid = fr.action_input.id.value
        if aid == 0 and full_reset:
            self.n_actions, self.level_actions, self._level_start = 0, [], 0
        elif aid in range(0, 8):
            self.n_actions += 1
        while len(self.level_actions) < fr.levels_completed:
            self.level_actions.append(self.n_actions - self._level_start)
            self._level_start = self.n_actions

    def _obs(self, fr: FrameDataRaw, ignored: bool = False) -> Obs:
        return Obs(frames=list(fr.frame), state=fr.state.value if hasattr(fr.state, "value") else str(fr.state),
                   levels_completed=fr.levels_completed, win_levels=fr.win_levels,
                   available_actions=list(fr.available_actions), level_index=self.game.level_index,
                   ignored=ignored)

    def reset(self) -> Obs:
        return self.step(0)

    def step(self, action: Any, x: int | None = None, y: int | None = None) -> Obs:
        aid, data = parse_action(action)
        if x is not None and y is not None:
            data = {"x": int(x), "y": int(y)}
        if aid == 6 and not data:
            raise ValueError("ACTION6 needs x,y in 0..63")
        self.history.append((aid, dict(data)))
        if aid == 0 and self._started and self.competition and self.game._action_count == 0 \
                and self.game._state != GameState.WIN:
            # Kaggle gateway: RESET at the very start of a level is swallowed but still charged.
            if self.card is not None and self.last is not None:
                self.card.update_scorecard(self.guid, self.last, False)
            self.n_actions += 1
            return self._obs(self.last, ignored=True)
        fr = self._perform(aid, data)
        if len(fr.frame) == 0:  # action after GAME_OVER/WIN: engine ignores it, server returns 400
            return Obs([], fr.state.value, fr.levels_completed, fr.win_levels, list(fr.available_actions),
                       self.game.level_index, ignored=True)
        full_reset = bool(fr.full_reset) or (aid == 0 and not self._started)
        self._started = True
        self._account(fr, full_reset)
        self.last = fr
        return self._obs(fr)

    # ---- cloning ---------------------------------------------------------------------------
    def clone(self) -> "ArcEnv":
        """Exact, independent copy (engine state + scorecard + counters)."""
        new = copy.copy(self)
        use_remap = self.clone_mode == "remap" or (self.clone_mode == "auto" and self._uses_id)
        new.game = deepcopy_remap_ids(self.game) if use_remap else copy.deepcopy(self.game)
        new.card = copy.deepcopy(self.card)
        new.history = list(self.history)
        new.level_actions = list(self.level_actions)
        new.last = self.last  # FrameDataRaw is treated as immutable
        return new

    snapshot = clone

    def restore(self, snap: "ArcEnv") -> None:
        c = snap.clone()
        self.__dict__.update(c.__dict__)

    # ---- introspection ---------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self.last.state.value if self.last is not None else "NOT_PLAYED"

    @property
    def levels_completed(self) -> int:
        return self.last.levels_completed if self.last is not None else 0

    @property
    def level_index(self) -> int:
        return self.game.level_index

    @property
    def available_actions(self) -> list[int]:
        return list(self.game._available_actions)

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self.last.frame[-1] if self.last is not None and self.last.frame else None

    def state_key(self) -> bytes:
        """Cheap hashable key of what the agent can see (level + last frame)."""
        f = self.frame
        return bytes([self.game.level_index]) + (f.tobytes() if f is not None else b"")

    def valid_clicks(self) -> list[tuple[int, int]]:
        """Engine-internal list of meaningful ACTION6 targets (sys_click / placeable sprites).
        Offline research only - the Kaggle API does not expose this."""
        return [(a.data["x"], a.data["y"]) for a in self.game._get_valid_actions() if a.id == GameAction.ACTION6]

    def score(self) -> dict:
        """Official per-game score (arc_agi EnvironmentScorecard.from_scorecard on this game's card)."""
        if self.card is None:
            lv = self.levels_completed
            return {"score": score_from_level_actions(self.level_actions + [0] * len(self.baseline), self.baseline, lv)}
        info = EnvironmentInfo(game_id=self.game_id, baseline_actions=self.baseline, tags=self.tags)
        es = EnvironmentScorecard.from_scorecard(self.card, [info])
        if not es.environments:
            return {"score": 0.0, "levels_completed": 0, "actions": 0}
        env = es.environments[0]
        best = max(env.runs, key=lambda r: r.score)
        return {"score": env.score, "levels_completed": env.levels_completed, "actions": env.actions,
                "resets": env.resets, "level_scores": best.level_scores, "level_actions": best.level_actions,
                "baseline": best.level_baseline_actions, "plays": len(env.runs)}

    def render(self, crop: bool = False) -> str:
        return ascii_frame(self.frame, crop=crop) if self.frame is not None else "<no frame>"


# --------------------------------------------------------------------------------------------
# Online (three.arcprize.org) helpers - read-only listing; each cmd call costs real actions.
# --------------------------------------------------------------------------------------------
def online_games(base_url: str = "https://three.arcprize.org") -> list[dict]:
    import requests

    key = os.environ.get("ARC_API_KEY", "")
    r = requests.get(f"{base_url}/api/games", headers={"X-API-Key": key, "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def _cli_list() -> None:
    for g in list_games():
        _, meta, uses_id = load_game_class(g)
        b = meta.get("baseline_actions") or []
        print(f"{meta['game_id']:16s} tags={','.join(meta.get('tags') or ['-']):14s} levels={len(b):2d} "
              f"baseline_sum={sum(b):5d} baseline={b}{'  [id()-keyed state]' if uses_id else ''}")


def _cli_show(game: str, actions: list[str]) -> None:
    env = ArcEnv(game)
    print(f"# {env.game_id} available={env.available_actions} levels={len(env.baseline)}")
    print(env.render())
    for a in actions:
        obs = env.step(a)
        print(f"\n# after {a}: state={obs.state} levels={obs.levels_completed} level_idx={obs.level_index} "
              f"frames={len(obs.frames)} ignored={obs.ignored} actions={env.n_actions}")
        print(env.render())
    print(json.dumps(env.score(), default=str))


def _trajectory(env: ArcEnv, acts: list, reset_on_end: bool = True) -> list:
    out = []
    for a in acts:
        o = env.step(a)
        out.append((o.state, o.levels_completed, o.frame.copy() if o.frame is not None else None, len(o.frames)))
        if o.state in ("GAME_OVER", "WIN") and reset_on_end:
            env.step(0)
    return out


def _cli_test(games: list[str], n: int = 200) -> int:
    import random
    import time

    bad = 0
    for g in games or list_games():
        t0 = time.time()
        env = ArcEnv(g, competition=True)
        rng = random.Random(7)
        av = env.available_actions
        acts = [(a, rng.randrange(64), rng.randrange(64)) if a == 6 else a for a in (rng.choice(av) for _ in range(n))]
        # reference run with snapshots every 25 actions
        snaps, ref = {}, []
        for i, a in enumerate(acts):
            if i % 25 == 0:
                snaps[i] = env.clone()
            ref.extend(_trajectory(env, [a]))
        # replay determinism (fresh instance)
        rep = _trajectory(ArcEnv(g, competition=True), acts)
        det = all(r[0] == q[0] and r[1] == q[1] and np.array_equal(r[2], q[2]) for r, q in zip(ref, rep))
        # clone fidelity: each snapshot must reproduce the reference suffix exactly (state, levels, last frame)
        ok = True
        nframes_ok = True
        for k, sn in snaps.items():
            suf = _trajectory(sn, acts[k:])
            for r, q in zip(ref[k:], suf):
                if not (r[0] == q[0] and r[1] == q[1] and np.array_equal(r[2], q[2])):
                    ok = False
                if r[3] != q[3]:
                    nframes_ok = False
        # official vs closed-form scorer on this (random) play
        sc = env.score()
        cf = score_from_level_actions(env.level_actions + [0] * len(env.baseline), env.baseline, env.levels_completed)
        agree = abs(sc["score"] - cf) < 1e-9 if env.card is not None and sc.get("plays", 1) == 1 else True
        status = "OK " if (det and ok and agree) else "BAD"
        bad += status == "BAD"
        print(f"{status} {g}: replay_deterministic={det} clone_exact={ok} anim_len_match={nframes_ok} "
              f"scorer_agree={agree} random_levels={env.levels_completed} actions={env.n_actions} "
              f"({time.time() - t0:.1f}s)", flush=True)
    # closed-form scorer sanity against the official calculator on synthetic plays
    if HAVE_ARC_AGI:
        from arc_agi.scorecard import EnvironmentScoreCalculator

        rng = np.random.default_rng(0)
        for _ in range(200):
            nlev = int(rng.integers(3, 11))
            base = [int(v) for v in rng.integers(5, 300, nlev)]
            done = int(rng.integers(0, nlev + 1))
            acts_l = [int(v) for v in rng.integers(1, 900, nlev)]
            calc = EnvironmentScoreCalculator()
            for i in range(nlev):
                calc.add_level(i + 1, i < done, acts_l[i], base[i])
            if abs(calc.to_score().score - score_from_level_actions(acts_l, base, done)) > 1e-9:
                print("closed-form scorer mismatch", base, done, acts_l)
                bad += 1
                break
        else:
            print("closed-form scorer == EnvironmentScoreCalculator on 200 synthetic plays")
    return bad


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        _cli_list()
    elif cmd == "show":
        _cli_show(rest[0], rest[1:])
    elif cmd == "test":
        return 1 if _cli_test(rest) else 0
    elif cmd == "online":
        gs = online_games()
        print(f"{len(gs)} online games")
        for g in gs:
            print(g["game_id"], g.get("tags"), g.get("baseline_actions"))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
