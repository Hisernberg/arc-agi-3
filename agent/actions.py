"""actions.py - guarded, accounted action channel for ARC-AGI-3 (backlog B05).

Wraps any env whose ``step`` returns frames/state/levels (``agent/arcenv.ArcEnv`` offline, or the
arc_agi remote wrapper via ``step_fn``) and makes every action cheaper to decide and safer to take:

* plan-with-expectations: ``execute([("ACTION4", "avatar moves right"), ...])`` runs a batch step by step
  and halts on the first step whose observed effect contradicts its expectation, returning the diff;
* repeat-no-op guard: (level, HUD-masked frame hash, action) already seen to change nothing is refused
  *without* spending an env action (the refusal is free; the repeat would cost 1 action);
* death memory: the exact (level, state, action) that produced GAME_OVER is banned;
* reset discipline: RESET at the very start of a level is refused (the Kaggle gateway swallows it but
  still charges 1 action; in local/online NORMAL mode it is a FULL reset to level 1);
* verified-prefix replay: every level attempt is recorded with post-state hashes; after GAME_OVER /
  RESET ``replay_prefix(n)`` re-executes the known-good prefix deterministically, verifying each state
  and skipping recorded no-ops;
* exact accounting consistent with the official scorer (arc_agi 0.9.8 ``Scorecard``): every ACTION1-7
  that reaches the game costs 1 (no-ops and dead clicks included), every RESET costs 1 (level reset, or a
  swallowed competition-mode reset), actions sent after GAME_OVER/WIN are ignored (not charged), and the
  per-level count runs from the moment the level starts until it is completed.

Nothing here is game-specific. Guards can be bypassed per call with ``force=True``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

try:  # package import (agent.actions) or flat import (sys.path = agent/)
    from .perception import (DIR_NAMES, Perceiver, Percept, action_label, masked_hash, norm_action, to_frames,
                             to_grid)
except ImportError:  # pragma: no cover
    from perception import (DIR_NAMES, Perceiver, Percept, action_label, masked_hash, norm_action, to_frames,
                            to_grid)

__all__ = ["Observation", "normalize_obs", "Expectation", "parse_expectation", "parse_plan", "StepResult",
           "BatchResult", "ActionChannel"]

_DIR_ALIASES = {"up": (0, -1), "u": (0, -1), "north": (0, -1), "down": (0, 1), "d": (0, 1), "south": (0, 1),
                "left": (-1, 0), "l": (-1, 0), "west": (-1, 0), "right": (1, 0), "r": (1, 0), "east": (1, 0)}


# ------------------------------------------------------------------------------------------------
# observations
# ------------------------------------------------------------------------------------------------
@dataclass
class Observation:
    frames: list  # list[np.ndarray] (all animation frames of the step; last = settled)
    state: str  # NOT_FINISHED | WIN | GAME_OVER | NOT_PLAYED
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    ignored: bool = False  # the env did not execute the action (empty frame list / swallowed RESET)

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self.frames[-1] if self.frames else None


def _state_str(s: Any) -> str:
    if s is None:
        return "NOT_PLAYED"
    v = getattr(s, "value", s)
    if isinstance(v, str):
        return v.split(".")[-1]
    return str(v)


def normalize_obs(o: Any) -> Observation:
    """arcenv.Obs, arcengine FrameData/FrameDataRaw, arc_agi FrameData, or a REST json dict."""
    if isinstance(o, Observation):
        return o
    if isinstance(o, dict):
        get = o.get
    else:
        def get(k, default=None):
            return getattr(o, k, default)
    frames = get("frames")
    if frames is None:
        frames = get("frame")
    frames = to_frames(frames) if frames is not None else []
    lv = get("levels_completed")
    if lv is None:
        lv = get("score", 0)
    wl = get("win_levels")
    if wl is None:
        wl = get("win_score", 0)
    av = get("available_actions") or []
    av = [int(getattr(a, "value", a)) for a in av]
    ignored = bool(get("ignored", False)) or len(frames) == 0
    return Observation(frames, _state_str(get("state")), int(lv or 0), int(wl or 0), av, ignored)


# ------------------------------------------------------------------------------------------------
# expectations
# ------------------------------------------------------------------------------------------------
@dataclass
class CheckContext:
    prev: np.ndarray
    cur: Optional[np.ndarray]
    percept: Percept
    obs: Observation
    prev_obs: Observation
    perceiver: Perceiver


@dataclass
class Expectation:
    """A predicted effect of one action. ``kind`` in: no_change, change, avatar_moves, avatar_stays,
    obj_moves, obj_disappears, color_disappears, color_appears, color_decreases, color_increases,
    cell, level_complete, no_level_change, game_over, not_game_over, all (conjunction)."""
    kind: str
    args: dict = field(default_factory=dict)
    text: str = ""

    def check(self, ctx: CheckContext) -> tuple[bool, str]:
        fn = getattr(self, f"_c_{self.kind}", None)
        if fn is None:
            return False, f"unknown expectation kind {self.kind!r}"
        return fn(ctx)

    # -- individual checks ------------------------------------------------------------------
    def _c_all(self, ctx):
        for e in self.args["items"]:
            ok, why = e.check(ctx)
            if not ok:
                return False, f"{e.text or e.kind}: {why}"
        return True, "ok"

    def _c_no_change(self, ctx):
        if ctx.obs.levels_completed != ctx.prev_obs.levels_completed:
            return False, "level changed"
        return (True, "ok") if ctx.percept.noop else (False, f"{ctx.percept.delta.n} cells changed")

    def _c_change(self, ctx):
        if ctx.obs.levels_completed != ctx.prev_obs.levels_completed:
            return True, "ok"
        return (False, "nothing changed") if ctx.percept.noop else (True, "ok")

    def _moved(self, ctx, colours: Optional[set] = None, obj=None):
        out = []
        for e in ctx.percept.events:
            if e.kind != "moved":
                continue
            if colours is not None and e.color not in colours:
                continue
            if obj is not None and not (e.before.id == obj.id or (
                    e.before.color == obj.color and e.before.r0 <= obj.r1 and obj.r0 <= e.before.r1
                    and e.before.c0 <= obj.c1 and obj.c0 <= e.before.c1)):
                continue
            out.append(e)
        return out

    def _dir_ok(self, e, want, n):
        sx, sy = int(np.sign(e.dx)), int(np.sign(e.dy))
        if want is not None and (sx, sy) != want:
            return False
        if n is not None and max(abs(e.dx), abs(e.dy)) != n:
            return False
        return True

    def _c_avatar_moves(self, ctx):
        want, n = self.args.get("dir"), self.args.get("n")
        av = set(ctx.perceiver.avatar.avatars())
        moved = self._moved(ctx, av if av else None)
        if not moved:
            return False, ("avatar did not move" if av else "nothing moved") + f" ({ctx.percept.text[:80]})"
        if any(self._dir_ok(e, want, n) for e in moved):
            return True, "ok"
        e = moved[0]
        return False, f"moved ({e.dx:+d},{e.dy:+d}) instead"

    def _c_avatar_stays(self, ctx):
        av = set(ctx.perceiver.avatar.avatars())
        if not av:
            return (True, "ok") if not self._moved(ctx) else (False, "something moved")
        moved = self._moved(ctx, av)
        return (True, "ok") if not moved else (False, f"avatar moved ({moved[0].dx:+d},{moved[0].dy:+d})")

    def _obj_at(self, ctx):
        x, y = self.args["x"], self.args["y"]
        c = ctx.perceiver.seg(ctx.prev).comp_at(x, y)
        return c

    def _c_obj_moves(self, ctx):
        obj = self._obj_at(ctx)
        if obj is None or obj.bg:
            return False, "no object at that position"
        moved = self._moved(ctx, obj=obj)
        if not moved:
            return False, "object did not move"
        want, n = self.args.get("dir"), self.args.get("n")
        if any(self._dir_ok(e, want, n) for e in moved):
            return True, "ok"
        return False, f"moved ({moved[0].dx:+d},{moved[0].dy:+d}) instead"

    def _c_obj_disappears(self, ctx):
        obj = self._obj_at(ctx)
        if obj is None or obj.bg:
            return False, "no object at that position"
        if ctx.cur is None:
            return False, "no frame"
        m = ctx.perceiver.seg(ctx.prev).mask_of(obj.id)
        gone_here = bool((ctx.cur[m] != obj.color).all())
        seg_c = ctx.perceiver.seg(ctx.cur)
        before = sum(1 for c in ctx.perceiver.seg(ctx.prev).comps if c.ctype == obj.ctype)
        after = sum(1 for c in seg_c.comps if c.ctype == obj.ctype)
        if gone_here and after < before:
            return True, "ok"
        return False, "still there" if not gone_here else "moved rather than vanished"

    def _count(self, g, colour, ctx):
        m = ctx.perceiver.hud_mask()
        return int(((g == colour) & ~m).sum())

    def _c_color_disappears(self, ctx):
        k = self._count(ctx.cur, self.args["color"], ctx) if ctx.cur is not None else -1
        return (True, "ok") if k == 0 else (False, f"{k} cells of c{self.args['color']} remain")

    def _c_color_appears(self, ctx):
        a = self._count(ctx.prev, self.args["color"], ctx)
        b = self._count(ctx.cur, self.args["color"], ctx) if ctx.cur is not None else 0
        return (True, "ok") if b > a else (False, f"c{self.args['color']} count {a}->{b}")

    _c_color_increases = _c_color_appears

    def _c_color_decreases(self, ctx):
        a = self._count(ctx.prev, self.args["color"], ctx)
        b = self._count(ctx.cur, self.args["color"], ctx) if ctx.cur is not None else a
        return (True, "ok") if b < a else (False, f"c{self.args['color']} count {a}->{b}")

    def _c_cell(self, ctx):
        r, c, k = self.args["r"], self.args["c"], self.args["color"]
        if ctx.cur is None:
            return False, "no frame"
        v = int(ctx.cur[r, c])
        return (True, "ok") if v == k else (False, f"cell {r},{c} is {v}")

    def _c_level_complete(self, ctx):
        if ctx.obs.levels_completed > ctx.prev_obs.levels_completed or ctx.obs.state == "WIN":
            return True, "ok"
        return False, "level not completed"

    def _c_no_level_change(self, ctx):
        ok = ctx.obs.levels_completed == ctx.prev_obs.levels_completed
        return (True, "ok") if ok else (False, "level completed")

    def _c_game_over(self, ctx):
        return (True, "ok") if ctx.obs.state == "GAME_OVER" else (False, f"state {ctx.obs.state}")

    def _c_not_game_over(self, ctx):
        return (True, "ok") if ctx.obs.state != "GAME_OVER" else (False, "GAME_OVER")

    def _c_frame(self, ctx):  # used by replay: recorded post-state, compared with the current HUD mask
        if ctx.cur is None:
            return False, "no frame"
        m = ctx.perceiver.hud_mask()
        diff = (ctx.cur != self.args["frame"]) & ~m
        if not diff.any():
            return True, "ok"
        return False, f"state differs from the recording in {int(diff.sum())} cells"


_NUM = r"(-?\d+)"


def parse_expectation(e: Any) -> Optional[Expectation]:
    """None | Expectation | dict(kind=..., **args) | text. Text forms (case-insensitive, joined by 'and'):
    'no change' | 'nothing' ; 'change' | 'something changes' ; 'avatar moves right [3]' | 'moves up' ;
    'avatar stays' ; 'object at x,y moves left' ; 'object at x,y disappears' ; 'c8 disappears' |
    'color 8 disappears' ; 'c8 appears' | 'c8 increases' | 'c8 decreases' ; 'cell r,c becomes 5' |
    'cell r,c = 5' ; 'level completes' | 'win' ; 'no level change' ; 'game over' ; 'not game over'."""
    if e is None or isinstance(e, Expectation):
        return e
    if isinstance(e, dict):
        d = dict(e)
        kind = d.pop("kind", d.pop("type", None))
        if kind == "all":
            d["items"] = [parse_expectation(i) for i in d.get("items", [])]
        if "dir" in d and isinstance(d["dir"], str):
            d["dir"] = _DIR_ALIASES[d["dir"].lower()]
        return Expectation(kind, d, str(e))
    text = str(e).strip()
    parts = [p.strip() for p in re.split(r"\s+and\s+|\s*&\s*", text, flags=re.I) if p.strip()]
    if len(parts) > 1:
        return Expectation("all", {"items": [parse_expectation(p) for p in parts]}, text)
    t = text.lower().strip(" .")
    dirs = "|".join(_DIR_ALIASES)
    if t in ("no change", "nothing", "noop", "no-op", "no effect", "nothing changes"):
        return Expectation("no_change", {}, text)
    if t in ("change", "changes", "something changes", "any change", "effect"):
        return Expectation("change", {}, text)
    if t in ("level completes", "level complete", "level up", "win", "level completed", "completes level"):
        return Expectation("level_complete", {}, text)
    if t in ("no level change", "level continues"):
        return Expectation("no_level_change", {}, text)
    if t in ("game over", "dies", "death"):
        return Expectation("game_over", {}, text)
    if t in ("not game over", "no game over", "survives", "alive"):
        return Expectation("not_game_over", {}, text)
    if t in ("avatar stays", "avatar does not move", "avatar blocked", "blocked"):
        return Expectation("avatar_stays", {}, text)
    m = re.fullmatch(rf"(?:avatar|player|it)?\s*moves?\s*(?:({dirs}))?(?:\s*(?:by\s*)?(\d+))?", t)
    if m and (m.group(1) or "avatar" in t or "player" in t or t.startswith("moves")):
        return Expectation("avatar_moves", {"dir": _DIR_ALIASES[m.group(1)] if m.group(1) else None,
                                            "n": int(m.group(2)) if m.group(2) else None}, text)
    m = re.fullmatch(rf"(?:object|obj)\s*(?:at)?\s*\(?x?{_NUM}\s*,\s*y?{_NUM}\)?\s*moves?\s*({dirs})?(?:\s*(?:by\s*)?(\d+))?", t)
    if m:
        return Expectation("obj_moves", {"x": int(m.group(1)), "y": int(m.group(2)),
                                         "dir": _DIR_ALIASES[m.group(3)] if m.group(3) else None,
                                         "n": int(m.group(4)) if m.group(4) else None}, text)
    m = re.fullmatch(rf"(?:object|obj)\s*(?:at)?\s*\(?x?{_NUM}\s*,\s*y?{_NUM}\)?\s*(?:disappears|vanishes|is removed|gone)", t)
    if m:
        return Expectation("obj_disappears", {"x": int(m.group(1)), "y": int(m.group(2))}, text)
    m = re.fullmatch(r"(?:c|colou?r\s*)(\d+)\s*(disappears|vanishes|gone|appears|increases|grows|decreases|shrinks)", t)
    if m:
        k = {"disappears": "color_disappears", "vanishes": "color_disappears", "gone": "color_disappears",
             "appears": "color_appears", "increases": "color_increases", "grows": "color_increases",
             "decreases": "color_decreases", "shrinks": "color_decreases"}[m.group(2)]
        return Expectation(k, {"color": int(m.group(1))}, text)
    m = re.fullmatch(rf"cell\s*\(?r?{_NUM}\s*,\s*c?{_NUM}\)?\s*(?:becomes|=|->|to|is)\s*(?:c|colou?r\s*)?(\d+)", t)
    if m:
        return Expectation("cell", {"r": int(m.group(1)), "c": int(m.group(2)), "color": int(m.group(3))}, text)
    raise ValueError(f"cannot parse expectation {text!r}")


def parse_plan(text: str) -> list[tuple[Any, Optional[str]]]:
    """'A4 => avatar moves right; A4; A6 32 20 => cell 20,32 becomes 5; RESET' -> [(action, expect), ...].
    Items are separated by ';' or newlines; '=>' introduces the expectation."""
    out = []
    for item in re.split(r"[;\n]", text):
        item = item.strip()
        if not item:
            continue
        act, _, exp = item.partition("=>")
        toks = act.replace(",", " ").split()
        head = toks[0].upper()
        if head.startswith("A") and head[1:].isdigit():
            head = "ACTION" + head[1:]
        if len(toks) >= 3:
            action: Any = (head, int(toks[1]), int(toks[2]))
        else:
            action = head
        out.append((action, exp.strip() or None))
    return out


# ------------------------------------------------------------------------------------------------
# results
# ------------------------------------------------------------------------------------------------
@dataclass
class StepResult:
    action: tuple[int, Optional[int], Optional[int]]
    executed: bool  # an env action was sent
    blocked: Optional[str] = None  # guard name if refused (no env action spent)
    percept: Optional[Percept] = None
    obs: Optional[Observation] = None
    expect: Optional[Expectation] = None
    matched: Optional[bool] = None  # None when there was no expectation
    mismatch: str = ""
    charged: int = 0  # actions charged by the scorer for this call

    @property
    def text(self) -> str:
        if self.blocked:
            return f"{action_label(self.action)}: BLOCKED ({self.blocked}), no action spent"
        t = self.percept.text if self.percept is not None else action_label(self.action)
        if self.matched is False:
            t += f" | EXPECTED '{self.expect.text}' -> MISMATCH: {self.mismatch}"
        return t


@dataclass
class BatchResult:
    steps: list[StepResult]
    halted: bool
    reason: str
    planned: int

    @property
    def executed(self) -> int:
        return sum(1 for s in self.steps if s.executed)

    @property
    def blocked(self) -> int:
        return sum(1 for s in self.steps if s.blocked)

    @property
    def charged(self) -> int:
        return sum(s.charged for s in self.steps)

    def text(self, max_lines: int = 6) -> str:
        head = f"ran {self.executed}/{self.planned} actions" + (f", {self.blocked} blocked (free)" if self.blocked else "")
        if self.halted:
            head += f"; HALTED: {self.reason}"
        lines = [head]
        shown = self.steps if len(self.steps) <= max_lines else self.steps[:2] + self.steps[-(max_lines - 2):]
        if len(self.steps) > max_lines:
            lines.append(f"  ... {len(self.steps) - len(shown)} steps omitted")
        for s in shown:
            lines.append("  " + s.text)
        return "\n".join(lines)


@dataclass
class StepRecord:
    action: tuple[int, Optional[int], Optional[int]]
    frame_after: np.ndarray  # settled post-step frame (int8)
    noop: bool
    n_frames: int
    state: str


@dataclass
class Attempt:
    level: int
    start: np.ndarray  # frame at the start of the attempt (level start or after RESET)
    steps: list = field(default_factory=list)  # list[StepRecord]
    end: str = "open"  # open | reset | game_over | won
    budget_death: bool = False  # ended in GAME_OVER with the timer/budget bar (nearly) empty


# ------------------------------------------------------------------------------------------------
# the channel
# ------------------------------------------------------------------------------------------------
class ActionChannel:
    """Guarded, accounted, replayable action channel around one game env.

    ``env``: an object with ``step(...)``. By default ``env.step(aid)`` / ``env.step(6, x=x, y=y)`` is used
    (the ``arcenv.ArcEnv`` signature). Pass ``step_fn(aid, x, y) -> obs`` for any other env (e.g. the
    arc_agi wrapper: ``lambda a, x, y: w.step(GameAction.from_id(a), data={'x': x, 'y': y} if a == 6 else {})``).
    ``initial``: the current observation (defaults to ``env.last`` / env attributes for ArcEnv).
    ``competition``: Kaggle gateway RESET semantics (RESET with 0 actions in the level is swallowed but
    charged); False = local NORMAL mode (such a RESET is a full restart and a new scorecard play)."""

    def __init__(self, env: Any, perceiver: Optional[Perceiver] = None, step_fn: Optional[Callable] = None,
                 initial: Any = None, competition: bool = True, guard_noops: bool = True,
                 guard_deaths: bool = True, guard_reset_at_start: bool = True, guard_unavailable: bool = True,
                 noop_exempt: Iterable[int] = (7,), block_animated_noops: bool = False):
        self.env = env
        self.P = perceiver or Perceiver()
        self._step_fn = step_fn or self._default_step
        self.competition = competition
        self.guard_noops = guard_noops
        self.guard_deaths = guard_deaths
        self.guard_reset_at_start = guard_reset_at_start
        self.guard_unavailable = guard_unavailable
        # ACTION7 is UNDO in the ARC-AGI-3 action set: its effect depends on history, not on the frame.
        self.noop_exempt = set(noop_exempt)
        # a step that animates but ends where it started may still change hidden state (e.g. a rewind
        # that records a ghost, a demo that plays once): not blocked unless asked
        self.block_animated_noops = block_animated_noops
        self.obs = normalize_obs(initial if initial is not None else self._initial_obs(env))
        self.P.reset(self.obs.frame, self.obs.levels_completed)
        # accounting (scorer-consistent)
        self.total_actions = 0  # charged actions in the current play
        self.level_actions: list[int] = []  # charged actions per completed level (current play)
        self.cur_level_actions = 0  # charged since the current level started
        self.since_reset = 0  # non-RESET actions executed since level start / last level reset
        self.plays = 1
        self.resets = 0
        self.blocked_count: Counter = Counter()
        # memories
        self.noop_mem: Counter = Counter()  # (level, hash, key) -> times seen with no effect
        self.anim_noop_mem: Counter = Counter()
        self.death_mem: set = set()
        self.ok_mem: set = set()  # (level, masked hash, key) executed without GAME_OVER
        self.attempt = Attempt(self.level, self._frame_copy())
        self.attempts: dict[int, list[Attempt]] = {}  # level -> closed attempts
        self.solutions: dict[int, list[tuple]] = {}  # level -> winning action list
        self.history: list[StepResult] = []

    # -- env plumbing -----------------------------------------------------------------------
    @staticmethod
    def _initial_obs(env: Any) -> Any:
        last = getattr(env, "last", None)
        if last is not None:
            o = normalize_obs(last)
            return o
        return {"frame": getattr(env, "frame", None), "state": getattr(env, "state", None),
                "levels_completed": getattr(env, "levels_completed", 0),
                "available_actions": getattr(env, "available_actions", [])}

    def _default_step(self, aid: int, x: Optional[int], y: Optional[int]) -> Any:
        if aid == 6:
            return self.env.step(6, x=x, y=y)
        return self.env.step(aid)

    # -- keys / hashes ----------------------------------------------------------------------
    @property
    def level(self) -> int:
        return self.obs.levels_completed

    @property
    def state(self) -> str:
        return self.obs.state

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self.obs.frame

    def _frame_copy(self) -> np.ndarray:
        f = self.obs.frame
        return np.zeros((64, 64), np.int8) if f is None else np.asarray(f).astype(np.int8).copy()

    def _same_state(self, a: np.ndarray, b: np.ndarray) -> bool:
        return not bool(((a != b) & ~self.P.hud_mask()).any())

    def _exact_hash(self, frame: Optional[np.ndarray] = None) -> str:
        f = self.obs.frame if frame is None else frame
        return "none" if f is None else "x" + masked_hash(f, None, self.obs.levels_completed)

    def _budget_nearly_out(self, frame: Optional[np.ndarray]) -> bool:
        """True when a detected budget/timer bar has (almost) no 'remaining'-colour cells left, i.e. a
        GAME_OVER now is probably budget exhaustion rather than caused by the action itself."""
        if frame is None:
            return False
        for bar in self.P.hud.bars:
            if bar.remaining(frame) <= 2 * max(bar.tick_px, bar.hi - bar.lo + 1):
                return True
        return False

    def _hash(self, frame: Optional[np.ndarray] = None) -> str:
        f = self.obs.frame if frame is None else frame
        if f is None:
            return "none"
        return masked_hash(f, self.P.hud_mask(), self.obs.levels_completed)

    def action_keys(self, a: tuple[int, Optional[int], Optional[int]]) -> list[tuple]:
        aid, x, y = a
        if aid != 6:
            return [(aid,)]
        keys = [(6, x, y)]
        s, oy, ox = self.P.scale()
        if s > 1 and x is not None:
            keys.append((6, "cell", (y - oy) // s, (x - ox) // s))
        return keys

    # -- guards -----------------------------------------------------------------------------
    def check_guards(self, action: Any) -> Optional[str]:
        """Name of the guard that would refuse ``action`` in the current state, or None."""
        a = norm_action(action)
        aid, x, y = a
        if aid < 0 or aid > 7:
            return "invalid action id"
        if aid == 6 and (x is None or y is None or not (0 <= x < 64 and 0 <= y < 64)):
            return "ACTION6 needs x,y in 0..63"
        if aid == 0:
            if self.state == "WIN":
                return "game already won (RESET would restart from level 1)"
            if self.guard_reset_at_start and self.since_reset == 0 and self.state != "GAME_OVER":
                return "RESET at level start (charged, does nothing / full restart)"
            return None
        if self.state in ("GAME_OVER", "WIN"):
            return f"{self.state}: only RESET is accepted"
        if self.guard_unavailable and self.obs.available_actions and aid not in self.obs.available_actions:
            return f"ACTION{aid} not available {self.obs.available_actions}"
        h = self._hash()
        he = self._exact_hash()
        for k in self.action_keys(a):
            mk = (self.level, h, k)
            if self.guard_deaths and (mk in self.death_mem or (self.level, he, k) in self.death_mem):
                return "known GAME_OVER transition"
            if self.guard_noops and aid not in self.noop_exempt and (
                    self.noop_mem[mk] >= 1 or (self.block_animated_noops and self.anim_noop_mem[mk] >= 2)):
                return "known no-op here"
        return None

    # -- one step ---------------------------------------------------------------------------
    def step(self, action: Any, expect: Any = None, force: bool = False) -> StepResult:
        a = norm_action(action)
        exp = parse_expectation(expect)
        if not force:
            g = self.check_guards(a)
            if g is not None:
                self.blocked_count[g] += 1
                r = StepResult(a, False, blocked=g, expect=exp, matched=False if exp else None,
                               mismatch="blocked" if exp else "")
                self.history.append(r)
                return r
        prev_obs = self.obs
        prev_frame = prev_obs.frame
        h_before = self._hash()
        raw = self._step_fn(a[0], a[1], a[2])
        obs = normalize_obs(raw)
        charged = self._account(a, prev_obs, obs)
        pc = self.P.observe(prev_frame, obs.frames, a, level=obs.levels_completed, state=obs.state)
        if not obs.frames:  # engine ignored it (e.g. sent after GAME_OVER): keep the prior frame
            obs = Observation(prev_obs.frames, obs.state or prev_obs.state, obs.levels_completed,
                              obs.win_levels or prev_obs.win_levels, obs.available_actions or prev_obs.available_actions,
                              True)
        self.obs = obs
        self._remember(a, prev_obs, obs, pc, h_before)
        r = StepResult(a, True, percept=pc, obs=obs, expect=exp, charged=charged)
        if exp is not None:
            ctx = CheckContext(to_grid(prev_frame), obs.frame, pc, obs, prev_obs, self.P)
            ok, why = exp.check(ctx)
            r.matched, r.mismatch = ok, ("" if ok else why)
        self.history.append(r)
        return r

    def _account(self, a, prev_obs: Observation, obs: Observation) -> int:
        """Mirror of the arc_agi Scorecard bookkeeping for one sent action; returns actions charged."""
        aid = a[0]
        if aid != 0 and prev_obs.state in ("GAME_OVER", "WIN"):
            return 0  # the engine returns no frame and the server rejects it: nothing is charged
        if aid == 0:
            if prev_obs.state == "WIN" or (not self.competition and self.since_reset == 0):
                # full reset back to level 1 = a new scorecard play; the RESET itself is not charged
                self.plays += 1
                self.total_actions, self.level_actions, self.cur_level_actions, self.since_reset = 0, [], 0, 0
                return 0
            # level reset (or a competition-mode RESET swallowed at level start): charged to the level
            self.total_actions += 1
            self.cur_level_actions += 1
            self.resets += 1
            self.since_reset = 0
            return 1
        self.total_actions += 1
        self.cur_level_actions += 1
        self.since_reset += 1
        while len(self.level_actions) < obs.levels_completed:
            self.level_actions.append(self.cur_level_actions)
            self.cur_level_actions = 0
            self.since_reset = 0
        return 1

    def _remember(self, a, prev_obs: Observation, obs: Observation, pc: Percept, h_before: str) -> None:
        lvl = prev_obs.levels_completed
        if a[0] == 0:
            if not obs.ignored or self.attempt.steps:
                self._close_attempt("reset")
            self.attempt = Attempt(self.level, self._frame_copy())
            return
        if obs.ignored:
            return
        rec = StepRecord(a, self._frame_copy(), pc.noop, len(obs.frames), obs.state)
        self.attempt.steps.append(rec)
        keys = self.action_keys(a)
        if obs.levels_completed > lvl:
            self.solutions[lvl] = [r.action for r in self.attempt.steps]
            self._close_attempt("won")
            self.attempt = Attempt(self.level, self._frame_copy())
            return
        if obs.state == "GAME_OVER":
            # Exact (unmasked) key always. The HUD-masked key only when the death is (a) not plausibly
            # budget exhaustion (with the timer masked, a budget death would ban a harmless action) and
            # (b) not history-dependent: the same visible (state, action) never survived before.
            he = "x" + masked_hash(prev_obs.frame, None, lvl) if prev_obs.frame is not None else "none"
            budget = self._budget_nearly_out(prev_obs.frame)
            for k in keys:
                self.death_mem.add((lvl, he, k))
                if not budget and (lvl, h_before, k) not in self.ok_mem:
                    self.death_mem.add((lvl, h_before, k))
            self.attempt.budget_death = budget
            self._close_attempt("game_over")
            self.attempt = Attempt(lvl, self._frame_copy())  # placeholder until the RESET
            return
        for k in keys:
            self.ok_mem.add((lvl, h_before, k))
        if pc.noop:
            mem = self.noop_mem if len(obs.frames) <= 1 else self.anim_noop_mem
            for k in keys:
                mem[(lvl, h_before, k)] += 1

    def _close_attempt(self, end: str) -> None:
        at = self.attempt
        if not at.steps:
            return
        at.end = end
        self.attempts.setdefault(at.level, []).append(at)

    # -- batches ----------------------------------------------------------------------------
    def execute(self, plan: Any, halt_on_mismatch: bool = True, halt_on_blocked: bool = True,
                stop_on_level: bool = True, stop_on_game_over: bool = True, force: bool = False) -> BatchResult:
        """Run a plan: list of actions / (action, expectation) / {'action':..,'expect':..} or plan text.
        Halts on the first expectation mismatch, a refused action, a level completion (unless expected)
        or GAME_OVER, and returns every step's observed diff."""
        items = parse_plan(plan) if isinstance(plan, str) else list(plan)
        steps: list[StepResult] = []
        for i, it in enumerate(items):
            if isinstance(it, dict):
                act, exp = it.get("action"), it.get("expect")
            elif isinstance(it, tuple) and len(it) == 2 and (it[1] is None or isinstance(it[1], (str, dict, Expectation))):
                act, exp = it
            else:
                act, exp = it, None
            r = self.step(act, exp, force=force)
            steps.append(r)
            if r.blocked and halt_on_blocked:
                return BatchResult(steps, True, f"step {i + 1} {action_label(r.action)} blocked: {r.blocked}", len(items))
            if r.matched is False and halt_on_mismatch:
                return BatchResult(steps, True, f"step {i + 1} {action_label(r.action)} expected "
                                                f"'{r.expect.text}' but {r.mismatch}", len(items))
            if r.obs is not None and r.executed:
                if stop_on_game_over and r.obs.state == "GAME_OVER":
                    return BatchResult(steps, True, f"step {i + 1}: GAME_OVER", len(items))
                if r.obs.state == "WIN":
                    return BatchResult(steps, i + 1 < len(items), "game WON", len(items))
                if stop_on_level and r.percept is not None and r.percept.level_changed and i + 1 < len(items):
                    exp_level = r.expect is not None and r.expect.kind in ("level_complete",)
                    return BatchResult(steps, not exp_level, f"step {i + 1}: level completed", len(items))
        return BatchResult(steps, False, "done", len(items))

    # -- verified-prefix replay -------------------------------------------------------------
    def last_attempt(self, level: Optional[int] = None) -> Optional[Attempt]:
        lst = self.attempts.get(self.level if level is None else level) or []
        return lst[-1] if lst else None

    def known_good_prefix(self, level: Optional[int] = None, skip_noops: bool = True,
                          compress_loops: bool = False) -> list[StepRecord]:
        """Steps of this level's last closed attempt without the fatal (GAME_OVER) action, with
        single-frame no-ops dropped and, optionally, state loops cut (revisited states)."""
        at = self.last_attempt(level)
        if at is None:
            return []
        recs = list(at.steps[:-1] if at.end == "game_over" else at.steps)
        if skip_noops:
            recs = [r for r in recs if not (r.noop and r.n_frames <= 1)]
        if compress_loops:
            m = self.P.hud_mask()

            def key(f):
                g = np.asarray(f).astype(np.int8).copy()
                g[m] = -1
                return g.tobytes()

            out: list[StepRecord] = []
            pos: dict[bytes, int] = {key(at.start): -1}
            for r in recs:
                k = key(r.frame_after)
                if k in pos:
                    out = out[:pos[k] + 1]
                    pos = {kk: i for kk, i in pos.items() if i < len(out)}
                else:
                    out.append(r)
                    pos[k] = len(out) - 1
            recs = out
        return recs

    def replay_prefix(self, n: Optional[int] = None, level: Optional[int] = None, skip_noops: bool = True,
                      compress_loops: Optional[bool] = None, verify: bool = True) -> BatchResult:
        """Re-execute the first ``n`` steps (default: all) of the known-good prefix of this level's last
        attempt. Must be called in the attempt's start state (i.e. right after the RESET that follows a
        GAME_OVER or a voluntary level reset). Every step is verified against the recorded post-step frame
        (HUD masked); the replay halts at the first divergence. Guards are bypassed (the actions are known).
        ``compress_loops`` (cut revisited states) defaults to True only when the attempt died of budget
        exhaustion - replaying all of it would walk straight back into the same budget death."""
        at = self.last_attempt(level)
        if at is None:
            return BatchResult([], True, "no recorded attempt for this level", 0)
        if self.state in ("GAME_OVER", "WIN"):
            return BatchResult([], True, f"{self.state}: RESET first", 0)
        if not self._same_state(self._frame_copy(), at.start):
            return BatchResult([], True, "not at the recorded start state of the attempt (RESET first)", 0)
        if compress_loops is None:
            compress_loops = at.budget_death
        recs = self.known_good_prefix(level, skip_noops, compress_loops)
        if n is not None:
            recs = recs[:n]
        plan = [(r.action, Expectation("frame", {"frame": r.frame_after}, "recorded state") if verify else None)
                for r in recs]
        res = self.execute(plan, halt_on_mismatch=True, stop_on_level=True, force=True)
        if at.budget_death and not res.halted:
            res.reason = (f"done; the previous attempt ran out of budget after {len(at.steps)} actions "
                          f"(replayed {res.executed}, state loops cut)")
        return res

    def replay_solution(self, level: int) -> BatchResult:
        """Re-run the recorded winning action list of ``level`` (e.g. after a full restart)."""
        acts = self.solutions.get(level)
        if not acts:
            return BatchResult([], True, f"no recorded solution for level {level}", 0)
        return self.execute([(a, None) for a in acts], force=True, stop_on_level=False)

    # -- accounting -------------------------------------------------------------------------
    def accounting(self) -> dict:
        return {"total": self.total_actions, "level_actions": list(self.level_actions),
                "current_level": self.cur_level_actions, "level": self.level, "resets": self.resets,
                "plays": self.plays, "blocked": sum(self.blocked_count.values()),
                "blocked_by": dict(self.blocked_count)}
