#!/usr/bin/env python3
"""Progress-aware game scheduler for the 110-game Kaggle run (backlog B08).

Why: the stock TAAF loop (`HarnessSolver._run_games`) runs `concurrency` games back to back, each for a fixed
`max_runtime_s_per_game` of wall clock, in gateway order. With 110 hidden games, 28 slots and 7,920 s per game,
four waves need 31.7 k s but only ~28 k s remain after serving setup, so the last wave (26 games) is cut short in
every stock submission, and a game that is stuck on level 1 burns the same 2.2 h as one clearing a level every
ten minutes. Unfinished levels cost nothing (scorer), so wall-clock allocation is pure upside.

What: the unit of work is one *turn* (one `ToolAgent.analyze()` call: LLM requests until an action executes or the
turn yields). `slots` turns run at once (= the LLM concurrency). Between turns every game is parked; the scheduler
decides which parked game gets the next free slot.

* **Budgets from the real clock.** `plan_budget()` is computed when the run starts (after serving setup), from the
  actual remaining wall clock to the soft deadline, the number of games the gateway reported and the slot count:
  fair share = slots x window / games; per-game service *floor* (default 30 % of the fair share) and *cap*
  (default 3x). Nothing assumes 9 h or 110 games.
* **Everyone is played.** Games are admitted in gateway order while the active set is below `max_active`, and a
  linear *start curve* forces admission so the last game starts by half of `window - floor` (the tail
  is not cut short: simulated tail/head service ~0.85 vs 0.55 for stock TAAF); when the active set
  is full, the most stagnant game that already had its floor is retired to make room. A newly admitted game gets
  its first turn before any other non-urgent game.
* **Progress-aware priority.** Weighted fair queuing on per-game virtual time: *hot* (cleared a level within the
  last `hot_turns` turns) weight 4, *fresh* (below its floor) 2, *normal* 1, *stagnant* (>= `stagnant_turns`
  action turns on the same level) 0.1. A game that clears a level jumps to the front (its virtual time is pulled
  down to the current minimum). *Dead* games (floor met, >= `dead_turns` on one level) are swapped out for pending
  games.
* **Keepalive.** An active game idle for `keepalive_trigger_s` (default 7 min) takes the next free slot, so no active
  game waits more than `keepalive_s` (10 min; the Kaggle kill is 15 min without interaction). `max_active` is sized
  from the measured turn time so keepalive turns stay below half of the capacity.
* **Clean stop.** No turn is granted after `deadline - stop_margin_s`; parked games are retired, running turns see
  `is_retired()` through the harness's `should_stop` and wind down before TAAF's soft deadline cancels anything.

Layers
------
`ProgressScheduler`  pure decision core, clock passed in explicitly (used by the simulator and the gate).
`SchedulerGate`      thread-safe blocking wrapper used by the Duck hook (`[DUCK-PATCH B08_SCHEDULER]` in
                     `inference/framework/solver.py`): `acquire()` before a turn, `release()` after it.
`simulate()`         discrete-event simulation with synthetic games (random per-level turn requirements) and a
                     shared GPU throughput model; baselines `taaf_stock` (FIFO, fixed 7,920 s), `wave_fit` (FIFO,
                     equal share = slots x window / games) and `round_robin` (all games, cyclic turns).

    python agent/scheduler.py --simulate --seeds 5            # compare policies (110 games, 28 slots, 28 k s)
    python agent/scheduler.py --plan --games 110 --slots 28 --window 28000 --turn 170

Pure standard library; importable on Kaggle without the Duck sources.
"""
from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import math
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

PENDING, WAITING, RUNNING, RETIRED = "pending", "waiting", "running", "retired"
HOT, FRESH, NORMAL, STAGNANT = "hot", "fresh", "normal", "stagnant"


# ----------------------------------------------------------------------------------------------------------------
# Configuration and budget plan
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    slots: int = 28                    # concurrent turns (= LLM concurrency / HarnessSolver.concurrency)
    keepalive_s: float = 600.0         # hard bound: an active game never waits longer than this between turns
    keepalive_trigger_s: float = 420.0  # idle this long -> takes the next free slot (margin for the queue)
    stagnant_turns: int = 12           # action turns on one level without a clear -> demoted
    dead_turns: int = 24               # ... this many (and floor met) -> may be swapped out for a pending game
    hot_turns: int = 6                 # a level clear keeps a game "hot" for this many turns
    floor_frac: float = 0.3            # floor = floor_frac x fair share (service seconds), unless floor_s is set
    floor_s: float | None = None
    min_floor_s: float = 240.0
    cap_frac: float = 3.0              # cap = cap_frac x fair share, unless cap_s is set
    cap_s: float | None = None
    stop_margin_s: float = 90.0        # no turn starts later than deadline - stop_margin_s
    start_curve_frac: float = 0.5      # last game admitted by t0 + start_curve_frac x (window - floor)
    max_active: int | None = None      # None: sized from the measured turn time (keepalive feasibility)
    keepalive_util: float = 0.5        # share of capacity keepalive turns may take when sizing max_active
    turn_estimate_s: float = 150.0     # prior for the turn-duration EWMA (09-22 production: ~170 s at 28 games)
    w_hot: float = 4.0
    w_fresh: float = 2.0
    w_normal: float = 1.0
    w_stagnant: float = 0.1
    status_every_s: float = 300.0      # gate: one status line this often (Kaggle logs are capped at 10 MB)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SchedulerConfig":
        data = dict(data or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known - {"window_s"})
        if unknown:
            raise ValueError(f"unknown scheduler config keys: {unknown} (known: {sorted(known)})")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class BudgetPlan:
    n_games: int
    slots: int
    window_s: float        # seconds from the plan's t0 until no more turns start (deadline - stop margin - t0)
    capacity_s: float      # slot-seconds available = slots x window
    fair_share_s: float    # capacity / games
    floor_s: float
    cap_s: float
    last_start_s: float    # the start curve reaches 100 % at t0 + last_start_s
    max_active: int
    turn_s: float          # turn-duration estimate used for max_active

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in dataclasses.asdict(self).items()}


def auto_max_active(cfg: SchedulerConfig, n_games: int, turn_s: float) -> int:
    """Largest active set whose keepalive turns stay within `keepalive_util` of the capacity; never below the slot
    count (then every active game always holds a slot and cannot idle)."""
    if cfg.max_active is not None:
        return max(1, min(n_games, int(cfg.max_active)))
    turn_s = max(1.0, float(turn_s))
    bound = int(cfg.keepalive_util * cfg.slots * cfg.keepalive_trigger_s / turn_s)
    return max(1, min(n_games, max(cfg.slots, bound)))


def plan_budget(n_games: int, window_s: float, cfg: SchedulerConfig, turn_s: float | None = None) -> BudgetPlan:
    """Per-game budgets from the wall clock that is actually left (`window_s`, measured at run start)."""
    n = max(1, int(n_games))
    window = max(1.0, float(window_s))
    capacity = cfg.slots * window
    fair = capacity / n
    floor = float(cfg.floor_s) if cfg.floor_s is not None else max(cfg.min_floor_s, cfg.floor_frac * fair)
    floor = min(floor, 0.9 * fair, window)  # every game must be able to get its floor
    cap = float(cfg.cap_s) if cfg.cap_s is not None else max(cfg.cap_frac * fair, floor)
    cap = max(cap, floor)
    last_start = max(1.0, (window - floor) * cfg.start_curve_frac)
    turn = float(turn_s if turn_s is not None else cfg.turn_estimate_s)
    return BudgetPlan(n, cfg.slots, window, capacity, fair, floor, cap, last_start, auto_max_active(cfg, n, turn), turn)


# ----------------------------------------------------------------------------------------------------------------
# Decision core
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class GameRecord:
    gid: str
    order: int
    state: str = PENDING
    ready: bool = True             # the game's worker is parked in acquire() (always True in the simulator)
    admitted_at: float | None = None
    first_turn_at: float | None = None
    turn_started_at: float | None = None
    last_turn_end: float | None = None
    service_s: float = 0.0
    turns: int = 0
    action_turns: int = 0
    levels: int = 0
    turns_on_level: int = 0
    last_clear_turn: int | None = None
    vtime: float = 0.0
    turn_weight: float = 1.0
    max_idle_s: float = 0.0
    retired_at: float | None = None
    retire_reason: str | None = None
    won: bool = False


class ProgressScheduler:
    """Decision core. Every method takes the current time explicitly (seconds, any monotonic origin)."""

    def __init__(self, game_ids: Iterable[str], cfg: SchedulerConfig | None = None, *, now: float,
                 deadline: float, turn_estimate_s: float | None = None):
        self.cfg = cfg or SchedulerConfig()
        ids = list(game_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate game ids")
        if not ids:
            raise ValueError("no games")
        self.games: dict[str, GameRecord] = {gid: GameRecord(gid, i) for i, gid in enumerate(ids)}
        self.order = ids
        self.t0 = float(now)
        self.deadline = float(deadline)
        # never let the stop margin eat a short window (smoke runs, rehearsals): at most 20 % of it
        self.stop_margin_s = min(self.cfg.stop_margin_s, 0.2 * max(0.0, self.deadline - self.t0))
        self.stop_at = self.deadline - self.stop_margin_s
        self.d_est = float(turn_estimate_s or self.cfg.turn_estimate_s)
        self.plan = plan_budget(len(ids), self.stop_at - self.t0, self.cfg, self.d_est)
        self.running = 0
        self.stopped: str | None = None
        self.events: list[tuple[float, str, str, str]] = []
        self._next_pending = 0

    # ---- classification -------------------------------------------------------------------------------------
    def klass(self, g: GameRecord) -> str:
        if g.last_clear_turn is not None and g.turns - g.last_clear_turn < self.cfg.hot_turns:
            return HOT
        if g.service_s < self.plan.floor_s:
            return FRESH
        if g.turns_on_level >= self.cfg.stagnant_turns:
            return STAGNANT
        return NORMAL

    def weight(self, g: GameRecord) -> float:
        k = self.klass(g)
        return {HOT: self.cfg.w_hot, FRESH: self.cfg.w_fresh, NORMAL: self.cfg.w_normal,
                STAGNANT: self.cfg.w_stagnant}[k]

    def active(self) -> list[GameRecord]:
        return [g for g in self.games.values() if g.state in (WAITING, RUNNING)]

    def _vmin(self, exclude: GameRecord | None = None) -> float:
        act = [g.vtime for g in self.active() if g is not exclude]
        return min(act) if act else 0.0

    def _log(self, now: float, kind: str, gid: str = "", detail: str = "") -> None:
        self.events.append((round(now - self.t0, 1), kind, gid, detail))

    # ---- admission / retirement -----------------------------------------------------------------------------
    def max_active(self) -> int:
        return auto_max_active(self.cfg, len(self.order), self.d_est)

    def required_started(self, now: float) -> int:
        frac = min(1.0, max(0.0, (now - self.t0) / self.plan.last_start_s))
        return min(len(self.order), math.ceil(len(self.order) * frac))

    def admitted_count(self) -> int:
        return self._next_pending

    def _admit_next(self, now: float, why: str) -> GameRecord | None:
        if self._next_pending >= len(self.order):
            return None
        g = self.games[self.order[self._next_pending]]
        self._next_pending += 1
        g.state = WAITING
        g.admitted_at = now
        g.vtime = self._vmin()
        self._log(now, "admit", g.gid, why)
        return g

    def _retire(self, g: GameRecord, now: float, reason: str) -> None:
        if g.state == RETIRED:
            return
        if g.state == RUNNING:
            self.running -= 1
        elif g.state == WAITING and g.last_turn_end is not None:  # parked until now: that idle counts too
            g.max_idle_s = max(g.max_idle_s, now - g.last_turn_end)
        g.state = RETIRED
        g.retired_at = now
        g.retire_reason = reason
        self._log(now, "retire", g.gid, reason)

    def _victim(self, forced: bool) -> GameRecord | None:
        """A parked game that already had its floor; `forced` (start curve) accepts any such game, otherwise only
        dead ones. Most stagnant first, then least progress."""
        cands = [g for g in self.games.values()
                 if g.state == WAITING and g.service_s >= self.plan.floor_s and self.klass(g) != HOT]
        if not forced:
            cands = [g for g in cands if g.turns_on_level >= self.cfg.dead_turns]
        if not cands:
            return None
        return max(cands, key=lambda g: (g.turns_on_level, -g.levels, g.service_s))

    def tick(self, now: float) -> None:
        """Deadline + admissions. Call before `dispatch`."""
        if self.stopped:
            return
        if now >= self.stop_at:
            self.stop_all(now, "deadline")
            return
        m = self.max_active()
        while self._next_pending < len(self.order):
            n_active = len(self.active())
            behind = self.required_started(now) > self._next_pending
            if n_active < m:
                self._admit_next(now, "slot")
            elif behind:
                victim = self._victim(forced=True)
                if victim is not None:
                    self._retire(victim, now, "start_curve")
                self._admit_next(now, "start_curve")
            else:
                victim = self._victim(forced=False)
                if victim is None:
                    break
                self._retire(victim, now, "stagnant")
                self._admit_next(now, "swap")

    def stop_all(self, now: float, reason: str) -> None:
        if self.stopped:
            return
        self.stopped = reason
        for g in self.games.values():
            if g.state in (PENDING, WAITING):
                self._retire(g, now, reason)
        self._log(now, "stop", "", reason)

    # ---- turns ----------------------------------------------------------------------------------------------
    def _urgent(self, g: GameRecord, now: float) -> bool:
        return g.turns > 0 and g.last_turn_end is not None and now - g.last_turn_end >= self.cfg.keepalive_trigger_s

    def dispatch(self, now: float) -> list[str]:
        """Fill free slots; returns the game ids granted a turn now."""
        granted: list[str] = []
        if self.stopped:
            return granted
        while self.running < self.cfg.slots:
            cands = [g for g in self.games.values() if g.state == WAITING and g.ready]
            if not cands:
                break
            urgent = [g for g in cands if self._urgent(g, now)]
            if urgent:
                g = min(urgent, key=lambda x: (x.last_turn_end, x.order))
            else:
                first = [g for g in cands if g.turns == 0]
                if first:
                    g = min(first, key=lambda x: x.order)
                else:
                    g = min(cands, key=lambda x: (x.vtime, x.last_turn_end or 0.0, x.order))
            self._start_turn(g, now)
            granted.append(g.gid)
        return granted

    def _start_turn(self, g: GameRecord, now: float) -> None:
        if g.last_turn_end is not None:
            g.max_idle_s = max(g.max_idle_s, now - g.last_turn_end)
        if g.first_turn_at is None:
            g.first_turn_at = now
        g.turn_weight = self.weight(g)
        g.state = RUNNING
        g.ready = False
        g.turn_started_at = now
        self.running += 1

    def on_turn_end(self, gid: str, now: float, *, levels: int | None = None, won: bool = False,
                    executed: bool = True, ready: bool = True) -> None:
        """Report a finished turn. `executed`: the turn executed at least one action (only those count towards
        stagnation, so an LLM outage does not demote every game). `ready`: the game can be granted its next turn
        right away (the gate passes False: its worker must come back to `acquire()` first)."""
        g = self.games[gid]
        g.ready = ready
        if g.turn_started_at is None:
            return
        dur = max(0.0, now - g.turn_started_at)
        g.turn_started_at = None
        if g.state == RUNNING:
            self.running -= 1
            g.state = WAITING
        g.service_s += dur
        g.turns += 1
        if dur > 0:
            self.d_est = 0.9 * self.d_est + 0.1 * dur
        g.vtime += dur / max(1e-6, g.turn_weight)
        if executed:
            g.action_turns += 1
        if levels is not None and levels > g.levels:
            g.levels = int(levels)
            g.turns_on_level = 0
            g.last_clear_turn = g.turns
            g.vtime = min(g.vtime, self._vmin(exclude=g) - 1e-6)  # a clear moves the game to the front
            self._log(now, "clear", gid, f"level={g.levels}")
        elif executed:
            g.turns_on_level += 1
        g.last_turn_end = now
        if g.state == RETIRED:  # ended by the harness (finish()) while its turn was running
            return
        if won:
            g.won = True
            self._retire(g, now, "won")
        elif g.service_s >= self.plan.cap_s:
            self._retire(g, now, "cap")
        elif self.stopped:
            self._retire(g, now, self.stopped)

    def finish(self, gid: str, now: float, reason: str) -> None:
        """The harness ended a game on its own (win, action cap, crash) outside a turn."""
        g = self.games[gid]
        if g.turn_started_at is not None:
            self.on_turn_end(gid, now, executed=False, ready=False)
        self._retire(g, now, reason)

    # ---- queries --------------------------------------------------------------------------------------------
    def is_retired(self, gid: str) -> bool:
        g = self.games[gid]
        return g.state == RETIRED or (self.stopped is not None and g.state == RUNNING)

    def time_remaining(self, gid: str, now: float) -> float:
        g = self.games[gid]
        return max(0.0, min(self.plan.cap_s - g.service_s, self.stop_at - now))

    def snapshot(self, now: float) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for g in self.games.values():
            key = g.state if g.state != WAITING and g.state != RUNNING else self.klass(g)
            counts[key] = counts.get(key, 0) + 1
        return {
            "t": round(now - self.t0, 1), "left_s": round(self.stop_at - now, 1), "running": self.running,
            "active": len(self.active()), "max_active": self.max_active(), "admitted": self._next_pending,
            "started": sum(1 for g in self.games.values() if g.first_turn_at is not None),
            "levels": sum(g.levels for g in self.games.values()), "turn_s": round(self.d_est, 1), **counts,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "config": self.cfg.to_dict(),
            "plan": self.plan.to_dict(),
            "stopped": self.stopped,
            "games": {
                g.gid: {
                    "state": g.state, "reason": g.retire_reason, "levels": g.levels, "turns": g.turns,
                    "service_s": round(g.service_s, 1), "max_idle_s": round(g.max_idle_s, 1),
                    "first_turn_s": None if g.first_turn_at is None else round(g.first_turn_at - self.t0, 1),
                    "retired_s": None if g.retired_at is None else round(g.retired_at - self.t0, 1),
                }
                for g in self.games.values()
            },
        }


# ----------------------------------------------------------------------------------------------------------------
# Thread-safe gate for the Duck hook
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Grant:
    ok: bool
    reason: str = ""


class SchedulerGate:
    """Blocking wrapper: one worker thread per game calls `acquire()` before each turn and `release()` after it."""

    def __init__(self, game_ids: Iterable[str], cfg: SchedulerConfig | None = None, *, deadline_in_s: float,
                 clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] | None = print,
                 poll_s: float = 1.0):
        self._clock = clock
        self._cond = threading.Condition()
        now = clock()
        self.core = ProgressScheduler(game_ids, cfg, now=now, deadline=now + float(deadline_in_s))
        self._log = log
        self._poll_s = poll_s
        self._last_status = now
        for g in self.core.games.values():
            g.ready = False
        self._emit(f"HYDRA_SCHED start games={len(self.core.order)} slots={self.core.cfg.slots} "
                   f"plan={json.dumps(self.core.plan.to_dict(), sort_keys=True)}")

    def _emit(self, line: str) -> None:
        if self._log is not None:
            try:
                self._log(line)
            except Exception:
                pass

    def _pump(self, now: float) -> None:
        core = self.core
        before = len(core.events)
        core.tick(now)
        if core.dispatch(now):
            self._cond.notify_all()
        for t, kind, gid, detail in core.events[before:]:
            if kind in ("retire", "stop") or (kind == "admit" and detail != "slot"):
                self._emit(f"HYDRA_SCHED {kind} t={t} {gid} {detail}".rstrip())
        if now - self._last_status >= core.cfg.status_every_s:
            self._last_status = now
            self._emit(f"HYDRA_SCHED status {json.dumps(core.snapshot(now), sort_keys=True)}")

    def acquire(self, gid: str, abort: Callable[[], bool] | None = None) -> Grant:
        with self._cond:
            g = self.core.games[gid]
            if g.state == RUNNING:  # re-entrant: turn already granted
                return Grant(True)
            g.ready = True
            while True:
                if g.state == RETIRED:
                    g.ready = False
                    return Grant(False, g.retire_reason or "retired")
                if abort is not None:
                    try:
                        if abort():
                            g.ready = False
                            return Grant(False, "stopped")
                    except Exception:
                        pass
                self._pump(self._clock())
                if g.state == RUNNING:
                    return Grant(True)
                self._cond.wait(timeout=self._poll_s)

    def release(self, gid: str, *, levels: int | None = None, won: bool = False, executed: bool = True) -> None:
        with self._cond:
            now = self._clock()
            self.core.on_turn_end(gid, now, levels=levels, won=won, executed=executed, ready=False)
            self._pump(now)
            self._cond.notify_all()

    def finish(self, gid: str, reason: str) -> None:
        with self._cond:
            now = self._clock()
            self.core.finish(gid, now, reason)
            self._pump(now)
            self._cond.notify_all()

    def shutdown(self, reason: str = "stopped") -> None:
        with self._cond:
            now = self._clock()
            self.core.stop_all(now, reason)
            self._cond.notify_all()
            self._emit(f"HYDRA_SCHED final {json.dumps(self.core.snapshot(now), sort_keys=True)}")

    def is_retired(self, gid: str) -> bool:
        with self._cond:
            return self.core.is_retired(gid)

    def time_remaining(self, gid: str) -> float:
        with self._cond:
            return self.core.time_remaining(gid, self._clock())

    def summary(self) -> dict[str, Any]:
        with self._cond:
            out = self.core.summary()
            out["snapshot"] = self.core.snapshot(self._clock())
            return out


# ----------------------------------------------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class SynthGame:
    gid: str
    level_turns: list[float]   # action turns needed for each level; inf = never solved

    @property
    def n_levels(self) -> int:
        return len(self.level_turns)

    def levels_after(self, turns: float) -> int:
        done, acc = 0, 0.0
        for need in self.level_turns:
            acc += need
            if turns >= acc:
                done += 1
            else:
                break
        return done


def synth_games(n: int = 110, seed: int = 0, *, scale_turns: float = 12.0, scale_sigma: float = 0.9,
                level_sigma: float = 0.6, level_growth: float = 0.25, opaque: float = 0.15,
                wall_base: float = 0.12, wall_growth: float = 0.05) -> list[SynthGame]:
    """Synthetic hidden games: per-game aptitude (lognormal turns scale), per-level requirement growing with depth
    (lognormal noise), some games opaque from level 1, and a per-level chance of a wall the agent never passes.
    Calibrated loosely on the 09-22 public-25 run (~42 turns/game -> ~1.8 levels/game)."""
    rng = random.Random(seed)
    games = []
    for i in range(n):
        n_levels = rng.randint(5, 9)
        scale = scale_turns * math.exp(rng.gauss(0.0, scale_sigma))
        opaque_game = rng.random() < opaque
        need = []
        for k in range(1, n_levels + 1):
            wall = (k == 1 and opaque_game) or rng.random() < min(0.6, wall_base + wall_growth * (k - 1))
            if wall:
                need.append(math.inf)
            else:
                need.append(max(1.0, scale * (1 + level_growth * (k - 1)) * math.exp(rng.gauss(0.0, level_sigma))))
        games.append(SynthGame(f"g{i:03d}", need))
    return games


def game_score(levels: int, n_levels: int) -> float:
    """Level-index-weighted completion (efficiency ignored): the scorer's cap for `levels` cleared levels."""
    total = n_levels * (n_levels + 1) / 2
    return 100.0 * (levels * (levels + 1) / 2) / total


class _Policy:
    name = "policy"

    def start(self, games: list[SynthGame], slots: int, now: float, deadline: float, turn_s: float) -> None: ...
    def dispatch(self, now: float, free: int) -> list[str]: ...
    def turn_end(self, gid: str, now: float, levels: int, won: bool) -> None: ...
    def tick(self, now: float) -> None: ...
    def stopped(self, now: float) -> bool: ...


class ProgressPolicy(_Policy):
    name = "progress"

    def __init__(self, cfg: SchedulerConfig | None = None):
        self.cfg = cfg

    def start(self, games, slots, now, deadline, turn_s):
        cfg = dataclasses.replace(self.cfg or SchedulerConfig(), slots=slots)
        # the real scheduler does not know the throughput in advance: start from the config prior, learn online
        self.core = ProgressScheduler([g.gid for g in games], cfg, now=now, deadline=deadline)

    def tick(self, now):
        self.core.tick(now)

    def dispatch(self, now, free):
        return self.core.dispatch(now)

    def turn_end(self, gid, now, levels, won):
        self.core.on_turn_end(gid, now, levels=levels, won=won)

    def stopped(self, now):
        return self.core.stopped is not None


class FixedWavePolicy(_Policy):
    """Stock TAAF: `slots` games play back to back in gateway order, each until `per_game_s` of its own wall clock
    (== service: it plays continuously) or a win; the soft deadline cancels whatever still runs."""

    def __init__(self, per_game_s: float | None = None, name: str = "taaf_stock"):
        self.per_game_s = per_game_s
        self.name = name

    def start(self, games, slots, now, deadline, turn_s):
        self.queue = [g.gid for g in games]
        self.deadline = deadline
        self.per_game = self.per_game_s if self.per_game_s is not None else slots * (deadline - now) / len(games)
        self.started: dict[str, float] = {}
        self.playing: set[str] = set()
        self.idle: list[str] = []   # playing games whose turn just ended (ready for their next turn)
        self.slots = slots

    def tick(self, now):
        pass

    def dispatch(self, now, free):
        out = []
        if now >= self.deadline:
            return out
        while self.idle and free > 0:
            out.append(self.idle.pop(0))
            free -= 1
        while self.queue and len(self.playing) < self.slots and free > 0:
            gid = self.queue.pop(0)
            self.playing.add(gid)
            self.started[gid] = now
            out.append(gid)
            free -= 1
        return out

    def turn_end(self, gid, now, levels, won):
        if won or now - self.started[gid] >= self.per_game:
            self.playing.discard(gid)
        else:
            self.idle.append(gid)

    def stopped(self, now):
        return now >= self.deadline


class RoundRobinPolicy(_Policy):
    """All games active from the start; turns strictly cyclic (least recently served first)."""
    name = "round_robin"

    def start(self, games, slots, now, deadline, turn_s):
        self.deadline = deadline
        self.last: dict[str, float] = {g.gid: -1.0 - i * 1e-6 for i, g in enumerate(games)}
        self.ready = set(self.last)

    def tick(self, now):
        pass

    def dispatch(self, now, free):
        out = []
        if now >= self.deadline:
            return out
        for gid in sorted(self.ready, key=lambda x: self.last[x])[:free]:
            self.ready.discard(gid)
            out.append(gid)
        return out

    def turn_end(self, gid, now, levels, won):
        self.last[gid] = now
        if not won:
            self.ready.add(gid)

    def stopped(self, now):
        return now >= self.deadline


@dataclass
class SimResult:
    policy: str
    levels: int
    score: float
    turns: int
    started: int
    unstarted: list[str]
    max_idle_s: float
    last_turn_end: float
    deadline: float
    per_game: dict[str, dict[str, Any]] = field(repr=False)

    def brief(self) -> dict[str, Any]:
        return {"policy": self.policy, "levels": self.levels, "score": round(self.score, 2), "turns": self.turns,
                "started": self.started, "max_idle_s": round(self.max_idle_s, 1),
                "last_turn_end": round(self.last_turn_end, 1)}


def simulate(games: list[SynthGame], policy: _Policy, *, slots: int = 28, window_s: float = 28_000.0,
             turns_per_s: float = 0.163, min_turn_s: float = 15.0, turn_sigma: float = 0.35, seed: int = 0,
             tick_s: float = 30.0, keepalive_s: float | None = None) -> SimResult:
    """Discrete-event run. Throughput model shared by every policy: with n turns in flight each turn takes
    max(min_turn_s, n / turns_per_s) x lognormal(0, turn_sigma) seconds (the GPU's aggregate rate is fixed; fewer
    concurrent requests -> faster individual turns). A turn still running at the deadline is cut (no credit).
    Progress: game g clears level k once it has had sum(level_turns[:k]) turns. Idle = gap between two turns of a
    game that had not finished (for stock/wave policies a parked-in-queue game is not idle: it never started)."""
    rng = random.Random(seed * 7919 + 1)
    by_id = {g.gid: g for g in games}
    now, deadline = 0.0, float(window_s)
    policy.start(games, slots, now, deadline, slots / turns_per_s)
    heap: list[tuple[float, int, str, bool]] = []   # (end, seq, gid, completes)
    seq = 0
    turns = {g.gid: 0 for g in games}
    levels = {g.gid: 0 for g in games}
    won = {g.gid: False for g in games}
    first: dict[str, float] = {}
    last_end: dict[str, float] = {}
    max_idle = {g.gid: 0.0 for g in games}
    service = {g.gid: 0.0 for g in games}
    in_flight: dict[str, float] = {}
    last_turn_end = 0.0

    def launch(t: float) -> None:
        nonlocal seq
        policy.tick(t)
        for gid in policy.dispatch(t, slots - len(in_flight)):
            if gid in in_flight:
                raise AssertionError(f"{policy.name}: {gid} granted twice")
            n = len(in_flight) + 1
            dur = max(min_turn_s, n / turns_per_s) * math.exp(rng.gauss(0.0, turn_sigma))
            end = t + dur
            completes = end <= deadline
            end = min(end, deadline)
            if gid in last_end:
                max_idle[gid] = max(max_idle[gid], t - last_end[gid])
            first.setdefault(gid, t)
            in_flight[gid] = t
            heapq.heappush(heap, (end, seq, gid, completes))
            seq += 1
        if len(in_flight) > slots:
            raise AssertionError(f"{policy.name}: {len(in_flight)} turns in flight > {slots} slots")

    launch(now)
    while True:
        next_event = heap[0][0] if heap else math.inf
        if not heap and (now >= deadline or policy.stopped(now)):
            break
        if next_event <= now + tick_s:
            end, _, gid, completes = heapq.heappop(heap)
            now = end
            start = in_flight.pop(gid)
            service[gid] += now - start
            last_turn_end = max(last_turn_end, now)
            if completes:
                turns[gid] += 1
                new_levels = by_id[gid].levels_after(turns[gid])
                levels[gid] = new_levels
                won[gid] = new_levels >= by_id[gid].n_levels
            last_end[gid] = now
            policy.turn_end(gid, now, levels[gid], won[gid])
        else:
            now = min(now + tick_s, deadline) if not heap else now + tick_s
        if now < deadline:
            launch(now)
    if isinstance(policy, RoundRobinPolicy):  # parked until the end: the idle tail counts
        for gid, t in last_end.items():
            if not won[gid]:
                max_idle[gid] = max(max_idle[gid], deadline - t)
    if isinstance(policy, ProgressPolicy):
        for gid, rec in policy.core.games.items():
            max_idle[gid] = max(max_idle[gid], rec.max_idle_s)
    per_game = {g.gid: {"levels": levels[g.gid], "n_levels": g.n_levels, "turns": turns[g.gid],
                        "service_s": round(service[g.gid], 1), "first_turn_s": first.get(g.gid),
                        "max_idle_s": round(max_idle[g.gid], 1), "won": won[g.gid]} for g in games}
    return SimResult(
        policy=policy.name,
        levels=sum(levels.values()),
        score=statistics.fmean(game_score(levels[g.gid], g.n_levels) for g in games),
        turns=sum(turns.values()),
        started=len(first),
        unstarted=[g.gid for g in games if g.gid not in first],
        max_idle_s=max(max_idle.values()) if max_idle else 0.0,
        last_turn_end=last_turn_end,
        deadline=deadline,
        per_game=per_game,
    )


def compare(seeds: Iterable[int] = range(5), *, n_games: int = 110, slots: int = 28, window_s: float = 28_000.0,
            turns_per_s: float = 0.163, cfg: SchedulerConfig | None = None) -> dict[str, list[SimResult]]:
    out: dict[str, list[SimResult]] = {}
    for seed in seeds:
        games = synth_games(n_games, seed)
        fair = slots * window_s / n_games
        for policy in (ProgressPolicy(cfg), FixedWavePolicy(7920.0, "taaf_stock"), FixedWavePolicy(fair, "wave_fit"),
                       RoundRobinPolicy()):
            res = simulate(games, policy, slots=slots, window_s=window_s, turns_per_s=turns_per_s, seed=seed)
            out.setdefault(policy.name, []).append(res)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--games", type=int, default=110)
    ap.add_argument("--slots", type=int, default=28)
    ap.add_argument("--window", type=float, default=28_000.0, help="seconds from run start to the soft deadline")
    ap.add_argument("--turn", type=float, default=None, help="turn-duration estimate (s) for max_active")
    ap.add_argument("--turns-per-s", type=float, default=0.163, help="simulated aggregate GPU rate (turns/s)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--config", default="{}", help="SchedulerConfig overrides as JSON")
    a = ap.parse_args(argv)
    cfg = SchedulerConfig.from_mapping({"slots": a.slots, **json.loads(a.config)})
    if a.plan or not a.simulate:
        plan = plan_budget(a.games, a.window - cfg.stop_margin_s, cfg, a.turn)
        print(json.dumps(plan.to_dict(), indent=1))
    if a.simulate:
        res = compare(range(a.seeds), n_games=a.games, slots=a.slots, window_s=a.window, turns_per_s=a.turns_per_s,
                      cfg=cfg)
        for name, runs in res.items():
            print(f"{name:12s} levels={statistics.fmean(r.levels for r in runs):7.1f} "
                  f"score={statistics.fmean(r.score for r in runs):6.2f} "
                  f"started={min(r.started for r in runs):3d}/{a.games} "
                  f"max_idle={max(r.max_idle_s for r in runs):7.1f}s turns={statistics.fmean(r.turns for r in runs):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
