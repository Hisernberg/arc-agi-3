"""Tests for agent/actions.py (B05).

1. Exact semantics on a tiny deterministic fake game (no engine needed).
2. Measurements on the 25 public games (accounting vs the official scorecard, guard refusals verified on
   clones, replay_prefix after GAME_OVER) and on the version-matched arcprize.org recordings
   (replay_prefix re-wins recorded levels; guarded re-play reaches the same levels).
"""

import random

import numpy as np
import pytest

import perception as P
from actions import ActionChannel, BatchResult, Expectation, parse_expectation, parse_plan
from conftest import all_games, requires_env


# ------------------------------------------------------------------------------------------------
# a tiny deterministic game with the engine's RESET/accounting semantics
# ------------------------------------------------------------------------------------------------
class Corridor:
    """Avatar (2x2, colour 9) on row 30 of a 64x64 frame. ACTION3/4 move it 4 px left/right; ACTION1/2 do
    nothing; walls at both ends. Stepping onto x <= 4 (pit, colour 8) is GAME_OVER; reaching x >= 52
    (goal, colour 14) completes the level (3 levels, identical). Budget bar on row 63: 11 -> 5 one cell
    per action; 40 actions per level attempt, then GAME_OVER. RESET: level reset after >= 1 action in
    the level; with 0 actions it is swallowed but charged (competition=True) or a full restart."""

    def __init__(self, competition=True, budget=40):
        self.competition = competition
        self.budget = budget
        self.level = 0
        self.n_actions = 0  # scorecard count (current play)
        self.level_actions = []
        self._level_start = 0
        self._reset_level()

    def _reset_level(self):
        self.pos, self.used, self.count, self.state = 24, 0, 0, "NOT_FINISHED"

    def frame(self):
        g = np.zeros((64, 64), np.int16)
        g[28:34, 0:2] = 3
        g[28:34, 62:64] = 3
        g[32:34, 2:6] = 8
        g[30:32, 54:58] = 14
        g[30:32, self.pos:self.pos + 2] = 9
        g[63, :self.budget] = 11  # the bar spans the budget, one cell per action
        g[63, :self.used] = 5
        return g

    def obs(self, ignored=False):
        return {"frame": [self.frame()], "state": self.state, "levels_completed": self.level,
                "win_levels": 3, "available_actions": [1, 2, 3, 4], "ignored": ignored}

    def step(self, aid, x=None, y=None):
        if aid == 0:
            if self.state == "WIN" or (self.count == 0 and not self.competition):
                self.level, self.n_actions, self.level_actions, self._level_start = 0, 0, [], 0
                self._reset_level()
                return self.obs()
            self.n_actions += 1
            if self.count == 0:  # competition gateway: swallowed, charged
                return self.obs(ignored=True)
            self._reset_level()
            return self.obs()
        if self.state in ("GAME_OVER", "WIN"):
            return {"frame": [], "state": self.state, "levels_completed": self.level, "win_levels": 3,
                    "available_actions": [1, 2, 3, 4]}
        self.n_actions += 1
        self.count += 1
        self.used += 1
        if aid == 3:
            self.pos = max(2, self.pos - 4)
        elif aid == 4:
            self.pos = min(60, self.pos + 4)
        if self.pos <= 4:
            self.state = "GAME_OVER"
        elif self.pos >= 52:
            self.level_actions.append(self.n_actions - self._level_start)
            self._level_start = self.n_actions
            self.level += 1
            if self.level == 3:
                self.state = "WIN"
            else:
                self._reset_level()
        elif self.used >= self.budget:
            self.state = "GAME_OVER"
        return self.obs()


def make(competition=True, budget=40):
    env = Corridor(competition, budget)
    return env, ActionChannel(env, initial=env.obs(), competition=competition)


def test_accounting_matches_on_fake_game():
    env, ch = make()
    ch.execute(["ACTION4"] * 7)  # 24 -> 52: level 1 done in 7
    assert ch.level == 1 and ch.accounting()["level_actions"] == [7] == env.level_actions
    r = ch.step(0)  # RESET at level start: refused, free
    assert r.blocked and env.n_actions == 7
    ch.step(0, force=True)  # forced: swallowed by the gateway but charged
    assert ch.total_actions == env.n_actions == 8
    ch.execute(["ACTION3", "ACTION4", "ACTION4"])
    ch.step(0)  # a real level reset
    assert ch.total_actions == env.n_actions == 12 and ch.resets == 2
    ch.execute(["ACTION4"] * 7)
    assert ch.accounting()["level_actions"] == env.level_actions == [7, 12]


def test_noop_guard_blocks_repeat_without_env_action():
    env, ch = make()
    r1 = ch.step("ACTION1")
    assert r1.executed and r1.percept.noop  # only the budget tick changed
    n = env.n_actions
    r2 = ch.step("ACTION1")
    assert r2.blocked == "known no-op here" and env.n_actions == n  # free refusal
    assert ch.step("ACTION1", force=True).executed and env.n_actions == n + 1


def test_death_ban_and_verified_prefix_replay():
    env, ch = make()
    ch.execute(["ACTION4", "ACTION1", "ACTION3", "ACTION3", "ACTION3"], stop_on_game_over=False)
    # 24 -> 28 -> 28 -> 24 -> 20 -> 16 ... not dead yet
    b = ch.execute(["ACTION3"] * 4)
    assert ch.state == "GAME_OVER" and b.halted and "GAME_OVER" in b.reason
    fatal_from = ch.last_attempt().steps[-2].frame_after
    assert ch.step("ACTION4").blocked  # only RESET after GAME_OVER
    ch.step(0)
    res = ch.replay_prefix()
    assert not res.halted, res.text()
    # no-op ACTION1 was skipped: the prefix is shorter than the attempt minus the fatal action
    assert res.executed == len(ch.last_attempt().steps) - 2
    assert not ((ch.frame != fatal_from) & ~ch.P.hud_mask()).any()  # back in the pre-death state
    assert ch.check_guards("ACTION3") == "known GAME_OVER transition"
    assert ch.check_guards("ACTION4") is None


def test_budget_death_is_not_banned_on_the_masked_state():
    env, ch = make(budget=6)
    ch.execute(["ACTION4", "ACTION3"] * 3, stop_on_game_over=False)
    assert ch.state == "GAME_OVER" and ch.last_attempt().budget_death
    ch.step(0)
    ch.replay_prefix()
    assert ch.check_guards("ACTION3") is None  # same place, but budget left: allowed


def test_execute_halts_on_first_mismatch_and_reports_diff():
    env, ch = make()
    ch.execute(["ACTION4", "ACTION3", "ACTION4", "ACTION3"])  # learn the avatar
    assert ch.P.avatar.avatars() == [9]
    n = env.n_actions
    b = ch.execute([("ACTION4", "avatar moves right"), ("ACTION2", "avatar moves down"), ("ACTION4", None)])
    assert b.halted and b.executed == 2 and env.n_actions == n + 2
    assert b.steps[0].matched is True and b.steps[1].matched is False
    assert "expected 'avatar moves down'" in b.reason and "MISMATCH" in b.text()
    b = ch.execute("A4 => avatar moves right 4; A3 => moves left & cell 30,28 becomes 9; A1 => no change")
    assert not b.halted, b.text()


def test_expectation_parser():
    k = lambda t: parse_expectation(t).kind  # noqa: E731
    assert k("no change") == "no_change" and k("something changes") == "change"
    e = parse_expectation("avatar moves right 3")
    assert e.kind == "avatar_moves" and e.args == {"dir": (1, 0), "n": 3}
    assert k("moves up") == "avatar_moves" and k("avatar stays") == "avatar_stays"
    e = parse_expectation("object at 12,40 moves left")
    assert e.kind == "obj_moves" and (e.args["x"], e.args["y"], e.args["dir"]) == (12, 40, (-1, 0))
    assert k("object at (x3,y4) disappears") == "obj_disappears"
    assert k("c8 disappears") == "color_disappears" and k("color 8 appears") == "color_appears"
    e = parse_expectation("cell 10,20 becomes 5")
    assert e.kind == "cell" and e.args == {"r": 10, "c": 20, "color": 5}
    assert k("level completes") == "level_complete" and k("game over") == "game_over"
    assert k("win and not game over") == "all"
    with pytest.raises(ValueError):
        parse_expectation("the answer is 42")


def test_parse_plan():
    plan = parse_plan("A4 => avatar moves right; ACTION6 12 40; RESET\nA1")
    assert plan == [("ACTION4", "avatar moves right"), (("ACTION6", 12, 40), None), ("RESET", None),
                    ("ACTION1", None)]


# ------------------------------------------------------------------------------------------------
# the 25 public games
# ------------------------------------------------------------------------------------------------
def _policy_action(ch, rng, kb):
    if kb and rng.random() < 0.7:
        return rng.choice(kb)
    av = ch.obs.available_actions
    a = rng.choice(av)
    if a == 6:
        c = ch.P.candidates(ch.frame)
        if c and rng.random() < 0.75:
            k = c[min(len(c) - 1, int(rng.expovariate(0.35)))]
            return (6, k.x, k.y)
        return (6, rng.randrange(64), rng.randrange(64))
    return a


@requires_env
@pytest.mark.parametrize("game", all_games())
def test_accounting_guards_and_prefix_replay_on_rollouts(game):
    """Random guarded rollout: the channel's action count equals the official scorecard; every refusal
    is verified on an exact clone; replay_prefix after GAME_OVER never diverges."""
    from arcenv import ArcEnv
    from eval_perception_actions import gt_hud_mask

    env = ArcEnv(game, competition=True)
    ch = ActionChannel(env)
    gtm = gt_hud_mask(game)
    rng = random.Random(3)
    kb = [a for a in (1, 2, 3, 4) if a in ch.obs.available_actions]
    false_refusals, prefix_fail = [], []
    for t in range(200):
        if ch.state == "WIN":
            break
        if ch.state == "GAME_OVER":
            ch.step(0)
            b = ch.replay_prefix()
            if b.halted and not b.reason.endswith("level completed") and b.planned:
                prefix_fail.append(b.reason)
            continue
        act = _policy_action(ch, rng, kb)
        if rng.random() < 0.03:
            act = 0
        g = ch.check_guards(act)
        if g in ("known no-op here", "known GAME_OVER transition"):
            c = env.clone()
            a0 = P.norm_action(act)
            o = c.step(6, x=a0[1], y=a0[2]) if a0[0] == 6 else c.step(a0[0])
            if g == "known no-op here":
                bad = o.levels_completed != env.levels_completed or (
                    o.state != "GAME_OVER" and bool(((o.frame != env.frame) & ~gtm).any()))
            else:
                bad = o.state != "GAME_OVER"
            if bad:
                false_refusals.append((t, g, act))
        n = env.n_actions
        r = ch.step(act)
        if r.blocked:
            assert env.n_actions == n  # refusals never spend an action
    acc = ch.accounting()
    assert acc["total"] == env.n_actions and acc["level_actions"] == env.level_actions
    assert env.score()["level_actions"][:len(acc["level_actions"])] == acc["level_actions"]
    assert not prefix_fail, prefix_fail
    assert len(false_refusals) <= 1, false_refusals


@requires_env
def test_reset_at_level_start_is_refused_and_forced_reset_is_charged():
    from arcenv import ArcEnv

    env = ArcEnv("ls20", competition=True)
    ch = ActionChannel(env)
    r = ch.step(0)
    assert r.blocked and env.n_actions == 0
    r = ch.step(0, force=True)  # swallowed by the gateway but charged
    assert r.executed and r.charged == 1 and env.n_actions == 1 == ch.total_actions
    ch.step(4)
    r = ch.step(0)  # level reset after an action: allowed and charged
    assert r.executed and ch.total_actions == env.n_actions == 3


@requires_env
def test_expectations_on_a_real_game():
    from arcenv import ArcEnv

    env = ArcEnv("ls20")
    ch = ActionChannel(env)
    # learn which cell ACTION4 changes (outside the HUD rows), on a clone, then predict it
    c = env.clone()
    before = env.frame.copy()
    after = c.step(4).frame
    r_, c_ = map(int, next(p for p in np.argwhere(before != after) if p[0] < 60))
    b = ch.execute([("ACTION4", f"cell {r_},{c_} becomes {int(after[r_, c_])}"),
                    ("ACTION2", "avatar moves down"), ("ACTION4", None)])
    assert b.steps[0].matched is True
    assert b.halted and b.executed == 2  # ACTION2 runs into a wall: mismatch, ACTION4 not sent


@requires_env
def test_replay_prefix_rewins_recorded_levels():
    """For every level won in the version-matched recordings (5 frontier + 2 human): play the winning
    attempt minus its last action, RESET, replay_prefix(), last action -> the level is won again."""
    from eval_perception_actions import VERSION_MATCHED, load_replays, prefix_win_eval

    results = []
    for rep in load_replays():
        if rep["game"] in VERSION_MATCHED:
            results += [(rep["game"], x) for x in prefix_win_eval(rep) if x["ok"] is not None]
    assert len(results) >= 40
    bad = [(g, x) for g, x in results if not x["ok"]]
    assert not bad, bad


@requires_env
def test_guarded_replay_of_recordings_reaches_same_levels():
    from eval_perception_actions import VERSION_MATCHED, guarded_replay, load_replays

    for rep in load_replays():
        if rep["game"] in VERSION_MATCHED:
            r = guarded_replay(rep)
            assert r["levels_reached"] >= r["levels_target"], r
