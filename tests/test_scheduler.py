"""B08: progress-aware scheduler (agent/scheduler.py) + its Duck hook ([DUCK-PATCH B08_SCHEDULER]).

    $SCRATCH/venv/bin/python -m pytest tests/test_scheduler.py -q

Simulation (synthetic games with random per-level turn requirements, one shared GPU throughput model):
110 games, 28 slots, 28,000 s left after setup (8h35m soft deadline - ~48 min setup). Proves: every game is
started, no active game idles > 10 min, everything stops before the deadline, more levels than the fixed
baselines (TAAF stock FIFO with 7,920 s/game, wave-fit FIFO with an equal share, turn-level round robin), and
that stock TAAF cuts the last wave short while the scheduler does not. The hook test runs the real Duck loop
(TAAF Benchmark -> HarnessSolver -> ToolAgent) against the scripted mock LLM with the switch on.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent"))

import scheduler as S  # noqa: E402

N_GAMES, SLOTS, WINDOW = 110, 28, 28_000.0
SLOW, FAST = 0.163, 0.65   # turns/s: the 09-22 production rate (42 turns/game/2 h at 28 games) and the M1 target (4x)
SEEDS = range(5)


def _run(policy, seed, tps, *, n=N_GAMES, slots=SLOTS, window=WINDOW):
    return S.simulate(S.synth_games(n, seed), policy, slots=slots, window_s=window, turns_per_s=tps, seed=seed)


# ------------------------------------------------------------------------------------------------ budget plan


def test_budget_plan_comes_from_the_measured_window():
    cfg = S.SchedulerConfig()
    plan = S.plan_budget(110, 28_000 - cfg.stop_margin_s, cfg, turn_s=170)
    assert plan.fair_share_s == pytest.approx(28 * plan.window_s / 110)
    assert plan.floor_s == pytest.approx(0.3 * plan.fair_share_s)
    assert plan.cap_s == pytest.approx(3.0 * plan.fair_share_s)
    assert plan.floor_s * 110 <= plan.capacity_s
    # keepalive feasibility at the 09-22 turn time: 0.5 * 28 * 420 / 170 = 34 active games, never below the slots
    assert plan.max_active == 34
    assert S.plan_budget(110, 28_000, cfg, turn_s=40).max_active == 110
    assert S.plan_budget(110, 28_000, cfg, turn_s=2000).max_active == 28
    # a slower setup (less wall clock left) shrinks every per-game budget; fewer games grow it
    late = S.plan_budget(110, 20_000, cfg, turn_s=170)
    assert late.fair_share_s < plan.fair_share_s and late.floor_s < plan.floor_s
    assert S.plan_budget(55, 28_000, cfg).fair_share_s == pytest.approx(2 * S.plan_budget(110, 28_000, cfg).fair_share_s)
    # a pinned floor is clipped so that every game can still get it
    assert S.plan_budget(110, 1_000, S.SchedulerConfig(floor_s=5_000)).floor_s <= 0.9 * 28 * 1_000 / 110


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        S.SchedulerConfig.from_mapping({"stagant_turns": 3})
    assert S.SchedulerConfig.from_mapping({"window_s": 5, "slots": 4}).slots == 4


# ------------------------------------------------------------------------------------------------ simulation


@pytest.mark.parametrize("tps", [SLOW, FAST])
def test_110_games_all_started_kept_alive_and_stopped_before_deadline(tps):
    for seed in SEEDS:
        policy = S.ProgressPolicy()
        res = _run(policy, seed, tps)
        core = policy.core
        assert res.started == N_GAMES and not res.unstarted
        assert res.max_idle_s <= 600.0, (seed, res.max_idle_s)
        assert res.last_turn_end <= res.deadline
        assert core.stopped == "deadline"
        assert all(g.state == S.RETIRED and g.retired_at <= core.deadline for g in core.games.values())
        # no turn was granted after deadline - stop_margin
        assert max(g.first_turn_at for g in core.games.values()) <= core.stop_at
        # every game started early enough to get its floor, and got it unless it won first
        floor = core.plan.floor_s
        for g in core.games.values():
            assert g.first_turn_at <= core.deadline - floor
            assert g.won or g.service_s >= floor * 0.98, (g.gid, g.service_s, floor)


@pytest.mark.parametrize("tps", [SLOW, FAST])
def test_more_levels_than_fixed_baselines_under_the_same_throughput(tps):
    ours, wave, stock, rr = [], [], [], []
    for seed in SEEDS:
        fair = SLOTS * WINDOW / N_GAMES
        ours.append(_run(S.ProgressPolicy(), seed, tps))
        wave.append(_run(S.FixedWavePolicy(fair, "wave_fit"), seed, tps))
        stock.append(_run(S.FixedWavePolicy(7920.0, "taaf_stock"), seed, tps))
        rr.append(_run(S.RoundRobinPolicy(), seed, tps))
    # same simulated GPU: every policy gets (nearly) the same number of turns
    turns = [statistics.fmean(r.turns for r in xs) for xs in (ours, wave, stock, rr)]
    assert max(turns) / min(turns) < 1.02, turns
    lv = {name: statistics.fmean(r.levels for r in xs)
          for name, xs in (("ours", ours), ("wave_fit", wave), ("taaf_stock", stock), ("round_robin", rr))}
    assert lv["ours"] > lv["wave_fit"] * 1.02, lv
    assert lv["ours"] > lv["taaf_stock"] * 1.02, lv
    assert lv["ours"] > lv["round_robin"] * 1.02, lv
    assert sum(o.levels >= w.levels for o, w in zip(ours, wave)) >= 4
    assert statistics.fmean(r.score for r in ours) > statistics.fmean(r.score for r in wave)


def test_stock_taaf_cuts_the_last_wave_short_and_the_scheduler_does_not():
    """Coordinator intel (research/10): 110 games x 7,920 s at 28 slots need four waves (31.7 k s) but only ~28 k s
    remain after setup, so the stock last wave is cut short. The scheduler plans from the measured window."""
    tail = 110 - 3 * 28  # the 26 games of the fourth wave (gateway order)
    for seed in range(3):
        games = S.synth_games(N_GAMES, seed)
        ids = [g.gid for g in games]
        stock = S.simulate(games, S.FixedWavePolicy(7920.0, "taaf_stock"), window_s=WINDOW, turns_per_s=SLOW, seed=seed)
        ours = S.simulate(games, S.ProgressPolicy(), window_s=WINDOW, turns_per_s=SLOW, seed=seed)

        def svc(res, gids):
            return statistics.fmean(res.per_game[g]["service_s"] for g in gids)

        head_ids, tail_ids = ids[:-tail], ids[-tail:]
        assert svc(stock, tail_ids) < 0.65 * 7920.0              # the last wave loses > 1/3 of its budget
        assert svc(stock, tail_ids) < 0.7 * svc(stock, head_ids)
        assert svc(ours, tail_ids) >= 0.85 * svc(ours, head_ids)  # no tail penalty
        assert ours.started == N_GAMES
        assert ours.levels > stock.levels


def test_stock_taaf_leaves_games_unplayed_when_slots_or_time_shrink():
    # 16 slots (the community's vLLM sweet spot) or a 20 k s window: stock never starts some games (score 0).
    for slots, window in ((16, 28_000.0), (28, 20_000.0)):
        games = S.synth_games(N_GAMES, 0)
        stock = S.simulate(games, S.FixedWavePolicy(7920.0), slots=slots, window_s=window, turns_per_s=0.3)
        ours = S.simulate(games, S.ProgressPolicy(), slots=slots, window_s=window, turns_per_s=0.3)
        assert stock.started < N_GAMES
        assert ours.started == N_GAMES and ours.max_idle_s <= 600.0
        assert ours.levels > stock.levels


def test_round_robin_over_all_games_breaks_keepalive_at_production_throughput():
    # Why max_active is sized from the turn time: 110 games cycled over 28 slots at ~170 s/turn wait > 10 min.
    res = _run(S.RoundRobinPolicy(), 0, SLOW)
    assert res.max_idle_s > 600.0


def test_level_clear_moves_a_game_to_the_front_and_stagnation_demotes_it():
    cfg = S.SchedulerConfig(slots=1, stagnant_turns=2, hot_turns=3, floor_s=0.0, min_floor_s=0.0, max_active=3)
    core = S.ProgressScheduler(["a", "b", "c"], cfg, now=0.0, deadline=10_000.0)
    t = 0.0

    def turn(expect=None, a_clears=False):
        nonlocal t
        core.tick(t)
        (gid,) = core.dispatch(t)
        if expect is not None:
            assert gid == expect
        t += 10.0
        core.on_turn_end(gid, t, levels=core.games[gid].levels + (1 if a_clears and gid == "a" else 0))
        return gid

    assert [turn() for _ in range(3)] == ["a", "b", "c"]          # first turns in gateway order
    for _ in range(6):                                           # nobody progresses -> all stagnant
        turn()
    assert all(core.klass(g) == S.STAGNANT for g in core.games.values())
    turn("a", a_clears=True)                                     # a clears a level ...
    assert core.klass(core.games["a"]) == S.HOT
    assert turn() == "a"                                         # ... jumps to the front ...
    nxt = [turn(a_clears=i % 2 == 1) for i in range(12)]
    assert nxt.count("a") >= 9                                   # ... and gets the GPU while it keeps clearing


def test_keepalive_preempts_priority():
    cfg = S.SchedulerConfig(slots=1, keepalive_trigger_s=100.0, floor_s=0.0, min_floor_s=0.0, max_active=2,
                            hot_turns=1000)
    core = S.ProgressScheduler(["hot", "cold"], cfg, now=0.0, deadline=100_000.0)
    t = 0.0
    served = []
    for i in range(40):
        core.tick(t)
        (gid,) = core.dispatch(t)
        served.append((t, gid))
        t += 30.0
        core.on_turn_end(gid, t, levels=i + 1 if gid == "hot" else 0)  # "hot" clears a level every turn
    cold = [s for s, g in served if g == "cold"]
    gaps = [b - a for a, b in zip(cold, cold[1:])]
    assert cold and max(gaps) <= 100.0 + 2 * 30.0  # served within one turn of crossing the trigger


# ------------------------------------------------------------------------------------------------ threaded gate


def test_gate_threads_respect_slots_and_stop_cleanly():
    cfg = S.SchedulerConfig(slots=3, stop_margin_s=0.2, keepalive_trigger_s=0.3, min_floor_s=0.01, turn_estimate_s=0.02,
                            status_every_s=0.5)
    ids = [f"g{i}" for i in range(12)]
    lines: list[str] = []
    gate = S.SchedulerGate(ids, cfg, deadline_in_s=1.5, log=lines.append, poll_s=0.02)
    lock = threading.Lock()
    live = {"n": 0, "max": 0, "turns": 0, "after_stop": 0}
    stop_at = time.monotonic() + 1.5 - 0.2

    def worker(gid: str, seed: int) -> None:
        rng = random.Random(seed)
        levels = 0
        while True:
            grant = gate.acquire(gid)
            if not grant.ok:
                return
            with lock:
                live["n"] += 1
                live["max"] = max(live["max"], live["n"])
                live["turns"] += 1
                if time.monotonic() > stop_at + 0.05:
                    live["after_stop"] += 1
            time.sleep(rng.uniform(0.005, 0.03))
            if rng.random() < 0.1:
                levels += 1
            with lock:
                live["n"] -= 1
            gate.release(gid, levels=levels)

    threads = [threading.Thread(target=worker, args=(gid, i)) for i, gid in enumerate(ids)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=5.0)
    assert not any(th.is_alive() for th in threads)
    assert live["max"] <= 3 and live["turns"] > 12 and live["after_stop"] == 0
    summary = gate.summary()
    assert all(g["first_turn_s"] is not None for g in summary["games"].values())
    assert all(g["state"] == "retired" for g in summary["games"].values())
    assert any(line.startswith("HYDRA_SCHED start") for line in lines)
    assert summary["stopped"] == "deadline"


def test_gate_shutdown_wakes_parked_games():
    gate = S.SchedulerGate(["a", "b"], S.SchedulerConfig(slots=1), deadline_in_s=3600, log=None, poll_s=5.0)
    assert gate.acquire("a").ok
    result = {}
    th = threading.Thread(target=lambda: result.setdefault("b", gate.acquire("b")))
    th.start()
    time.sleep(0.1)
    gate.shutdown("cancelled")
    th.join(timeout=2.0)
    assert not th.is_alive() and result["b"].ok is False and result["b"].reason == "cancelled"
    assert gate.is_retired("a")  # its running turn winds down through should_stop


# ------------------------------------------------------------------------------------------------ Duck hook


@pytest.fixture()
def duck():
    pytest.importorskip("arcengine")
    pytest.importorskip("arc_agi")
    import duck_offline

    if not Path(duck_offline.default_env_dir()).is_dir():
        pytest.skip("offline environment_files not found")
    return duck_offline


def _run_duck(duck, tmp_path, monkeypatch, *, enabled: bool, games=("ls20", "vc33", "ft09", "ar25"), steps=6):
    import asyncio

    monkeypatch.setenv("DUCK_PATCH_B08_SCHEDULER", "1" if enabled else "0")
    monkeypatch.setenv("DUCK_PATCH_B08_SCHEDULER_CONFIG",
                       json.dumps({"window_s": 600, "stop_margin_s": 5, "status_every_s": 1}))
    env_dir = duck.default_env_dir()
    ids = duck.resolve_game_ids(list(games), env_dir)
    (tmp_path / "git_status.txt").write_text("test\n")
    lock = threading.Lock()
    stats = {"now": 0, "max": 0, "order": []}
    with duck.patch_profile("patched"), duck.MockLLMServer(duck.ScriptedPolicy(seed=0)) as server:
        base = duck.make_mock_analyzer_factory(server, {})

        def factory(game, index):
            agent = base(game, index)
            orig = agent.analyze

            def analyze(*args, **kwargs):
                with lock:
                    stats["now"] += 1
                    stats["max"] = max(stats["max"], stats["now"])
                    stats["order"].append(game.game_run.game_id)
                try:
                    return orig(*args, **kwargs)
                finally:
                    with lock:
                        stats["now"] -= 1

            agent.analyze = analyze
            return agent

        solver = duck.shipped_solver(max_actions_per_game=steps, max_runtime_s_per_game=600.0, concurrency=2,
                                     analyzer_factory=factory)
        bm = duck.build_benchmark(ids, job_dir=tmp_path, solver=solver, env_dir=env_dir)
        asyncio.run(bm.run(soft_end_time=None, runtime_environment=None, minimal_diagnostics=True))
    return bm, stats, ids


def test_hook_gates_the_real_duck_loop(duck, tmp_path, monkeypatch):
    bm, stats, ids = _run_duck(duck, tmp_path, monkeypatch, enabled=True)
    assert stats["max"] <= 2                                   # slots = HarnessSolver.concurrency
    runs = {r.game_id: r for r in bm.game_runs}
    assert all(len(runs[g].history) == 6 and runs[g].state == "gave_up" for g in ids)
    # all four games were playing before the first one finished (turn-level interleaving, not waves)
    first_seen = [stats["order"].index(g) for g in ids]
    assert max(first_seen) < len(stats["order"]) // 2
    summary = json.loads((tmp_path / "hydra_scheduler.json").read_text())
    assert summary["plan"]["slots"] == 2 and summary["plan"]["n_games"] == 4
    assert all(g["first_turn_s"] is not None and g["state"] == "retired" for g in summary["games"].values())
    # per-game state isolation still holds (MISTAKES #2): one transcript per game, no shared prompt log
    assert sorted(p.stem for p in (tmp_path / "transcripts").glob("*.txt")) == sorted(f"{g}_p0" for g in ids)


def test_hook_off_keeps_the_stock_loop(duck, tmp_path, monkeypatch):
    bm, stats, ids = _run_duck(duck, tmp_path, monkeypatch, enabled=False, games=("ls20", "vc33", "ft09"), steps=4)
    assert not (tmp_path / "hydra_scheduler.json").exists()
    assert all(len(r.history) == 4 for r in bm.game_runs)
    assert stats["max"] <= 3
