#!/usr/bin/env python3
"""eval_perception_actions.py - offline evaluation of agent/perception.py (B04) and agent/actions.py (B05)
on the 25 public games (random/scripted rollouts via agent/arcenv.py) and on the arcprize.org replays in the
repo root (*.json: 5 frontier-model + 11 human recordings). Writes analysis/02_perception_actions_eval.md.

    SCRATCH/venv/bin/python tools/eval_perception_actions.py [--steps 300] [--games ls20 ft09] [--out FILE]

The labels below (HUD_GT) are EVALUATION ground truth only: the agent code never sees game ids.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

import perception as P  # noqa: E402
from actions import ActionChannel  # noqa: E402

try:
    from arcenv import ArcEnv, list_games  # noqa: E402
    HAVE_ENV = True
except Exception:  # pragma: no cover
    HAVE_ENV = False

# Per-action budget/timer bar of each public game: (rows, cols) whose cells the bar occupies.
# Read off offline probes (random rollouts, cells that flip with one fixed transition per action) and
# cross-checked with research/20_game_mechanics.md. Evaluation labels only.
HUD_GT: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "ar25": ((), (63,)), "bp35": ((63,), ()), "cd82": ((63,), ()), "cn04": ((0,), ()), "dc22": ((63,), ()),
    "ft09": ((63,), ()), "g50t": ((63,), ()), "ka59": ((63,), ()), "lf52": ((0,), ()), "lp85": ((), (0,)),
    "ls20": ((61, 62), ()), "m0r0": ((0, 63), ()), "r11l": ((), (0,)), "re86": ((63,), ()), "s5i5": ((63,), ()),
    "sb26": ((53,), ()), "sc25": ((), (62, 63)), "sk48": ((53,), ()), "sp80": ((0,), ()), "su15": ((63,), ()),
    "tn36": ((1,), ()), "tr87": ((63,), ()), "tu93": ((63,), ()), "vc33": ((0,), ()), "wa30": ((63,), ()),
}
# games whose replay recording uses the same game version as the local environment files
VERSION_MATCHED = {"ar25", "ft09", "lp85", "r11l", "s5i5", "cd82", "ls20"}


def gt_hud_mask(game: str) -> np.ndarray:
    rows, cols = HUD_GT[game[:4]]
    m = np.zeros((64, 64), bool)
    for r in rows:
        m[r, :] = True
    for c in cols:
        m[:, c] = True
    return m


# ------------------------------------------------------------------------------------------------
# replays
# ------------------------------------------------------------------------------------------------
def load_replay(path: str) -> dict:
    """Normalise one arcprize.org replay (both JSONL formats) into
    {game, version, source, steps: [{aid, x, y, frame, n_frames, levels, state}]} (step 0 = initial frame)."""
    steps, game_id, source = [], None, "human"
    for line in open(path):
        rec = json.loads(line)
        d = rec.get("data", rec)
        game_id = d.get("game_id", game_id)
        ai = d.get("action_input") or {}
        if ai.get("reasoning"):
            source = "model"
        aid = ai.get("id")
        if aid is None:
            aid = -1
        elif isinstance(aid, str):
            aid = 0 if aid == "RESET" else int(aid.replace("ACTION", ""))
        data = ai.get("data") or {}
        fr = d.get("frame") or []
        steps.append({"aid": int(aid), "x": data.get("x"), "y": data.get("y"),
                      "frame": np.asarray(fr[-1], dtype=np.int16) if fr else None, "n_frames": len(fr),
                      "frames": fr, "levels": int(d.get("levels_completed", d.get("score")) or 0),
                      "state": d.get("state")})
    g, _, ver = (game_id or os.path.basename(path)[:4]).partition("-")
    return {"game": g, "version": ver, "source": source, "path": path, "steps": steps}


def load_replays(root: str = ROOT) -> list[dict]:
    return [load_replay(p) for p in sorted(glob.glob(os.path.join(root, "*-*-*-*-*-*.json")))]


def replay_action(st: dict) -> Any:
    return (6, st["x"], st["y"]) if st["aid"] == 6 else st["aid"]


# ------------------------------------------------------------------------------------------------
# click-candidate recall on replays
# ------------------------------------------------------------------------------------------------
def replay_click_ranks(rep: dict) -> list[dict]:
    """For every ACTION6 of a replay: rank of the clicked object among the candidates computed from the
    pre-click frame (perceiver state accumulated along the replay, i.e. with dead-signature suppression)
    and without history; whether the click was effective and whether it completed a level."""
    per = P.Perceiver()
    out = []
    prev = None
    lv = 0
    for st in rep["steps"]:
        if st["frame"] is None:
            continue
        if prev is None:
            per.reset(st["frame"], st["levels"])
            prev, lv = st["frame"], st["levels"]
            continue
        a = replay_action(st)
        info = None
        if st["aid"] == 6 and st["x"] is not None:
            seg = per.seg(prev)
            comp = seg.comp_at(int(st["x"]), int(st["y"]))
            cands = per.candidates(prev)
            fresh = P.rank_click_candidates(seg, per.hud_mask())
            info = {"bg": comp is None or comp.bg, "rank": None, "rank_fresh": None, "type_rank": None, "shown": False,
                    "n_cands": len(cands), "n_types": len({c.ctype for c in cands})}
            if comp is not None and not comp.bg:
                grp = seg.group_of(comp.id)
                for i, c in enumerate(cands):
                    if comp.id in c.comp.members:
                        info["rank"] = i
                        break
                for i, c in enumerate(fresh):
                    if comp.id in c.comp.members:
                        info["rank_fresh"] = i
                        break
                types = []
                for c in cands:
                    if c.ctype not in types:
                        types.append(c.ctype)
                info["type_rank"] = types.index(grp.ctype) if grp.ctype in types else None
                info["shown"] = grp.ctype in P.shown_types(cands)
        pc = per.observe(prev, st["frames"] or [st["frame"]], a, level=st["levels"], state=st["state"])
        if info is not None:
            info["effective"] = (not pc.noop) or st["levels"] > lv
            info["level_win"] = st["levels"] > lv
            out.append(info)
        prev, lv = st["frame"], st["levels"]
    return out


def recall_at(ranks: list[Optional[int]], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r < k) / max(1, len(ranks))


# ------------------------------------------------------------------------------------------------
# rollouts (random / scripted) on the offline env
# ------------------------------------------------------------------------------------------------
_SPRITE_UID = __import__("itertools").count(1)


def _sprite_uid(s) -> int:
    """Stable ground-truth identity for an engine sprite. `id(s)` is NOT stable: sprites are re-created on level
    reset (and spawned/removed in e.g. g50t), a freed address can be reused by a different sprite, and the reuse
    pattern varies across processes — that silently merged the movement histories of unrelated sprites and made the
    avatar GT (and the avatar test) depend on PYTHONHASHSEED."""
    uid = getattr(s, "_eval_gt_uid", None)
    if uid is None:
        uid = next(_SPRITE_UID)
        try:
            object.__setattr__(s, "_eval_gt_uid", uid)
        except Exception:
            return id(s)
    return uid


def _sprites(env) -> dict[int, tuple]:
    try:
        return {_sprite_uid(s): (s.x, s.y, s.width, s.height) for s in env.game.current_level.get_sprites()}
    except Exception:
        return {}


def _sprite_display_bbox(env, sid_info) -> tuple[int, int, int, int]:
    cam = env.game.camera
    s, xo, yo = cam._calculate_scale_and_offset()
    x, y, w, h = sid_info
    c0 = (x - cam.x) * s + xo
    r0 = (y - cam.y) * s + yo
    return r0, c0, r0 + h * s - 1, c0 + w * s - 1


def _changed_outside(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> bool:
    return bool(((a != b) & ~m).any())


def rollout_eval(game: str, n_steps: int = 300, seed: int = 0, click_every: int = 12, k_eval: int = 10,
                 prefix: Optional[list] = None) -> dict:
    """One guarded rollout of a mixed random policy (arrows 70% when available, clicks on ranked candidates
    or random pixels, occasional voluntary RESET) with every metric collected on the way."""
    rng = random.Random(seed)
    env = ArcEnv(game, competition=True)
    ch = ActionChannel(env)
    per = ch.P
    gtm = gt_hud_mask(game)
    if prefix:
        for a in prefix:
            ch.step(a, force=True)
    res: dict[str, Any] = defaultdict(float)
    res["game"] = game
    res["blocked"] = 0
    toks, ms = [], []
    hud_changed_gt = hud_masked_gt = masked_total = masked_outside = 0
    gt_noop_steps = our_noop_on_gt = our_noop_steps = false_noop = 0
    gt_ev: dict[int, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    avatar_hits = avatar_checks = 0
    first_avatar = None
    click_stats = defaultdict(list)
    guard_checked = guard_false = 0
    gameovers, replays_ok, replays_n, replay_saved = 0, 0, 0, []
    kb = [a for a in (1, 2, 3, 4) if a in ch.obs.available_actions]
    others = [a for a in ch.obs.available_actions if a not in (1, 2, 3, 4)]
    for t in range(n_steps):
        if ch.state == "WIN":
            break
        if ch.state == "GAME_OVER":
            gameovers += 1
            ch.step(0)
            la = ch.last_attempt()
            b = ch.replay_prefix()
            if la is not None and la.end == "game_over":
                replays_n += 1
                replays_ok += (not b.halted) or b.reason.endswith("level completed")
                replay_saved.append((len(la.steps) - 1, b.executed))
            continue
        # -- choose an action ------------------------------------------------------------------
        if kb and rng.random() < 0.7:
            act: Any = rng.choice(kb)
        elif others:
            a = rng.choice(others)
            if a == 6:
                cands = per.candidates(ch.frame)
                if cands and rng.random() < 0.75:
                    c = cands[min(len(cands) - 1, int(rng.expovariate(0.35)))]
                    act = (6, c.x, c.y)
                else:
                    act = (6, rng.randrange(64), rng.randrange(64))
            elif a == 7 and rng.random() < 0.5:
                act = rng.choice(kb) if kb else 7
            else:
                act = a
        else:
            act = rng.choice(kb)
        if rng.random() < 0.02 and ch.since_reset > 3:
            act = 0
        # -- click-candidate evaluation against engine-valid clicks (on clones) -------------------
        if 6 in ch.obs.available_actions and t % click_every == 0:
            _click_eval(env, per, gtm, rng, click_stats, k_eval)
        # -- guard safety: verify a refusal on a clone ---------------------------------------------
        g = ch.check_guards(act)
        if g in ("known no-op here", "known GAME_OVER transition"):
            c = env.clone()
            a0 = P.norm_action(act)
            o = c.step(6, x=a0[1], y=a0[2]) if a0[0] == 6 else c.step(a0[0])
            guard_checked += 1
            if g == "known no-op here":
                # a refused no-op that would have ended the game (budget exhausted) is a saving, not an error
                bad = o.levels_completed != env.levels_completed or (
                    o.state != "GAME_OVER" and _changed_outside(env.frame, o.frame, gtm))
            else:
                bad = o.state != "GAME_OVER"
            guard_false += bool(bad)
            if bad and os.environ.get("EVAL_DEBUG"):
                print("FALSE REFUSAL", game, t, g, act, np.argwhere((env.frame != o.frame) & ~gtm)[:8].tolist(),
                      o.state, len(o.frames))
        prev = ch.frame.copy()
        prev_lv = ch.level
        spr_before = _sprites(env)
        t0 = time.perf_counter()
        r = ch.step(act)
        if r.blocked or not r.executed:
            res["blocked"] += 1
            continue
        ms.append((time.perf_counter() - t0) * 1000)
        pc = r.percept
        toks.append(P.approx_tokens(pc.text))
        cur = ch.frame
        aid = P.norm_action(act)[0]
        if aid == 0 or ch.level != prev_lv or cur is None:
            continue
        # -- HUD metrics ----------------------------------------------------------------------------
        ch_mask = prev != cur
        m = per.hud_mask()
        hud_changed_gt += int((ch_mask & gtm).sum())
        hud_masked_gt += int((ch_mask & gtm & m).sum())
        masked_total += int((ch_mask & m).sum())
        masked_outside += int((ch_mask & m & ~gtm).sum())
        gt_noop = not (ch_mask & ~gtm).any()
        gt_noop_steps += gt_noop
        our_noop_steps += pc.noop
        our_noop_on_gt += gt_noop and pc.noop
        false_noop += pc.noop and not gt_noop
        # -- avatar ground truth from engine sprites --------------------------------------------------
        if aid in (1, 2, 3, 4):
            spr_after = _sprites(env)
            for sid, (x0, y0, w0, h0) in spr_before.items():
                if sid in spr_after:
                    x1, y1 = spr_after[sid][:2]
                    if (x1, y1) != (x0, y0):
                        gt_ev[sid][aid][(int(np.sign(x1 - x0)), int(np.sign(y1 - y0)))] += 1
            gt_ids = _gt_avatars(gt_ev)
            ours = per.avatar.groups(per.seg(cur))
            if ours and first_avatar is None:
                first_avatar = t
            boxes = [_sprite_display_bbox(env, spr_after[s]) for s in gt_ids if s in spr_after]
            if boxes and ours:  # (sprite objects are re-created on level reset: no GT until re-learned)
                avatar_checks += 1
                avatar_hits += any(_overlap(bb, ob) for _, ob in ours for bb in boxes)
    acc = ch.accounting()
    res.update({
        "steps": len(toks), "tokens_p50": statistics.median(toks) if toks else 0,
        "tokens_p95": float(np.percentile(toks, 95)) if toks else 0, "tokens_max": max(toks) if toks else 0,
        "ms_p50": statistics.median(ms) if ms else 0,
        "hud_recall": hud_masked_gt / hud_changed_gt if hud_changed_gt else None,
        "hud_changed_gt": hud_changed_gt,
        "hud_fp_rate": masked_outside / masked_total if masked_total else 0.0, "hud_masked": masked_total,
        "hud_bars": per.hud.describe(),
        "noop_recall": our_noop_on_gt / gt_noop_steps if gt_noop_steps else None, "gt_noop_steps": gt_noop_steps,
        "false_noop": false_noop, "our_noop_steps": our_noop_steps,
        "avatar_gt": bool(_gt_avatars(gt_ev)), "avatar_acc": avatar_hits / avatar_checks if avatar_checks else None,
        "avatar_checks": avatar_checks, "avatar_first": first_avatar, "avatar_text": per.avatar_text(ch.frame) if ch.frame is not None else "",
        "scale": per.scale(), "camera": tuple(int(v) for v in env.game.camera._calculate_scale_and_offset()),
        "guard_checked": guard_checked, "guard_false": guard_false, "blocked_by": acc["blocked_by"],
        "accounting_ok": acc["total"] == env.n_actions and acc["level_actions"] == env.level_actions,
        "actions": acc["total"], "levels": ch.level, "gameovers": gameovers, "replays_n": replays_n,
        "replays_ok": replays_ok, "replay_saved": replay_saved,
    })
    for k, v in click_stats.items():
        res["click_" + k] = v
    return dict(res)


def _gt_avatars(gt_ev) -> list[int]:
    scored = []
    for sid, per_a in gt_ev.items():
        moves = sum(sum(c.values()) for c in per_a.values())
        cons = sum(c.most_common(1)[0][1] for c in per_a.values())
        if moves >= 3 and cons >= 0.75 * moves:
            scored.append((cons, sid))
    if not scored:
        return []
    top = max(s for s, _ in scored)
    return [sid for s, sid in scored if s >= 0.5 * top]


def _overlap(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _click_eval(env, per, gtm, rng, stats, k_eval: int) -> None:
    """Engine-valid clicks (offline oracle) vs our ranked candidates, both executed on clones."""
    base = env.frame
    vc = sorted(set(env.valid_clicks()))
    if len(vc) > 40:
        vc = rng.sample(vc, 40)
    seg = per.seg(base)
    cands = per.candidates(base)

    def effect(x, y):
        c = env.clone()
        o = c.step(6, x=x, y=y)
        if o.frame is None:
            return False, b""
        eff = o.levels_completed != env.levels_completed or o.state != env.state or _changed_outside(base, o.frame, gtm)
        g = o.frame.astype(np.int8).copy()
        g[gtm] = -1
        return eff, g.tobytes() + bytes([o.levels_completed]) + o.state.encode()

    gt_eff = []
    for x, y in vc:
        eff, key = effect(x, y)
        if eff:
            comp = seg.comp_at(x, y)
            gt_eff.append((comp, key))
    if not gt_eff:
        return
    top = cands[:max(k_eval, 20)]
    cand_eff = [effect(c.x, c.y) for c in top]
    for comp, key in gt_eff:
        rank = None
        if comp is not None:
            rank = next((i for i, c in enumerate(cands) if comp.id in c.comp.members), None)
        stats["obj_rank"].append(rank)
        stats["bg_target"].append(comp is None or comp.bg)
        eff_rank = next((i for i, (e, kk) in enumerate(cand_eff) if e and kk == key), None)
        stats["effect_rank"].append(eff_rank)
    stats["precision"].append(sum(1 for e, _ in cand_eff[:k_eval] if e) / max(1, min(k_eval, len(cand_eff))))
    # distinct effects reachable from the top-k vs from all valid clicks
    stats["distinct_gt"].append(len({k for _, k in gt_eff}))
    stats["distinct_topk"].append(len({kk for e, kk in cand_eff[:k_eval] if e}))


# ------------------------------------------------------------------------------------------------
# replays through the action channel (version-matched games only)
# ------------------------------------------------------------------------------------------------
def guarded_replay(rep: dict) -> dict:
    """Feed a whole recording through a guarded channel (NORMAL-mode env, as recorded online). Refused
    actions are skipped. Reports whether every recorded level completion is still reached and how many
    actions the guards saved; recorded full resets are forced so the trajectory stays aligned."""
    env = ArcEnv(rep["game"], competition=False)
    ch = ActionChannel(env, competition=False)
    gtm = gt_hud_mask(rep["game"])
    steps = [s for s in rep["steps"][1:] if s["aid"] >= 0]
    target = max(s["levels"] for s in rep["steps"])
    blocked, diverged, charged = Counter(), 0, 0
    for i, st in enumerate(steps):
        a = replay_action(st)
        force = False
        if st["aid"] == 0:
            nxt = st["levels"]
            force = nxt < ch.level  # the recording did a full reset: mirror it
        r = ch.step(a, force=force)
        if r.blocked:
            blocked[r.blocked] += 1
            continue
        charged += r.charged
        if ch.frame is not None and st["frame"] is not None and st["levels"] == ch.level and \
                _changed_outside(ch.frame, st["frame"], gtm):
            diverged += 1
    return {"game": rep["game"], "source": rep["source"], "recorded_actions": len(steps),
            "levels_target": target, "levels_reached": max(ch.level, ch.obs.levels_completed),
            "blocked": sum(blocked.values()), "blocked_by": dict(blocked), "diverged_steps": diverged}


def prefix_win_eval(rep: dict) -> list[dict]:
    """For every level won in a version-matched recording: play up to the level, play its winning attempt
    minus the last action, RESET (level reset), replay_prefix(), then the last action -> must win."""
    steps = [s for s in rep["steps"][1:] if s["aid"] >= 0]
    # split into (level, attempt start index, completion index)
    wins = []
    lv, att_start = 0, 0
    for i, st in enumerate(steps):
        if st["aid"] == 0:
            att_start = i + 1
            if st["levels"] < lv:
                lv = st["levels"]
            continue
        if st["levels"] > lv:
            wins.append((lv, att_start, i))
            lv, att_start = st["levels"], i + 1
    out = []
    for level, s0, s1 in wins:
        env = ArcEnv(rep["game"], competition=False)
        ch = ActionChannel(env, competition=False)
        for st in steps[:s0]:
            ch.step(replay_action(st), force=True)
        if ch.level != level:
            out.append({"level": level, "ok": False, "why": "could not reach level"})
            continue
        att = [replay_action(st) for st in steps[s0:s1 + 1] if st["aid"] != 0]
        for a in att[:-1]:
            ch.step(a, force=True)
        if len(att) <= 1:
            out.append({"level": level, "ok": None, "why": "one-action level", "attempt": len(att)})
            continue
        before = env.n_actions
        rr = ch.step(0)
        b = ch.replay_prefix()
        last = ch.step(att[-1], force=True)
        ok = (not b.halted) and ch.level == level + 1
        out.append({"level": level, "ok": ok, "attempt": len(att), "replayed": b.executed,
                    "why": b.reason if b.halted else "", "reset_charged": rr.charged,
                    "charged": env.n_actions - before})
    return out


# ------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------
def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def run_all(games: list[str], n_steps: int, seed: int) -> dict:
    t0 = time.time()
    out = {"rollouts": [], "replay_clicks": [], "guarded": [], "prefix": []}
    for g in games:
        r = rollout_eval(g, n_steps=n_steps, seed=seed)
        out["rollouts"].append(r)
        print(f"[rollout] {g} {time.time() - t0:.0f}s hud_rec={_pct(r['hud_recall'])} fp={_pct(r['hud_fp_rate'])} "
              f"avatar={_pct(r['avatar_acc'])} acct={r['accounting_ok']} guard_false={r['guard_false']}", flush=True)
    for rep in load_replays():
        if rep["game"] not in games:
            continue
        out["replay_clicks"].append({"game": rep["game"], "source": rep["source"], "ranks": replay_click_ranks(rep)})
        if rep["game"] in VERSION_MATCHED:
            out["guarded"].append(guarded_replay(rep))
            out["prefix"].append({"game": rep["game"], "source": rep["source"], "levels": prefix_win_eval(rep)})
        print(f"[replay] {rep['game']} {time.time() - t0:.0f}s", flush=True)
    out["secs"] = time.time() - t0
    return out


def write_report(res: dict, path: str, n_steps: int, seed: int) -> None:
    L = []
    w = L.append
    ro = res["rollouts"]
    w("# 02 - Perception pack (B04) and action channel (B05): offline evaluation")
    w("")
    w(f"_Generated by `tools/eval_perception_actions.py --steps {n_steps} --seed {seed}` "
      f"({res['secs']:.0f} s on CPU). Code: `agent/perception.py`, `agent/actions.py`; tests: "
      f"`tests/test_perception.py`, `tests/test_actions.py`._")
    w("")
    w("Everything is game-agnostic: the modules never see a game id. The per-game HUD labels used below "
      "(`HUD_GT` in the eval tool) are evaluation ground truth only.")
    w("")
    # ---- headline -----------------------------------------------------------------------------
    rc_all = [x for r in res["replay_clicks"] for x in r["ranks"]]
    eff = [x for x in rc_all if x["effective"]]
    eff_obj = [x for x in eff if not x["bg"]]
    wins = [x for x in rc_all if x["level_win"]]
    wins_obj = [x for x in wins if not x["bg"]]
    obj_ranks = [r for rr in ro for r in rr.get("click_obj_rank", [])]
    eff_ranks = [r for rr in ro for r in rr.get("click_effect_rank", [])]
    kb_games = [r for r in ro if r["avatar_gt"]]
    av = [r["avatar_acc"] for r in kb_games if r["avatar_acc"] is not None]
    hud_rec = [r["hud_recall"] for r in ro if r["hud_recall"] is not None]
    fp_num = sum(r["hud_fp_rate"] * r["hud_masked"] for r in ro)
    fp_den = sum(r["hud_masked"] for r in ro)
    pw = [lv for p in res["prefix"] for lv in p["levels"] if lv["ok"] is not None]
    w("## Headline metrics")
    w("")
    w("| metric | value |")
    w("|---|---|")
    w(f"| click recall, winning clicks in replays (clicked object in top-5 / top-10 / anywhere in list), "
      f"object clicks | {_pct(recall_at([x['rank'] for x in wins_obj], 5))} / {_pct(recall_at([x['rank'] for x in wins_obj], 10))} / "
      f"{_pct(recall_at([x['rank'] for x in wins_obj], 10 ** 6))} of {len(wins_obj)} level-completing clicks "
      f"({len(wins) - len(wins_obj)} more hit empty space) |")
    w(f"| click recall, all effective replay clicks on objects: top-5 / top-10 / top-20 / anywhere | "
      f"{_pct(recall_at([x['rank'] for x in eff_obj], 5))} / {_pct(recall_at([x['rank'] for x in eff_obj], 10))} / "
      f"{_pct(recall_at([x['rank'] for x in eff_obj], 20))} / {_pct(recall_at([x['rank'] for x in eff_obj], 10 ** 6))} "
      f"of {len(eff_obj)} ({len(eff) - len(eff_obj)} effective clicks on empty space) |")
    w(f"| same, type-level (clicked object's type among the first 5 / 10 types) | "
      f"{_pct(recall_at([x['type_rank'] for x in eff_obj], 5))} / {_pct(recall_at([x['type_rank'] for x in eff_obj], 10))} |")
    w(f"| engine-valid effective clicks (rollouts, 25 games): object in top-10 / top-20 / list | "
      f"{_pct(recall_at(obj_ranks, 10))} / {_pct(recall_at(obj_ranks, 20))} / {_pct(recall_at(obj_ranks, 10 ** 6))} of {len(obj_ranks)} |")
    w(f"| same, *effect* reproduced by a top-10 / top-20 candidate click (same next frame) | "
      f"{_pct(recall_at(eff_ranks, 10))} / {_pct(recall_at(eff_ranks, 20))} |")
    w(f"| avatar detection accuracy, games with an engine-verified controlled sprite ({len(kb_games)}) | "
      f"mean {_pct(statistics.mean(av) if av else None)}; {sum(1 for a in av if a >= 0.8)}/{len(av)} games >= 80% |")
    w(f"| HUD/timer mask: recall of per-action bar cells (mean over games) | {_pct(statistics.mean(hud_rec) if hud_rec else None)} |")
    w(f"| HUD mask false-positive rate (masked changed cells outside the true bar) | {_pct(fp_num / fp_den if fp_den else 0)} "
      f"({int(fp_num)} of {fp_den} cells) |")
    w(f"| false no-op calls (we said 'no effect' but non-HUD cells changed) | "
      f"{sum(r['false_noop'] for r in ro)} of {sum(r['our_noop_steps'] for r in ro)} no-op calls |")
    w(f"| describe() length, approx. tokens (digits counted singly) p50 / p95 / max | "
      f"{statistics.median([r['tokens_p50'] for r in ro]):.0f} / {max(r['tokens_p95'] for r in ro):.0f} / "
      f"{max(r['tokens_max'] for r in ro)} |")
    w(f"| action accounting == official scorecard (random rollouts incl. RESET/GAME_OVER) | "
      f"{sum(r['accounting_ok'] for r in ro)}/{len(ro)} games |")
    w(f"| guard refusals verified on clones (would the refused action have changed anything?) | "
      f"{sum(r['guard_false'] for r in ro)} false of {sum(r['guard_checked'] for r in ro)} checked |")
    rn, rk = sum(r["replays_n"] for r in ro), sum(r["replays_ok"] for r in ro)
    w(f"| replay_prefix after GAME_OVER reproduces the recorded prefix (rollouts) | {rk}/{rn} |")
    w(f"| replay_prefix + last action re-wins a recorded level (version-matched replays) | "
      f"{sum(1 for x in pw if x['ok'])}/{len(pw)} levels |")
    gr = res["guarded"]
    w(f"| guarded re-play of whole recordings: same levels reached / actions saved by guards | "
      f"{sum(1 for g in gr if g['levels_reached'] >= g['levels_target'])}/{len(gr)} recordings; "
      f"{sum(g['blocked'] for g in gr)} of {sum(g['recorded_actions'] for g in gr)} recorded actions refused |")
    w(f"| perception cost | {statistics.median([r['ms_p50'] for r in ro]):.1f} ms per step (median; perceive + guard) |")
    w("")
    # ---- per-game rollouts -------------------------------------------------------------------------
    w("## Per-game rollouts (random mixed policy, guarded channel)")
    w("")
    w(f"{n_steps} policy steps per game, seed {seed}: arrows 70 % when available, otherwise clicks "
      "(75 % on a ranked candidate, 25 % random pixel) / ACTION5 / UNDO, 2 % voluntary RESET. GAME_OVER is "
      "followed by RESET + `replay_prefix()`. HUD recall/FP use the labelled bar lines; 'false no-op' = we "
      "called a step 'no effect' although a non-bar cell changed.")
    w("")
    w("| game | scale (ours / camera) | bars found | HUD recall | HUD FP | no-op recall | false no-op | avatar acc (checks) | tokens p50/max | acct | refused (false) | GAME_OVER / prefix ok |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ro:
        sc = r["scale"]
        cam = r["camera"]
        w(f"| {r['game']} | {sc[0]}@{sc[2]},{sc[1]} / {cam[0]}@{cam[1]},{cam[2]} | {r['hud_bars'] or '-'} | "
          f"{_pct(r['hud_recall'])} | {_pct(r['hud_fp_rate'])} | {_pct(r['noop_recall'])} | {r['false_noop']} | "
          f"{_pct(r['avatar_acc'])} ({r['avatar_checks']}) | {r['tokens_p50']:.0f}/{r['tokens_max']} | "
          f"{'ok' if r['accounting_ok'] else 'MISMATCH'} | {int(r['blocked'])} ({r['guard_false']}) | "
          f"{r['gameovers']} / {r['replays_ok']}/{r['replays_n']} |")
    w("")
    w("Scale column: `s@x,y` = cell size and grid offset. The camera value is the engine's upscale factor; "
      "games that draw at 1-px resolution inside a scaled camera (ar25 overlays, bp35/lf52 custom renderers) "
      "correctly fall back to 1, and tu93 draws its content on a 3-px grid inside a 1x camera.")
    w("")
    # ---- click candidates -------------------------------------------------------------------------
    w("## Click candidates")
    w("")
    w("### Replays (clicked object's rank in the candidate list built from the pre-click frame)")
    w("")
    w("`rank` uses the perceiver state accumulated along the replay (dead-signature suppression on); `fresh` "
      "ranks without any click history. Effective = the click changed something outside the HUD.")
    w("")
    w("| game | source | clicks | effective | on empty space | eff. obj top-5 | top-10 | top-20 | any | fresh top-10 | level-winning clicks (obj top-10) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["replay_clicks"]:
        rk = r["ranks"]
        if not rk:
            w(f"| {r['game']} | {r['source']} | 0 | - | - | - | - | - | - | - | - |")
            continue
        e = [x for x in rk if x["effective"]]
        eo = [x for x in e if not x["bg"]]
        wn = [x for x in rk if x["level_win"]]
        wno = [x for x in wn if not x["bg"]]
        w(f"| {r['game']} | {r['source']} | {len(rk)} | {len(e)} | {len(e) - len(eo)} | "
          f"{_pct(recall_at([x['rank'] for x in eo], 5))} | {_pct(recall_at([x['rank'] for x in eo], 10))} | "
          f"{_pct(recall_at([x['rank'] for x in eo], 20))} | {_pct(recall_at([x['rank'] for x in eo], 10 ** 6))} | "
          f"{_pct(recall_at([x['rank_fresh'] for x in eo], 10))} | "
          f"{sum(1 for x in wno if x['rank'] is not None and x['rank'] < 10)}/{len(wno)} (+{len(wn) - len(wno)} empty) |")
    w("")
    w("### Engine-valid clicks on rollout states (offline oracle `valid_clicks()`, executed on clones)")
    w("")
    w("Only clicks that change something count. `effect@k`: some top-k candidate click produces exactly the same "
      "next frame. `precision@10`: share of the top-10 candidates whose click changes something.")
    w("")
    w("| game | states | effective valid clicks | on empty space | obj top-10 | obj top-20 | obj any | effect@10 | effect@20 | precision@10 | distinct effects top-10 / all |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ro:
        orr = r.get("click_obj_rank")
        if not orr:
            continue
        er = r["click_effect_rank"]
        w(f"| {r['game']} | {len(r['click_precision'])} | {len(orr)} | {sum(r['click_bg_target'])} | "
          f"{_pct(recall_at(orr, 10))} | {_pct(recall_at(orr, 20))} | {_pct(recall_at(orr, 10 ** 6))} | "
          f"{_pct(recall_at(er, 10))} | {_pct(recall_at(er, 20))} | {_pct(statistics.mean(r['click_precision']))} | "
          f"{sum(r['click_distinct_topk'])}/{sum(r['click_distinct_gt'])} |")
    w("")
    # ---- avatar ---------------------------------------------------------------------------------
    w("## Avatar detection")
    w("")
    w("Ground truth: engine sprites whose position changes consistently (>= 75 %) with ACTION1-4 during the rollout. "
      "A check passes when our avatar box overlaps a ground-truth sprite's display box. Only games where such a "
      "sprite exists are listed.")
    w("")
    w("| game | accuracy | checks | first detected at step | our avatar + learned controls (end of rollout) |")
    w("|---|---|---|---|---|")
    for r in kb_games:
        w(f"| {r['game']} | {_pct(r['avatar_acc'])} | {r['avatar_checks']} | {r['avatar_first']} | {r['avatar_text'] or '-'} |")
    w("")
    # ---- actions ------------------------------------------------------------------------------------
    w("## Action channel")
    w("")
    w("### Guarded re-play of whole recordings (version-matched games)")
    w("")
    w("The recorded actions are fed through a guarded channel; refused actions are skipped (not sent). "
      "'diverged' counts executed steps whose frame differs from the recording outside the HUD line "
      "(expected after a skipped action only if the guard was wrong, or when timers differ).")
    w("")
    w("| game | source | recorded actions | refused (free) | by guard | levels reached / recorded | diverged steps |")
    w("|---|---|---|---|---|---|---|")
    for g in gr:
        w(f"| {g['game']} | {g['source']} | {g['recorded_actions']} | {g['blocked']} | {g['blocked_by'] or '-'} | "
          f"{g['levels_reached']}/{g['levels_target']} | {g['diverged_steps']} |")
    w("")
    w("### replay_prefix reproduces recorded level wins")
    w("")
    w("Per recorded level: play the winning attempt minus its last action, RESET (a charged level reset), "
      "`replay_prefix()` (verified step by step, no-ops skipped), then the last action.")
    w("")
    w("| game | levels re-won / tested | prefix actions replayed vs recorded attempt | failures |")
    w("|---|---|---|---|")
    for p in res["prefix"]:
        lv = [x for x in p["levels"] if x["ok"] is not None]
        rep_n = sum(x.get("replayed", 0) for x in lv)
        att_n = sum(x.get("attempt", 1) - 1 for x in lv)
        fails = "; ".join(f"L{x['level'] + 1}: {x['why']}" for x in lv if not x["ok"]) or "-"
        w(f"| {p['game']} | {sum(1 for x in lv if x['ok'])}/{len(lv)} | {rep_n} / {att_n} | {fails} |")
    w("")
    w("### Accounting and guards on rollouts")
    w("")
    w("| game | charged actions | levels | refusals by guard | GAME_OVERs | prefix replay: recorded attempt -> replayed |")
    w("|---|---|---|---|---|---|")
    for r in ro:
        sv = ", ".join(f"{a}->{b}" for a, b in r["replay_saved"][:4]) or "-"
        w(f"| {r['game']} | {r['actions']} | {r['levels']} | {r['blocked_by'] or '-'} | {r['gameovers']} | {sv} |")
    w("")
    open(path, "w").write("\n".join(L) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis", "02_perception_actions_eval.md"))
    ap.add_argument("--json", default=None, help="also dump raw results here")
    args = ap.parse_args(argv)
    games = args.games or list_games()
    res = run_all(games, args.steps, args.seed)
    if args.json:
        json.dump(res, open(args.json, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_report(res, args.out, args.steps, args.seed)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
