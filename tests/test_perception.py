"""Tests for agent/perception.py (B04).

Synthetic unit tests (no engine) plus measurements on all 25 public games and on the arcprize.org
replays in the repo root. Thresholds are set a little below the measured values reported in
analysis/02_perception_actions_eval.md so that regressions fail loudly.
"""

import random
import statistics

import numpy as np
import pytest

import perception as P
from conftest import all_games, requires_env


# ------------------------------------------------------------------------------------------------
# synthetic helpers
# ------------------------------------------------------------------------------------------------
def blank(bg=0):
    return np.full((64, 64), bg, dtype=np.int16)


def put(g, r, c, h, w, colour):
    g = g.copy()
    g[r:r + h, c:c + w] = colour
    return g


# ------------------------------------------------------------------------------------------------
# inputs / delta
# ------------------------------------------------------------------------------------------------
def test_norm_action_forms():
    assert P.norm_action("ACTION3") == (3, None, None)
    assert P.norm_action(4) == (4, None, None)
    assert P.norm_action("RESET") == (0, None, None)
    assert P.norm_action("6:12,40") == (6, 12, 40)
    assert P.norm_action((6, 5, 7)) == (6, 5, 7)
    assert P.norm_action({"id": 6, "data": {"x": 1, "y": 2}}) == (6, 1, 2)
    assert P.action_label((6, 5, 7)) == "A6(x5,y7)"


def test_to_grid_takes_last_animation_frame():
    f1, f2 = blank(), put(blank(), 0, 0, 1, 1, 3)
    assert P.to_grid([f1, f2])[0, 0] == 3
    assert P.to_grid(f1.tolist()).shape == (64, 64)
    assert len(P.to_frames([f1, f2])) == 2


def test_frame_delta_cells_and_rle():
    a = blank()
    b = a.copy()
    b[10, 20] = 7
    d = P.frame_delta(a, b)
    assert d.n == 1 and d.bbox == (10, 20, 10, 20) and d.transitions[(0, 7)] == 1
    assert P.delta_text(a, b) == "1 cells(r,c): 10,20:0>7"
    # a 5x5 block moving right by 5: rectangle RLE
    a = put(blank(), 40, 13, 5, 5, 9)
    b = put(blank(), 40, 18, 5, 5, 9)
    t = P.delta_text(a, b)
    assert "r40-44 c13-17:9>0" in t and "r40-44 c18-22:0>9" in t
    assert P.frame_delta(a, b).bbox == (40, 13, 44, 22)
    # masked cells are excluded and counted separately
    m = np.zeros((64, 64), bool)
    m[40:45, 13:18] = True
    d = P.frame_delta(a, b, m)
    assert d.n == 25 and d.hud_changed == 25


# ------------------------------------------------------------------------------------------------
# segmentation / scale
# ------------------------------------------------------------------------------------------------
def test_segment_components_background_and_shape_hash():
    g = put(put(put(blank(), 5, 5, 3, 3, 5), 30, 40, 3, 3, 5), 50, 10, 1, 4, 7)
    seg = P.segment(g)
    objs = seg.objects()
    assert seg.bg_color == 0
    assert sorted((o.color, o.w, o.h) for o in objs) == [(5, 3, 3), (5, 3, 3), (7, 4, 1)]
    sq = [o for o in objs if o.color == 5]
    assert sq[0].ctype == sq[1].ctype  # translation-invariant type
    assert sq[0].shape == P.segment(put(blank(), 1, 1, 3, 3, 9)).objects()[0].shape  # colour-free shape
    assert seg.comp_at(41, 31).color == 5 and seg.comp_at(0, 0).bg


def test_diagonal_outline_groups_into_one_object():
    g = blank()
    for i in range(5):
        g[10 + i, 10 + i] = 4
    seg = P.segment(g)
    assert len([o for o in seg.objects() if o.color == 4]) == 5  # 4-connected: five 1x1 pieces
    groups = [x for x in seg.groups() if not x.bg]
    assert len(groups) == 1 and len(groups[0].members) == 5 and (groups[0].w, groups[0].h) == (5, 5)
    cands = P.rank_click_candidates(seg)
    assert cands[0].comp.members == groups[0].members


def test_detect_scale_synthetic():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 6, size=(16, 16))
    g = np.kron(small, np.ones((4, 4), int)).astype(np.int16)
    assert P.detect_scale(g) == (4, 0, 0)
    # 20x20 camera at scale 3 with 2 px letterbox, plus a 1-px HUD bar on the last row
    small = rng.integers(0, 6, size=(20, 20))
    g = np.full((64, 64), 5, np.int16)
    g[2:62, 2:62] = np.kron(small, np.ones((3, 3), int))
    g[63, :40] = 11
    assert P.detect_scale(g) == (3, 2, 2)
    # pixel noise: no grid
    assert P.detect_scale(rng.integers(0, 16, size=(64, 64)))[0] == 1


def test_object_diff_kinds():
    a = put(put(put(blank(), 10, 10, 3, 3, 5), 30, 30, 2, 2, 8), 50, 50, 2, 4, 6)
    a = put(a, 20, 40, 3, 3, 12)
    b = put(put(put(blank(), 10, 12, 3, 3, 5), 50, 50, 2, 6, 6), 5, 55, 1, 1, 14)
    b = put(b, 20, 40, 3, 3, 13)
    d = P.frame_delta(a, b)
    ev = P.object_diff(P.segment(a), P.segment(b), d.changed)
    kinds = {(e.kind, e.color) for e in ev}
    assert ("moved", 5) in kinds and ("despawned", 8) in kinds and ("spawned", 14) in kinds
    assert ("resized", 6) in kinds and ("recolored", 12) in kinds
    mv = next(e for e in ev if e.kind == "moved")
    assert (mv.dx, mv.dy) == (2, 0) and "right 2" in mv.text()


# ------------------------------------------------------------------------------------------------
# HUD / avatar / clicks / animation (synthetic)
# ------------------------------------------------------------------------------------------------
def _bar_frame(used, avatar_x, extra=None):
    g = blank()
    g[63, :] = 11
    g[63, :used] = 5
    g = put(g, 30, avatar_x, 3, 3, 9)
    if extra is not None:
        g = extra(g)
    return g


def test_hud_bar_detected_and_masked_but_content_is_not():
    per = P.Perceiver()
    f = _bar_frame(0, 10)
    per.reset(f)
    for t in range(1, 6):
        nf = _bar_frame(t, 10 + 3 * t)
        pc = per.observe(f, nf, 4)
        f = nf
        assert not pc.noop  # the avatar moved
    m = per.hud_mask()
    assert m[63].all() and not m[:63].any()
    assert per.hud.bars and per.hud.bars[0].a == 11 and per.hud.bars[0].b == 5
    # a pure timer tick is now a no-op
    nf = _bar_frame(6, 10 + 15)
    pc = per.observe(f, nf, 1)
    assert pc.noop and "no effect" in pc.text


def test_hud_not_fooled_by_thin_moving_object_or_toggles():
    hud = P.HudTracker()
    # a 1x3 vertical line moving right one pixel per step (leading edge 0->7, trailing 7->0)
    prev = put(blank(), 20, 20, 3, 1, 7)
    for t in range(1, 8):
        cur = put(blank(), 20, 20 + t, 3, 1, 7)
        hud.update(prev, cur)
        prev = cur
    assert not hud.bars
    # isolated content toggles (bits) 2 px apart in one row, one per step
    hud = P.HudTracker()
    prev = blank()
    for t in range(4):
        cur = prev.copy()
        cur[40, 20 + 3 * t] = 5
        hud.update(prev, cur)
        prev = cur
    assert not hud.bars


def test_avatar_tracker_learns_controls():
    per = P.Perceiver()
    pos = [20, 20]
    f = put(put(blank(), pos[0], pos[1], 2, 2, 9), 50, 50, 4, 4, 3)
    per.reset(f)
    moves = {1: (-2, 0), 2: (2, 0), 3: (0, -2), 4: (0, 2)}
    rng = random.Random(0)
    for _ in range(12):
        a = rng.choice([1, 2, 3, 4])
        pos = [pos[0] + moves[a][0], pos[1] + moves[a][1]]
        nf = put(put(blank(), pos[0], pos[1], 2, 2, 9), 50, 50, 4, 4, 3)
        per.observe(f, nf, a)
        f = nf
    assert per.avatar.avatars() == [9]
    assert per.avatar.controls(9) == {1: "up", 2: "down", 3: "left", 4: "right"}
    grp = per.avatar.groups(per.seg(f))
    assert grp[0][1] == (pos[0], pos[1], pos[0] + 1, pos[1] + 1)
    assert "avatar c9" in per.summary(f)


def test_click_candidates_rank_and_dead_suppression():
    g = blank()
    for i in range(4):  # a row of four 4x4 'buttons' (colour 3)
        g = put(g, 10, 5 + 8 * i, 4, 4, 3)
    g = put(g, 40, 40, 4, 4, 12)  # a unique 4x4 object (colour 12)
    g = put(g, 60, 0, 1, 64, 2)  # a frame-wide strip
    per = P.Perceiver()
    per.reset(g)
    cands = per.candidates(g)
    assert all(c.comp.color != 0 for c in cands[:-1]) and cands[-1].ctype.startswith("bg:")
    top_types = [c.comp.color for c in cands[:3]]
    assert 12 in top_types and 3 in top_types  # both kinds of object are offered first
    # a dead click on the unique object demotes it below every untested candidate
    x, y = next((c.x, c.y) for c in cands if c.comp.color == 12)
    per.observe(g, g, (6, x, y))
    cands = per.candidates(g)
    order = [c.comp.color for c in cands if not c.ctype.startswith("bg:")]
    assert order[-1] == 12 and cands[order.index(12)].status == "dead"
    # a live click on one button puts all buttons first
    g2 = g.copy()
    g2[10:14, 5:9] = 4
    x, y = next((c.x, c.y) for c in cands if c.comp.color == 3)
    pc = per.observe(g, g2, (6, x, y))
    assert not pc.noop
    cands = per.candidates(g)
    assert [c.comp.color for c in cands[:4]] == [3, 3, 3, 3] and cands[0].status == "live"


def test_animation_summary_transient():
    prev = blank()
    frames = [put(prev, 10, 10 + i, 2, 2, 6) for i in range(5)] + [prev.copy()]
    frames[-1][0, 0] = 1
    a = P.animation_summary(prev, frames)
    assert a["n_frames"] == 6 and a["transient_bbox"] == (10, 10, 11, 15) and a["peak_frame"] is not None


def test_describe_is_compact_and_deterministic():
    a = put(put(blank(), 40, 13, 5, 5, 9), 5, 5, 3, 3, 2)
    b = put(put(blank(), 40, 18, 5, 5, 9), 5, 5, 3, 3, 2)
    t1 = P.describe(a, b, "ACTION4")
    t2 = P.describe(a, b, "ACTION4")
    assert t1 == t2 and t1.startswith("A4:") and "moved c9 5x5 right 5" in t1
    assert P.approx_tokens(t1) <= 150
    assert P.describe(a, a, (6, 6, 6)).endswith("type marked dead")


# ------------------------------------------------------------------------------------------------
# all 25 public games
# ------------------------------------------------------------------------------------------------
@requires_env
@pytest.mark.parametrize("game", all_games())
def test_scale_is_camera_or_safe_fallback(game):
    """A detected cell size s > 1 must really tile the frame (>= 95% of s x s blocks uniform); where the
    game renders its camera grid without 1-px overlays the detected grid equals the engine camera's."""
    from arcenv import ArcEnv

    env = ArcEnv(game)
    f = P.to_grid(env.frame)
    s, oy, ox = P.detect_scale(f)
    cam_s, cam_x, cam_y = env.game.camera._calculate_scale_and_offset()
    if s > 1:
        blocks = f[oy:oy + ((64 - oy) // s) * s, ox:ox + ((64 - ox) // s) * s]
        hh, ww = blocks.shape[0] // s, blocks.shape[1] // s
        b = blocks.reshape(hh, s, ww, s)
        uniform = (b == b[:, :1, :, :1]).all(axis=(1, 3))
        assert uniform[1:-1, 1:-1].mean() >= 0.95  # (outer ring skipped: HUD lines)
    if game[:4] in ("cn04", "ft09", "lp85", "m0r0", "sp80", "vc33"):
        assert (s, oy % s, ox % s) == (cam_s, cam_y % cam_s, cam_x % cam_s)


def _random_rollout(game, n, seed):
    from arcenv import ArcEnv
    from eval_perception_actions import gt_hud_mask

    env = ArcEnv(game)
    per = P.Perceiver()
    per.reset(env.frame, env.levels_completed)
    gtm = gt_hud_mask(game)
    rng = random.Random(seed)
    stats = {"masked_out": 0, "masked": 0, "gt_changed": 0, "gt_masked": 0, "false_noop": 0, "tokens": []}
    av = env.available_actions
    for _ in range(n):
        prev = env.frame.copy()
        a = rng.choice(av)
        if a == 6:
            c = per.candidates(prev)
            k = c[min(len(c) - 1, int(rng.expovariate(0.3)))]
            act = (6, k.x, k.y)
        else:
            act = a
        lv = env.levels_completed
        o = env.step(act)
        pc = per.observe(prev, o.frames, act, level=o.levels_completed, state=o.state)
        stats["tokens"].append(P.approx_tokens(pc.text))
        if o.state in ("GAME_OVER", "WIN"):
            prev = env.frame.copy()
            o = env.step(0)
            per.observe(prev, o.frames, 0, level=o.levels_completed, state=o.state)
            continue
        if o.levels_completed != lv:
            continue
        ch = prev != env.frame
        m = per.hud_mask()
        stats["masked"] += int((ch & m).sum())
        stats["masked_out"] += int((ch & m & ~gtm).sum())
        stats["gt_changed"] += int((ch & gtm).sum())
        stats["gt_masked"] += int((ch & gtm & m).sum())
        stats["false_noop"] += pc.noop and bool((ch & ~gtm).any())
    return per, stats


@requires_env
@pytest.mark.parametrize("game", all_games())
def test_rollout_hud_mask_noop_and_text_budget(game):
    per, st = _random_rollout(game, 120, seed=1)
    # masked cells that changed must lie on the true timer/budget line (false-positive rate <= 5 %)
    assert st["masked_out"] <= 0.05 * max(1, st["masked"]), (per.hud.describe(), st)
    # the per-action bar is found and hidden whenever it ticked enough to be learned
    if st["gt_changed"] >= 10:
        assert st["gt_masked"] >= 0.8 * st["gt_changed"], (per.hud.describe(), st)
    assert st["false_noop"] == 0
    assert max(st["tokens"]) <= 155 and statistics.median(st["tokens"]) <= 130


@requires_env
def test_avatar_detection_on_keyboard_games():
    from eval_perception_actions import rollout_eval

    accs = {}
    for g in all_games():
        r = rollout_eval(g, n_steps=120, seed=2, click_every=10 ** 9)
        if r["avatar_gt"] and r["avatar_checks"] >= 10:
            accs[g] = r["avatar_acc"]
    assert len(accs) >= 8, accs
    good = [g for g, a in accs.items() if a >= 0.8]
    assert len(good) >= 0.8 * len(accs), accs
    assert statistics.mean(accs.values()) >= 0.8, accs


@requires_env
def test_click_candidate_recall_on_replays():
    """Winning / effective clicks of the human + frontier replays land on a top-ranked candidate."""
    from eval_perception_actions import load_replays, recall_at, replay_click_ranks

    ranks = [x for rep in load_replays() for x in replay_click_ranks(rep)]
    eff = [x for x in ranks if x["effective"] and not x["bg"]]
    wins = [x for x in ranks if x["level_win"] and not x["bg"]]
    assert len(eff) >= 300 and len(wins) >= 20
    assert recall_at([x["rank"] for x in eff], 10 ** 6) >= 0.95  # every clicked object is offered
    assert recall_at([x["rank"] for x in eff], 20) >= 0.75
    assert recall_at([x["rank"] for x in eff], 10) >= 0.55
    assert recall_at([x["rank"] for x in wins], 10) >= 0.6
    # history (dead-signature suppression + live boost) must help over the history-free prior
    assert recall_at([x["rank"] for x in eff], 10) > recall_at([x["rank_fresh"] for x in eff], 10)


@requires_env
def test_perception_is_deterministic_on_a_replay():
    from eval_perception_actions import load_replays, replay_action

    rep = next(r for r in load_replays() if r["game"] == "s5i5")

    def run():
        per, out, prev = P.Perceiver(), [], None
        for st in rep["steps"][:120]:
            if st["frame"] is None:
                continue
            if prev is None:
                per.reset(st["frame"])
            else:
                pc = per.observe(prev, st["frames"], replay_action(st), level=st["levels"], state=st["state"])
                out.append((pc.text, [(c.x, c.y) for c in per.candidates(st["frame"])[:10]]))
            prev = st["frame"]
        return out

    assert run() == run()
