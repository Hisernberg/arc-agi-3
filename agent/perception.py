"""perception.py - deterministic, game-agnostic perception pack for ARC-AGI-3 frames (backlog B04).

Everything here works on raw 64x64 integer frames (colours 0..15) and on nothing else: no game ids,
no per-game tables, no sprite names. The hidden evaluation games are different from the 25 public ones,
so every rule below is a generic visual heuristic that is measured (not tuned) on the public games
(tests/test_perception.py, analysis/02_perception_actions_eval.md).

Pieces
  frame_delta / delta_text     raw-int frame delta: changed cells as ``r,c:old>new`` or row/rect RLE, active bbox
  detect_scale                 integer upscale factor + offset of the camera grid (from edge alignment)
  segment                      4-connected same-colour components, background detection, shape hashes
  object_diff                  spawned / despawned / moved / recoloured / resized objects between two frames
  HudTracker                   finds per-action counter/timer bars (fixed colour transition that advances
                               one step per action) and masks them out of change detection and state hashes
  AvatarTracker                the object(s) that move consistently with the directional actions
  rank_click_candidates        one ACTION6 target per distinct object/cell, ranked by rarity/size/shape,
                               with dead-signature suppression (object types whose clicks did nothing)
  animation_summary            multi-frame steps: frame count, transient bbox
  Perceiver                    stateful per-game facade tying the above together; ``describe()`` renders a
                               compact (<= ~150 tokens) text for the LLM prompt
  describe(prev, cur, action)  convenience wrapper

Coordinates: grid cells are (r, c) = (row, column) = (y, x). ACTION6 takes display (x, y) = (column, row).
Text output labels them explicitly: ``r,c`` in cell lists, ``(x..,y..)`` for object positions / clicks.
Only numpy is required.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "to_grid", "to_frames", "norm_action", "action_label", "frame_delta", "delta_text", "Delta",
    "detect_scale", "Comp", "Segmentation", "segment", "ObjEvent", "object_diff", "HudTracker",
    "AvatarTracker", "ClickCandidate", "rank_click_candidates", "candidates_text", "animation_summary",
    "Percept", "Perceiver", "describe", "masked_hash", "DIR_NAMES",
]

H = W = 64
DIR_NAMES = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}  # (dx, dy) -> name


# ------------------------------------------------------------------------------------------------
# input normalisation
# ------------------------------------------------------------------------------------------------
def to_grid(frame: Any) -> np.ndarray:
    """Return one 2-D int16 grid. Accepts an ndarray, nested lists, a list of animation frames
    (the LAST one is the settled state), or an object with a ``.frame``/``.frames`` attribute."""
    if frame is None:
        raise ValueError("no frame")
    for attr in ("frames", "frame"):
        if not isinstance(frame, (np.ndarray, list, tuple)) and hasattr(frame, attr):
            frame = getattr(frame, attr)
            break
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a[-1]
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {a.shape}")
    return a.astype(np.int16, copy=False)


def to_frames(frames: Any) -> list[np.ndarray]:
    """List of 2-D int16 grids (all animation frames of one step)."""
    if frames is None:
        return []
    if not isinstance(frames, (np.ndarray, list, tuple)):
        frames = getattr(frames, "frames", None) or getattr(frames, "frame", None) or []
    a = frames
    if isinstance(a, np.ndarray):
        return [a.astype(np.int16)] if a.ndim == 2 else [f.astype(np.int16) for f in a]
    if len(a) == 0:
        return []
    first = np.asarray(a[0])
    if first.ndim == 1:  # a single frame given as nested lists
        return [np.asarray(a).astype(np.int16)]
    return [np.asarray(f).astype(np.int16) for f in a]


def norm_action(a: Any) -> tuple[int, Optional[int], Optional[int]]:
    """(action_id, x, y). Accepts 0-7, 'ACTION6', 'RESET', '6:x,y', (6, x, y), {'id':6,'x':..}, GameAction."""
    if a is None:
        return -1, None, None
    if isinstance(a, dict):
        aid = norm_action(a.get("id", a.get("action")))[0]
        d = a.get("data") or a
        x, y = d.get("x"), d.get("y")
        return aid, (None if x is None else int(x)), (None if y is None else int(y))
    if isinstance(a, (tuple, list)):
        aid = norm_action(a[0])[0]
        if len(a) >= 3 and a[1] is not None:
            return aid, int(a[1]), int(a[2])
        if len(a) == 2 and isinstance(a[1], dict):
            return aid, a[1].get("x"), a[1].get("y")
        return aid, None, None
    if hasattr(a, "value") and not isinstance(a, (int, np.integer)):  # enum (GameAction)
        return int(a.value), None, None
    if isinstance(a, (int, np.integer)):
        return int(a), None, None
    s = str(a).strip().upper()
    if ":" in s:
        head, tail = s.split(":", 1)
        xs, ys = tail.replace(" ", "").split(",")
        return norm_action(head)[0], int(xs), int(ys)
    if s.startswith("ACTION"):
        return int(s[6:]), None, None
    if s in ("RESET", "R"):
        return 0, None, None
    if s.isdigit():
        return int(s), None, None
    names = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "INTERACT": 5, "UNDO": 7}
    if s in names:
        return names[s], None, None
    raise ValueError(f"cannot parse action {a!r}")


def action_label(a: Any) -> str:
    aid, x, y = norm_action(a)
    if aid == 0:
        return "RESET"
    if aid == 6 and x is not None:
        return f"A6(x{x},y{y})"
    return f"A{aid}" if aid >= 0 else "?"


# ------------------------------------------------------------------------------------------------
# frame delta
# ------------------------------------------------------------------------------------------------
@dataclass
class Delta:
    changed: np.ndarray  # bool mask of changed cells (HUD cells excluded when a mask was given)
    n: int
    bbox: Optional[tuple[int, int, int, int]]  # (r0, c0, r1, c1) inclusive, or None
    transitions: Counter  # (old, new) -> count
    hud_changed: int = 0  # changed cells that fell inside the HUD mask

    @property
    def empty(self) -> bool:
        return self.n == 0


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    rows = np.flatnonzero(mask.any(axis=1))
    if len(rows) == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def frame_delta(prev: Any, cur: Any, mask: Optional[np.ndarray] = None) -> Delta:
    a, b = to_grid(prev), to_grid(cur)
    ch = a != b
    hud = 0
    if mask is not None:
        hud = int((ch & mask).sum())
        ch = ch & ~mask
    n = int(ch.sum())
    tr: Counter = Counter()
    if n:
        ys, xs = np.nonzero(ch)
        pairs = a[ys, xs].astype(np.int32) * 16 + b[ys, xs].astype(np.int32)
        for v, k in zip(*np.unique(pairs, return_counts=True)):
            tr[(int(v) // 16, int(v) % 16)] = int(k)
    return Delta(ch, n, _bbox(ch) if n else None, tr, hud)


def _fmt_bbox(bb: Optional[tuple[int, int, int, int]]) -> str:
    if bb is None:
        return "-"
    r0, c0, r1, c1 = bb
    rr = f"r{r0}" if r0 == r1 else f"r{r0}-{r1}"
    cc = f"c{c0}" if c0 == c1 else f"c{c0}-{c1}"
    return f"{rr} {cc}"


def delta_text(prev: Any, cur: Any, mask: Optional[np.ndarray] = None, max_cells: int = 12,
               max_chars: int = 240, delta: Optional[Delta] = None) -> str:
    """Compact raw-int delta. Small changes: ``r,c:old>new`` list. Larger: rectangle RLE
    ``r40-44 c13-17:3>9`` (runs of one transition merged over consecutive rows). Falls back to a
    transition histogram + bbox when even that exceeds ``max_chars``."""
    a, b = to_grid(prev), to_grid(cur)
    d = delta if delta is not None else frame_delta(a, b, mask)
    if d.n == 0:
        return "no change" + (f" (hud {d.hud_changed})" if d.hud_changed else "")
    if d.n <= max_cells:
        ys, xs = np.nonzero(d.changed)
        s = " ".join(f"{y},{x}:{a[y, x]}>{b[y, x]}" for y, x in zip(ys.tolist(), xs.tolist()))
        if len(s) <= max_chars:
            return f"{d.n} cells(r,c): {s}"
    # row runs of one transition
    runs: list[tuple[int, int, int, int, int]] = []  # (r, c0, c1, old, new)
    ys, xs = np.nonzero(d.changed)
    cur_run = None
    for y, x in zip(ys.tolist(), xs.tolist()):
        o, nw = int(a[y, x]), int(b[y, x])
        if cur_run and cur_run[0] == y and cur_run[2] == x - 1 and cur_run[3] == o and cur_run[4] == nw:
            cur_run[2] = x
        else:
            if cur_run:
                runs.append(tuple(cur_run))
            cur_run = [y, x, x, o, nw]
    if cur_run:
        runs.append(tuple(cur_run))
    # merge identical runs on consecutive rows into rectangles
    open_rects: dict[tuple[int, int, int, int], list[int]] = {}
    rects: list[tuple[int, int, int, int, int, int]] = []
    for r, c0, c1, o, nw in runs:
        k = (c0, c1, o, nw)
        rr = open_rects.get(k)
        if rr is not None and rr[1] == r - 1:
            rr[1] = r
        else:
            if rr is not None:
                rects.append((rr[0], rr[1], *k))
            open_rects[k] = [r, r]
    rects.extend((v[0], v[1], *k) for k, v in open_rects.items())
    rects.sort()
    parts = []
    for r0, r1, c0, c1, o, nw in rects:
        rs = f"r{r0}" if r0 == r1 else f"r{r0}-{r1}"
        cs = f"c{c0}" if c0 == c1 else f"c{c0}-{c1}"
        parts.append(f"{rs} {cs}:{o}>{nw}")
    s = "; ".join(parts)
    head = f"{d.n} cells in {_fmt_bbox(d.bbox)}"
    if len(s) <= max_chars:
        return f"{head}: {s}"
    tr = " ".join(f"{o}>{nw}x{k}" for (o, nw), k in d.transitions.most_common(5))
    return f"{head}; {len(rects)} rects; {tr}"


# ------------------------------------------------------------------------------------------------
# grid scale (camera upscaling) detection
# ------------------------------------------------------------------------------------------------
def _edge_hist(g: np.ndarray, ignore: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """hc[c] = #rows where g[r,c] != g[r,c-1]; vr[r] = #cols where g[r,c] != g[r-1,c]."""
    dh = g[:, 1:] != g[:, :-1]
    dv = g[1:, :] != g[:-1, :]
    if ignore is not None:
        dh &= ~(ignore[:, 1:] | ignore[:, :-1])
        dv &= ~(ignore[1:, :] | ignore[:-1, :])
    hc = np.zeros(g.shape[1], np.int64)
    hc[1:] = dh.sum(axis=0)
    vr = np.zeros(g.shape[0], np.int64)
    vr[1:] = dv.sum(axis=1)
    return hc, vr


def _best_scale(hc: np.ndarray, vr: np.ndarray, max_scale: int = 8, tol: float = 0.03,
                min_edges: int = 12) -> tuple[int, int, int]:
    tot_h, tot_v = int(hc.sum()), int(vr.sum())
    if tot_h + tot_v < min_edges:
        return 1, 0, 0
    idx_c, idx_r = np.arange(len(hc)), np.arange(len(vr))
    for s in range(max_scale, 1, -1):
        best = None
        for ox in range(s):
            bad_h = tot_h - int(hc[(idx_c % s) == ox].sum())
            if bad_h > tol * max(tot_h, 1):
                continue
            for oy in range(s):
                bad_v = tot_v - int(vr[(idx_r % s) == oy].sum())
                if bad_v <= tol * max(tot_v, 1):
                    cand = (bad_h + bad_v, oy, ox)
                    best = cand if best is None or cand < best else best
        if best is not None:
            return s, best[1], best[2]
    return 1, 0, 0


def detect_scale(frame: Any, ignore: Optional[np.ndarray] = None, max_scale: int = 8) -> tuple[int, int, int]:
    """(scale, off_y, off_x): the largest s <= max_scale such that >= 97% of colour edges lie on the
    lines r = off_y (mod s), c = off_x (mod s). Games render a small camera grid upscaled by an integer
    factor (plus letterbox padding), so one grid cell = an s x s block. Returns (1, 0, 0) if none fits."""
    hc, vr = _edge_hist(to_grid(frame), ignore)
    return _best_scale(hc, vr, max_scale)


# ------------------------------------------------------------------------------------------------
# segmentation
# ------------------------------------------------------------------------------------------------
@dataclass
class Comp:
    id: int
    color: int
    n: int
    r0: int
    c0: int
    r1: int
    c1: int
    cy: float
    cx: float
    shape: str  # colour-free translation-invariant shape hash
    ctype: str  # colour + shape hash ("object type")
    bg: bool = False  # background / large region (not an object)

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r1, self.c1

    def short(self) -> str:
        return f"c{self.color} {self.w}x{self.h}"


@dataclass
class Segmentation:
    grid: np.ndarray
    labels: np.ndarray  # int32 component id per cell
    comps: list[Comp]
    bg_color: int
    scale: tuple[int, int, int] = (1, 0, 0)

    def comp_at(self, x: int, y: int) -> Optional[Comp]:
        if not (0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]):
            return None
        return self.comps[int(self.labels[y, x])]

    def objects(self) -> list[Comp]:
        return [c for c in self.comps if not c.bg]

    def mask_of(self, cid: int) -> np.ndarray:
        return self.labels == cid


def _label_runs(g: np.ndarray):
    """4-connected components via row runs + union-find. Returns (labels, run arrays, root ids)."""
    Hh, Ww = g.shape
    starts = np.ones((Hh, Ww), bool)
    starts[:, 1:] = g[:, 1:] != g[:, :-1]
    rs, cs = np.nonzero(starts)
    nrun = len(rs)
    ce = np.empty(nrun, np.int64)
    ce[:-1] = cs[1:] - 1
    ce[-1] = Ww - 1
    last = np.append(rs[1:] != rs[:-1], True)
    ce[last] = Ww - 1
    col = g[rs, cs]
    row_start = np.searchsorted(rs, np.arange(Hh + 1)).tolist()
    parent = list(range(nrun))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    csl, cel, coll = cs.tolist(), ce.tolist(), col.tolist()
    for r in range(1, Hh):
        i, iend = row_start[r], row_start[r + 1]
        j, jend = row_start[r - 1], row_start[r]
        while i < iend and j < jend:
            if coll[i] == coll[j] and csl[i] <= cel[j] and csl[j] <= cel[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    if ri < rj:
                        parent[rj] = ri
                    else:
                        parent[ri] = rj
            if cel[i] < cel[j]:
                i += 1
            elif cel[j] < cel[i]:
                j += 1
            else:
                i += 1
                j += 1
    roots = [find(i) for i in range(nrun)]
    # components numbered in order of first run (reading order of top-left-most cell)
    remap: dict[int, int] = {}
    comp_of_run = np.empty(nrun, np.int32)
    for i, rt in enumerate(roots):
        k = remap.get(rt)
        if k is None:
            k = remap[rt] = len(remap)
        comp_of_run[i] = k
    lengths = ce - cs + 1
    labels = np.repeat(comp_of_run, lengths).reshape(Hh, Ww)
    return labels, rs, cs, ce, col, comp_of_run, len(remap)


def _hash_bytes(b: bytes) -> str:
    return f"{zlib.crc32(b) & 0xffffffff:08x}"


def segment(frame: Any, hud_mask: Optional[np.ndarray] = None, scale: Optional[tuple[int, int, int]] = None,
            bg_color: Optional[int] = None) -> Segmentation:
    """Segment into 4-connected same-colour components. Background = most frequent colour (HUD cells
    excluded); a component is flagged ``bg`` if it has the background colour and >= 2% of the frame, or
    is any single region covering >= 20% of the frame, or lies entirely inside the HUD mask."""
    g = to_grid(frame)
    labels, rs, cs, ce, col, comp_of_run, ncomp = _label_runs(g)
    total = g.size
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=ncomp)
    yy, xx = np.divmod(np.arange(total), g.shape[1])
    sum_r = np.bincount(flat, weights=yy, minlength=ncomp)
    sum_c = np.bincount(flat, weights=xx, minlength=ncomp)
    order = np.argsort(comp_of_run, kind="stable")
    bounds = np.flatnonzero(np.r_[True, np.diff(comp_of_run[order]) != 0])
    r0 = np.minimum.reduceat(rs[order], bounds)
    r1 = np.maximum.reduceat(rs[order], bounds)
    c0 = np.minimum.reduceat(cs[order], bounds)
    c1 = np.maximum.reduceat(ce[order], bounds)
    colors = np.empty(ncomp, np.int64)
    colors[comp_of_run[order][bounds]] = col[order][bounds]
    if bg_color is None:
        vals = g if hud_mask is None else g[~hud_mask]
        bg_color = int(np.bincount(vals.ravel().astype(np.int64) % 256).argmax()) if vals.size else 0
    hud_frac = None
    if hud_mask is not None and hud_mask.any():
        hud_frac = np.bincount(flat, weights=hud_mask.ravel().astype(np.float64), minlength=ncomp)
    comps: list[Comp] = []
    for k in range(ncomp):
        a0, b0, a1, b1 = int(r0[k]), int(c0[k]), int(r1[k]), int(c1[k])
        n = int(counts[k])
        crop = labels[a0:a1 + 1, b0:b1 + 1] == k
        key = bytes((a1 - a0 + 1, b1 - b0 + 1)) + np.packbits(crop).tobytes()
        shp = _hash_bytes(key)
        color = int(colors[k])
        is_bg = (color == bg_color and n >= 0.02 * total) or n >= 0.20 * total
        if hud_frac is not None and hud_frac[k] >= 0.999 * n:
            is_bg = True
        comps.append(Comp(k, color, n, a0, b0, a1, b1, float(sum_r[k] / n), float(sum_c[k] / n), shp,
                          f"{color}:{shp}", is_bg))
    return Segmentation(g, labels, comps, bg_color, scale or (1, 0, 0))


# ------------------------------------------------------------------------------------------------
# object diff
# ------------------------------------------------------------------------------------------------
@dataclass
class ObjEvent:
    kind: str  # moved | recolored | resized | spawned | despawned
    color: int
    before: Optional[Comp] = None
    after: Optional[Comp] = None
    dx: int = 0
    dy: int = 0
    new_color: Optional[int] = None

    @property
    def comp(self) -> Comp:
        return self.after if self.after is not None else self.before

    def text(self) -> str:
        c = self.comp
        if self.kind == "moved":
            b, a = self.before, self.after
            dname = DIR_NAMES.get((int(np.sign(self.dx)), int(np.sign(self.dy)))) if (self.dx == 0 or self.dy == 0) else None
            dd = f"{dname} {max(abs(self.dx), abs(self.dy))}" if dname else f"({self.dx:+d},{self.dy:+d})"
            return f"moved {b.short()} {dd} (x{b.c0},y{b.r0})->(x{a.c0},y{a.r0})"
        if self.kind == "recolored":
            return f"recolor {c.w}x{c.h} c{self.color}>c{self.new_color} @(x{c.c0},y{c.r0})"
        if self.kind == "resized":
            b, a = self.before, self.after
            return f"resize c{self.color} {b.w}x{b.h}>{a.w}x{a.h} @(x{a.c0},y{a.r0})"
        if self.kind == "spawned":
            return f"new {c.short()} @(x{c.c0},y{c.r0})"
        return f"gone {c.short()} @(x{c.c0},y{c.r0})"


def object_diff(seg_a: Segmentation, seg_b: Segmentation, changed: np.ndarray, max_events: int = 64) -> list[ObjEvent]:
    """Match the non-background components touched by the change mask between two segmentations."""
    if not changed.any():
        return []
    la = np.unique(seg_a.labels[changed])
    lb = np.unique(seg_b.labels[changed])
    A = [seg_a.comps[i] for i in la.tolist() if not seg_a.comps[i].bg]
    B = [seg_b.comps[i] for i in lb.tolist() if not seg_b.comps[i].bg]
    if len(A) > 400 or len(B) > 400:  # texture explosion: skip object-level diff
        return []
    used_a: set[int] = set()
    used_b: set[int] = set()
    events: list[ObjEvent] = []
    # 1) identical type (colour+shape) at a different position -> moved (closest first)
    by_type_b: dict[str, list[Comp]] = defaultdict(list)
    for c in B:
        by_type_b[c.ctype].append(c)
    pairs = []
    for a in A:
        for b in by_type_b.get(a.ctype, ()):
            d = abs(b.r0 - a.r0) + abs(b.c0 - a.c0)
            if d > 0:
                pairs.append((d, a.id, b.id, a, b))
    pairs.sort(key=lambda t: t[:3])
    for d, ia, ib, a, b in pairs:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        events.append(ObjEvent("moved", a.color, a, b, b.c0 - a.c0, b.r0 - a.r0))
    # 2) same shape + same place, other colour -> recoloured
    rest_b = [b for b in B if b.id not in used_b]
    for a in A:
        if a.id in used_a:
            continue
        for b in rest_b:
            if b.id not in used_b and b.shape == a.shape and b.bbox == a.bbox and b.color != a.color:
                used_a.add(a.id)
                used_b.add(b.id)
                events.append(ObjEvent("recolored", a.color, a, b, new_color=b.color))
                break
    # 3) same colour, overlapping boxes -> resized / reshaped
    for a in A:
        if a.id in used_a:
            continue
        best = None
        for b in B:
            if b.id in used_b or b.color != a.color:
                continue
            ov_r = min(a.r1, b.r1) - max(a.r0, b.r0) + 1
            ov_c = min(a.c1, b.c1) - max(a.c0, b.c0) + 1
            if ov_r > 0 and ov_c > 0:
                sc = ov_r * ov_c
                if best is None or sc > best[0]:
                    best = (sc, b)
        if best is not None:
            used_a.add(a.id)
            used_b.add(best[1].id)
            events.append(ObjEvent("resized", a.color, a, best[1]))
    for a in A:
        if a.id not in used_a:
            events.append(ObjEvent("despawned", a.color, a, None))
    for b in B:
        if b.id not in used_b:
            events.append(ObjEvent("spawned", b.color, None, b))
    # biggest first, moved first
    rank = {"moved": 0, "spawned": 1, "despawned": 1, "recolored": 2, "resized": 3}
    events.sort(key=lambda e: (rank[e.kind], -e.comp.n, e.comp.r0, e.comp.c0))
    return events[:max_events]


# ------------------------------------------------------------------------------------------------
# HUD / timer bars
# ------------------------------------------------------------------------------------------------
@dataclass
class HudBar:
    a: int  # colour before a tick
    b: int  # colour after a tick
    axis: str  # "h": ticks advance along columns (horizontal bar); "v": along rows
    lo: int  # perpendicular band [lo, hi] (rows for "h", cols for "v")
    hi: int
    start: int  # along-axis extent [start, end]
    end: int
    ticks: int = 1
    confirmed: bool = False

    def region(self) -> np.ndarray:
        m = np.zeros((H, W), bool)
        if self.axis == "h":
            m[self.lo:self.hi + 1, self.start:self.end + 1] = True
        else:
            m[self.start:self.end + 1, self.lo:self.hi + 1] = True
        return m

    def text(self) -> str:
        if self.axis == "h":
            return f"bar r{self.lo}-{self.hi} c{self.start}-{self.end} {self.a}>{self.b}"
        return f"bar c{self.lo}-{self.hi} r{self.start}-{self.end} {self.a}>{self.b}"


class HudTracker:
    """Detects counter/timer bars: every (or every k-th) action flips a few cells of a thin line from
    colour a to colour b, the flipped cells advancing along the line. Such cells carry no information
    about what the action did, so they are excluded from change detection and state hashing.

    A *tick* = an 8-connected blob of changed cells with one (a -> b) transition, <= ``max_tick`` cells
    and thickness <= 3. A bar is confirmed by two ticks with the same transition in the same band
    advancing along it, or immediately by one tick on the outermost frame row/column whose line consists
    only of the two bar colours. The masked region is the band x the contiguous run of bar-coloured
    cells through the ticks (sticky: it only grows)."""

    def __init__(self, max_tick: int = 8, max_gap: int = 8):
        self.max_tick = max_tick
        self.max_gap = max_gap
        self.bars: list[HudBar] = []
        self.pending: list[tuple[int, int, int, int, int, int]] = []  # (a, b, r0, c0, r1, c1) unconfirmed ticks
        self._mask = np.zeros((H, W), bool)

    # -- public ------------------------------------------------------------------------------
    def mask(self) -> np.ndarray:
        return self._mask

    def reset_pending(self) -> None:
        self.pending = []

    def update(self, prev: np.ndarray, cur: np.ndarray) -> list[tuple]:
        """Feed one settled transition; returns the tick blobs attributed to bars in this step."""
        ch = prev != cur
        if not ch.any():
            return []
        blobs = self._tick_blobs(prev, cur, ch)
        used = []
        new_pending = []
        for bl in blobs:
            a, b, r0, c0, r1, c1 = bl
            bar = self._bar_for(bl)
            if bar is not None:
                bar.ticks += 1
                self._extend(bar, cur, (r0, c0, r1, c1))
                used.append(bl)
                continue
            partner = self._pending_partner(bl)
            if partner is not None:
                self.bars.append(self._make_bar(partner, bl, cur))
                used.append(bl)
                continue
            axis = self._edge_bar(bl, cur)
            if axis is not None:
                lo, hi = (r0, r1) if axis == "h" else (c0, c1)
                st, en = (c0, c1) if axis == "h" else (r0, r1)
                bar = HudBar(a, b, axis, lo, hi, st, en, 1, True)
                self._extend(bar, cur, (r0, c0, r1, c1))
                self.bars.append(bar)
                used.append(bl)
                continue
            new_pending.append(bl)
        self.pending = (self.pending + new_pending)[-24:]
        self._rebuild()
        return used

    def is_tick_only(self, prev: np.ndarray, cur: np.ndarray) -> bool:
        """True if every changed cell is either masked or a single small unattributed tick-like blob on
        the outermost frame line (first-step heuristic for no-op detection before confirmation)."""
        ch = (prev != cur) & ~self._mask
        if not ch.any():
            return True
        blobs = self._tick_blobs(prev, cur, ch)
        tot = sum((r1 - r0 + 1) * (c1 - c0 + 1) for _, _, r0, c0, r1, c1 in blobs)
        if len(blobs) != 1 or tot != int(ch.sum()):
            return False
        _, _, r0, c0, r1, c1 = blobs[0]
        return r0 in (0, H - 1) or r1 in (0, H - 1) or c0 in (0, W - 1) or c1 in (0, W - 1)

    def describe(self) -> str:
        return "; ".join(b.text() for b in self.bars)

    # -- internals ---------------------------------------------------------------------------
    def _tick_blobs(self, prev, cur, ch) -> list[tuple[int, int, int, int, int, int]]:
        """8-connected blobs of changed cells that look like ticks (small, thin, one transition)."""
        n = int(ch.sum())
        if n == 0 or n > 400:
            return []
        ys, xs = np.nonzero(ch)
        pts = list(zip(ys.tolist(), xs.tolist()))
        seen = set()
        pset = set(pts)
        out = []
        for p in pts:
            if p in seen:
                continue
            stack, blob = [p], []
            seen.add(p)
            while stack:
                y, x = stack.pop()
                blob.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        q = (y + dy, x + dx)
                        if q in pset and q not in seen:
                            seen.add(q)
                            stack.append(q)
            if len(blob) > self.max_tick:
                continue
            trs = {(int(prev[y, x]), int(cur[y, x])) for y, x in blob}
            if len(trs) != 1:
                continue
            r0 = min(y for y, _ in blob)
            r1 = max(y for y, _ in blob)
            c0 = min(x for _, x in blob)
            c1 = max(x for _, x in blob)
            if min(r1 - r0, c1 - c0) + 1 > 3 or (r1 - r0 + 1) * (c1 - c0 + 1) != len(blob):
                continue
            a, b = trs.pop()
            out.append((a, b, r0, c0, r1, c1))
        # a timer tick is isolated (no other change within 2 cells) and never comes with the reverse
        # transition on its own line band in the same step (a moving thin object produces both edges)
        good = []
        for bl in out:
            a, b, r0, c0, r1, c1 = bl
            win = ch[max(0, r0 - 2):r1 + 3, max(0, c0 - 2):c1 + 3]
            if int(win.sum()) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            rev = False
            for o in out:
                if o is bl or (o[0], o[1]) != (b, a):
                    continue
                if (o[2] <= r1 + 1 and o[4] >= r0 - 1) or (o[3] <= c1 + 1 and o[5] >= c0 - 1):
                    rev = True
                    break
            if not rev:
                good.append(bl)
        return good

    def _bar_for(self, bl) -> Optional[HudBar]:
        a, b, r0, c0, r1, c1 = bl
        for bar in self.bars:
            if {a, b} - {bar.a, bar.b} and not (bar.a == a or bar.b == b):
                continue
            if bar.axis == "h":
                if r0 >= bar.lo - 1 and r1 <= bar.hi + 1 and c1 >= bar.start - self.max_gap and c0 <= bar.end + self.max_gap:
                    return bar
            else:
                if c0 >= bar.lo - 1 and c1 <= bar.hi + 1 and r1 >= bar.start - self.max_gap and r0 <= bar.end + self.max_gap:
                    return bar
        return None

    def _pending_partner(self, bl):
        a, b, r0, c0, r1, c1 = bl
        for p in reversed(self.pending):
            pa, pb, q0, d0, q1, d1 = p
            if (pa, pb) != (a, b):
                continue
            same_rows = (q0, q1) == (r0, r1)
            same_cols = (d0, d1) == (c0, c1)
            if same_rows and not same_cols and min(abs(c0 - d1), abs(d0 - c1)) <= self.max_gap:
                return p
            if same_cols and not same_rows and min(abs(r0 - q1), abs(q0 - r1)) <= self.max_gap:
                return p
        return None

    def _make_bar(self, p, bl, cur) -> HudBar:
        a, b, r0, c0, r1, c1 = bl
        _, _, q0, d0, q1, d1 = p
        if (q0, q1) == (r0, r1):
            bar = HudBar(a, b, "h", r0, r1, min(c0, d0), max(c1, d1), 2, True)
        else:
            bar = HudBar(a, b, "v", c0, c1, min(r0, q0), max(r1, q1), 2, True)
        self._extend(bar, cur, (r0, c0, r1, c1))
        try:
            self.pending.remove(p)
        except ValueError:
            pass
        return bar

    def _edge_bar(self, bl, cur) -> Optional[str]:
        """Axis of a bar when one tick sits on the outermost frame row/column whose full line holds only
        the two bar colours (first-step confirmation), else None."""
        a, b, r0, c0, r1, c1 = bl
        cols = {a, b}
        if r0 == r1 and r0 in (0, H - 1) and set(np.unique(cur[r0]).tolist()) <= cols:
            return "h"
        if c0 == c1 and c0 in (0, W - 1) and set(np.unique(cur[:, c0]).tolist()) <= cols:
            return "v"
        return None

    def _extend(self, bar: HudBar, cur: np.ndarray, tick) -> None:
        """Grow the along-axis extent over contiguous cells whose band is entirely bar-coloured."""
        cols = np.array(sorted({bar.a, bar.b}))
        if bar.axis == "h":
            band = cur[bar.lo:bar.hi + 1, :]
            ok = np.isin(band, cols).all(axis=0)
            lo, hi = min(bar.start, tick[1]), max(bar.end, tick[3])
        else:
            band = cur[:, bar.lo:bar.hi + 1]
            ok = np.isin(band, cols).all(axis=1)
            lo, hi = min(bar.start, tick[0]), max(bar.end, tick[2])
        while lo - 1 >= 0 and ok[lo - 1]:
            lo -= 1
        while hi + 1 < len(ok) and ok[hi + 1]:
            hi += 1
        bar.start, bar.end = min(bar.start, lo), max(bar.end, hi)

    def _rebuild(self) -> None:
        m = np.zeros((H, W), bool)
        for bar in self.bars:
            m |= bar.region()
        self._mask = m


def masked_hash(frame: Any, mask: Optional[np.ndarray] = None, level: Any = None) -> str:
    """Deterministic hash of a frame with HUD cells blanked (and an optional level tag)."""
    g = to_grid(frame).astype(np.int8)
    if mask is not None and mask.any():
        g = g.copy()
        g[mask] = -1
    pre = b"" if level is None else str(level).encode() + b"|"
    return f"{zlib.crc32(pre + g.tobytes()) & 0xffffffff:08x}{zlib.adler32(g.tobytes()) & 0xffffffff:08x}"


# ------------------------------------------------------------------------------------------------
# avatar detection
# ------------------------------------------------------------------------------------------------
class AvatarTracker:
    """Learns which object(s) move consistently with the non-click actions.

    Evidence is keyed by the moved component's colour (shape can change when an avatar turns). For each
    (colour, action) we count displacement directions; a colour is controlled when it moved under >= 2
    steps and, per action, its dominant direction explains >= 75% of its moves. Mass scrolls (most moved
    cells share one displacement and cover > 25% of the frame) are ignored."""

    def __init__(self):
        self.ev: dict[int, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.step_mag: dict[int, Counter] = defaultdict(Counter)
        self.last_pos: dict[int, Comp] = {}
        self.steps_seen: Counter = Counter()

    def update(self, aid: int, events: list[ObjEvent], changed_n: int) -> None:
        if aid in (0, 6, 7) or aid < 0:
            return
        self.steps_seen[aid] += 1
        moved = [e for e in events if e.kind == "moved"]
        if not moved:
            return
        tot = sum(e.before.n for e in moved)
        if changed_n > 0.25 * H * W:
            disp = Counter()
            for e in moved:
                disp[(e.dx, e.dy)] += e.before.n
            if disp.most_common(1)[0][1] > 0.6 * tot and len(moved) > 3:
                return  # camera scroll
        # one vote per colour per step (largest moved component of that colour)
        best: dict[int, ObjEvent] = {}
        for e in moved:
            if e.color not in best or e.before.n > best[e.color].before.n:
                best[e.color] = e
        for colr, e in best.items():
            d = (int(np.sign(e.dx)), int(np.sign(e.dy)))
            self.ev[colr][aid][d] += 1
            self.step_mag[colr][max(abs(e.dx), abs(e.dy))] += 1
            self.last_pos[colr] = e.after

    def ranking(self) -> list[tuple[float, int]]:
        out = []
        for colr, per in self.ev.items():
            moves = sum(sum(c.values()) for c in per.values())
            if moves < 2:
                continue
            cons = sum(c.most_common(1)[0][1] for c in per.values())
            if cons < 0.75 * moves:
                continue
            dirs = {c.most_common(1)[0][0] for c in per.values()}
            score = cons + 0.5 * len(dirs) - 0.5 * (moves - cons)
            out.append((score, colr))
        out.sort(key=lambda t: (-t[0], t[1]))
        return out

    def avatars(self, max_n: int = 2) -> list[int]:
        r = self.ranking()
        if not r:
            return []
        top = r[0][0]
        return [c for s, c in r[:max_n] if s >= 0.6 * top]

    def controls(self, colr: int) -> dict[int, str]:
        """action id -> direction name for a controlled colour."""
        out = {}
        for aid, c in sorted(self.ev.get(colr, {}).items()):
            d = c.most_common(1)[0][0]
            out[aid] = DIR_NAMES.get(d, f"({d[0]:+d},{d[1]:+d})")
        return out

    def locate(self, seg: Segmentation, colr: int) -> Optional[Comp]:
        """Current component of an avatar colour: same shape as last seen, else nearest same colour."""
        last = self.last_pos.get(colr)
        cands = [c for c in seg.comps if c.color == colr and not c.bg]
        if not cands:
            return None
        if last is not None:
            same = [c for c in cands if c.shape == last.shape]
            pool = same or cands
            return min(pool, key=lambda c: (abs(c.cy - last.cy) + abs(c.cx - last.cx), c.id))
        return max(cands, key=lambda c: (c.n, -c.id))


# ------------------------------------------------------------------------------------------------
# click candidates
# ------------------------------------------------------------------------------------------------
@dataclass
class ClickCandidate:
    x: int
    y: int
    comp: Comp
    ctype: str
    count: int  # instances of this type in the frame
    score: float
    status: str  # live | dead | untested
    rank_type: int = 0  # rank of this object's type among types

    def short(self) -> str:
        c = self.comp
        mult = f" x{self.count}" if self.count > 1 else ""
        st = {"live": "+", "dead": "-", "untested": ""}[self.status]
        return f"(x{self.x},y{self.y}) c{c.color} {c.w}x{c.h}{mult}{st}"


def click_point(c: Comp, seg: Segmentation) -> tuple[int, int]:
    """A display (x, y) inside component c: the member cell nearest its centroid, snapped to the
    centre of its s x s grid block when that block lies in the component."""
    s, oy, ox = seg.scale
    m = seg.labels[c.r0:c.r1 + 1, c.c0:c.c1 + 1] == c.id
    ys, xs = np.nonzero(m)
    ys = ys + c.r0
    xs = xs + c.c0
    d = (ys - c.cy) ** 2 + (xs - c.cx) ** 2
    i = int(np.lexsort((xs, ys, d))[0])
    y, x = int(ys[i]), int(xs[i])
    if s > 1:
        by = oy + ((y - oy) // s) * s
        bx = ox + ((x - ox) // s) * s
        cy, cx = by + s // 2, bx + s // 2
        if 0 <= cy < H and 0 <= cx < W and seg.labels[cy, cx] == c.id:
            y, x = cy, cx
    return x, y


def rank_click_candidates(seg: Segmentation, hud_mask: Optional[np.ndarray] = None,
                          dead: Optional[Counter] = None, live: Optional[Counter] = None,
                          max_n: int = 64, include_bg: bool = True) -> list[ClickCandidate]:
    """One candidate per (non-background) object, deduplicated per grid cell, ranked by
    rarity / size / shape; object types whose clicks changed nothing (``dead`` without ``live``) sink to
    the bottom. The list is type-diverse: the best instance of every type first, then the duplicates."""
    dead = dead or Counter()
    live = live or Counter()
    s, oy, ox = seg.scale
    objs = [c for c in seg.comps if not c.bg]
    if hud_mask is not None and hud_mask.any():
        objs = [c for c in objs if not hud_mask[int(round(c.cy)), int(round(c.cx))] or c.n > 64]
    if not objs and not include_bg:
        return []
    fg_total = sum(c.n for c in objs) or 1
    color_px = Counter()
    for c in objs:
        color_px[c.color] += c.n
    type_count = Counter(c.ctype for c in objs)
    cell = float(s * s)
    scored = []
    for c in objs:
        cells = c.n / cell
        share = color_px[c.color] / fg_total
        sc = 1.5 * (1.0 - share) + 0.8 * min(1.0, -math.log10(max(share, 1e-3)) / 2.0)
        if cells < 1:
            sc -= 2.0
        elif cells == 1:
            sc += 0.2
        elif cells <= 64:
            sc += 1.0
        elif cells <= 256:
            sc += 0.2
        else:
            sc -= 1.0
        m = type_count[c.ctype]
        if m == 1:
            sc += 0.5
        elif m <= 12:
            sc += 0.3
        else:
            sc -= 1.5
        fill = c.n / float(c.w * c.h)
        if fill >= 0.6:
            sc += 0.3
        if c.color == seg.bg_color:  # holes / gaps in the background colour are rarely targets
            sc -= 1.5
        if c.r0 == 0 or c.c0 == 0 or c.r1 == H - 1 or c.c1 == W - 1:
            sc -= 0.5
        if live[c.ctype]:
            status = "live"
            sc += 3.0
        elif dead[c.ctype]:
            status = "dead"
            sc -= 8.0 + dead[c.ctype]
        else:
            status = "untested"
        scored.append((sc, c, status))
    scored.sort(key=lambda t: (-t[0], t[1].r0, t[1].c0))
    out: list[ClickCandidate] = []
    seen_cells: set = set()
    type_rank: dict[str, int] = {}
    firsts, dups = [], []
    for sc, c, status in scored:
        x, y = click_point(c, seg)
        cellkey = ((y - oy) // s, (x - ox) // s) if s > 1 else (y, x)
        if cellkey in seen_cells:
            continue
        seen_cells.add(cellkey)
        if c.ctype not in type_rank:
            type_rank[c.ctype] = len(type_rank)
            firsts.append(ClickCandidate(x, y, c, c.ctype, type_count[c.ctype], round(sc, 3), status))
        else:
            dups.append(ClickCandidate(x, y, c, c.ctype, type_count[c.ctype], round(sc, 3), status))
    # live/untested duplicates before dead representatives
    good_firsts = [k for k in firsts if k.status != "dead"]
    dead_firsts = [k for k in firsts if k.status == "dead"]
    good_dups = [k for k in dups if k.status != "dead"]
    dead_dups = [k for k in dups if k.status == "dead"]
    ordered = good_firsts + good_dups + dead_firsts + dead_dups
    for k in ordered:
        k.rank_type = type_rank[k.ctype]
    if include_bg:
        bgc = [c for c in seg.comps if c.bg and c.color == seg.bg_color]
        if bgc:
            c = max(bgc, key=lambda q: (q.n, -q.id))
            x, y = click_point(c, seg)
            st = "live" if live[f"bg:{c.color}"] else ("dead" if dead[f"bg:{c.color}"] else "untested")
            ordered.append(ClickCandidate(x, y, c, f"bg:{c.color}", 1, -9.0, st, len(type_rank)))
    return ordered[:max_n]


def candidates_text(cands: list[ClickCandidate], k: int = 8, max_pos: int = 4) -> str:
    """Group candidates by type: ``c11 3x3 x8 @(36,36)(44,36)..; c14 8x8 @(4,29)``."""
    groups: dict[str, list[ClickCandidate]] = {}
    for cd in cands:
        if cd.ctype.startswith("bg:"):
            continue
        groups.setdefault(cd.ctype, []).append(cd)
    parts = []
    for ct, lst in list(groups.items())[:k]:
        c = lst[0].comp
        st = {"live": "+", "dead": "-", "untested": ""}[lst[0].status]
        pos = "".join(f"({q.x},{q.y})" for q in lst[:max_pos]) + ("…" if len(lst) > max_pos else "")
        mult = f"x{lst[0].count} " if lst[0].count > 1 else ""
        parts.append(f"c{c.color} {c.w}x{c.h}{st} {mult}@{pos}")
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------------
# animation summary
# ------------------------------------------------------------------------------------------------
def animation_summary(prev: Any, frames: Sequence[Any], mask: Optional[np.ndarray] = None) -> dict:
    """n_frames, the bbox of cells touched during the animation, the transient cells (changed at some
    point but back to their pre-step value at the end) and the frame with the largest deviation."""
    fr = to_frames(frames)
    out = {"n_frames": len(fr), "touched_bbox": None, "transient_bbox": None, "transient_n": 0, "peak_frame": None}
    if len(fr) <= 1 or prev is None:
        return out
    p = to_grid(prev)
    touched = np.zeros(p.shape, bool)
    best = (-1, None)
    for i, f in enumerate(fr):
        d = f != p
        if mask is not None:
            d &= ~mask
        touched |= d
        k = int(d.sum())
        if k > best[0]:
            best = (k, i)
    final = fr[-1] != p
    if mask is not None:
        final &= ~mask
    trans = touched & ~final
    out.update(touched_bbox=_bbox(touched) if touched.any() else None,
               transient_bbox=_bbox(trans) if trans.any() else None,
               transient_n=int(trans.sum()), peak_frame=best[1])
    return out


# ------------------------------------------------------------------------------------------------
# stateful facade
# ------------------------------------------------------------------------------------------------
@dataclass
class Percept:
    action: tuple[int, Optional[int], Optional[int]]
    delta: Delta
    events: list[ObjEvent]
    anim: dict
    noop: bool  # nothing changed except HUD/timer cells
    level_changed: bool
    state: Optional[str]
    clicked: Optional[Comp] = None
    avatar_moves: list[ObjEvent] = field(default_factory=list)
    text: str = ""


class Perceiver:
    """Per-game perception state: HUD bars, avatar evidence, dead/live click types, grid scale.

    Typical use::

        P = Perceiver()
        P.reset(frame0)                                   # at game start
        pc = P.observe(prev_frame, obs.frames, action, level=obs.levels_completed, state=obs.state)
        prompt += pc.text                                 # compact <=150-token description
        cands = P.candidates(obs.frame)                   # ranked ACTION6 targets
    """

    def __init__(self):
        self.hud = HudTracker()
        self.avatar = AvatarTracker()
        self.dead: Counter = Counter()
        self.live: Counter = Counter()
        self.level: Any = None
        self._edges_h = np.zeros(W, np.int64)
        self._edges_v = np.zeros(H, np.int64)
        self._scale = (1, 0, 0)
        self._seg_cache: tuple[Optional[bytes], Optional[Segmentation]] = (None, None)
        self.level_start: Optional[np.ndarray] = None
        self.n_steps = 0

    # -- geometry ---------------------------------------------------------------------------
    def scale(self) -> tuple[int, int, int]:
        return self._scale

    def _add_edges(self, g: np.ndarray) -> None:
        hc, vr = _edge_hist(g, self.hud.mask())
        self._edges_h += hc
        self._edges_v += vr
        self._scale = _best_scale(self._edges_h, self._edges_v)

    def seg(self, frame: Any) -> Segmentation:
        g = to_grid(frame)
        key = g.astype(np.int8).tobytes() + self.hud.mask().tobytes()
        if self._seg_cache[0] == key:
            return self._seg_cache[1]
        s = segment(g, hud_mask=self.hud.mask(), scale=self._scale)
        self._seg_cache = (key, s)
        return s

    def hud_mask(self) -> np.ndarray:
        return self.hud.mask()

    def state_hash(self, frame: Any, level: Any = None) -> str:
        return masked_hash(frame, self.hud.mask(), self.level if level is None else level)

    # -- lifecycle --------------------------------------------------------------------------
    def reset(self, frame: Any, level: Any = 0) -> None:
        g = to_grid(frame)
        self.level = level
        self.level_start = g.copy()
        self._edges_h[:] = 0
        self._edges_v[:] = 0
        self._add_edges(g)

    def observe(self, prev: Any, frames: Any, action: Any, level: Any = None, state: Optional[str] = None) -> Percept:
        fr = to_frames(frames)
        aid, x, y = norm_action(action)
        if not fr:
            d = Delta(np.zeros((H, W), bool), 0, None, Counter())
            pc = Percept((aid, x, y), d, [], {"n_frames": 0}, True, False, state)
            pc.text = f"{action_label(action)}: ignored (no frame; state {state})"
            return pc
        cur = fr[-1]
        p = to_grid(prev) if prev is not None else cur
        level_changed = level is not None and self.level is not None and level != self.level
        seg_prev = self.seg(p)
        clicked = seg_prev.comp_at(x, y) if aid == 6 and x is not None else None
        if level_changed or aid == 0:
            # new level / reset: no causal diff; restart per-level geometry
            self.hud.reset_pending()
            if level is not None:
                self.level = level
            self.level_start = cur.copy()
            self._edges_h[:] = 0
            self._edges_v[:] = 0
            self._add_edges(cur)
            d = frame_delta(p, cur, self.hud.mask())
            pc = Percept((aid, x, y), d, [], animation_summary(p, fr, self.hud.mask()), False, level_changed, state,
                         clicked)
            pc.text = self._text(pc, p, cur)
            self.n_steps += 1
            return pc
        if self.level is None:
            self.level = level
        self.hud.update(p, cur)
        mask = self.hud.mask()
        d = frame_delta(p, cur, mask)
        noop = d.n == 0 or (self.n_steps < 3 and self.hud.is_tick_only(p, cur))
        events: list[ObjEvent] = []
        if d.n:
            events = object_diff(self.seg(p), self.seg(cur), d.changed)
        self._add_edges(cur)
        self.avatar.update(aid, events, d.n)
        if aid == 6:
            key = clicked.ctype if clicked is not None and not clicked.bg else (
                f"bg:{clicked.color}" if clicked is not None else "offscreen")
            if noop and len(fr) <= 1:
                self.dead[key] += 1
            elif not noop:
                self.live[key] += 1
        av = set(self.avatar.avatars())
        pc = Percept((aid, x, y), d, events, animation_summary(p, fr, mask), noop, False, state, clicked,
                     [e for e in events if e.kind == "moved" and e.color in av])
        pc.text = self._text(pc, p, cur)
        self.n_steps += 1
        return pc

    # -- outputs ----------------------------------------------------------------------------
    def candidates(self, frame: Any, max_n: int = 64) -> list[ClickCandidate]:
        return rank_click_candidates(self.seg(frame), self.hud.mask(), self.dead, self.live, max_n)

    def avatar_text(self, frame: Any) -> str:
        cols = self.avatar.avatars()
        if not cols:
            return ""
        seg = self.seg(frame)
        parts = []
        for colr in cols:
            c = self.avatar.locate(seg, colr)
            ctl = " ".join(f"A{a}={dn}" for a, dn in self.avatar.controls(colr).items())
            where = f" @(x{c.c0},y{c.r0}) {c.w}x{c.h}" if c is not None else ""
            parts.append(f"c{colr}{where} [{ctl}]")
        return "avatar " + "; ".join(parts)

    def _text(self, pc: Percept, prev: np.ndarray, cur: np.ndarray, max_chars: int = 560) -> str:
        head = action_label(pc.action)
        if pc.action[0] == 6 and pc.clicked is not None:
            c = pc.clicked
            head += f" on {'bg ' if c.bg else ''}{c.short()}"
        bits = []
        if pc.state in ("GAME_OVER", "WIN"):
            bits.append(pc.state)
        if pc.level_changed:
            bits.append(f"NEW LEVEL {pc.level_changed and self.level}")
        if pc.anim.get("n_frames", 1) > 1:
            a = pc.anim
            t = f"anim {a['n_frames']}f"
            if a.get("transient_bbox"):
                t += f" transient {_fmt_bbox(a['transient_bbox'])} ({a['transient_n']})"
            bits.append(t)
        if pc.level_changed or pc.action[0] == 0:
            return f"{head}: " + ", ".join(bits + [f"{pc.delta.n} cells differ"])
        if pc.noop:
            bits.append("no effect" + (" (timer only)" if pc.delta.hud_changed or pc.delta.n else ""))
            if pc.action[0] == 6 and pc.clicked is not None and not pc.clicked.bg:
                bits.append("type demoted")
            return f"{head}: " + ", ".join(bits)
        bits.append(f"{pc.delta.n} cells {_fmt_bbox(pc.delta.bbox)}")
        text = f"{head}: " + ", ".join(bits)
        ev = [e.text() for e in pc.events[:4]]
        if len(pc.events) > 4:
            ev.append(f"+{len(pc.events) - 4} more")
        if ev:
            text += ". " + "; ".join(ev)
        if pc.avatar_moves:
            text += ". " + self.avatar_text(cur)
        rest = max_chars - len(text) - 2
        if rest > 40:
            dt = delta_text(prev, cur, self.hud.mask(), max_cells=10, max_chars=min(rest, 220), delta=pc.delta)
            if len(dt) <= rest:
                text += ". " + dt
        return text[:max_chars]

    def describe(self, prev: Any, cur: Any, action: Any, level: Any = None, state: Optional[str] = None) -> str:
        return self.observe(prev, cur, action, level=level, state=state).text


def describe(prev: Any, cur: Any, action: Any, perceiver: Optional[Perceiver] = None, level: Any = None,
             state: Optional[str] = None) -> str:
    """Compact (<= ~150 tokens) text of what ``action`` did between ``prev`` and ``cur`` (cur may be the
    list of animation frames of the step). Pass the game's ``Perceiver`` to use its accumulated HUD mask,
    avatar evidence and click history; without one a fresh perceiver is used (stateless call)."""
    P = perceiver if perceiver is not None else Perceiver()
    if perceiver is None:
        P.reset(prev, level)
    return P.describe(prev, cur, action, level=level, state=state)
