"""Normalise ARC-AGI-3 trajectories from several sources into one step-level parquet schema.

Sources handled:
  * arcprize.org replay recordings (jsonl, one line per step: {"timestamp", "data": {...}})
    -> frontier-model runs (with `action_input.reasoning`) and human/playtest runs (no reasoning).
  * TAAF / Duck harness run artifacts (`artifacts/<game>_p0_events.jsonl` with board + action).

Schema (one row per environment step):
  source, run_id, game_id, game_family, step, action, x, y, reasoning, state, levels_completed,
  win_levels, available_actions, frame (64x64 uint8 flattened -> bytes), frame_shape, n_anim_frames, timestamp

Usage:
  python tools/build_trace_dataset.py --replays . --taaf-artifacts <results/artifacts> --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ACTION_NAMES = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4",
                5: "ACTION5", 6: "ACTION6", 7: "ACTION7"}


def _frame_bytes(frame) -> tuple[bytes | None, list[int] | None, int]:
    """Return (last frame as uint8 bytes, shape, number of animation frames)."""
    if frame is None:
        return None, None, 0
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim == 3:  # list of animation frames -> keep the final one
        n = arr.shape[0]
        arr = arr[-1]
    else:
        n = 1
    return arr.tobytes(), list(arr.shape), n


def _action_of(d: dict) -> tuple[str | None, int | None, int | None, str | None]:
    ai = d.get("action_input") or {}
    aid = ai.get("id")
    if isinstance(aid, int):
        aid = ACTION_NAMES.get(aid, str(aid))
    data = ai.get("data") or {}
    reasoning = ai.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = json.dumps(reasoning)
    return aid, data.get("x"), data.get("y"), reasoning


def load_replay(path: str) -> list[dict]:
    rows = []
    run_id = Path(path).stem
    game_id = None
    with open(path) as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    for step, rec in enumerate(lines):
        d = rec.get("data", rec)
        gid = d.get("game_id") or ((d.get("action_input") or {}).get("data") or {}).get("game_id")
        game_id = gid or game_id
        aid, x, y, reasoning = _action_of(d)
        fb, shape, nfr = _frame_bytes(d.get("frame"))
        has_reasoning = reasoning is not None
        rows.append(dict(
            run_id=run_id, game_id=game_id, step=step, action=aid, x=x, y=y, reasoning=reasoning,
            state=d.get("state"), levels_completed=d.get("levels_completed"), win_levels=d.get("win_levels"),
            available_actions=json.dumps(d.get("available_actions")), frame=fb, frame_shape=json.dumps(shape),
            n_anim_frames=nfr, timestamp=rec.get("timestamp"), _has_reasoning=has_reasoning,
        ))
    src = "arcprize_replay_model" if any(r["_has_reasoning"] for r in rows) else "arcprize_replay_human"
    for r in rows:
        r.pop("_has_reasoning")
        r["source"] = src
        r["game_id"] = r["game_id"] or game_id
        r["game_family"] = (r["game_id"] or "")[:4]
    return rows


_MOUSE_RE = __import__("re").compile(r"row=(\d+),\s*col=(\d+)")


def load_taaf_events(path: str, run_label: str) -> list[dict]:
    """TAAF events: 'initial' | 'analysis' (LLM transcript) | 'action' (board after action).

    The most recent analysis transcript is attached as `reasoning` to the next action row.
    MOUSE(row=r, col=c) is ACTION6 with x=c, y=r.
    """
    rows = []
    game_id = Path(path).name.split("_p0")[0]
    step = 0
    pending_transcript = None
    with open(path) as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = e.get("type")
            if etype == "analysis":
                pending_transcript = e.get("transcript")
                continue
            if etype not in ("initial", "action"):
                continue
            x = y = None
            m = _MOUSE_RE.search(e.get("action_display") or "")
            if m:
                y, x = int(m.group(1)), int(m.group(2))
            fb, shape, nfr = _frame_bytes(e.get("board"))
            rows.append(dict(
                source=f"taaf_run:{run_label}", run_id=f"{run_label}:{game_id}", game_id=game_id,
                game_family=game_id[:4], step=step,
                action=e.get("action_name") if etype == "action" else "INITIAL", x=x, y=y,
                reasoning=pending_transcript if etype == "action" else None,
                state=e.get("state"), levels_completed=(e.get("level") or 1) - 1,
                win_levels=None, available_actions=None, frame=fb, frame_shape=json.dumps(shape),
                n_anim_frames=nfr, timestamp=None, board_changed=e.get("board_changed"),
                level_completed=e.get("level_completed"), game_over=e.get("game_over"),
                analysis_step=e.get("analysis_step"), batch_index=e.get("batch_index"),
                batch_size=e.get("batch_size"),
            ))
            if etype == "action":
                pending_transcript = None
            step += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=None, help="dir with arcprize replay *.json(l)")
    ap.add_argument("--taaf-artifacts", default=None, help="dir with *_events.jsonl from a TAAF run")
    ap.add_argument("--taaf-label", default="run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.replays:
        rows = []
        for f in sorted(glob.glob(os.path.join(args.replays, "*.json")) + glob.glob(os.path.join(args.replays, "*.jsonl"))):
            try:
                rows.extend(load_replay(f))
            except Exception as exc:  # keep going on odd files
                print("skip", f, exc)
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(args.out, "replays.parquet"), index=False)
        print("replays:", df.shape, df.groupby(["source", "game_id"]).size().to_dict())

    if args.taaf_artifacts:
        rows = []
        for f in sorted(glob.glob(os.path.join(args.taaf_artifacts, "*_events.jsonl"))):
            rows.extend(load_taaf_events(f, args.taaf_label))
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(args.out, f"taaf_{args.taaf_label}.parquet"), index=False)
        print("taaf:", df.shape)


if __name__ == "__main__":
    main()
