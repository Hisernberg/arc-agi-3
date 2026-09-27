#!/usr/bin/env python3
"""Serving benchmark for the Duck agent workload (backlog B02, experiment EXP-001).

Replays real Duck traffic (the 09-22 Flash-Next validation run) as K concurrent multi-turn agents
against an OpenAI-compatible server (vLLM) and measures what matters for ARC-AGI-3: how many LLM
decisions each game gets per hour of wall clock.

Sub-commands
------------
build-pack   Parse `results/transcripts/*.txt` (+ `prompts/*.log`, `artifacts/*_events.jsonl`,
             `vllm-metrics-final.prom`) into a compact replay pack (`*.jsonl.gz`, one line per game).
run          Replay the pack against a server (`--base-url`) or an in-process mock (`--mock`), write
             `report.json`, `summary.md`, `requests.jsonl.gz` and `metrics_samples.jsonl` to `--out`.
compare      Compare several `report.json` files (ratios vs a baseline, EXP-001 verdict).
mock-server  Run the mock OpenAI/vLLM server standalone (used by `serving/launch.py` profile `mock_cpu`).

Agent model (mirrors ARC3-Inference `tool_agent.py` as shipped in the 09-22 run)
-------------------------------------------------------------------------------
* Each agent replays one recorded game turn by turn. A turn starts with the recorded user prompt
  (+ the recorded board rendered as a 4x-upscaled PNG, like `MULTIMODAL_CONTEXT=current_grid`).
* Inside a turn the agent keeps calling the model until the model's own tool call contains
  `action(` (Duck: `step_executed`), or `yield_seconds` (60 s) elapsed (Duck: `turn_time_budget`).
  Tool results are the recorded ones (realistic lengths); the assistant messages appended are the
  model's *actual* replies (reasoning + content + tool calls), so context grows realistically.
  `--history recorded` instead appends the recorded assistant turns (identical prompt stream for
  every server; request count per turn follows the recording).
* Context: Duck's estimator (json chars / 3, AGENTFIX-F1: 120 tokens per image), budget
  `context_window - reply_reserve(512) - safety_margin(6144)`, oldest history blocks dropped first,
  at most 30 assistant turns persisted, images kept on the newest `images_in_history` user turns.
* Request: temperature 0.6, top_p 0.95, top_k 20, `enable_thinking`, tools=[python], tool_choice
  auto, `max_tokens` = server default (Duck `LOCAL_ANALYZER_MAX_OUTPUT=0`) unless `--max-tokens`.

"Decision" = a completed request whose reply is a valid `python` tool call containing `action(`
(the call that executes environment actions in Duck). Headline metric: decisions per game-hour
(= per agent per hour of the measurement window).

The module only needs the standard library; it uses aiohttp or httpx when importable.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import math
import os
import random
import re
import statistics
import struct
import sys
import threading
import time
import zlib
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

PACK_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = REPO_ROOT / "automation" / "serving_pack" / "duck_0922.jsonl.gz"

# Duck constants (ARC3-Inference tool_agent.py + the 09-22 analyzer status block).
DUCK_REPLY_RESERVE = 512
DUCK_SAFETY_MARGIN = 6144  # "request_safety_margin_tokens: 6144" in the 09-22 transcripts
DUCK_PERSISTENT_ASSISTANT_TURNS = 30
DUCK_IMAGE_TOKENS = 120  # AGENTFIX F1 flat per-image estimate
DUCK_YIELD_SECONDS = 60.0
DUCK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)
FOLLOWUP_PROMPT = (
    "You have not acted yet. Investigate first. "
    "Then investigate and revise your working world model of what the level contains, what actions appear to do, "
    "what the current goal seems to be, and what plan looks best. "
    "Call the `python` tool with code that inspects `current_frame`, `previous_frame`, `last_transition`, `history`, "
    "or `valid_actions`, then call `action(actions)` inside Python with the best valid action or ordered batch that "
    "your code selected. " + TOOL_CALL_FORMAT_GUIDANCE
)
DEFAULT_PYTHON_TOOL_DESCRIPTION = (
    "Run one ephemeral Python snippet against preloaded ASCII game state. Available globals: `current_frame`, "
    "`previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`, and "
    "`action(actions)` for executing one or more real environment actions."
)
ACTION_RE = re.compile(r"\baction\s*\(")
MARKUP_RE = re.compile(r"<tool_call>|<function=|<parameter=")

# ARC palette (vision_context.py) -> RGB
ARC_RGB = {
    0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102),
    4: (51, 51, 51), 5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204),
    8: (249, 60, 49), 9: (30, 147, 255), 10: (136, 216, 241), 11: (255, 220, 0),
    12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48), 15: (163, 86, 214),
}


# ----------------------------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(statistics.fmean(vals)) if vals else None


def rnd(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def open_text(path: Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8")
    return open(path, mode, encoding=None if "b" in mode else "utf-8")


def grid_to_png(grid: list[list[int]], scale: int = 4) -> bytes:
    """Encode an ARC grid as an RGB PNG (nearest-neighbour upscale), stdlib only."""
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0)
    if rows == 0 or cols == 0:
        raise ValueError("empty grid")
    raw = bytearray()
    for r in grid:
        line = bytearray()
        for c in range(cols):
            v = r[c] if c < len(r) else 0
            line += bytes(ARC_RGB.get(int(v), ARC_RGB[0])) * scale
        scanline = b"\x00" + bytes(line)
        raw += scanline * scale

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", cols * scale, rows * scale, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


# ----------------------------------------------------------------------------------------------
# Pack builder
# ----------------------------------------------------------------------------------------------
TURN_RE = re.compile(r"^--- analysis_step=(\S+) \| action=(\S+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$")
SECTION_LABELS = {
    "SYSTEM PROMPT", "USER PROMPT", "MODEL RESPONSE META", "THINKING", "ASSISTANT",
    "TOOL CALL: python", "TOOL RESULT: python", "ANALYZER STATUS",
}
SECTION_RE = re.compile(r"^\[([A-Z][A-Z _]*(?:: [a-z_]+)?)\]$")


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def split_transcript(text: str) -> list[dict[str, Any]]:
    """Split a Duck transcript into turns, each a list of (label, body) sections."""
    turns: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    label: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if cur is not None and label is not None:
            cur["sections"].append((label, "\n".join(buf).strip("\n")))

    for line in text.split("\n"):
        m = TURN_RE.match(line)
        if m:
            flush()
            label, buf = None, []
            cur = {
                "header": line,
                "analysis_step": _to_int(m.group(1)),
                "action": _to_int(m.group(2)),
                "clock": m.group(3),
                "sections": [],
            }
            turns.append(cur)
            continue
        m = SECTION_RE.match(line)
        if m and m.group(1) in SECTION_LABELS and cur is not None:
            flush()
            label, buf = m.group(1), []
            continue
        buf.append(line)
    flush()
    return turns


def _parse_meta(body: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    head, sep, tail = body.partition("raw_tool_calls:")
    for line in head.splitlines():
        key, s, value = line.partition(":")
        if s:
            meta[key.strip()] = value.strip()
    calls: list[dict[str, Any]] = []
    if sep:
        try:
            raw = json.loads(tail.strip() or "[]")
        except json.JSONDecodeError:
            raw = []
        for call in raw if isinstance(raw, list) else []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            calls.append({
                "id": str(call.get("id", "")),
                "name": str(fn.get("name", "")),
                "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments", {})),
            })
    meta["tool_calls"] = calls
    return meta


def _parse_status(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body.splitlines():
        key, s, value = line.partition(":")
        if s and key.strip() in {"step_executed", "message", "request_error", "history_messages", "context_budget_tokens",
                                 "request_safety_margin_tokens", "reply_reserve_tokens", "yield_seconds"}:
            out[key.strip()] = value.strip()
    return out


def parse_turns(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (turns, system_prompts_seen) for one transcript."""
    turns_out: list[dict[str, Any]] = []
    systems: list[str] = []
    for raw in split_transcript(text):
        turn: dict[str, Any] = {
            "analysis_step": raw["analysis_step"], "action": raw["action"], "clock": raw["clock"],
            "header": raw["header"], "user": "", "requests": [], "status": {},
        }
        for label, body in raw["sections"]:
            if label == "SYSTEM PROMPT":
                systems.append(body)
            elif label == "USER PROMPT":
                if not turn["requests"]:
                    turn["user"] = body
                else:
                    turn["requests"][-1]["followup"] = body
            elif label == "MODEL RESPONSE META":
                meta = _parse_meta(body)
                turn["requests"].append({
                    "finish_reason": meta.get("finish_reason", ""),
                    "reasoning": "",
                    "content": "",
                    "tool_calls": meta["tool_calls"],
                    "tool_results": [],
                })
            elif label == "THINKING" and turn["requests"]:
                turn["requests"][-1]["reasoning"] = body
            elif label == "ASSISTANT" and turn["requests"]:
                turn["requests"][-1]["content"] = body
            elif label == "TOOL RESULT: python" and turn["requests"]:
                turn["requests"][-1]["tool_results"].append(body)
            elif label == "ANALYZER STATUS":
                turn["status"] = _parse_status(body)
        for req in turn["requests"]:
            codes = []
            for call in req["tool_calls"]:
                try:
                    codes.append(str(json.loads(call["arguments"]).get("code", "")))
                except (json.JSONDecodeError, AttributeError):
                    pass
            req["action"] = any(ACTION_RE.search(c) for c in codes)
        turns_out.append(turn)
    return turns_out, systems


def _parse_tool_description(prompt_log: Path) -> str | None:
    if not prompt_log.is_file():
        return None
    text = prompt_log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\[AVAILABLE TOOLS\]\n- python: (.*?)\n\n\[MODEL INPUT\]", text, flags=re.S)
    return m.group(1).strip() if m else None


def _load_boards(events_path: Path) -> tuple[dict[str, list[list[int]]], list[tuple[int, list[list[int]]]]]:
    by_header: dict[str, list[list[int]]] = {}
    by_action: list[tuple[int, list[list[int]]]] = []
    if not events_path.is_file():
        return by_header, by_action
    with open(events_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            board = ev.get("board")
            if not isinstance(board, list) or not board:
                continue
            if ev.get("type") == "analysis" and isinstance(ev.get("transcript"), str):
                header = ev["transcript"].lstrip("\n").split("\n", 1)[0].strip()
                by_header.setdefault(header, board)
            action_num = _to_int(ev.get("action_num"))
            if action_num is not None:
                by_action.append((action_num, board))
    by_action.sort(key=lambda x: x[0])
    return by_header, by_action


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    line_re = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+([^\s]+)(\s+\d+)?$")
    label_re = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line.strip())
        if not m:
            continue
        name, labels_raw, value_raw = m.group(1), m.group(3) or "", m.group(4)
        try:
            value = float(value_raw)
        except ValueError:
            continue
        labels = {k: v for k, v in label_re.findall(labels_raw)}
        out.setdefault(name, []).append((labels, value))
    return out


def metric_sum(parsed: dict[str, list[tuple[dict[str, str], float]]], names: str | list[str], **flt: str) -> float | None:
    for name in [names] if isinstance(names, str) else names:
        rows = parsed.get(name)
        if rows is None:
            continue
        total = 0.0
        for labels, value in rows:
            if all(labels.get(k) == v for k, v in flt.items()):
                total += value
        return total
    return None


def recorded_run_stats(results_dir: Path) -> dict[str, Any]:
    """Headline numbers of the recorded run (09-22) from its vLLM final metrics + transcripts."""
    stats: dict[str, Any] = {}
    prom = results_dir / "vllm-metrics-final.prom"
    if prom.is_file():
        p = parse_prometheus(prom.read_text(encoding="utf-8", errors="replace"))
        req = metric_sum(p, "vllm:request_success_total")
        stats.update({
            "requests": req,
            "prompt_tokens": metric_sum(p, "vllm:prompt_tokens_total"),
            "generation_tokens": metric_sum(p, "vllm:generation_tokens_total"),
            "preemptions": metric_sum(p, ["vllm:num_preemptions_total", "vllm:num_preemptions"]),
            "queue_time_s_sum": metric_sum(p, "vllm:request_queue_time_seconds_sum"),
            "e2e_s_sum": metric_sum(p, "vllm:e2e_request_latency_seconds_sum"),
            "spec_draft_tokens": metric_sum(p, "vllm:spec_decode_num_draft_tokens_total"),
            "spec_accepted_tokens": metric_sum(p, "vllm:spec_decode_num_accepted_tokens_total"),
            "mm_cache_queries": metric_sum(p, "vllm:mm_cache_queries_total"),
        })
        if req:
            stats["mean_prompt_tokens"] = (stats["prompt_tokens"] or 0) / req
            stats["mean_generation_tokens"] = (stats["generation_tokens"] or 0) / req
            stats["queue_share_of_e2e"] = (stats["queue_time_s_sum"] or 0) / max(1e-9, stats["e2e_s_sum"] or 0)
            stats["images_per_request"] = (stats["mm_cache_queries"] or 0) / req
    ident = results_dir / "vllm-server-identity.json"
    if ident.is_file():
        try:
            doc = json.loads(ident.read_text())
            stats["server_argv"] = doc.get("argv")
            if doc.get("ready_epoch"):
                stats["ready_epoch"] = doc["ready_epoch"]
        except json.JSONDecodeError:
            pass
    return stats


def build_pack(results_dir: Path, out_path: Path, *, image_scale: int = 4) -> dict[str, Any]:
    results_dir = Path(results_dir)
    transcripts = sorted((results_dir / "transcripts").glob("*.txt"))
    if not transcripts:
        raise FileNotFoundError(f"no transcripts under {results_dir / 'transcripts'}")
    systems: dict[str, str] = {}
    tool_description: str | None = None
    games: list[dict[str, Any]] = []
    totals = {"turns": 0, "requests": 0, "tool_calls": 0, "action_requests": 0, "step_executed_turns": 0,
              "yield_turns": 0, "request_errors": 0, "images": 0}
    for tpath in transcripts:
        game = tpath.stem.rsplit("_p", 1)[0]
        turns, sys_seen = parse_turns(tpath.read_text(encoding="utf-8", errors="replace"))
        sys_ids = []
        for s in sys_seen:
            key = "sys-" + hashlib.sha1(s.encode()).hexdigest()[:10]
            systems.setdefault(key, s)
            sys_ids.append(key)
        tool_description = tool_description or _parse_tool_description(results_dir / "prompts" / f"{tpath.stem}.log")
        by_header, by_action = _load_boards(results_dir / "artifacts" / f"{tpath.stem}_events.jsonl")
        images: dict[str, str] = {}
        out_turns = []
        for i, t in enumerate(turns):
            board = by_header.get(t["header"])
            if board is None and by_action and t["action"] is not None:
                cands = [b for a, b in by_action if a <= t["action"]]
                board = cands[-1] if cands else by_action[0][1]
            img_key = None
            if board is not None:
                png = grid_to_png(board, scale=image_scale)
                img_key = hashlib.sha1(png).hexdigest()[:12]
                images.setdefault(img_key, base64.b64encode(png).decode("ascii"))
            status = t["status"]
            totals["turns"] += 1
            totals["requests"] += len(t["requests"])
            totals["tool_calls"] += sum(len(r["tool_calls"]) for r in t["requests"])
            totals["action_requests"] += sum(1 for r in t["requests"] if r["action"])
            totals["step_executed_turns"] += int(status.get("step_executed") == "True")
            totals["yield_turns"] += int("turn_time_budget" in status.get("message", ""))
            totals["request_errors"] += int("request_error" in status)
            out_turns.append({
                "i": i,
                "analysis_step": t["analysis_step"],
                "action": t["action"],
                "clock": t["clock"],
                "system": sys_ids[i] if i < len(sys_ids) else (sys_ids[-1] if sys_ids else None),
                "user": t["user"],
                "image": img_key,
                "requests": [
                    {k: r[k] for k in ("finish_reason", "reasoning", "content", "tool_calls", "tool_results", "action")}
                    | ({"followup": r["followup"]} if r.get("followup") else {})
                    for r in t["requests"]
                ],
                "status": {k: v for k, v in status.items() if k in ("step_executed", "message", "request_error")},
            })
        totals["images"] += len(images)
        games.append({"type": "game", "game": game, "turns": out_turns, "images": images})
    recorded = recorded_run_stats(results_dir)
    recorded.update(totals)
    recorded["games"] = len(games)
    if recorded.get("requests") is None:
        recorded["requests"] = totals["requests"]
    # 09-22 validation cap: 2 h per game (every game hit the cap).
    recorded.setdefault("wall_hours_per_game", 2.0)
    g = max(1, len(games))
    recorded["requests_per_game_hour"] = totals["requests"] / g / recorded["wall_hours_per_game"]
    recorded["decisions_per_game_hour"] = totals["action_requests"] / g / recorded["wall_hours_per_game"]
    recorded["turns_per_game_hour"] = totals["turns"] / g / recorded["wall_hours_per_game"]
    header = {
        "type": "header",
        "version": PACK_VERSION,
        "source": str(results_dir),
        "built_epoch": time.time(),
        "image_scale": image_scale,
        "systems": systems,
        "tools": [{
            "type": "function",
            "function": {
                "name": "python",
                "description": tool_description or DEFAULT_PYTHON_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python code to run. The snippet is ephemeral and is not saved across tool calls."}},
                    "required": ["code"],
                },
            },
        }],
        "sampling": dict(DUCK_SAMPLING),
        "duck": {"reply_reserve": DUCK_REPLY_RESERVE, "safety_margin": DUCK_SAFETY_MARGIN,
                 "context_window": 32768, "yield_seconds": DUCK_YIELD_SECONDS,
                 "persistent_assistant_turns": DUCK_PERSISTENT_ASSISTANT_TURNS, "image_tokens": DUCK_IMAGE_TOKENS},
        "recorded_run": recorded,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(out_path, "wt") as fh:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for game in games:
            fh.write(json.dumps(game, separators=(",", ":")) + "\n")
    return {"path": str(out_path), "bytes": out_path.stat().st_size, "games": len(games), **totals,
            "systems": len(systems)}


# ----------------------------------------------------------------------------------------------
# Pack loading
# ----------------------------------------------------------------------------------------------
@dataclass
class Game:
    name: str
    turns: list[dict[str, Any]]
    images: dict[str, str]
    pool: list[str] = field(default_factory=list)  # recorded tool results, in order


def load_pack(path: Path) -> tuple[dict[str, Any], list[Game]]:
    header: dict[str, Any] | None = None
    games: list[Game] = []
    with open_text(Path(path), "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("type") == "header":
                header = doc
            elif doc.get("type") == "game":
                images = {k: "data:image/png;base64," + v for k, v in doc.get("images", {}).items()}
                pool = [r for t in doc["turns"] for q in t["requests"] for r in q.get("tool_results", [])]
                games.append(Game(doc["game"], doc["turns"], images, pool or ["ok"]))
    if header is None:
        raise ValueError(f"{path}: no header line")
    if header.get("version") != PACK_VERSION:
        raise ValueError(f"{path}: pack version {header.get('version')} != {PACK_VERSION}")
    return header, games


# ----------------------------------------------------------------------------------------------
# Duck context management (ported from ARC3-Inference tool_agent.py)
# ----------------------------------------------------------------------------------------------
def _strip_images(msg: dict[str, Any]) -> dict[str, Any]:
    content = msg.get("content")
    if msg.get("role") != "user" or not isinstance(content, list):
        return msg
    text = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return {"role": "user", "content": text}


def _n_images(msg: dict[str, Any]) -> int:
    content = msg.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")


class TokenEstimator:
    """Duck `_estimate_tokens` over json (chars+2)//3 with AGENTFIX-F1 image accounting; cached per message."""

    def __init__(self, tools: list[dict[str, Any]]):
        self._tools_chars = len(json.dumps({"tools": tools, "tool_choice": "auto"}, ensure_ascii=True, sort_keys=True))
        self._cache: dict[int, tuple[dict[str, Any], int, int]] = {}

    def _msg(self, m: dict[str, Any]) -> tuple[int, int]:
        hit = self._cache.get(id(m))
        if hit is not None and hit[0] is m:
            return hit[1], hit[2]
        n_img = _n_images(m)
        if n_img:
            stripped = dict(m)
            stripped["content"] = [p if p.get("type") != "image_url" else {"type": "image_url"} for p in m["content"]]
            chars = len(json.dumps(stripped, ensure_ascii=True, sort_keys=True, default=str))
        else:
            chars = len(json.dumps(m, ensure_ascii=True, sort_keys=True, default=str))
        if len(self._cache) > 50000:
            self._cache.clear()
        self._cache[id(m)] = (m, chars, n_img)
        return chars, n_img

    def __call__(self, messages: list[dict[str, Any]]) -> int:
        chars = self._tools_chars + 16
        imgs = 0
        for m in messages:
            c, n = self._msg(m)
            chars += c + 2
            imgs += n
        return max(1, (chars + 2) // 3) + DUCK_IMAGE_TOKENS * imgs


def drop_oldest_history_block(history: list[dict[str, Any]], preserve_recent: int) -> bool:
    if len(history) - preserve_recent <= 0:
        return False
    first = history.pop(0)
    if first.get("role") in ("assistant", "tool"):
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        return True
    while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
        history.pop(0)
    while history and history[0].get("role") != "user" and len(history) > preserve_recent:
        history.pop(0)
    return True


def drop_until_first_user(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(history)
    while out and out[0].get("role") != "user":
        out.pop(0)
    return out


def trim_messages(messages: list[dict[str, Any]], budget: int, est: Callable[[list[dict[str, Any]]], int],
                  preserve_recent: int = 1) -> list[dict[str, Any]]:
    """Duck `_trim_messages_for_context`: drop oldest history blocks until the estimate fits."""
    if not messages:
        return []
    system, history = messages[0], list(messages[1:])
    while history and est([system, *history]) > budget:
        if not drop_oldest_history_block(history, preserve_recent):
            break
    history = drop_until_first_user(history)
    return [system, *history]


def keep_recent_turns(history: list[dict[str, Any]], max_turns: int) -> list[dict[str, Any]]:
    if max_turns <= 0 or not history:
        return []
    kept: list[dict[str, Any]] = []
    n = 0
    for m in reversed(history):
        kept.append(m)
        if m.get("role") == "assistant":
            n += 1
            if n >= max_turns:
                break
    kept.reverse()
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)
    return kept


def persistent_history(messages: list[dict[str, Any]], budget: int, est: Callable[[list[dict[str, Any]]], int],
                       max_turns: int = DUCK_PERSISTENT_ASSISTANT_TURNS) -> list[dict[str, Any]]:
    """Duck `_persistent_history_messages` (history carried into the next turn, system dropped)."""
    trimmed = trim_messages(messages, budget, est)
    if not trimmed:
        return []
    th = trimmed[1:]
    hist = keep_recent_turns(th, max_turns)
    if hist and hist[0].get("role") != "user" and len(th) > len(hist):
        prev = th[len(th) - len(hist) - 1]
        if prev.get("role") == "user":
            hist = [prev, *hist]
    return drop_until_first_user(hist)


def apply_image_policy(messages: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Keep image parts on the newest `keep` image-bearing user messages (AGENTFIX F1 keeps 2)."""
    if keep < 0:
        return messages
    out = list(messages)
    seen = 0
    for i in range(len(out) - 1, -1, -1):
        if _n_images(out[i]):
            seen += 1
            if seen > keep:
                out[i] = _strip_images(out[i])
    return out


# ----------------------------------------------------------------------------------------------
# HTTP transports (aiohttp -> httpx -> urllib in threads)
# ----------------------------------------------------------------------------------------------
def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


class HttpClient:
    name = "base"

    async def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any, str]:
        raise NotImplementedError

    async def get_text(self, url: str, timeout: float) -> tuple[int, str]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class UrllibClient(HttpClient):
    name = "urllib-threads"

    def __init__(self, base_url: str):
        import urllib.request
        handlers = [urllib.request.ProxyHandler({})] if _is_loopback(base_url) else []
        self._opener = urllib.request.build_opener(*handlers)

    def _post(self, url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any, str]:
        import urllib.error
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                status = resp.status
        except urllib.error.HTTPError as exc:
            return exc.code, None, exc.read().decode("utf-8", errors="replace")[:2000]
        try:
            return status, json.loads(text), ""
        except json.JSONDecodeError:
            return status, None, text[:2000]

    def _get(self, url: str, timeout: float) -> tuple[int, str]:
        import urllib.error
        import urllib.request
        try:
            with self._opener.open(urllib.request.Request(url), timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""

    async def post_json(self, url, payload, timeout):
        return await asyncio.to_thread(self._post, url, payload, timeout)

    async def get_text(self, url, timeout):
        return await asyncio.to_thread(self._get, url, timeout)


class AiohttpClient(HttpClient):
    name = "aiohttp"

    def __init__(self, base_url: str):
        import aiohttp  # noqa: F401
        self._aiohttp = aiohttp
        self._trust_env = not _is_loopback(base_url)
        self._session = None

    def _s(self):
        if self._session is None:
            conn = self._aiohttp.TCPConnector(limit=0)
            self._session = self._aiohttp.ClientSession(connector=conn, trust_env=self._trust_env)
        return self._session

    async def post_json(self, url, payload, timeout):
        t = self._aiohttp.ClientTimeout(total=timeout)
        async with self._s().post(url, json=payload, timeout=t, headers={"Authorization": "Bearer local"}) as resp:
            text = await resp.text()
            try:
                return resp.status, json.loads(text), ""
            except json.JSONDecodeError:
                return resp.status, None, text[:2000]

    async def get_text(self, url, timeout):
        t = self._aiohttp.ClientTimeout(total=timeout)
        async with self._s().get(url, timeout=t) as resp:
            return resp.status, await resp.text()

    async def close(self):
        if self._session is not None:
            await self._session.close()


class HttpxClient(HttpClient):
    name = "httpx"

    def __init__(self, base_url: str):
        import httpx
        self._client = httpx.AsyncClient(trust_env=not _is_loopback(base_url), timeout=None,
                                         limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))

    async def post_json(self, url, payload, timeout):
        resp = await self._client.post(url, json=payload, timeout=timeout, headers={"Authorization": "Bearer local"})
        try:
            return resp.status_code, resp.json(), ""
        except ValueError:
            return resp.status_code, None, resp.text[:2000]

    async def get_text(self, url, timeout):
        resp = await self._client.get(url, timeout=timeout)
        return resp.status_code, resp.text

    async def close(self):
        await self._client.aclose()


def make_client(base_url: str, prefer: str = "auto") -> HttpClient:
    order = {"auto": ["aiohttp", "httpx", "urllib"], "aiohttp": ["aiohttp"], "httpx": ["httpx"], "urllib": ["urllib"]}[prefer]
    for name in order:
        try:
            if name == "aiohttp":
                return AiohttpClient(base_url)
            if name == "httpx":
                return HttpxClient(base_url)
            return UrllibClient(base_url)
        except ImportError:
            continue
    return UrllibClient(base_url)


# ----------------------------------------------------------------------------------------------
# Replay agents
# ----------------------------------------------------------------------------------------------
@dataclass
class BenchConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    pack: str = str(DEFAULT_PACK)
    label: str = "run"
    out: str = "serving_bench_out"
    agents: int = 28
    duration_s: float = 720.0
    warmup_s: float = 90.0
    grace_s: float = 5.0
    context_window: int = 32768
    reply_reserve: int = DUCK_REPLY_RESERVE
    safety_margin: int = DUCK_SAFETY_MARGIN
    max_tokens: int | None = None
    images: bool = True
    images_in_history: int = 2
    yield_seconds: float = DUCK_YIELD_SECONDS
    max_requests_per_turn: int = 12
    history_mode: str = "actual"  # actual | recorded
    request_timeout_s: float = 900.0
    metrics_interval_s: float = 5.0
    model: str | None = None
    seed: int = 0
    transport: str = "auto"
    think_gap_s: float = 0.0
    games: list[str] | None = None
    profile: dict[str, Any] | None = None

    @property
    def budget(self) -> int:
        return max(1024, self.context_window - self.reply_reserve - self.safety_margin)


@dataclass
class ReqRecord:
    agent: int
    game: str
    turn: int
    req_in_turn: int
    t_start: float
    t_end: float
    latency_s: float
    http_status: int
    category: str  # valid_tool_call | invalid_tool_args | markup_in_text | no_tool_call | truncated | error
    action: bool
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    est_prompt_tokens: int
    n_messages: int
    n_images: int
    reasoning_chars: int
    content_chars: int
    error: str = ""


@dataclass
class TurnRecord:
    agent: int
    game: str
    turn: int
    t_start: float
    t_end: float
    requests: int
    acted: bool
    reason: str  # acted | yield | max_requests | recorded_end | error | stopped


def classify_reply(body: dict[str, Any] | None) -> dict[str, Any]:
    """Tool-call validity of one chat completion (Duck semantics: exactly one `python` call with {code})."""
    out = {"category": "error", "action": False, "finish_reason": "", "tool_calls": [], "reasoning": "", "content": ""}
    if not isinstance(body, dict) or not body.get("choices"):
        return out
    choice = body["choices"][0] or {}
    msg = choice.get("message") or {}
    out["finish_reason"] = str(choice.get("finish_reason") or "")
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    out["reasoning"], out["content"] = str(reasoning), str(content)
    calls = msg.get("tool_calls") or []
    parsed = []
    valid = bool(calls)
    for c in calls:
        fn = (c or {}).get("function") or {}
        args = fn.get("arguments", "")
        code = None
        try:
            obj = json.loads(args) if isinstance(args, str) else args
            if isinstance(obj, dict) and isinstance(obj.get("code"), str):
                code = obj["code"]
        except (json.JSONDecodeError, TypeError):
            code = None
        if fn.get("name") != "python" or code is None:
            valid = False
        parsed.append({"id": str(c.get("id") or f"call-{len(parsed)}"), "name": str(fn.get("name", "")),
                       "arguments": args if isinstance(args, str) else json.dumps(args), "code": code or ""})
    out["tool_calls"] = parsed
    if calls:
        out["category"] = "valid_tool_call" if valid else "invalid_tool_args"
        out["action"] = valid and any(ACTION_RE.search(p["code"]) for p in parsed)
    elif out["finish_reason"] == "length":
        out["category"] = "truncated"
    elif MARKUP_RE.search(out["content"]) or MARKUP_RE.search(out["reasoning"]):
        out["category"] = "markup_in_text"
    else:
        out["category"] = "no_tool_call"
    return out


class Agent:
    def __init__(self, idx: int, game: Game, n_games: int, header: dict[str, Any], cfg: BenchConfig,
                 model: str, est: TokenEstimator):
        self.idx = idx
        self.game = game
        self.cfg = cfg
        self.header = header
        self.model = model
        self.est = est
        self.rng = random.Random(cfg.seed * 1000 + idx)
        self.lane_tag = f"[replay lane {idx}] " if idx >= n_games else ""
        # duplicate lanes start mid-game so they never share a prefix with their twin
        self.turn_ptr = (idx // max(1, n_games)) * (len(game.turns) // 2) % max(1, len(game.turns))
        self.history: list[dict[str, Any]] = []
        self.pool_ptr = self.rng.randrange(len(game.pool))
        self.tools = header["tools"]
        self.systems = header["systems"]

    def _system(self, turn: dict[str, Any]) -> dict[str, Any]:
        sid = turn.get("system")
        text = self.systems.get(sid) if sid else None
        if text is None:
            text = next(iter(self.systems.values()), "You are a coding agent solving a grid-based puzzle game.")
        return {"role": "system", "content": text}

    def _user(self, turn: dict[str, Any]) -> dict[str, Any]:
        text = self.lane_tag + turn["user"]
        url = self.game.images.get(turn.get("image") or "") if self.cfg.images else None
        if not url:
            return {"role": "user", "content": text}
        return {"role": "user", "content": [{"type": "text", "text": f"{text}\n\nCurrent grid image:"},
                                            {"type": "image_url", "image_url": {"url": url}}]}

    def _next_pool_result(self) -> str:
        r = self.game.pool[self.pool_ptr % len(self.game.pool)]
        self.pool_ptr += 1
        return r

    def _payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **{k: v for k, v in self.header.get("sampling", DUCK_SAMPLING).items()},
            "chat_template_kwargs": {"enable_thinking": True},
            "tools": self.tools,
            "tool_choice": "auto",
        }
        if self.cfg.max_tokens:
            payload["max_tokens"] = int(self.cfg.max_tokens)
        return payload

    async def run(self, client: HttpClient, stop: asyncio.Event, reqs: list[ReqRecord], turns: list[TurnRecord]) -> None:
        cfg = self.cfg
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        while not stop.is_set():
            if self.turn_ptr >= len(self.game.turns):
                self.turn_ptr = 0
                self.history = []  # new episode
                self.est._cache.clear()
            turn = self.game.turns[self.turn_ptr]
            system = self._system(turn)
            messages = [system, *self.history, self._user(turn)]
            turn_start = time.monotonic()
            n_req = 0
            acted = False
            reason = "stopped"
            rec_i = 0
            turn_results = [r for q in turn["requests"] for r in q.get("tool_results", [])]
            res_i = 0
            while not stop.is_set():
                if n_req > 0 and cfg.history_mode == "actual" and time.monotonic() - turn_start >= cfg.yield_seconds:
                    reason = "yield"
                    break
                if n_req >= cfg.max_requests_per_turn:
                    reason = "max_requests"
                    break
                messages = trim_messages(messages, cfg.budget, self.est, preserve_recent=1)
                send = apply_image_policy(messages, cfg.images_in_history + 1)
                est_tokens = self.est(send)
                payload = self._payload(send)
                t0 = time.monotonic()
                status, body, err = 0, None, ""
                try:
                    status, body, err = await client.post_json(url, payload, cfg.request_timeout_s)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # timeouts, disconnects
                    err = f"{type(exc).__name__}: {exc}"[:500]
                t1 = time.monotonic()
                n_req += 1
                info = classify_reply(body if status == 200 else None)
                usage = (body or {}).get("usage") or {} if isinstance(body, dict) else {}
                details = usage.get("prompt_tokens_details") or {}
                reqs.append(ReqRecord(
                    agent=self.idx, game=self.game.name, turn=int(turn.get("i", self.turn_ptr)), req_in_turn=n_req - 1,
                    t_start=t0, t_end=t1, latency_s=t1 - t0, http_status=status,
                    category=info["category"] if status == 200 else "error", action=bool(info["action"]),
                    finish_reason=info["finish_reason"], prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    cached_tokens=details.get("cached_tokens") if isinstance(details, dict) else None,
                    est_prompt_tokens=est_tokens, n_messages=len(send), n_images=sum(_n_images(m) for m in send),
                    reasoning_chars=len(info["reasoning"]), content_chars=len(info["content"]),
                    error=(err or ("" if status == 200 else f"HTTP {status}"))[:300],
                ))
                if status != 200:
                    reason = "error"
                    await asyncio.sleep(min(5.0, 0.5 + self.rng.random()))
                    break
                if cfg.history_mode == "recorded":
                    rec = turn["requests"][rec_i] if rec_i < len(turn["requests"]) else None
                    rec_i += 1
                    if rec is None:
                        reason = "recorded_end"
                        break
                    am: dict[str, Any] = {"role": "assistant"}
                    if rec.get("reasoning"):
                        am["reasoning"] = rec["reasoning"]
                    if rec.get("content"):
                        am["content"] = rec["content"]
                    if rec.get("tool_calls"):
                        am["tool_calls"] = [{"id": c["id"] or f"call-{j}", "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for j, c in enumerate(rec["tool_calls"])]
                    messages.append(am)
                    results = list(rec.get("tool_results") or [])
                    for j, c in enumerate(am.get("tool_calls", [])):
                        messages.append({"role": "tool", "tool_call_id": c["id"],
                                         "content": results[j] if j < len(results) else self._next_pool_result()})
                    acted = acted or bool(info["action"])
                    if rec_i >= len(turn["requests"]):
                        reason = "recorded_end"
                        break
                    continue
                # actual mode: append the model's real reply, recorded tool output
                am = {"role": "assistant"}
                if info["reasoning"]:
                    am["reasoning"] = info["reasoning"]
                if info["tool_calls"] and info["category"] in ("valid_tool_call", "invalid_tool_args"):
                    if info["content"]:
                        am["content"] = info["content"]
                    am["tool_calls"] = [{"id": c["id"], "type": "function",
                                         "function": {"name": c["name"] or "python", "arguments": c["arguments"]}}
                                        for c in info["tool_calls"]]
                    messages.append(am)
                    for c in info["tool_calls"]:
                        if res_i < len(turn_results):
                            result = turn_results[res_i]
                            res_i += 1
                        else:
                            result = self._next_pool_result()
                        messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                    if info["action"]:
                        acted = True
                        reason = "acted"
                        break
                    if time.monotonic() - turn_start >= cfg.yield_seconds:
                        reason = "yield"
                        break
                    continue
                if info["content"]:
                    am["content"] = info["content"]
                elif info["reasoning"]:
                    am["content"] = None
                if info["content"] or info["reasoning"]:
                    messages.append(am)
                if time.monotonic() - turn_start >= cfg.yield_seconds:
                    reason = "yield"
                    break
                messages.append({"role": "user", "content": FOLLOWUP_PROMPT})
            turns.append(TurnRecord(self.idx, self.game.name, int(turn.get("i", self.turn_ptr)), turn_start,
                                    time.monotonic(), n_req, acted, reason))
            if reason != "error":
                self.history = persistent_history(messages, cfg.budget, self.est)
            self.turn_ptr += 1
            if cfg.think_gap_s:
                await asyncio.sleep(cfg.think_gap_s)


# ----------------------------------------------------------------------------------------------
# Metrics sampling
# ----------------------------------------------------------------------------------------------
COUNTERS = {
    "prompt_tokens": ["vllm:prompt_tokens_total"],
    "generation_tokens": ["vllm:generation_tokens_total"],
    "prompt_tokens_cached": ["vllm:prompt_tokens_cached_total"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "prefix_queries": ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries"],
    "prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits"],
    "mm_queries": ["vllm:mm_cache_queries_total"],
    "mm_hits": ["vllm:mm_cache_hits_total"],
    "spec_drafts": ["vllm:spec_decode_num_drafts_total"],
    "spec_draft_tokens": ["vllm:spec_decode_num_draft_tokens_total"],
    "spec_accepted_tokens": ["vllm:spec_decode_num_accepted_tokens_total"],
    "requests_success": ["vllm:request_success_total"],
}
HISTOGRAMS = {
    "queue_time_s": "vllm:request_queue_time_seconds",
    "e2e_s": "vllm:e2e_request_latency_seconds",
    "prefill_s": "vllm:request_prefill_time_seconds",
    "decode_s": "vllm:request_decode_time_seconds",
    "ttft_s": "vllm:time_to_first_token_seconds",
    "prompt_len": "vllm:request_prompt_tokens",
    "gen_len": "vllm:request_generation_tokens",
}
GAUGES = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "kv_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
}


def snapshot_from_prom(text: str) -> dict[str, Any]:
    p = parse_prometheus(text)
    snap: dict[str, Any] = {"counters": {}, "hist": {}, "gauges": {}, "info": {}}
    for key, names in COUNTERS.items():
        snap["counters"][key] = metric_sum(p, names)
    for key, name in HISTOGRAMS.items():
        buckets: dict[str, float] = {}
        for labels, value in p.get(name + "_bucket", []):
            le = labels.get("le")
            if le is not None:
                buckets[le] = buckets.get(le, 0.0) + value
        snap["hist"][key] = {"sum": metric_sum(p, name + "_sum"), "count": metric_sum(p, name + "_count"), "buckets": buckets}
    for key, names in GAUGES.items():
        snap["gauges"][key] = metric_sum(p, names)
    rows = p.get("vllm:cache_config_info")
    if rows:
        snap["info"]["cache_config"] = rows[0][0]
    per_pos = p.get("vllm:spec_decode_num_accepted_tokens_per_pos_total") or []
    if per_pos:
        snap["counters"]["spec_accepted_per_pos"] = {lab.get("position", "?"): val for lab, val in per_pos}
    return snap


def hist_quantile(buckets: dict[str, float], q: float) -> float | None:
    rows = []
    for le, cnt in buckets.items():
        try:
            rows.append((math.inf if le in ("+Inf", "inf") else float(le), cnt))
        except ValueError:
            continue
    rows.sort()
    if not rows or rows[-1][1] <= 0:
        return None
    total = rows[-1][1]
    target = q * total
    prev_le, prev_cnt = 0.0, 0.0
    for le, cnt in rows:
        if cnt >= target:
            if math.isinf(le):
                return prev_le
            if cnt == prev_cnt:
                return le
            return prev_le + (le - prev_le) * (target - prev_cnt) / (cnt - prev_cnt)
        prev_le, prev_cnt = le, cnt
    return rows[-1][0]


def diff_snapshots(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    if not a or not b:
        return {}
    out: dict[str, Any] = {"counters": {}, "hist": {}}
    for key in COUNTERS:
        va, vb = a["counters"].get(key), b["counters"].get(key)
        out["counters"][key] = (vb - va) if (va is not None and vb is not None) else None
    for key in HISTOGRAMS:
        ha, hb = a["hist"].get(key) or {}, b["hist"].get(key) or {}
        s = (hb.get("sum") - ha.get("sum")) if hb.get("sum") is not None and ha.get("sum") is not None else None
        c = (hb.get("count") - ha.get("count")) if hb.get("count") is not None and ha.get("count") is not None else None
        bk = {le: hb.get("buckets", {}).get(le, 0.0) - ha.get("buckets", {}).get(le, 0.0) for le in hb.get("buckets", {})}
        out["hist"][key] = {"sum": s, "count": c, "mean": (s / c) if s is not None and c else None,
                            "p50": hist_quantile(bk, 0.5), "p90": hist_quantile(bk, 0.9)}
    return out


class MetricsSampler:
    def __init__(self, client: HttpClient, base_url: str, interval: float, sink: Path | None):
        root = base_url.rstrip("/")
        self.url = (root[:-3] if root.endswith("/v1") else root) + "/metrics"
        self.client = client
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.sink = sink
        self.available: bool | None = None
        self.last_text = ""

    async def scrape(self) -> dict[str, Any] | None:
        try:
            status, text = await self.client.get_text(self.url, 15.0)
        except Exception:
            self.available = False if self.available is None else self.available
            return None
        if status != 200:
            self.available = False if self.available is None else self.available
            return None
        self.available = True
        self.last_text = text
        snap = snapshot_from_prom(text)
        snap["t"] = time.monotonic()
        return snap

    async def loop(self, stop: asyncio.Event, t0: float) -> None:
        fh = open(self.sink, "w", encoding="utf-8") if self.sink else None
        try:
            while not stop.is_set():
                snap = await self.scrape()
                if snap is not None:
                    row = {"t": round(snap["t"] - t0, 2), **{k: v for k, v in snap["gauges"].items()},
                           "generation_tokens": snap["counters"].get("generation_tokens"),
                           "prompt_tokens": snap["counters"].get("prompt_tokens"),
                           "preemptions": snap["counters"].get("preemptions")}
                    self.samples.append(row)
                    if fh:
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            if fh:
                fh.close()


# ----------------------------------------------------------------------------------------------
# Bench driver
# ----------------------------------------------------------------------------------------------
async def discover_model(client: HttpClient, base_url: str) -> tuple[str | None, dict[str, Any]]:
    try:
        status, text = await client.get_text(base_url.rstrip("/") + "/models", 30.0)
        if status == 200:
            doc = json.loads(text)
            data = doc.get("data") or []
            if data:
                return str(data[0].get("id")), data[0]
    except Exception:
        pass
    return None, {}


def summarize(cfg: BenchConfig, header: dict[str, Any], reqs: list[ReqRecord], turns: list[TurnRecord],
              w0: float, w1: float, snap0: dict[str, Any] | None, snap1: dict[str, Any] | None,
              samples: list[dict[str, Any]], t_origin: float, model_info: dict[str, Any], transport: str) -> dict[str, Any]:
    window = max(1e-9, w1 - w0)
    hours = window / 3600.0
    in_w = [r for r in reqs if w0 <= r.t_end <= w1]
    ok = [r for r in in_w if r.http_status == 200]
    turns_w = [t for t in turns if w0 <= t.t_end <= w1]
    cats: dict[str, int] = {}
    for r in in_w:
        cats[r.category] = cats.get(r.category, 0) + 1
    n_ok = max(1, len(ok))
    decisions = sum(1 for r in ok if r.category == "valid_tool_call" and r.action)
    comp = [r.completion_tokens for r in ok if r.completion_tokens is not None]
    prm = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    k = max(1, cfg.agents)
    d = diff_snapshots(snap0, snap1)
    c = d.get("counters", {})
    h = d.get("hist", {})
    wsamples = [s for s in samples if (w0 - t_origin) <= s["t"] <= (w1 - t_origin)]

    def gauge(key: str, fn) -> float | None:
        vals = [s.get(key) for s in wsamples if s.get(key) is not None]
        return fn(vals) if vals else None

    prefix_rate = None
    if c.get("prefix_queries"):
        prefix_rate = (c.get("prefix_hits") or 0.0) / c["prefix_queries"]
    spec_rate = spec_len = None
    if c.get("spec_draft_tokens"):
        spec_rate = (c.get("spec_accepted_tokens") or 0.0) / c["spec_draft_tokens"]
    if c.get("spec_drafts"):
        spec_len = 1.0 + (c.get("spec_accepted_tokens") or 0.0) / c["spec_drafts"]
    rec = header.get("recorded_run", {})
    summary = {
        "label": cfg.label,
        "model": model_info.get("id") or cfg.model,
        "server_max_model_len": model_info.get("max_model_len"),
        "base_url": cfg.base_url,
        "transport": transport,
        "agents": cfg.agents,
        "window_s": window,
        "warmup_s": cfg.warmup_s,
        "history_mode": cfg.history_mode,
        "context_window": cfg.context_window,
        "prompt_budget_est_tokens": cfg.budget,
        "max_tokens": cfg.max_tokens,
        # headline
        "decisions_per_game_hour": decisions / k / hours,
        "requests_per_game_hour": len(ok) / k / hours,
        "turns_per_game_hour": len(turns_w) / k / hours,
        "acted_turns_per_game_hour": sum(1 for t in turns_w if t.acted) / k / hours,
        "requests_completed": len(ok),
        "requests_failed": len(in_w) - len(ok),
        "decisions": decisions,
        "turns_completed": len(turns_w),
        "turn_end_reasons": {r: sum(1 for t in turns_w if t.reason == r) for r in sorted({t.reason for t in turns_w})},
        "requests_per_turn_mean": mean([t.requests for t in turns_w]),
        # throughput
        "client_gen_tok_s": sum(comp) / window if comp else None,
        "client_prompt_tok_s": sum(prm) / window if prm else None,
        "server_gen_tok_s": (c["generation_tokens"] / window) if c.get("generation_tokens") is not None else None,
        "server_prompt_tok_s": (c["prompt_tokens"] / window) if c.get("prompt_tokens") is not None else None,
        "server_prompt_cached_tok_s": (c["prompt_tokens_cached"] / window) if c.get("prompt_tokens_cached") is not None else None,
        # latency
        "latency_p50_s": percentile([r.latency_s for r in ok], 0.5),
        "latency_p90_s": percentile([r.latency_s for r in ok], 0.9),
        "latency_mean_s": mean([r.latency_s for r in ok]),
        "turn_latency_p50_s": percentile([t.t_end - t.t_start for t in turns_w], 0.5),
        "turn_latency_p90_s": percentile([t.t_end - t.t_start for t in turns_w], 0.9),
        "queue_time_mean_s": (h.get("queue_time_s") or {}).get("mean"),
        "queue_time_p50_s": (h.get("queue_time_s") or {}).get("p50"),
        "queue_time_p90_s": (h.get("queue_time_s") or {}).get("p90"),
        "e2e_server_mean_s": (h.get("e2e_s") or {}).get("mean"),
        "prefill_time_mean_s": (h.get("prefill_s") or {}).get("mean"),
        "decode_time_mean_s": (h.get("decode_s") or {}).get("mean"),
        "ttft_mean_s": (h.get("ttft_s") or {}).get("mean"),
        "queue_share_of_e2e": ((h.get("queue_time_s") or {}).get("sum") or 0) / (h.get("e2e_s") or {}).get("sum")
        if (h.get("e2e_s") or {}).get("sum") else None,
        # scheduler state
        "running_mean": gauge("running", lambda v: statistics.fmean(v)),
        "running_max": gauge("running", max),
        "waiting_mean": gauge("waiting", lambda v: statistics.fmean(v)),
        "waiting_max": gauge("waiting", max),
        "kv_usage_mean": gauge("kv_usage", lambda v: statistics.fmean(v)),
        "kv_usage_max": gauge("kv_usage", max),
        "preemptions": c.get("preemptions"),
        "prefix_cache_hit_rate": prefix_rate,
        "mm_cache_hit_rate": ((c.get("mm_hits") or 0) / c["mm_queries"]) if c.get("mm_queries") else None,
        "spec_acceptance_rate": spec_rate,
        "spec_mean_acceptance_length": spec_len,
        # quality proxies
        "tool_call_valid_pct": 100.0 * cats.get("valid_tool_call", 0) / n_ok,
        "action_pct": 100.0 * decisions / n_ok,
        "categories": cats,
        "prompt_tokens_p50": percentile(prm, 0.5),
        "prompt_tokens_p90": percentile(prm, 0.9),
        "completion_tokens_p50": percentile(comp, 0.5),
        "completion_tokens_p90": percentile(comp, 0.9),
        "completion_tokens_mean": mean(comp),
        "est_prompt_tokens_p50": percentile([r.est_prompt_tokens for r in ok], 0.5),
        "images_per_request_mean": mean([r.n_images for r in ok]),
        "errors_sample": sorted({r.error for r in in_w if r.error})[:5],
        "error_pct": 100.0 * (len(in_w) - len(ok)) / max(1, len(in_w)),
        # context
        "server_cache_config": ((snap1 or snap0 or {}).get("info") or {}).get("cache_config"),
        "metrics_available": snap1 is not None,
        "recorded_0922": {
            "requests_per_game_hour": rec.get("requests_per_game_hour"),
            "decisions_per_game_hour": rec.get("decisions_per_game_hour"),
            "turns_per_game_hour": rec.get("turns_per_game_hour"),
            "mean_prompt_tokens": rec.get("mean_prompt_tokens"),
            "mean_generation_tokens": rec.get("mean_generation_tokens"),
            "queue_share_of_e2e": rec.get("queue_share_of_e2e"),
        },
    }
    if rec.get("decisions_per_game_hour"):
        summary["decisions_vs_recorded_0922"] = summary["decisions_per_game_hour"] / rec["decisions_per_game_hour"]
    return summary


FIELDS_MD = [
    ("decisions_per_game_hour", "decisions / game-hour (valid `python` call with `action(`)", 1),
    ("requests_per_game_hour", "LLM requests / game-hour", 1),
    ("turns_per_game_hour", "turns / game-hour", 1),
    ("server_gen_tok_s", "generated tok/s (server)", 0),
    ("client_gen_tok_s", "generated tok/s (client usage)", 0),
    ("server_prompt_tok_s", "prompt tok/s (server, incl. cached)", 0),
    ("latency_p50_s", "request latency p50 (s)", 1),
    ("latency_p90_s", "request latency p90 (s)", 1),
    ("turn_latency_p50_s", "turn latency p50 (s)", 1),
    ("turn_latency_p90_s", "turn latency p90 (s)", 1),
    ("queue_time_mean_s", "queue time mean (s, server)", 1),
    ("queue_time_p90_s", "queue time p90 (s, server)", 1),
    ("queue_share_of_e2e", "queue share of server e2e", 2),
    ("running_mean", "running requests (mean)", 1),
    ("waiting_mean", "waiting requests (mean)", 1),
    ("waiting_max", "waiting requests (max)", 0),
    ("kv_usage_mean", "KV usage (mean)", 2),
    ("preemptions", "preemptions", 0),
    ("prefix_cache_hit_rate", "prefix-cache hit rate", 3),
    ("spec_acceptance_rate", "spec-decode draft acceptance", 3),
    ("spec_mean_acceptance_length", "spec-decode mean acceptance length", 2),
    ("tool_call_valid_pct", "tool-call validity %", 1),
    ("action_pct", "requests with action() %", 1),
    ("prompt_tokens_p50", "prompt tokens p50", 0),
    ("completion_tokens_p50", "completion tokens p50", 0),
    ("completion_tokens_p90", "completion tokens p90", 0),
    ("error_pct", "request errors %", 1),
]


def _fmt(v: Any, digits: int) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    return str(v)


def summary_markdown(s: dict[str, Any]) -> str:
    lines = [f"# Serving bench — {s['label']}", "",
             f"- model `{s.get('model')}` at `{s['base_url']}` (max_model_len {s.get('server_max_model_len')})",
             f"- {s['agents']} agents, window {s['window_s']:.0f} s after {s['warmup_s']:.0f} s warm-up, "
             f"history `{s['history_mode']}`, context window {s['context_window']} (prompt budget ≈{s['prompt_budget_est_tokens']} est. tokens)",
             f"- completed {s['requests_completed']} requests / {s['turns_completed']} turns; failed {s['requests_failed']}; "
             f"turn ends {s['turn_end_reasons']}", "",
             "| metric | value |", "|---|---|"]
    for key, name, digits in FIELDS_MD:
        lines.append(f"| {name} | {_fmt(s.get(key), digits)} |")
    rec = s.get("recorded_0922") or {}
    if rec.get("decisions_per_game_hour"):
        lines += ["", f"Recorded 09-22 run (baseline profile, real games): {rec['decisions_per_game_hour']:.1f} decisions/game-hour, "
                      f"{rec['requests_per_game_hour']:.1f} requests/game-hour, mean prompt {rec.get('mean_prompt_tokens') or 0:,.0f} tok, "
                      f"queue share {rec.get('queue_share_of_e2e') or 0:.0%}. This run: ×{s.get('decisions_vs_recorded_0922', 0):.2f}."]
    if s.get("server_cache_config"):
        cc = s["server_cache_config"]
        keys = ["kv_cache_size_tokens", "kv_cache_max_concurrency", "block_size", "cache_dtype", "enable_prefix_caching",
                "mamba_cache_mode", "kv_cache_memory_bytes", "gpu_memory_utilization"]
        lines += ["", "Server cache config: " + ", ".join(f"{k}={cc.get(k)}" for k in keys if k in cc)]
    lines += ["", f"Categories: {s.get('categories')}", f"Errors: {s.get('errors_sample')}"]
    return "\n".join(lines) + "\n"


async def run_bench_async(cfg: BenchConfig) -> dict[str, Any]:
    header, games = load_pack(Path(cfg.pack))
    if cfg.games:
        games = [g for g in games if g.name.split("-")[0] in cfg.games or g.name in cfg.games] or games
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(cfg.base_url, cfg.transport)
    try:
        model, model_info = await discover_model(client, cfg.base_url)
        model = cfg.model or model
        if not model:
            raise RuntimeError(f"no model at {cfg.base_url}/models and --model not given")
        model_info.setdefault("id", model)
        est = TokenEstimator(header["tools"])
        agents = [Agent(i, games[i % len(games)], len(games), header, cfg, model, TokenEstimator(header["tools"]))
                  for i in range(cfg.agents)]
        del est
        reqs: list[ReqRecord] = []
        turns: list[TurnRecord] = []
        stop = asyncio.Event()
        sampler = MetricsSampler(client, cfg.base_url, cfg.metrics_interval_s, out_dir / "metrics_samples.jsonl")
        t_origin = time.monotonic()
        sampler_stop = asyncio.Event()
        sampler_task = asyncio.create_task(sampler.loop(sampler_stop, t_origin))
        tasks = [asyncio.create_task(a.run(client, stop, reqs, turns)) for a in agents]
        w0 = t_origin + cfg.warmup_s
        w1 = w0 + cfg.duration_s
        await asyncio.sleep(max(0.0, w0 - time.monotonic()))
        snap0 = await sampler.scrape()
        w0 = time.monotonic()
        w1 = w0 + cfg.duration_s
        last_print = w0
        while time.monotonic() < w1:
            await asyncio.sleep(min(5.0, max(0.0, w1 - time.monotonic())))
            if time.monotonic() - last_print >= 60:
                last_print = time.monotonic()
                done = sum(1 for r in reqs if r.t_end >= w0)
                print(f"[bench {cfg.label}] t={time.monotonic() - w0:5.0f}s requests={done} "
                      f"last={sampler.samples[-1] if sampler.samples else {}}", flush=True)
            if all(t.done() for t in tasks):
                break
        snap1 = await sampler.scrape()
        w1 = time.monotonic()
        stop.set()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=cfg.grace_s)
        except asyncio.TimeoutError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        sampler_stop.set()
        await asyncio.gather(sampler_task, return_exceptions=True)
        agent_errors = [repr(t.exception()) for t in tasks if t.done() and not t.cancelled() and t.exception()]
        summary = summarize(cfg, header, reqs, turns, w0, w1, snap0, snap1, sampler.samples, t_origin, model_info, client.name)
        summary["agent_exceptions"] = agent_errors[:5]
        report = {
            "summary": summary,
            "config": {k: v for k, v in asdict(cfg).items() if k != "profile"},
            "profile": cfg.profile,
            "metrics_window": {"start": snap0, "end": snap1},
            "pack": {"path": cfg.pack, "games": len(games), "recorded_run": header.get("recorded_run")},
            "created_epoch": time.time(),
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        (out_dir / "summary.md").write_text(summary_markdown(summary))
        if sampler.last_text:
            (out_dir / "metrics_final.prom").write_text(sampler.last_text)
        with gzip.open(out_dir / "requests.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in reqs:
                row = asdict(r)
                row["t_start"] = round(r.t_start - t_origin, 3)
                row["t_end"] = round(r.t_end - t_origin, 3)
                fh.write(json.dumps(row) + "\n")
        with gzip.open(out_dir / "turns.jsonl.gz", "wt", encoding="utf-8") as fh:
            for t in turns:
                row = asdict(t)
                row["t_start"] = round(t.t_start - t_origin, 3)
                row["t_end"] = round(t.t_end - t_origin, 3)
                fh.write(json.dumps(row) + "\n")
        return report
    finally:
        await client.close()


def run_bench(cfg: BenchConfig) -> dict[str, Any]:
    return asyncio.run(run_bench_async(cfg))


# ----------------------------------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------------------------------
COMPARE_COLS = [
    ("decisions_per_game_hour", "decisions/game-h", 1),
    ("requests_per_game_hour", "requests/game-h", 1),
    ("server_gen_tok_s", "gen tok/s", 0),
    ("server_prompt_tok_s", "prompt tok/s", 0),
    ("latency_p50_s", "lat p50 s", 1),
    ("latency_p90_s", "lat p90 s", 1),
    ("turn_latency_p50_s", "turn p50 s", 1),
    ("queue_time_mean_s", "queue mean s", 1),
    ("waiting_mean", "waiting", 1),
    ("preemptions", "preempt", 0),
    ("prefix_cache_hit_rate", "prefix hit", 2),
    ("spec_acceptance_rate", "spec acc", 2),
    ("tool_call_valid_pct", "tool valid %", 1),
    ("action_pct", "action %", 1),
    ("prompt_tokens_p50", "prompt p50", 0),
    ("completion_tokens_p50", "gen p50", 0),
    ("error_pct", "err %", 1),
]


def compare_reports(paths: list[Path], baseline: str | None = None) -> tuple[str, dict[str, Any]]:
    rows = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            p = p / "report.json"
        try:
            rep = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(rep.get("summary", rep))
    if not rows:
        return "No reports.\n", {}
    base = next((r for r in rows if r.get("label") == baseline), None) if baseline else None
    lines = ["| profile | " + " | ".join(n for _, n, _ in COMPARE_COLS) + " | × baseline |",
             "|---" * (len(COMPARE_COLS) + 2) + "|"]
    ratios: dict[str, float | None] = {}
    for r in rows:
        ratio = None
        if base and base.get("decisions_per_game_hour"):
            ratio = (r.get("decisions_per_game_hour") or 0.0) / base["decisions_per_game_hour"]
        ratios[r["label"]] = ratio
        lines.append(f"| {r['label']} | " + " | ".join(_fmt(r.get(k), d) for k, _, d in COMPARE_COLS)
                     + f" | {_fmt(ratio, 2)} |")
    verdict = []
    if base:
        best = max(rows, key=lambda r: (r.get("decisions_per_game_hour") or 0.0) * (1.0 if (r.get("error_pct") or 0) < 20 else 0.0))
        verdict.append(f"Winner by decisions/game-hour (error < 20 %): **{best['label']}** "
                       f"(×{ratios.get(best['label']) or 0:.2f} vs {baseline}).")
        others = [v for k, v in ratios.items() if k != baseline and v is not None]
        if others and max(others) < 1.5:
            verdict.append("EXP-001 falsifier hit: no profile beats the baseline by ≥ 1.5× → the bottleneck is not KV; "
                           "look at prefill cost / prompt size.")
        for r in rows:
            if (r.get("tool_call_valid_pct") or 0) < 90 and r.get("requests_completed"):
                verdict.append(f"Warning: {r['label']} tool-call validity {r['tool_call_valid_pct']:.1f} % (< 90 %).")
    md = "# Serving A/B (EXP-001)\n\n" + "\n".join(lines) + "\n\n" + "\n".join(verdict) + "\n"
    return md, {"rows": rows, "ratios": ratios, "baseline": baseline, "verdict": verdict}


# ----------------------------------------------------------------------------------------------
# Mock OpenAI / vLLM server (CPU testing)
# ----------------------------------------------------------------------------------------------
@dataclass
class MockConfig:
    model: str = "mock/duck-replay"
    max_model_len: int = 32768
    slots: int = 8
    prefill_tps: float = 200000.0
    decode_tps: float = 4000.0
    gen_mean: int = 160
    gen_sigma: float = 0.5
    p_action: float = 0.55
    p_markup: float = 0.02
    p_no_tool: float = 0.02
    cache_tokens: int = 400000
    block_tokens: int = 64
    spec_tokens: int = 2
    spec_accept: float = 0.6
    seed: int = 0


class MockState:
    BUCKETS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320, 640]

    def __init__(self, cfg: MockConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.sem = threading.BoundedSemaphore(max(1, cfg.slots))
        self.rng = random.Random(cfg.seed)
        self.c = {k: 0.0 for k in ("prompt", "gen", "cached", "pq", "ph", "req", "drafts", "draft_tok", "acc_tok", "mmq", "mmh")}
        self.h = {k: {"sum": 0.0, "count": 0, "b": [0] * len(self.BUCKETS)} for k in ("queue", "e2e", "prefill", "decode")}
        self.running = 0
        self.waiting = 0
        self.running_tokens = 0
        self.cache: OrderedDict[str, None] = OrderedDict()
        self.mm_seen: set[str] = set()
        self.n = 0

    def observe(self, key: str, value: float) -> None:
        h = self.h[key]
        h["sum"] += value
        h["count"] += 1
        for i, le in enumerate(self.BUCKETS):
            if value <= le:
                h["b"][i] += 1

    def prefix_blocks(self, messages: list[dict[str, Any]]) -> list[str]:
        text = json.dumps(messages, sort_keys=True)
        step = self.cfg.block_tokens * 4
        out, hsh = [], hashlib.sha1()
        for i in range(0, len(text) - step + 1, step):
            hsh.update(text[i:i + step].encode())
            out.append(hsh.hexdigest())
        return out

    def metrics(self) -> str:
        lab = f'engine="0",model_name="{self.cfg.model}"'
        with self.lock:
            lines = [
                f"vllm:num_requests_running{{{lab}}} {float(self.running)}",
                f"vllm:num_requests_waiting{{{lab}}} {float(self.waiting)}",
                f"vllm:kv_cache_usage_perc{{{lab}}} {min(1.0, self.running_tokens / max(1, self.cfg.cache_tokens))}",
                f"vllm:prompt_tokens_total{{{lab}}} {self.c['prompt']}",
                f"vllm:generation_tokens_total{{{lab}}} {self.c['gen']}",
                f"vllm:prompt_tokens_cached_total{{{lab}}} {self.c['cached']}",
                f"vllm:num_preemptions_total{{{lab}}} 0.0",
                f"vllm:prefix_cache_queries_total{{{lab}}} {self.c['pq']}",
                f"vllm:prefix_cache_hits_total{{{lab}}} {self.c['ph']}",
                f"vllm:mm_cache_queries_total{{{lab}}} {self.c['mmq']}",
                f"vllm:mm_cache_hits_total{{{lab}}} {self.c['mmh']}",
                f"vllm:request_success_total{{{lab},finished_reason=\"stop\"}} {self.c['req']}",
                f"vllm:spec_decode_num_drafts_total{{{lab}}} {self.c['drafts']}",
                f"vllm:spec_decode_num_draft_tokens_total{{{lab}}} {self.c['draft_tok']}",
                f"vllm:spec_decode_num_accepted_tokens_total{{{lab}}} {self.c['acc_tok']}",
                'vllm:cache_config_info{block_size="%d",cache_dtype="auto",enable_prefix_caching="True",'
                'kv_cache_size_tokens="%d",mamba_cache_mode="align",engine="0"} 1.0' % (self.cfg.block_tokens, self.cfg.cache_tokens),
            ]
            names = {"queue": "vllm:request_queue_time_seconds", "e2e": "vllm:e2e_request_latency_seconds",
                     "prefill": "vllm:request_prefill_time_seconds", "decode": "vllm:request_decode_time_seconds"}
            for key, name in names.items():
                h = self.h[key]
                for le, cnt in zip(self.BUCKETS, h["b"]):
                    lines.append(f'{name}_bucket{{{lab},le="{float(le)}"}} {float(cnt)}')
                lines.append(f'{name}_bucket{{{lab},le="+Inf"}} {float(h["count"])}')
                lines.append(f"{name}_sum{{{lab}}} {h['sum']}")
                lines.append(f"{name}_count{{{lab}}} {float(h['count'])}")
        return "\n".join(lines) + "\n"

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        cfg = self.cfg
        messages = body.get("messages") or []
        n_img = 0
        img_keys = []
        for m in messages:
            if isinstance(m.get("content"), list):
                for p in m["content"]:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        n_img += 1
                        img_keys.append(hashlib.sha1(str((p.get("image_url") or {}).get("url", "")).encode()).hexdigest())
        stripped = [(_strip_images(m) if isinstance(m.get("content"), list) else m) for m in messages]
        prompt_tokens = max(1, len(json.dumps(stripped)) // 4) + 64 * n_img
        blocks = self.prefix_blocks(stripped)
        t_arrive = time.monotonic()
        with self.lock:
            self.waiting += 1
            self.n += 1
            rid = self.n
            rng = random.Random(self.rng.random())
        self.sem.acquire()
        t_sched = time.monotonic()
        try:
            with self.lock:
                self.waiting -= 1
                self.running += 1
                hits = 0
                for b in blocks:
                    if b in self.cache:
                        hits += 1
                        self.cache.move_to_end(b)
                    else:
                        break
                for b in blocks:
                    self.cache[b] = None
                    self.cache.move_to_end(b)
                cap_blocks = max(1, cfg.cache_tokens // cfg.block_tokens)
                while len(self.cache) > cap_blocks:
                    self.cache.popitem(last=False)
                cached = min(prompt_tokens, hits * cfg.block_tokens)
                mm_hits = sum(1 for k in img_keys if k in self.mm_seen)
                self.mm_seen.update(img_keys)
                self.running_tokens += prompt_tokens
            limit = max(1, cfg.max_model_len - prompt_tokens)
            if body.get("max_tokens"):
                limit = min(limit, int(body["max_tokens"]))
            gen = int(max(8, min(limit, rng.lognormvariate(math.log(max(8, cfg.gen_mean)), cfg.gen_sigma))))
            prefill_s = (prompt_tokens - cached) / cfg.prefill_tps
            decode_s = gen / cfg.decode_tps
            time.sleep(prefill_s + decode_s)
        finally:
            with self.lock:
                self.running -= 1
                self.running_tokens -= prompt_tokens
            self.sem.release()
        t_done = time.monotonic()
        roll = rng.random()
        reasoning = ("Let me inspect the board and compare frames. " * max(1, gen // 10))[: gen * 4]
        finish = "tool_calls"
        tool_calls = None
        content: str | None = None
        if gen >= limit:
            finish = "length"
        elif roll < cfg.p_markup:
            content = "<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>"
            finish = "stop"
        elif roll < cfg.p_markup + cfg.p_no_tool:
            content = "World model: unchanged."
            finish = "stop"
        else:
            code = "print(current_frame.step)\n" + ("action(['ACTION1'])\n" if rng.random() < cfg.p_action else "")
            tool_calls = [{"id": f"chatcmpl-tool-{rid:08x}", "type": "function",
                           "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
        with self.lock:
            self.c["prompt"] += prompt_tokens
            self.c["gen"] += gen
            self.c["cached"] += cached
            self.c["pq"] += prompt_tokens
            self.c["ph"] += cached
            self.c["req"] += 1
            self.c["mmq"] += n_img
            self.c["mmh"] += mm_hits
            if cfg.spec_tokens > 0:
                drafts = max(1, gen // (1 + cfg.spec_tokens))
                self.c["drafts"] += drafts
                self.c["draft_tok"] += drafts * cfg.spec_tokens
                self.c["acc_tok"] += int(drafts * cfg.spec_tokens * cfg.spec_accept)
            self.observe("queue", t_sched - t_arrive)
            self.observe("e2e", t_done - t_arrive)
            self.observe("prefill", prefill_s)
            self.observe("decode", decode_s)
        msg: dict[str, Any] = {"role": "assistant", "content": content, "reasoning": reasoning}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {"id": f"chatcmpl-mock-{rid}", "object": "chat.completion", "created": int(time.time()), "model": cfg.model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": gen, "total_tokens": prompt_tokens + gen,
                          "prompt_tokens_details": {"cached_tokens": cached}}}


def make_mock_server(cfg: MockConfig, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, MockState]:
    state = MockState(cfg)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # silence
            return

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/v1/models", "/models"):
                doc = {"object": "list", "data": [{"id": cfg.model, "object": "model", "owned_by": "mock", "max_model_len": cfg.max_model_len}]}
                self._send(200, json.dumps(doc).encode())
            elif path == "/metrics":
                self._send(200, state.metrics().encode(), "text/plain; version=0.0.4")
            elif path in ("/health", "/ping", "/version"):
                self._send(200, b'{"ok": true}')
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            path = self.path.split("?", 1)[0].rstrip("/")
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._send(404, b'{"error": "not found"}')
                return
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self._send(400, b'{"error": "bad json"}')
                return
            try:
                out = state.complete(body)
            except (BrokenPipeError, ConnectionResetError):
                return
            try:
                self._send(200, json.dumps(out).encode())
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server, state


class MockServerThread:
    def __init__(self, cfg: MockConfig, host: str = "127.0.0.1", port: int = 0):
        self.server, self.state = make_mock_server(cfg, host, port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "MockServerThread":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"))
    p.add_argument("--pack", default=str(DEFAULT_PACK))
    p.add_argument("--label", default="run")
    p.add_argument("--out", default="serving_bench_out")
    p.add_argument("--agents", type=int, default=28)
    p.add_argument("--duration", type=float, default=720.0, help="measurement window seconds")
    p.add_argument("--warmup", type=float, default=90.0, help="seconds excluded before the window")
    p.add_argument("--context-window", type=int, default=32768)
    p.add_argument("--safety-margin", type=int, default=DUCK_SAFETY_MARGIN)
    p.add_argument("--max-tokens", type=int, default=None, help="default: server default like Duck")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--images-in-history", type=int, default=2)
    p.add_argument("--yield-seconds", type=float, default=DUCK_YIELD_SECONDS)
    p.add_argument("--history", choices=["actual", "recorded"], default="actual")
    p.add_argument("--request-timeout", type=float, default=900.0)
    p.add_argument("--metrics-interval", type=float, default=5.0)
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transport", choices=["auto", "aiohttp", "httpx", "urllib"], default="auto")
    p.add_argument("--games", default=None, help="comma list of game ids to replay (default all)")
    p.add_argument("--mock", action="store_true", help="run against an in-process mock server")
    p.add_argument("--mock-slots", type=int, default=8)
    p.add_argument("--mock-decode-tps", type=float, default=4000.0)
    p.add_argument("--mock-gen-mean", type=int, default=160)


def config_from_args(a: argparse.Namespace) -> BenchConfig:
    return BenchConfig(
        base_url=a.base_url, pack=a.pack, label=a.label, out=a.out, agents=a.agents, duration_s=a.duration,
        warmup_s=a.warmup, context_window=a.context_window, safety_margin=a.safety_margin, max_tokens=a.max_tokens,
        images=not a.no_images, images_in_history=a.images_in_history, yield_seconds=a.yield_seconds,
        history_mode=a.history, request_timeout_s=a.request_timeout, metrics_interval_s=a.metrics_interval,
        model=a.model, seed=a.seed, transport=a.transport, games=a.games.split(",") if a.games else None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-pack", help="build the replay pack from a results directory")
    b.add_argument("--results", required=True, help="dir with transcripts/, prompts/, artifacts/, vllm-metrics-final.prom")
    b.add_argument("--out", required=True, help="output .jsonl.gz (or .jsonl)")
    b.add_argument("--copy-to", default=None, help="also copy the pack here if smaller than --copy-max-mb")
    b.add_argument("--copy-max-mb", type=float, default=20.0)
    r = sub.add_parser("run", help="replay the pack against a server")
    _add_run_args(r)
    c = sub.add_parser("compare", help="compare report.json files")
    c.add_argument("reports", nargs="+")
    c.add_argument("--baseline", default="flashnext_baseline")
    c.add_argument("--out", default=None, help="write markdown here (json next to it)")
    m = sub.add_parser("mock-server", help="run the mock OpenAI/vLLM server")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=1234)
    m.add_argument("--slots", type=int, default=8)
    m.add_argument("--decode-tps", type=float, default=4000.0)
    m.add_argument("--gen-mean", type=int, default=160)
    m.add_argument("--model", default="mock/duck-replay")
    m.add_argument("--max-model-len", type=int, default=32768)
    a = ap.parse_args(argv)

    if a.cmd == "build-pack":
        info = build_pack(Path(a.results), Path(a.out))
        print(json.dumps(info, indent=1))
        if a.copy_to:
            size_mb = info["bytes"] / 1e6
            if size_mb < a.copy_max_mb:
                dst = Path(a.copy_to)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(Path(a.out).read_bytes())
                print(f"copied to {dst} ({size_mb:.1f} MB)")
            else:
                print(f"not copied: {size_mb:.1f} MB >= {a.copy_max_mb} MB")
        return 0
    if a.cmd == "run":
        cfg = config_from_args(a)
        if a.mock:
            mcfg = MockConfig(slots=a.mock_slots, decode_tps=a.mock_decode_tps, gen_mean=a.mock_gen_mean,
                              max_model_len=a.context_window, seed=a.seed)
            with MockServerThread(mcfg) as srv:
                cfg.base_url = srv.base_url
                report = run_bench(cfg)
        else:
            report = run_bench(cfg)
        print(summary_markdown(report["summary"]))
        print(f"wrote {Path(cfg.out) / 'report.json'}")
        return 0
    if a.cmd == "compare":
        md, doc = compare_reports([Path(p) for p in a.reports], a.baseline)
        print(md)
        if a.out:
            Path(a.out).write_text(md)
            Path(a.out).with_suffix(".json").write_text(json.dumps(doc, indent=1, default=str))
        return 0
    if a.cmd == "mock-server":
        mcfg = MockConfig(model=a.model, slots=a.slots, decode_tps=a.decode_tps, gen_mean=a.gen_mean, max_model_len=a.max_model_len)
        server, _ = make_mock_server(mcfg, a.host, a.port)
        print(f"mock server on http://{a.host}:{server.server_address[1]}/v1", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
