#!/usr/bin/env python3
"""duck_offline.py - run the vendored Duck harness end-to-end on CPU against a scripted mock LLM.

What runs (nothing is re-implemented; every piece below is the shipped code path):
    taaf.Benchmark.run -> HarnessSolver._play_one -> _HarnessGameSession.play -> ToolAgent.analyze
    -> HTTP POST /chat/completions (to the mock) -> python tool sandbox (subprocess) -> action(...)
    -> step_env -> taaf.GameAPI (arc_agi OFFLINE Arcade on the competition environment_files)
    -> runtime state / transcripts / prompt logs / viewer artifacts -> benchmark.json
    -> inference.tools.eval.evaluate_runs + save_score_file -> score.json   (as notebook cell 16)

The mock (`MockLLMServer`) is a real OpenAI-compatible HTTP server on 127.0.0.1. Its scripted policy
returns python tool calls that pick valid actions from the sandbox's own `valid_actions`, plus a fixed
opening script per game that hits every branch once: inspection-only calls, text-only replies (world
model lines), tool-call markup in text, invalid/unknown actions, MOUSE without coordinates, syntax and
runtime errors, malformed JSON arguments, an unknown tool, batches and multi-call snippets. It records
every request and estimates prompt tokens per request (chars / 3.5; images by pixel count).

Usage (python 3.12 venv with arc_agi 0.9.8 + arcengine 0.9.3, scipy, imageio, matplotlib, pillow, requests):
    SCRATCH/venv/bin/python agent/duck_offline.py                       # ls20, vc33, ar25 x 20 actions
    SCRATCH/venv/bin/python agent/duck_offline.py --games ls20,ft09 --steps 30 --profile shipped
    SCRATCH/venv/bin/python agent/duck_offline.py --compare-profiles    # token baseline, all profiles

Profiles (source patches, see agent/duck/PATCHES.md): `patched` (every default: B01 fixes + prompt v9
P1-P3), `v9` (patched + M1 retained reasoning + M2 verified facts), `pre_v9` (B01 fixes only), `shipped`
(the 09-22 notebook's AGENTFIX set: F1+F3), `stock` (DUCK_PATCHES=0: the bundle byte-for-byte).

Importing this module sets the shipped analyzer environment (setdefault) BEFORE the Duck modules are
imported, because tool_agent.py reads its LOCAL_ANALYZER_* knobs at import time.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import random
import re
import socket
import statistics
import struct
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

AGENT_DIR = Path(__file__).resolve().parent
REPO_DIR = AGENT_DIR.parent
DUCK_DIR = AGENT_DIR / "duck"
DUCK_SOURCE_ROOTS = (
    DUCK_DIR / "src" / "ARC3-Inference",
    DUCK_DIR / "src" / "tufa-arc-agi-framework" / "src",
)
_SCRATCH = Path("/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad")

# --------------------------------------------------------------------------------------------------
# Shipped configuration, rebuilt in code (instead of benchmark_initial.pkl / setup-env side effects)
# --------------------------------------------------------------------------------------------------

# Analyzer env the shipped serving_setup.py persists (persist_analyzer_environment), plus the notebook
# overrides that were live in the 09-22 run (cell 3, cell 10 ARM P). LOCAL_ANALYZER_CONTEXT_WINDOW:
# cell 3 sets 16384, but cell 9 re-applies the setup env (32768) before tool_agent is imported, so the
# analyzer effectively ran with 32768 even under the "ctx16k" profile name.
SHIPPED_ANALYZER_ENV: dict[str, str] = {
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "INFERENCE_ANALYZER_MODEL": "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1234/v1",  # replaced per game by the mock URL
    "OPENAI_API_KEY": "offline-kaggle-local-server",
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "12",  # ARM P (notebook cell 10) raised the bundle's 4 to 12
    "ONLY_RESET_LEVELS": "true",
    "TAAF_MINIMAL_DIAGNOSTICS": "1",
    "TAAF_RUN_AS_SUBMISSION": "0",
    "MPLBACKEND": "Agg",
}

# HarnessSolver fields of benchmark_initial.pkl (== notebook cell 14 overrides). The kaggle_* fields
# only drive TAAF's deploy path; they are kept for parity (tests compare against the pickle).
SHIPPED_SOLVER_FIELDS: dict[str, Any] = {
    "label": "duck-harness",
    "model": "local",
    "analyzer_timeout": 900.0,
    "max_actions_per_game": None,
    "max_runtime_s_per_game": 7920.0,
    "concurrency": 28,
    "save_request_logs": False,
    "start_local_server": False,
    "local_server_config": "",
    "local_server_api_key_file": "",
    "local_server_port": None,
    "local_server_tensor_parallel_size": None,
    "local_server_count": 1,
    "cancel_drain_timeout_s": 120.0,
    "kaggle_enable_vllm": True,
    "kaggle_model_dataset_source": "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
    "kaggle_served_model_name": "vrfai/Qwen3.6-27B-FP8",
    "kaggle_vllm_max_model_len": 65536,
    "kaggle_vllm_port": 1234,
    "kaggle_vllm_tensor_parallel_size": 1,
    "kaggle_wheelhouse_dataset_source": "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    "kaggle_wheelhouse_stamp_text": "vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\n",
}
SHIPPED_BENCHMARK_FIELDS: dict[str, Any] = {"label": "duck-harness-kaggle", "n_passes": 1, "game_weights": None}
# deploy_target.pkl (taaf.deploy_kaggle.KaggleTarget), the fields that matter for a submission.
SHIPPED_TARGET_FIELDS: dict[str, Any] = {
    "accelerator": "NvidiaRtxPro6000",
    "max_runtime_s": 32400.0,
    "enable_internet": False,
    "cpu_only": False,
    "run_as_submission": False,
}
# Notebook cell 16: the public-25 order the shipped run uses.
PUBLIC_GAME_IDS: tuple[str, ...] = (
    "tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940", "wa30-ee6fef47",
    "lp85-305b61c3", "r11l-495a7899", "tu93-0768757b", "sp80-589a99af", "m0r0-492f87ba",
    "vc33-5430563c", "ar25-0c556536", "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629",
    "dc22-fdcac232", "cd82-fb555c5d", "ft09-0d8bbf25", "g50t-5849a774", "ls20-9607627b",
    "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44", "su15-1944f8ab", "tr87-cd924810",
)

# Source-patch profiles (agent/duck/PATCHES.md). `patched` = every default (B01 fixes + prompt v9
# P1-P3; M1/M2 default off), `v9` = patched + M1 retained reasoning + M2 verified facts/compaction,
# `pre_v9` = the B01 patch set only (== the code before B06/B07, pinned by tests/golden).
PROFILES: dict[str, dict[str, str]] = {
    "patched": {"DUCK_PATCHES": "1"},
    "v9": {"DUCK_PATCHES": "1", "DUCK_PATCH_M1_THINK": "1", "DUCK_PATCH_M2_FACTS": "1"},
    "pre_v9": {"DUCK_PATCHES": "1", "DUCK_PATCHES_V9": "0"},
    "shipped": {"DUCK_PATCHES": "1", "DUCK_PATCHES_V9": "0", "DUCK_PATCH_F2_RESULT": "0",
                "DUCK_PATCH_S1_STATE_KEY": "0"},
    "stock": {"DUCK_PATCHES": "0"},
    # Hydra-1 layering (automation/hydra1): the notebook's AGENTFIX cell supplies F1/F3 (+F6/F9/F13/F19/ARM P) exactly
    # as on Kaggle, so our duplicate source fixes F1/F2/F3 stay off; ours add S1 + v9 prompt (P1-P3) [+ M2 memory].
    "hydra1a": {"DUCK_PATCHES": "1", "DUCK_PATCH_F1_IMAGES": "0", "DUCK_PATCH_F2_RESULT": "0",
                "DUCK_PATCH_F3_MEMORY": "0"},
    "hydra1b": {"DUCK_PATCHES": "1", "DUCK_PATCH_F1_IMAGES": "0", "DUCK_PATCH_F2_RESULT": "0",
                "DUCK_PATCH_F3_MEMORY": "0", "DUCK_PATCH_M2_FACTS": "1"},
}
_PATCH_ENV_KEYS = ("DUCK_PATCHES", "DUCK_PATCH_F1_IMAGES", "DUCK_PATCH_F2_RESULT", "DUCK_PATCH_F3_MEMORY",
                   "DUCK_PATCH_S1_STATE_KEY", "DUCK_PATCHES_V9", "DUCK_PATCH_P1_SCORING", "DUCK_PATCH_P2_UNDO",
                   "DUCK_PATCH_P3_DISCIPLINE", "DUCK_PATCH_M1_THINK", "DUCK_PATCH_M2_FACTS")

DEFAULT_GAMES = ("ls20", "vc33", "ar25")  # keyboard / click / keyboard+click with ACTION7
CHARS_PER_TOKEN = 3.5


def configure_environment() -> None:
    """Shipped analyzer env (setdefault, so an exported value wins) + importable Duck sources."""
    for key, value in SHIPPED_ANALYZER_ENV.items():
        os.environ.setdefault(key, value)
    for root in reversed(DUCK_SOURCE_ROOTS):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))


configure_environment()


def default_env_dir() -> str:
    from arcenv import DEFAULT_ENV_DIR  # agent/arcenv.py (read-only use)

    return DEFAULT_ENV_DIR


def resolve_game_ids(games: list[str] | tuple[str, ...], env_dir: str) -> list[str]:
    """`ls20` or `ls20-9607627b` -> full offline game id (from <env_dir>/<game>/<ver>/metadata.json)."""
    resolved: list[str] = []
    for name in games:
        short = name.split("-", 1)[0]
        metas = sorted(Path(env_dir).glob(f"{short}/*/metadata.json"))
        if not metas:
            raise ValueError(f"game {name!r} not found under {env_dir}")
        game_id = json.loads(metas[0].read_text())["game_id"]
        if "-" in name and name != game_id:
            raise ValueError(f"game {name!r} not found under {env_dir} (have {game_id})")
        resolved.append(game_id)
    return resolved


@contextlib.contextmanager
def patch_profile(profile: str) -> Iterator[None]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    saved = {key: os.environ.get(key) for key in _PATCH_ENV_KEYS}
    for key in _PATCH_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(PROFILES[profile])
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------------------------------
# Network guard: only the mock server's port may be connected to
# --------------------------------------------------------------------------------------------------


class NetworkGuard:
    """Patch socket connects process-wide; allow only 127.0.0.1:<allowed ports>. Records every attempt."""

    def __init__(self, allowed_ports: set[int]):
        self.allowed_ports = set(allowed_ports)
        self.attempts: list[tuple[str, int, bool]] = []
        self._lock = threading.Lock()
        self._orig_connect = None
        self._orig_connect_ex = None
        self._orig_getaddrinfo = None

    _LOOPBACK = {"127.0.0.1", "localhost", "::1"}

    def _check(self, address: Any) -> None:
        host, port = (address[0], int(address[1])) if isinstance(address, tuple) else (str(address), -1)
        ok = host in self._LOOPBACK and port in self.allowed_ports
        with self._lock:
            self.attempts.append((host, port, ok))
        if not ok:
            raise ConnectionRefusedError(f"duck_offline network guard blocked a connection to {address!r}")

    def __enter__(self) -> "NetworkGuard":
        guard = self
        self._orig_connect = socket.socket.connect
        self._orig_connect_ex = socket.socket.connect_ex

        def connect(sock: socket.socket, address: Any) -> Any:
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                guard._check(address)
            return guard._orig_connect(sock, address)

        def connect_ex(sock: socket.socket, address: Any) -> Any:
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                guard._check(address)
            return guard._orig_connect_ex(sock, address)

        self._orig_getaddrinfo = socket.getaddrinfo

        def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if host is not None and str(host) not in guard._LOOPBACK:
                with guard._lock:
                    guard.attempts.append((str(host), -1, False))
                raise socket.gaierror(f"duck_offline network guard blocked a DNS lookup of {host!r}")
            return guard._orig_getaddrinfo(host, *args, **kwargs)

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> None:
        socket.socket.connect = self._orig_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = self._orig_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = self._orig_getaddrinfo  # type: ignore[assignment]

    @property
    def blocked(self) -> list[tuple[str, int, bool]]:
        return [attempt for attempt in self.attempts if not attempt[2]]


# --------------------------------------------------------------------------------------------------
# Scripted mock policy
# --------------------------------------------------------------------------------------------------

# Runs inside the Duck python sandbox (restricted builtins, no hashlib): a frame fingerprint and a
# valid-action picker over the six basic names. ACTION7 is exercised separately by the
# `advertised_unmappable` step: unmappable in the shipped action_names.py, executed as UNDO with P2.
_PRELUDE = '''
def _fp(frame):
    h = 0
    for ch in frame.ascii:
        h = (h * 131 + ord(ch)) % 2305843009213693951
    return h

def _pick(k):
    acts = [a for a in valid_actions if a in ("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE")]
    if not acts:
        return None
    a = acts[k % len(acts)]
    if a != "MOUSE":
        return a
    nodes = sorted(current_frame.segmentation["nodes"], key=lambda n: (n["pixels"], n["id"]))
    node = nodes[k % len(nodes)] if nodes else None
    r, c = node["boundary"][0] if node else (32, 32)
    return {"action": "MOUSE", "row": int(r), "col": int(c)}

print("FP", _fp(current_frame), "step", current_frame.step, "level", current_frame.level)
'''

_ACT_TEMPLATES: dict[str, str] = {
    "inspect": (
        "seg = current_frame.segmentation\n"
        "sizes = sorted((n['pixels'] for n in seg['nodes']), reverse=True)[:5]\n"
        "print('objects', len(seg['nodes']), 'largest', sizes, 'valid', valid_actions)\n"
    ),
    "act_single": (
        "r = action([_pick({k})])\n"
        "print('ACTED', r.get('executed'), 'changed', r.get('board_changed'), 'FP_AFTER', _fp(current_frame),"
        " 'step', current_frame.step)\n"
    ),
    "act_batch": (
        "plan = [p for p in (_pick({k}), _pick({k} + 1), _pick({k} + 2)) if p is not None]\n"
        "r = action(plan)\n"
        "print('ACTED', r.get('executed'), 'n', r.get('executed_count'), 'FP_AFTER', _fp(current_frame),"
        " 'step', current_frame.step)\n"
    ),
    "act_result_var": "result = action([_pick({k})])\n",
    "act_multi_call": (
        "r1 = action([_pick({k})])\n"
        "if not (r1.get('game_over') or r1.get('level_completed') or r1.get('run_complete')):\n"
        "    r2 = action([_pick({k} + 1)])\n"
        "print('ACTED', r1.get('executed'), 'FP_AFTER', _fp(current_frame), 'step', current_frame.step)\n"
    ),
    "act_then_error": (
        "r = action([_pick({k})])\n"
        "print('ACTED', r.get('executed'), 'step', current_frame.step)\n"
        "boom = 1 / 0\n"
    ),
    "invalid_action": "r = action(['JUMP'])\nprint('ACTED', r.get('executed'), r.get('error'))\n",
    "advertised_unmappable": (
        "extra = [a for a in valid_actions if a not in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', 'MOUSE')]\n"
        "r = action([extra[0] if extra else 'ACTION9'])\n"
        "print('ACTED', r.get('executed'), r.get('error'))\n"
    ),
    "invalid_then_valid": (
        "r0 = action(['ACTION9'])\n"
        "r = action([_pick({k})])\n"
        "print('ACTED', r.get('executed'), 'first_error', r0.get('error'), 'step', current_frame.step)\n"
    ),
    "mouse_no_coords": "r = action([{'action': 'MOUSE'}])\nprint('ACTED', r.get('executed'), r.get('error'))\n",
    "reset": "r = action(['RESET'])\nprint('ACTED', r.get('executed'), 'step', current_frame.step)\n",
}

# Every game plays this opening once (covers each branch), then weighted random steps.
OPENING_SCRIPT: tuple[str, ...] = (
    "inspect", "text_only", "act_single", "markup", "invalid_action", "act_result_var",
    "mouse_no_coords", "act_batch", "syntax_error", "act_then_error", "bad_json", "unknown_tool",
    "act_multi_call", "advertised_unmappable", "invalid_then_valid", "content_and_tool",
)
RANDOM_WEIGHTS: dict[str, int] = {
    "act_single": 30, "act_batch": 15, "act_result_var": 8, "act_multi_call": 8, "content_and_tool": 8,
    "act_then_error": 3, "inspect": 8, "text_only": 4, "markup": 3, "invalid_action": 3,
    "invalid_then_valid": 3, "mouse_no_coords": 2, "syntax_error": 2, "bad_json": 1, "unknown_tool": 1,
    "reset": 1,
}
_NON_ACTING = {"inspect", "text_only", "invalid_action", "mouse_no_coords", "syntax_error", "bad_json",
               "unknown_tool", "advertised_unmappable"}

_FILLER = (
    "Looking at the segmentation, I compare previous_frame with current_frame to see which objects moved, "
    "separate HUD changes along the border from gameplay changes, and keep the world model consistent with "
    "the evidence. The next probe should discriminate between the remaining hypotheses. "
)


def _filler(n_chars: int, seed_text: str) -> str:
    text = f"[{seed_text}] "
    while len(text) < n_chars:
        text += _FILLER
    return text[:n_chars]


@dataclass
class RequestRecord:
    game: str
    index: int
    scenario: str
    new_turn: bool
    n_messages: int
    n_images: int
    image_px: list[tuple[int, int]]
    chars: dict[str, int]
    text_tokens: float
    image_tokens: float
    reasoning_tokens: float
    prompt_tokens: float
    completion_tokens: float


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _png_size(data_url: str) -> tuple[int, int] | None:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1][:64])
        if raw[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width, height = struct.unpack(">II", raw[16:24])
        return int(width), int(height)
    except Exception:
        return None


def image_tokens_for(width: int, height: int) -> int:
    """Qwen-VL style estimate: 16 px patches merged 2x2 -> one token per 32x32 px (+2 delimiters)."""
    return math.ceil(width / 32) * math.ceil(height / 32) + 2


def measure_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Split one chat request into sections (chars) and estimate tokens (chars/3.5, images by pixels)."""
    chars: Counter[str] = Counter()
    images: list[tuple[int, int]] = []
    messages = payload.get("messages") or []
    last_user_index = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        content = message.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    size = _png_size(str((part.get("image_url") or {}).get("url", "")))
                    images.append(size or (0, 0))
        text_chars = sum(len(t) for t in texts)
        if role == "system":
            chars["system"] += text_chars
        elif role == "user":
            chars["user_latest" if index == last_user_index else "user_history"] += text_chars
        elif role == "assistant":
            chars["assistant_content"] += text_chars
            chars["assistant_reasoning"] += len(str(message.get("reasoning") or message.get("reasoning_content") or ""))
            for call in message.get("tool_calls") or []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                chars["assistant_tool_calls"] += len(str(function.get("arguments", ""))) + len(str(function.get("name", "")))
        elif role == "tool":
            chars["tool_results"] += text_chars
        chars["message_overhead"] += 16  # role markers / template tokens (~4-5 tokens)
    chars["tools_schema"] += len(json.dumps(payload.get("tools") or [], separators=(",", ":")))
    total_text = sum(chars.values())
    image_tokens = sum(image_tokens_for(w, h) for w, h in images if w and h)
    return {
        "chars": dict(chars),
        "images": images,
        "text_tokens": total_text / CHARS_PER_TOKEN,
        "reasoning_tokens": chars["assistant_reasoning"] / CHARS_PER_TOKEN,
        "image_tokens": float(image_tokens),
        "prompt_tokens": total_text / CHARS_PER_TOKEN + image_tokens,
    }


class ScriptedPolicy:
    """Deterministic per-game script. Thread-safe; one instance serves all games of a run."""

    def __init__(self, *, seed: int = 0, reasoning_chars: int = 4000, content_chars: int = 300):
        self.seed = seed
        self.reasoning_chars = reasoning_chars
        self.content_chars = content_chars
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._non_acting_streak: dict[str, int] = defaultdict(int)
        self._rngs: dict[str, random.Random] = {}
        self.records: list[RequestRecord] = []
        self.tool_results: dict[str, dict[str, Any]] = {}  # tool_call_id -> {game, scenario, content}
        self.issued: dict[str, str] = {}  # tool_call_id -> scenario

    # ---- helpers ------------------------------------------------------------------------------
    def _rng(self, game: str) -> random.Random:
        if game not in self._rngs:
            self._rngs[game] = random.Random(f"{self.seed}:{game}")
        return self._rngs[game]

    def _choose(self, game: str, index: int, last_role: str) -> str:
        if index < len(OPENING_SCRIPT):
            return OPENING_SCRIPT[index]
        if self._non_acting_streak[game] >= 2 or last_role == "tool":
            return "act_single" if self._rng(game).random() < 0.6 else "act_batch"
        names = list(RANDOM_WEIGHTS)
        return self._rng(game).choices(names, weights=[RANDOM_WEIGHTS[n] for n in names], k=1)[0]

    @staticmethod
    def _code(scenario: str, k: int) -> str:
        return _PRELUDE + _ACT_TEMPLATES[scenario].replace("{k}", str(k))

    def _tool_call(self, call_id: str, name: str, arguments: str) -> dict[str, Any]:
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}

    # ---- main entry ---------------------------------------------------------------------------
    def respond(self, game: str, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        last = messages[-1] if messages else {}
        last_role = str(last.get("role", ""))
        with self._lock:
            index = self._counters[game]
            self._counters[game] += 1
            for message in messages:  # harvest tool results (dedup by id) for isolation checks
                if message.get("role") == "tool":
                    call_id = str(message.get("tool_call_id", ""))
                    if call_id and call_id not in self.tool_results:
                        self.tool_results[call_id] = {
                            "game": game,
                            "scenario": self.issued.get(call_id, "?"),
                            "content": str(message.get("content", "")),
                        }
            scenario = self._choose(game, index, last_role)
            self._non_acting_streak[game] = self._non_acting_streak[game] + 1 if scenario in _NON_ACTING else 0

        k = index
        call_id = f"call-{game}-{index}"
        reasoning = _filler(self.reasoning_chars, f"{game} request {index} {scenario}")
        content = ""
        tool_calls: list[dict[str, Any]] = []
        if scenario == "text_only":
            content = (
                f"World model (revised): WM-MARKER-{game} the board has a few movable objects.\n"
                f"Goal model: reach the target configuration.\n"
                f"Plan (next): probe action {k} and compare frames.\n"
            )
        elif scenario == "markup":
            content = (
                "<tool_call>\n<function=python>\n<parameter=code>\n"
                + self._code("act_single", k)
                + "</parameter>\n</function>\n</tool_call>"
            )
        elif scenario == "content_and_tool":
            content = f"Recent findings: request {k} of {game}.\nPlan (next): execute one probe.\n"
            tool_calls = [self._tool_call(call_id, "python", json.dumps({"code": self._code("act_single", k)}))]
        elif scenario == "syntax_error":
            tool_calls = [self._tool_call(call_id, "python", json.dumps({"code": "for x in range(:\n    pass"}))]
        elif scenario == "bad_json":
            tool_calls = [self._tool_call(call_id, "python", '{"code": "print(1)"')]  # truncated JSON
        elif scenario == "unknown_tool":
            tool_calls = [self._tool_call(call_id, "bash", json.dumps({"cmd": "ls"}))]
        else:
            tool_calls = [self._tool_call(call_id, "python", json.dumps({"code": self._code(scenario, k)}))]
        if content and scenario not in ("text_only", "markup"):
            content = content + _filler(max(0, self.content_chars - len(content)), "note")

        message: dict[str, Any] = {"role": "assistant", "content": content or None, "reasoning_content": reasoning}
        if tool_calls:
            message["tool_calls"] = tool_calls
        measured = measure_request(payload)
        completion_chars = len(reasoning) + len(content) + sum(len(c["function"]["arguments"]) for c in tool_calls)
        completion_tokens = completion_chars / CHARS_PER_TOKEN
        new_turn = last_role == "user" and not _message_text(last).startswith(
            ("You have not acted yet", "You did not call a tool")
        )
        with self._lock:
            for call in tool_calls:
                self.issued[call["id"]] = scenario
            self.records.append(
                RequestRecord(
                    game=game,
                    index=index,
                    scenario=scenario,
                    new_turn=new_turn,
                    n_messages=len(messages),
                    n_images=len(measured["images"]),
                    image_px=list(measured["images"]),
                    chars=measured["chars"],
                    text_tokens=measured["text_tokens"],
                    image_tokens=measured["image_tokens"],
                    reasoning_tokens=measured["reasoning_tokens"],
                    prompt_tokens=measured["prompt_tokens"],
                    completion_tokens=completion_tokens,
                )
            )
        return {
            "id": f"chatcmpl-{call_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", "mock")),
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {
                "prompt_tokens": int(measured["prompt_tokens"]),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(measured["prompt_tokens"] + completion_tokens),
            },
        }


class MockLLMServer:
    """OpenAI-compatible chat server on 127.0.0.1:<random>. Games are told apart by URL:
    each game's analyzer gets base_url = <server>/g/<game_id>/v1."""

    _PATH_RE = re.compile(r"^/g/(?P<game>[^/]+)/v1/chat/completions$")

    def __init__(self, policy: ScriptedPolicy):
        self.policy = policy
        self.errors: list[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # silence
                return

            def _send(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.endswith("/v1/models"):
                    self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                match = server._PATH_RE.match(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if match is None:
                    self._send(404, {"error": f"unknown path {self.path}"})
                    return
                try:
                    payload = json.loads(raw)
                    body = server.policy.respond(match.group("game"), payload)
                except Exception as exc:  # surface policy bugs as HTTP 500 + record
                    server.errors.append(f"{type(exc).__name__}: {exc}")
                    self._send(500, {"error": str(exc)})
                    return
                self._send(200, body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="mock-llm", daemon=True)

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    def base_url_for(self, game_id: str) -> str:
        return f"http://127.0.0.1:{self.port}/g/{game_id}/v1"

    def __enter__(self) -> "MockLLMServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


# --------------------------------------------------------------------------------------------------
# Benchmark / solver construction (in code, equivalent to the shipped pickles)
# --------------------------------------------------------------------------------------------------


def shipped_solver(**overrides: Any) -> Any:
    from inference.framework.solver import HarnessSolver

    fields = dict(SHIPPED_SOLVER_FIELDS)
    fields.update(overrides)
    return HarnessSolver(**fields)


def offline_games(game_ids: list[str], env_dir: str) -> list[Any]:
    """Notebook cell 16 `_offline_games`: taaf GameAPI over an OFFLINE arc_agi Arcade."""
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def build_benchmark(game_ids: list[str], *, job_dir: Path, solver: Any, env_dir: str) -> Any:
    import taaf.benchmark

    bm = taaf.benchmark.Benchmark(
        label=SHIPPED_BENCHMARK_FIELDS["label"],
        games=offline_games(game_ids, env_dir),
        solver=solver,
        n_passes=SHIPPED_BENCHMARK_FIELDS["n_passes"],
        job_dir=job_dir,
        game_weights=SHIPPED_BENCHMARK_FIELDS["game_weights"],
    )
    return bm


def make_mock_analyzer_factory(server: MockLLMServer, registry: dict[str, Any]) -> Callable[[Any, int], Any]:
    """Same construction as HarnessSolver._make_analyzer (no local server), except that the base URL
    carries the game id so the mock can script each game separately."""

    def factory(game: Any, index: int) -> Any:
        from inference.agent.tool_agent import ToolAgent

        game_id = game.game_run.game_id if game.game_run is not None else str(index)
        agent = ToolAgent(
            model=SHIPPED_SOLVER_FIELDS["model"],
            timeout=SHIPPED_SOLVER_FIELDS["analyzer_timeout"],
            save_request_logs=SHIPPED_SOLVER_FIELDS["save_request_logs"],
            api_key=None,
            base_url=server.base_url_for(game_id),
            provider=None,
        )
        registry[game_id] = agent
        return agent

    return factory


# --------------------------------------------------------------------------------------------------
# Run + metrics
# --------------------------------------------------------------------------------------------------


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)]
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 1),
        "median": round(statistics.median(values), 1),
        "p90": round(p90, 1),
        "max": round(max(values), 1),
    }


def summarize_records(records: list[RequestRecord]) -> dict[str, Any]:
    sections: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key in ("system", "tools_schema", "user_latest", "user_history", "assistant_content",
                    "assistant_reasoning", "assistant_tool_calls", "tool_results", "message_overhead"):
            sections[key].append(record.chars.get(key, 0) / CHARS_PER_TOKEN)
        sections["images"].append(record.image_tokens)
    turn_records = [r for r in records if r.new_turn]
    return {
        "requests": len(records),
        "turn_starts": len(turn_records),
        "prompt_tokens_per_request": _dist([r.prompt_tokens for r in records]),
        "prompt_tokens_per_request_excl_reasoning": _dist([r.prompt_tokens - r.reasoning_tokens for r in records]),
        "prompt_tokens_at_turn_start": _dist([r.prompt_tokens for r in turn_records]),
        "text_tokens_per_request": _dist([r.text_tokens for r in records]),
        "image_tokens_per_request": _dist([r.image_tokens for r in records]),
        "images_per_request": _dist([float(r.n_images) for r in records]),
        "image_px": sorted({f"{w}x{h}" for r in records for (w, h) in r.image_px}),
        "section_tokens_mean": {key: round(statistics.fmean(vals), 1) for key, vals in sections.items() if vals},
        "completion_tokens_per_request": _dist([r.completion_tokens for r in records]),
        "scenarios": dict(Counter(r.scenario for r in records)),
    }


@dataclass
class OfflineRunResult:
    job_dir: Path
    profile: str
    game_ids: list[str]
    steps: int
    score: float
    score_path: Path
    games: dict[str, dict[str, Any]]
    token_summary: dict[str, Any]
    per_game_tokens: dict[str, dict[str, Any]]
    records: list[RequestRecord] = field(repr=False)
    tool_results: dict[str, dict[str, Any]] = field(repr=False)
    agents: dict[str, Any] = field(repr=False)
    network_attempts: list[tuple[str, int, bool]] = field(repr=False)
    mock_errors: list[str] = field(repr=False)
    wall_seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "job_dir": str(self.job_dir),
            "profile": self.profile,
            "game_ids": self.game_ids,
            "steps": self.steps,
            "score": self.score,
            "score_path": str(self.score_path),
            "games": self.games,
            "token_summary": self.token_summary,
            "per_game_tokens": self.per_game_tokens,
            "network_attempts": len(self.network_attempts),
            "network_blocked": [a for a in self.network_attempts if not a[2]],
            "mock_errors": self.mock_errors,
            "wall_seconds": round(self.wall_seconds, 1),
        }


def run_offline(
    games: list[str] | tuple[str, ...] = DEFAULT_GAMES,
    *,
    steps: int = 20,
    out_dir: str | Path | None = None,
    profile: str = "patched",
    seed: int = 0,
    env_dir: str | None = None,
    max_runtime_s_per_game: float = 600.0,
    block_network: bool = True,
    reasoning_chars: int = 4000,
    quiet: bool = False,
) -> OfflineRunResult:
    """Play `games` through the real Duck loop until each has executed `steps` actions."""
    started = time.monotonic()
    env_dir = env_dir or default_env_dir()
    game_ids = resolve_game_ids(list(games), env_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = Path(out_dir) if out_dir else _SCRATCH / "duck_offline" / f"{stamp}_{profile}"
    job_dir.mkdir(parents=True, exist_ok=True)
    # TAAF shells out to git for git_status.txt unless it already exists; keep the run git-free.
    (job_dir / "git_status.txt").write_text("duck_offline mock run (git status not captured)\n")
    (job_dir / "run_config.json").write_text(json.dumps({
        "model": "mock-scripted", "dataset": "offline-public", "games": game_ids, "concurrent_jobs": len(game_ids),
        "max_actions_per_game": steps, "profile": profile, "seed": seed,
    }, indent=2))

    policy = ScriptedPolicy(seed=seed, reasoning_chars=reasoning_chars)
    agents: dict[str, Any] = {}
    guard: NetworkGuard | None = None
    with patch_profile(profile), MockLLMServer(policy) as server:
        guard_cm: Any = NetworkGuard({server.port}) if block_network else contextlib.nullcontext()
        with guard_cm as guard_obj:
            guard = guard_obj if isinstance(guard_obj, NetworkGuard) else None
            from inference.tools.eval import evaluate_runs, save_score_file

            solver = shipped_solver(
                max_actions_per_game=steps,
                max_runtime_s_per_game=max_runtime_s_per_game,
                analyzer_factory=make_mock_analyzer_factory(server, agents),
            )
            bm = build_benchmark(game_ids, job_dir=job_dir, solver=solver, env_dir=env_dir)
            with contextlib.ExitStack() as stack:
                if quiet:
                    stack.enter_context(contextlib.redirect_stdout(stack.enter_context(open(os.devnull, "w"))))
                asyncio.run(bm.run(soft_end_time=None, runtime_environment=None, minimal_diagnostics=True))
            bm._save_json()  # notebook cell 16 does the same after a minimal-diagnostics run
            summary = evaluate_runs([job_dir])
            score_path = save_score_file(summary, run_dirs=[job_dir], output_path=job_dir / "score.json")

    runs = {run.game_id: run for run in bm.game_runs}
    games_out: dict[str, dict[str, Any]] = {}
    for game_id in game_ids:
        run = runs[game_id]
        games_out[game_id] = {
            "state": run.state,
            "actions": len(run.history),
            "levels_completed": run.levels_completed,
            "number_of_levels": run.number_of_levels,
            "final_score": run.final_score,
            "solver_note": run.solver_note,
            "action_ids": [rec.action.id.name for rec in run.history],
        }
    per_game_tokens = {
        game_id: summarize_records([r for r in policy.records if r.game == game_id]) for game_id in game_ids
    }
    result = OfflineRunResult(
        job_dir=job_dir,
        profile=profile,
        game_ids=game_ids,
        steps=steps,
        score=float(summary.overall_score),
        score_path=Path(score_path),
        games=games_out,
        token_summary=summarize_records(policy.records),
        per_game_tokens=per_game_tokens,
        records=list(policy.records),
        tool_results=dict(policy.tool_results),
        agents=agents,
        network_attempts=list(guard.attempts) if guard is not None else [],
        mock_errors=list(server.errors),
        wall_seconds=time.monotonic() - started,
    )
    (job_dir / "mock_metrics.json").write_text(json.dumps(
        {**result.to_json(), "records": [asdict(r) for r in policy.records], "tool_results": policy.tool_results},
        indent=2, default=str))
    return result


def _print_result(result: OfflineRunResult) -> None:
    print(f"\n== duck_offline profile={result.profile} job_dir={result.job_dir} ({result.wall_seconds:.1f}s)")
    for game_id, info in result.games.items():
        print(f"  {game_id:16s} state={info['state']:9s} actions={info['actions']:3d} "
              f"levels={info['levels_completed']}/{info['number_of_levels']} score={info['final_score']:.2f} "
              f"note={info['solver_note']}")
    print(f"  score.json: {result.score_path} (overall {result.score:.2f})")
    ts = result.token_summary
    print(f"  requests={ts['requests']} turn_starts={ts['turn_starts']} images/request={ts['images_per_request']}")
    print(f"  prompt tokens/request      : {ts['prompt_tokens_per_request']}")
    print(f"  ... excluding old reasoning: {ts['prompt_tokens_per_request_excl_reasoning']}")
    print(f"  prompt tokens at turn start: {ts['prompt_tokens_at_turn_start']}")
    print(f"  mean tokens by section     : {ts['section_tokens_mean']}")
    blocked = [a for a in result.network_attempts if not a[2]]
    print(f"  network: {len(result.network_attempts)} connects, {len(blocked)} blocked; mock errors: {result.mock_errors}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--games", default=",".join(DEFAULT_GAMES), help="comma list (ls20 or ls20-9607627b)")
    parser.add_argument("--steps", type=int, default=20, help="actions per game (HarnessSolver cap)")
    parser.add_argument("--profile", default="patched", choices=sorted(PROFILES))
    parser.add_argument("--compare-profiles", action="store_true", help="run every profile, same seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="job dir (default: SCRATCH/duck_offline/<stamp>_<profile>)")
    parser.add_argument("--reasoning-chars", type=int, default=4000, help="mock reasoning length per reply")
    parser.add_argument("--allow-network", action="store_true", help="do not install the network guard")
    parser.add_argument("--json", action="store_true", help="print the result JSON")
    args = parser.parse_args(argv)
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    profiles = sorted(PROFILES) if args.compare_profiles else [args.profile]
    results = []
    for profile in profiles:
        out = Path(args.out) / profile if (args.out and len(profiles) > 1) else args.out
        result = run_offline(games, steps=args.steps, out_dir=out, profile=profile, seed=args.seed,
                             block_network=not args.allow_network, reasoning_chars=args.reasoning_chars,
                             quiet=True)
        results.append(result)
        _print_result(result)
    if args.json:
        print(json.dumps([r.to_json() for r in results], indent=2, default=str))
    bad = [g for r in results for g, info in r.games.items() if info["state"] == "crashed" or info["actions"] < r.steps]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
