"""Switches for the campaign's source-level fixes to the shipped Duck agent.

Every fix lives directly in the source (tagged ``[DUCK-PATCH <ID>]``) and is guarded by
``enabled("<ID>")`` so it can be turned off per run without touching code:

- ``DUCK_PATCHES=0`` turns every fix off (byte-for-byte shipped-bundle behaviour).
- ``DUCK_PATCH_<ID>=0`` turns one fix off, e.g. ``DUCK_PATCH_F2_RESULT=0``.
- ``DUCK_PATCHES_V9=0`` turns the B06/B07 "v9" group (P1-P3, M1, M2) off: exactly the B01 patch
  set (F1/F2/F3/S1). ``DUCK_PATCH_<ID>=1`` turns on a fix whose default is off (M1, M2).

Switches are read at call time, never cached at import time, so a test or a runner can flip
them between runs in one process. See ``agent/duck/PATCHES.md`` for what each fix does.
"""
from __future__ import annotations

import os

PATCH_IDS = (
    "F1_IMAGES",
    "F2_RESULT",
    "F3_MEMORY",
    "S1_STATE_KEY",
    # [DUCK-PATCH P1-P3, M1, M2] B07 prompt v9 and B06 memory/compaction.
    "P1_SCORING",
    "P2_UNDO",
    "P3_DISCIPLINE",
    "M1_THINK",
    "M2_FACTS",
    # [DUCK-PATCH B08_SCHEDULER] progress-aware turn scheduler (agent/scheduler.py, framework/solver.py).
    "B08_SCHEDULER",
)

# The v9 group (DUCK_PATCHES_V9=0 turns all of these off).
V9_PATCH_IDS = ("P1_SCORING", "P2_UNDO", "P3_DISCIPLINE", "M1_THINK", "M2_FACTS")

# Off unless explicitly enabled: they change what the model remembers and need a GPU A/B first.
DEFAULT_OFF = frozenset({"M1_THINK", "M2_FACTS", "B08_SCHEDULER"})  # B08: changes scheduling, opt-in

_OFF = {"0", "false", "no", "off"}


def enabled(patch_id: str) -> bool:
    if os.environ.get("DUCK_PATCHES", "1").strip().lower() in _OFF:
        return False
    if patch_id in V9_PATCH_IDS and os.environ.get("DUCK_PATCHES_V9", "1").strip().lower() in _OFF:
        return False
    default = "0" if patch_id in DEFAULT_OFF else "1"
    return os.environ.get(f"DUCK_PATCH_{patch_id}", default).strip().lower() not in _OFF


def int_setting(name: str, default: int) -> int:
    raw = os.environ.get(f"DUCK_PATCH_{name}", "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def report() -> dict[str, bool]:
    return {patch_id: enabled(patch_id) for patch_id in PATCH_IDS}
