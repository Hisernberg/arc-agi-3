"""Switches for the campaign's source-level fixes to the shipped Duck agent.

Every fix lives directly in the source (tagged ``[DUCK-PATCH <ID>]``) and is guarded by
``enabled("<ID>")`` so it can be turned off per run without touching code:

- ``DUCK_PATCHES=0`` turns every fix off (byte-for-byte shipped-bundle behaviour).
- ``DUCK_PATCH_<ID>=0`` turns one fix off, e.g. ``DUCK_PATCH_F2_RESULT=0``.

Switches are read at call time, never cached at import time, so a test or a runner can flip
them between runs in one process. See ``agent/duck/PATCHES.md`` for what each fix does.
"""
from __future__ import annotations

import os

PATCH_IDS = ("F1_IMAGES", "F2_RESULT", "F3_MEMORY", "S1_STATE_KEY")

_OFF = {"0", "false", "no", "off"}


def enabled(patch_id: str) -> bool:
    if os.environ.get("DUCK_PATCHES", "1").strip().lower() in _OFF:
        return False
    return os.environ.get(f"DUCK_PATCH_{patch_id}", "1").strip().lower() not in _OFF


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
