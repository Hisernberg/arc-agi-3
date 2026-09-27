"""Shared test setup: import paths for agent/ and tools/, and a skip marker for tests that need the
offline game environments (arcengine wheels + environment_files, see agent/arcenv.py).

Run:  SCRATCH/venv/bin/python -m pytest tests/ -q
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("agent", "tools"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import arcenv  # noqa: F401

    HAVE_ENV = bool(arcenv.list_games())
except Exception:  # pragma: no cover - environment without the competition wheels
    HAVE_ENV = False

requires_env = pytest.mark.skipif(not HAVE_ENV, reason="offline ARC environments (arcengine + environment_files) not available")


def all_games():
    return arcenv.list_games() if HAVE_ENV else []
