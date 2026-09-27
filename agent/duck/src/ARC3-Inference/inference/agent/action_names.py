"""Shared action-name mapping between model-facing labels and engine actions."""
from __future__ import annotations

from typing import Iterable

from inference.agent import patches


ENGINE_TO_MODEL_ACTION = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
    "RESET": "RESET",
}

MODEL_TO_ENGINE_ACTION = {value: key for key, value in ENGINE_TO_MODEL_ACTION.items()}

# [DUCK-PATCH P2] ACTION7 ("simple undo action" in the ARC docs; ar25/bp35/lf52/sb26/sk48/su15
# implement it as undo) is advertised in available_actions, but the stock map above cannot resolve
# it, so the model saw "ACTION7" and every call was rejected as an unknown action (MISTAKES #11).
# With P2 it is exposed as UNDO and executable under both names.
_P2_ENGINE_TO_MODEL_ACTION = {**ENGINE_TO_MODEL_ACTION, "ACTION7": "UNDO"}
_P2_MODEL_TO_ENGINE_ACTION = {value: key for key, value in _P2_ENGINE_TO_MODEL_ACTION.items()}


def _maps() -> tuple[dict[str, str], dict[str, str]]:
    if patches.enabled("P2_UNDO"):
        return _P2_ENGINE_TO_MODEL_ACTION, _P2_MODEL_TO_ENGINE_ACTION
    return ENGINE_TO_MODEL_ACTION, MODEL_TO_ENGINE_ACTION


def to_model_action(name: str | None) -> str:
    engine_to_model, _ = _maps()
    raw = str(name or "").strip().upper()
    return engine_to_model.get(raw, raw)


def to_engine_action(name: str | None) -> str | None:
    engine_to_model, model_to_engine = _maps()
    raw = str(name or "").strip().upper()
    if not raw:
        return None
    if raw in engine_to_model:
        return raw
    return model_to_engine.get(raw)


def to_model_actions(names: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for name in names:
        label = to_model_action(name)
        if label and label not in resolved:
            resolved.append(label)
    return resolved
