"""[DUCK-PATCH M2] Per-game verified-facts block (B06).

The stock agent carries memory only as (a) chat history, evicted oldest-first once the request
estimate exceeds the budget, and (b) the model's own labelled notes ("World model:", "Plan:" ...),
which are wiped on every level transition. This module adds a compact block that survives history
trimming and is rebuilt every turn from two sources:

1. **Harness-verified effects** (never dropped): every executed action is read back from the
   runtime-state history the solver writes (the real before/after frames), classified as
   ``changed`` (cells outside the 2-cell border band changed), ``edge-only`` (only border cells
   changed: usually a HUD/timer bar), ``no change``, ``level up`` or ``GAME_OVER`` (from the
   harness's own action results), and aggregated per (level, action key). MOUSE clicks are keyed
   by the colour of the clicked cell (``MOUSE@R``) with sample hit / dead coordinates.
2. **Model-written notes** (truncated to fit): the latest labelled blocks the harness parsed, plus
   the last notes of every cleared level (archived instead of silently wiped).

Plus the level history from the solver (levels cleared, actions used vs the human baseline when
the environment exposes it). The block is capped (``max_chars``); only model-written notes are
truncated to meet the cap, verified facts switch to a compact rendering instead of being dropped.

One ledger belongs to one game session: ``ToolAgent._ensure_session`` creates a fresh ledger (keyed
by the S1 game key) whenever the state file changes, so facts never cross games.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from inference.utils.grid_utils import ARC_COLOR_CHARS

EDGE_BAND = 2  # rows/cols within this many cells of the border count as "edge" (HUD/timer bars)
MAX_MOUSE_KEYS = 6  # distinct MOUSE@colour lines per level before the rest are grouped
MAX_SAMPLES = 3  # hit / dead click coordinates kept per MOUSE key

_MOUSE_RE = re.compile(r"MOUSE\(row=(-?\d+),\s*col=(-?\d+)\)")

NOTE_LABELS = (
    ("world_model", "World model"),
    ("goal_model", "Goal model"),
    ("action_model", "Action model"),
    ("current_plan", "Plan"),
    ("recent_findings", "Recent findings"),
    ("open_questions", "Open questions"),
    ("cross_level_notes", "Cross-level notes"),
)


@dataclass
class EffectTally:
    """Harness-verified outcomes of one action key on one level."""

    n: int = 0
    changed: int = 0
    edge_only: int = 0
    no_change: int = 0
    level_up: int = 0
    game_over: int = 0
    cells: int = 0  # interior cells changed, summed over `changed` actions
    last_bbox: tuple[int, int, int, int] | None = None
    hits: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)

    def add(self, other: "EffectTally") -> None:
        self.n += other.n
        self.changed += other.changed
        self.edge_only += other.edge_only
        self.no_change += other.no_change
        self.level_up += other.level_up
        self.game_over += other.game_over
        self.cells += other.cells

    def outcome_text(self, *, detailed: bool) -> str:
        parts: list[str] = []
        if self.level_up:
            parts.append(f"level up {self.level_up}")
        if self.game_over:
            parts.append(f"GAME_OVER {self.game_over}")
        if self.changed:
            text = f"changed {self.changed}"
            if detailed:
                avg = max(1, round(self.cells / max(1, self.changed)))
                text += f" (~{avg} cells"
                if self.last_bbox is not None:
                    r0, r1, c0, c1 = self.last_bbox
                    text += f", last r{r0}-{r1} c{c0}-{c1}"
                text += ")"
            parts.append(text)
        if self.edge_only:
            parts.append(f"edge-only {self.edge_only}")
        if self.no_change:
            parts.append(f"no change {self.no_change}")
        text = ", ".join(parts) or "no effect recorded"
        if detailed and (self.hits or self.dead):
            if self.hits:
                text += f" [hit {' '.join(self.hits)}]"
            if self.dead:
                text += f" [dead {' '.join(self.dead)}]"
        return text


@dataclass
class LevelRecord:
    level: int
    first_action: int
    actions: int = 0
    cleared_at: int | None = None
    last_actions: list[str] = field(default_factory=list)  # most recent keys (for "cleared by")
    tallies: dict[str, EffectTally] = field(default_factory=dict)
    archived_notes: str = ""


def _diff(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]) -> tuple[int, int, tuple[int, int, int, int] | None]:
    """(interior changed cells, edge changed cells, interior bbox r0,r1,c0,c1)."""
    rows = max(len(before), len(after))
    interior = edge = 0
    r_min = c_min = 10**9
    r_max = c_max = -1
    for r in range(rows):
        row_a = before[r] if r < len(before) else ()
        row_b = after[r] if r < len(after) else ()
        if row_a == row_b:
            continue
        cols = max(len(row_a), len(row_b))
        for c in range(cols):
            a = row_a[c] if c < len(row_a) else None
            b = row_b[c] if c < len(row_b) else None
            if a == b:
                continue
            if r < EDGE_BAND or c < EDGE_BAND or r >= rows - EDGE_BAND or c >= cols - EDGE_BAND:
                edge += 1
                continue
            interior += 1
            r_min, r_max = min(r_min, r), max(r_max, r)
            c_min, c_max = min(c_min, c), max(c_max, c)
    bbox = (r_min, r_max, c_min, c_max) if interior else None
    return interior, edge, bbox


def _action_key(action: str, before: tuple[tuple[int, ...], ...]) -> tuple[str, str | None]:
    """History action display -> (aggregation key, click coordinate label or None)."""
    text = str(action or "").strip()
    match = _MOUSE_RE.fullmatch(text)
    if match:
        row, col = int(match.group(1)), int(match.group(2))
        colour = "?"
        if 0 <= row < len(before) and 0 <= col < len(before[row]):
            value = int(before[row][col])
            colour = ARC_COLOR_CHARS[max(0, min(15, value))]
        return f"MOUSE@{colour}", f"r{row}c{col}"
    return (text.upper() or "?"), None


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


class FactsLedger:
    """Per-game ledger of harness-verified action effects + level history + archived notes."""

    def __init__(self, game_key: str | None) -> None:
        self.game_key = game_key
        self.levels: dict[int, LevelRecord] = {}
        self.last_step = 0  # frame.step of the newest ingested history entry
        self.total_actions = 0
        self.game_over_steps: set[int] = set()
        self.level_up_steps: set[int] = set()
        self.current_level = 1

    # ------------------------------------------------------------------------------ ingestion
    def note_action_results(self, results: Iterable[dict[str, Any]]) -> None:
        """Record GAME_OVER / level completion from the harness's own action results (the frame
        alone cannot show GAME_OVER). ``action_num`` is the step after the terminal action."""
        for item in results:
            if not isinstance(item, dict) or not item.get("executed"):
                continue
            try:
                step = int(item.get("action_num"))
            except (TypeError, ValueError):
                continue
            if item.get("game_over"):
                self.game_over_steps.add(step)
            if item.get("level_completed") or item.get("run_complete"):
                self.level_up_steps.add(step)

    def _level(self, level: int, first_action: int) -> LevelRecord:
        record = self.levels.get(level)
        if record is None:
            record = LevelRecord(level=level, first_action=first_action)
            self.levels[level] = record
        return record

    def ingest(self, history: list[Any]) -> int:
        """Ingest runtime-state history entries (``.action``, ``.frame.grid/.step/.level``) newer
        than the last call. Returns the number of new actions."""
        if not history:
            return 0
        first = history[0].frame
        self.current_level = max(self.current_level, int(first.level))
        self._level(int(first.level), 1)
        added = 0
        for index in range(1, len(history)):
            entry = history[index]
            step = int(entry.frame.step)
            if step <= self.last_step:
                continue
            prev = history[index - 1].frame
            level = int(prev.level)
            key, coord = _action_key(entry.action, prev.grid)
            interior, edge, bbox = _diff(prev.grid, entry.frame.grid)
            outcome = EffectTally(n=1)
            leveled = int(entry.frame.level) > level or step in self.level_up_steps
            if step in self.game_over_steps:
                outcome.game_over = 1
            elif leveled:
                outcome.level_up = 1
            elif interior:
                outcome.changed, outcome.cells, outcome.last_bbox = 1, interior, bbox
            elif edge:
                outcome.edge_only = 1
            else:
                outcome.no_change = 1
            record = self._level(level, step)
            tally = record.tallies.setdefault(key, EffectTally())
            tally.add(outcome)
            if outcome.last_bbox is not None:
                tally.last_bbox = outcome.last_bbox
            if coord is not None:
                bucket = tally.hits if (outcome.changed or outcome.level_up) else tally.dead
                if coord not in bucket:
                    bucket.append(coord)
                    del bucket[:-MAX_SAMPLES]
            record.actions += 1
            record.last_actions.append(coord and f"{key}({coord})" or key)
            del record.last_actions[:-4]
            if leveled and record.cleared_at is None:
                record.cleared_at = step
            new_level = int(entry.frame.level)
            if new_level > level:
                self._level(new_level, step + 1)
            self.current_level = max(self.current_level, new_level)
            self.last_step = step
            self.total_actions += 1
            added += 1
        return added

    def archive_notes(self, level: int | None, notes: dict[str, str]) -> None:
        """Keep the last model notes of a level that is being wiped (level transition)."""
        text = "; ".join(
            f"{label}: {' '.join(str(notes.get(key, '')).split())}"
            for key, label in NOTE_LABELS
            if key in ("world_model", "goal_model", "action_model", "current_plan") and notes.get(key)
        )
        if not text:
            return
        record = self._level(int(level or self.current_level), self.last_step + 1)
        record.archived_notes = text

    # ------------------------------------------------------------------------------ rendering
    def _verified_lines(self, status: dict[str, Any] | None, *, detailed: bool) -> list[str]:
        status = status or {}
        lines: list[str] = []
        n_levels = status.get("number_of_levels")
        completed = status.get("levels_completed")
        cleared = sorted(r.level for r in self.levels.values() if r.cleared_at is not None)
        if completed is None:
            completed = len(cleared)
        per_level = list(status.get("actions_per_level") or [])
        baselines = status.get("baseline_actions") or None
        level_bits: list[str] = []
        for index in range(int(completed or 0)):
            used = per_level[index] if index < len(per_level) else (
                self.levels[index + 1].actions if (index + 1) in self.levels else None)
            if used is None:
                continue
            bit = f"L{index + 1} {used}"
            if baselines and index < len(baselines) and baselines[index]:
                bit += f"/{baselines[index]}"
            level_bits.append(bit)
        current = self.current_level
        head = f"- Levels cleared: {completed}" + (f" of {n_levels}" if n_levels else "")
        if level_bits:
            head += " (actions" + ("/human baseline" if baselines else "") + ": " + ", ".join(level_bits) + ")"
        record = self.levels.get(current)
        on_level = per_level[current - 1] if 0 < current <= len(per_level) else (record.actions if record else 0)
        head += f". Now L{current}: {on_level} actions on this level, {self.total_actions} in total."
        lines.append(head)

        if record is not None and record.tallies:
            lines.append(f"- L{current} effects: " + self._tally_text(record.tallies, detailed=detailed))
        others = [r for lvl, r in sorted(self.levels.items()) if lvl != current and r.tallies]
        if others:
            merged: dict[str, EffectTally] = {}
            for rec in others:
                for key, tally in rec.tallies.items():
                    merged.setdefault(key, EffectTally()).add(tally)
            lines.append("- Earlier levels: " + self._tally_text(merged, detailed=False))
        wins = [
            f"L{r.level} at action {r.cleared_at} (last: {', '.join(r.last_actions[-3:])})"
            for _, r in sorted(self.levels.items())
            if r.cleared_at is not None
        ]
        if wins:
            lines.append("- Cleared by: " + "; ".join(wins) + ".")
        return lines

    @staticmethod
    def _tally_text(tallies: dict[str, EffectTally], *, detailed: bool) -> str:
        keys = list(tallies)
        mouse = [k for k in keys if k.startswith("MOUSE@")]
        grouped: list[tuple[str, EffectTally]] = [(k, tallies[k]) for k in keys if k not in mouse]
        if len(mouse) > MAX_MOUSE_KEYS:
            ranked = sorted(mouse, key=lambda k: (-tallies[k].n, k))
            keep, rest = ranked[:MAX_MOUSE_KEYS - 1], ranked[MAX_MOUSE_KEYS - 1:]
            grouped.extend((k, tallies[k]) for k in keep)
            other = EffectTally()
            for k in rest:
                other.add(tallies[k])
            grouped.append(("MOUSE@{" + ",".join(k[6:] for k in sorted(rest)) + "}", other))
        else:
            grouped.extend((k, tallies[k]) for k in mouse)
        return "; ".join(f"{k} {t.n}x: {t.outcome_text(detailed=detailed)}" for k, t in grouped)

    def render(
        self,
        *,
        status: dict[str, Any] | None,
        notes: dict[str, str],
        max_chars: int,
    ) -> list[str]:
        header = "Verified facts (harness-checked from real frames; kept when old history is trimmed):"
        verified = self._verified_lines(status, detailed=True)
        if len("\n".join([header, *verified])) > int(max_chars * 0.65):
            verified = self._verified_lines(status, detailed=False)
        lines = [header, *verified]
        footer = "- Revise any note above immediately if `current_frame` or `history` contradicts it."
        remaining = max_chars - len("\n".join(lines)) - len(footer) - 2

        note_lines: list[str] = []
        candidates = [(label, notes.get(key, "")) for key, label in NOTE_LABELS if notes.get(key)]
        candidates += [
            (f"L{r.level} final notes", r.archived_notes)
            for _, r in sorted(self.levels.items(), reverse=True)
            if r.archived_notes
        ]
        if candidates:
            title = "Model notes (latest; not verified by the harness):"
            remaining -= len(title) + 1
            # Priority order (World model first); each note gets an equal share of what is left,
            # so a short note hands its unused room to the ones after it.
            for index, (label, value) in enumerate(candidates):
                prefix = f"- {label}: "
                budget = remaining // (len(candidates) - index) - len(prefix) - 1
                if budget < 24:
                    continue
                text = _clip(value, budget)
                if not text:
                    continue
                note_lines.append(prefix + text)
                remaining -= len(note_lines[-1]) + 1
            if note_lines:
                lines.append(title)
                lines.extend(note_lines)
        lines.append(footer)
        return lines
