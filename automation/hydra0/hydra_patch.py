"""HYDRA-0 runtime patch (applied from a notebook cell AFTER the AGENTFIX cell, before the benchmark runs).

H1  HYDRA_SCORING   Append two short lines to every user prompt: the true scoring rule and a pacing line with the
                    game's remaining wall-clock budget. Community-measured +28 % hidden together with click hints
                    (Kaggle discussion 743060); ~60 tokens per turn.
H2  (backlink)      `_HarnessGameSession.request_timeout_seconds` (called right before every analyzer request) stores
                    a back-reference `analyzer._hydra_session = session`, so the prompt builder can read
                    `session.timing_payload()`.

Every patch is fail-open: any exception leaves the stock behaviour untouched. Switch off with HYDRA_SCORING=0.
"""
from __future__ import annotations

import os

_ON = lambda k, d="1": os.environ.get(k, d).strip() not in ("0", "", "false", "False")

SCORING_LINE = (
    "Scoring: a level you finish scores min(1.15, (human_actions / your_actions)^2); level k weighs k; "
    "an unfinished level scores 0 and actions spent on it cost nothing. Clearing the next level is worth far more "
    "than saving a few actions, but once you know how to win a level, win it in as few actions as possible."
)


def _pacing_line(analyzer) -> str | None:
    session = getattr(analyzer, "_hydra_session", None)
    if session is None:
        return None
    timing = session.timing_payload()
    remaining = timing.get("time_remaining_seconds")
    elapsed = timing.get("run_elapsed_seconds") or 0.0
    if remaining is None:
        return None
    total = max(1.0, elapsed + remaining)
    level = None
    try:
        level = session.current_frame().level
    except Exception:
        pass
    lvl = f" You are on level {level}." if level is not None else ""
    return (f"Time left for this game: {remaining / 60:.0f} of {total / 60:.0f} min "
            f"({100 * remaining / total:.0f}%).{lvl} Pace yourself: commit to your best hypothesis and act.")


def install(tool_agent_module, solver_module) -> dict:
    report = {"installed": False}
    if not _ON("HYDRA_SCORING"):
        report["reason"] = "HYDRA_SCORING=0"
        return report
    try:
        TA = tool_agent_module.ToolAgent
        Session = solver_module._HarnessGameSession

        _orig_rts = Session.request_timeout_seconds

        def request_timeout_seconds(self):
            try:
                self.analyzer._hydra_session = self
            except Exception:
                pass
            return _orig_rts(self)

        Session.request_timeout_seconds = request_timeout_seconds

        _orig_bup = TA._build_user_prompt

        def _build_user_prompt(self, action_num, **kw):
            text = _orig_bup(self, action_num, **kw)
            try:
                extra = [SCORING_LINE]
                pacing = _pacing_line(self)
                if pacing:
                    extra.append(pacing)
                return text.rstrip("\n") + "\n" + "\n".join(extra) + "\n"
            except Exception:
                return text

        TA._build_user_prompt = _build_user_prompt
        report.update(installed=True, H1_scoring=True, H2_backlink=True)
    except Exception as exc:  # fail-open
        report["error"] = repr(exc)
    return report
