# STATUS (update at the end of every session)

_Last updated: 2026-09-27 09:00 UTC_

- **Best public LB**: 3.85 (2026-09-21/22). Latest: v7 nooa-lite 2.61 (09-26); taaf-flashnext-sheetu12b-0922 v1 submitted 09-27 07:01 UTC, **pending**.
- **Leaderboard**: Tufa 27.29 · Lord Han Solo 20.80 · Tong Hui Kang 20.53 · Yi-Chia Chen 18.80 · D. Franzen 16.68 · NVARC3 16.07.
- **GPU quota**: 27.6/30 h used; refresh **2026-10-03 00:00 UTC**. No ARC GPU use and no submissions until then (user directive).
- **Phase**: W0 (CPU-only build week). See `plans/BACKLOG.md`.
- **Next GPU action**: Sat 10-03 — (1) Hydra-0 smoke commit (~0.4 h) → submit (EXP-002); (2) serving A/B EXP-001 (~1.5 h).
- **Hydra-0**: built + CPU-validated (`notebooks/hydra0/`, tests `tests/test_hydra0_*.py`).
- **HF datasets (all private, Nabidnur/)**: `arc-agi-3-traces` (replays + our run steps), `arc-agi-3-kaggle-discussions`
  (225 topics / 1,144 posts + LB history), `arc-agi-3-frontier-replays` (25 GPT-6 Astra recordings, sessions, human baselines),
  `arc-agi-3-research` (raw web sources + docs snapshot). `arc-agi-3-kaggle-notebooks` (1,132 notebooks indexed, 456 scored sources, 43 with outputs, solver bundles, LB; 21,581 files).
- **Open questions for the user**: timezone for routines; confirm secrets added as environment variables.
