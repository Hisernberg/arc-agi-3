# ARC-AGI-3 campaign — operating rules for every Claude session

Competition: Kaggle `arc-prize-2026-arc-agi-3` (final deadline **2026-11-02 23:59 UTC**; entry/team-merge and
notebook-publishing cut-off **2026-10-26**). Kaggle user `kragglenote2forwork`, HF user `Nabidnur`,
GitHub `Hisernberg`. Companion repo with the earlier campaign (v4→v8 harness patches, docs 01-16):
`Hisernberg/arc-agi-3-campaign`.

Start every session by reading, in order:
1. `plans/MASTER_PLAN.md` — strategy, targets, calendar, gates.
2. `plans/STATUS.md` — where we are right now (last run, last submission, next action).
3. `plans/BACKLOG.md` — prioritised work items; take the top unblocked one.
4. `plans/MISTAKES.md` — things that already cost us a run. Do not repeat them.

## Hard rules
- **Never delete anything** from any repo or hub — GitHub (files, branches, history: no force-push/reset of pushed
  commits), Hugging Face (repos, files, revisions), Kaggle (datasets, notebooks, versions). The user does all
  deletions manually. Storage is plentiful: add new files/versions instead of replacing or pruning.
- **Secrets**: credentials come only from environment variables `KAGGLE_API_TOKEN`, `HF_TOKEN`, `ARC_API_KEY`,
  `GITHUB_TOKEN` (or a local `~/.secrets/arc.env` that is never committed). Never write a token into any file in
  this repo, a notebook, a Kaggle dataset, an HF repo, a commit message or a PR. If a variable is missing, say so
  and continue with work that doesn't need it.
- **Submissions**: at most **1 per UTC day** (the competition limit). `automation/kaggle_pipeline.py` enforces it.
  No submission before **2026-10-03** (GPU quota exhausted until the Saturday 00:00 UTC refresh).
- **GPU budget**: ≤ **15 GPU-hours per week** for ARC (Kaggle quota is 30 h/week shared with other comps). Every GPU
  use is logged in `plans/GPU_LEDGER.md` *before* launch with an estimate, and reconciled after.
  Check `python automation/kaggle_pipeline.py quota` first. Never run two ARC GPU kernels at once
  (Kaggle allows only 2 batch GPU sessions per account, and the user's other comps use them too).
- **Run gate** (nothing touches the GPU until all are true): written thesis in `plans/EXPERIMENTS.md`
  (what changed, expected effect, what would falsify it); CPU tests green (`tests/`); preflight green
  (`automation/preflight.py` once it exists); kernel title == existing slug title; quota check passes.
- **CPU is free**: all 25 public games run offline on CPU via `agent/arcenv.py` (state cloning works for all 25).
  Develop and test every harness change on CPU first.
- Never train on or hard-code solutions to the 25 public games: the hidden 110 games are different by design.
  Public games are for mechanism testing only.
- Local eval must reject the `ACTION6`-with-null-coordinates "win" bug (scores a false WIN on 18/25 games).
- Commit and push after every meaningful step. Keep large raw data out of git (Hugging Face datasets or scratch).

## Where things are
- `research/` — the research "brain" (Kaggle notebooks + discussions, web SOTA, HF/GitHub assets, env + scoring,
  game mechanics). `research/weekly/` — the Saturday research sweeps.
- `analysis/` — run post-mortems. `plans/` — plan, status, backlog, experiments, ledgers, submission log.
- `agent/` — our harness code (offline env `arcenv.py`, perception pack, action channel, memory).
- `automation/` — Kaggle pipeline (`kaggle_pipeline.py`), notebook builder, preflight, routine prompts.
- `tools/` — dataset builders (HF: `Nabidnur/arc-agi-3-*`, all private).
