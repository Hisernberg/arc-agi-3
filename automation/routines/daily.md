# Daily routine (fires once per day; follow exactly)

0. `cd` to the arc-agi-3 repo (clone https://github.com/Hisernberg/arc-agi-3 if missing), `git fetch origin` and
   check out the campaign branch `claude/charming-feynman-i0xee3` (pull latest). Read `CLAUDE.md`, then
   `plans/STATUS.md`, `plans/SUBMISSION_QUEUE.md`, `plans/MISTAKES.md`.
1. Credentials: need `KAGGLE_API_TOKEN` (and `HF_TOKEN` for datasets). If `~/.secrets/arc.env` exists, `source` it.
   If Kaggle auth is missing, write that into STATUS.md, notify the user, and do CPU-only work.
2. Sync: `python automation/kaggle_pipeline.py sync` (updates `plans/SUBMISSION_LOG.md`, `automation/state/`).
   Note any newly scored submission in STATUS.md and in the matching EXPERIMENTS.md entry (result + decision).
3. Submission decision (skip entirely before 2026-10-03 UTC):
   - If today's UTC slot is unused AND the top queue entry is READY (CPU tests green, preflight green, thesis written)
     AND `quota` leaves ≥ 1 h beyond this week's remaining plan AND no other GPU batch kernel of the user is running:
     log the estimate in GPU_LEDGER.md → `python automation/kaggle_pipeline.py daily --dir <notebook dir> -m "<ID>: <thesis>"`.
   - If nothing is READY: do NOT burn the slot on a random re-submission; only re-submit a top-2 candidate if the
     weekly plan explicitly scheduled a variance repeat for today.
4. Work: take the top unblocked item from `plans/BACKLOG.md`; implement + test on CPU; mark progress.
5. Wrap up: update STATUS.md (date, best LB, what was done, next action), GPU_LEDGER actuals, MISTAKES.md if anything
   went wrong; commit and push to the campaign branch. Keep the PR description current.
6. Report to the user in ≤ 8 lines: LB change, submission made (or why not), what was built, next step.
