# Weekly routine (Saturday; research sweep + re-plan; follow exactly)

1. Same setup as `daily.md` steps 0-2.
2. Research sweep (write `research/weekly/<YYYY-MM-DD>.md`):
   - Kaggle: new/updated discussion topics + top comments since last sweep (internal API with Bearer token, see
     research/11 method); new public notebooks with public score ≥ 4 (see research/10 method); leaderboard deltas
     for the top 20 (`automation/state/leaderboard_*.json`), who jumped and by how much.
   - Hugging Face: new models relevant to 96 GB offline serving (Qwen 4 / Qwen3.8 / Gemma / gpt-oss variants,
     NVFP4/FP8 builds), new ARC-AGI-3 trace datasets; GitHub: new ARC-AGI-3 repos by top teams.
   - Web: arcprize.org blog/leaderboard changes, rules/FAQ changes, papers on ARC-AGI-3 agents.
   - For every finding: what it implies for us, and whether it changes the plan.
3. Grade the week: planned vs actual GPU hours, submissions made, LB movement vs the week's target (MASTER_PLAN §5),
   which experiments confirmed/falsified their thesis.
4. Re-plan: rewrite next week's rows in MASTER_PLAN §5 (targets may change), reorder BACKLOG, queue the week's
   submissions in SUBMISSION_QUEUE.md (one per day, each with an EXPERIMENTS thesis), open a new GPU_LEDGER week.
   If the week missed its target by > 30 %, change strategy (not just parameters) and say why.
5. Update the HF datasets with the week's new runs/traces (private). Commit + push. Report to the user (≤ 15 lines).
