# STATUS (update at the end of every session)

_Last updated: 2026-09-28 07:10 UTC (daily routine)_

- **Best public LB**: 3.85 (09-21/22). Latest scored: base notebook `taaf-flashnext-sheetu12b-0922` v1 = **3.18** (09-27 sub,
  EXP-000). Base family (same code as Scott's 5.19) spans ~2.4–5.2 per submission.
- **Leaderboard 09-28**: Yi-Chia Chen **28.34** (+9.5 overnight) · Tufa 27.29 · Daniel Franzen 21.01 (+4.3) · Lord Han Solo 20.80 ·
  Tong Hui Kang 20.53 · Third Intelligence 18.29 · the last dance 16.84 · NVARC3 16.07 · rellik13 13.40. Top-3 now ≈ 21.
- **GPU quota**: 29.77/30 h used (0.23 h left); refresh **2026-10-03 00:00 UTC**. No ARC GPU use and no submissions until then.
- **Credentials**: this container still has `~/.secrets/arc.env`; `KAGGLE_API_TOKEN` / `HF_TOKEN` / `ARC_API_KEY` are NOT yet
  environment variables — a recycled container would lose Kaggle/HF access (asked the user to add + rotate them).
- **Phase**: W0 (CPU build week). Done: B01 B02 B03 B06 B07 B13 P01 P02 P03 **B14 (Hydra-1a/1b)**. Partial (agents stopped by user
  09-27): B04 B05 B08 B09 B10 B11. Not started: B12. 09-28: B04 avatar eval made deterministic (stable sprite ids,
  MISTAKES #15): 12/14 keyboard games >= 0.8 (weak: g50t 0.19, sk48 0.77). Full suite **222/222 green**.
- **Ready to ship (all preflight green, attachments verified online)**: Hydra-0 (EXP-002), Hydra-1a (EXP-004), Hydra-1b (EXP-005).
  Private Kaggle datasets: `arc3-serving-kit` (v2), `arc3-duck-bundle` (v1, 86 files verified == local manifest).

## Runbook — Sat 2026-10-03 (first GPU day; the daily routine executes this)
1. `kaggle_pipeline.py quota` → expect ~30 h free; `kernels list --mine` → no other GPU batch kernel running (MISTAKES #5).
2. GPU_LEDGER: log Hydra-0 smoke (0.4 h) + EXP-001 (1.7 h) before launch.
3. Rebuild + test: `python automation/hydra0/build_hydra0.py && pytest tests/test_hydra0_*.py`, preflight `--kaggle-check`.
4. `kaggle_pipeline.py daily --dir notebooks/hydra0 -m "EXP-002 Hydra-0: MTP off+10GiB KV, wave-fit, UNDO, scoring lines"`
   (push = smoke commit → wait COMPLETE → submit). Before submitting, read the commit log: `HYDRA0_WAVEFIT`, `HYDRA0_SMOKE`,
   `HYDRA0_PATCH installed`, vLLM started with MTP off (no `speculative_config`) and `kv_cache_memory_bytes=10737418240`.
   If vLLM failed to start: rebuild `--kv-gib 7`, retry once; if that fails too, submit nothing new today and write it up.
5. Only after the Hydra-0 commit has finished (never 2 ARC GPU kernels at once): push `automation/exp001_notebook`
   (`kaggle_pipeline.py push --dir automation/exp001_notebook`, ~1.7 h), wait, `kaggle kernels output` → write
   `analysis/04_exp001_serving.md` (decisions/game-hour per profile) → pick the serving profile for Hydra-2.
6. 10-04: Hydra-1a (EXP-004); 10-05: Hydra-1b (EXP-005); then re-plan on the three scores (weekly routine 10-03 20:43 UTC
   writes W1's remaining days).
