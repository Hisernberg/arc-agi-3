# Submission queue (the daily routine submits the top READY entry; 1 per UTC day; none before 2026-10-03)

| Order | Candidate | Kernel slug | Status | Thesis (EXPERIMENTS id) | Gates (CPU tests / preflight / quota) |
|---|---|---|---|---|---|
| 1 | **Hydra-0** (M0: MTP off + 10 GiB KV + 28 seqs + wave-fit + UNDO + scoring/pacing lines + timeout 1200 + anim frames) | kragglenote2forwork/arc3-hydra0 (`notebooks/hydra0/`, built by `automation/hydra0/build_hydra0.py`) | **CPU-READY** 09-27 (19 tests green); remaining gate: GPU smoke commit 10-03 (validates vLLM start with MTP off + 10 GiB KV) → submit | EXP-002 | CPU ✅ / preflight: pending B09 / quota: 10-03 |
| 2 | Hydra-1 (EXP-001 serving winner + Hydra-0 + perception pack) | TBD | NOT READY (needs EXP-001) | EXP-003 | – |
| – | Rollback: best scored version (3.85) | kragglenote2forwork/duck-qwen3-8-anim-base-35c14d (v?) | READY (re-submit only deliberately) | – | – |

Statuses: NOT READY → READY (all gates green) → SUBMITTED (date, ref) → SCORED (public).
