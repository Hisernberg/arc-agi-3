# Submission queue (the daily routine submits the top READY entry; 1 per UTC day; none before 2026-10-03)

| Order | Candidate | Kernel slug | Status | Thesis (EXPERIMENTS id) | Gates (CPU tests / preflight / quota) |
|---|---|---|---|---|---|
| 1 | **Hydra-0** (M0: MTP off + bigger KV + wave-fit + UNDO + scoring/pacing lines + ctx fix + anim frames) | TBD | NOT READY (build W0) | EXP-002 | – |
| 2 | Hydra-1 (EXP-001 serving winner + Hydra-0 + perception pack) | TBD | NOT READY (needs EXP-001) | EXP-003 | – |
| – | Rollback: best scored version (3.85) | kragglenote2forwork/duck-qwen3-8-anim-base-35c14d (v?) | READY (re-submit only deliberately) | – | – |

Statuses: NOT READY → READY (all gates green) → SUBMITTED (date, ref) → SCORED (public).
