# Submission queue (the daily routine submits the top READY entry; 1 per UTC day; none before 2026-10-03)

| Order | Planned date | Candidate | Kernel / build | Status | Thesis | Gates (CPU tests / preflight / quota) |
|---|---|---|---|---|---|---|
| 1 | Sat 10-03 | **Hydra-0** (MTP off + 10 GiB KV + 28 seqs + wave-fit + UNDO + scoring/pacing lines + timeout 1200 + anim frames) | `kragglenote2forwork/arc3-hydra0` · `notebooks/hydra0/` · `automation/hydra0/build_hydra0.py` | **CPU-READY** (preflight green 09-28, attachments verified) | EXP-002 | tests ✅ · preflight ✅ · quota 10-03 |
| 2 | Sun 10-04 | **Hydra-1a** (Hydra-0 on our Duck bundle: S1 + v9 prompt P1-P3) | `kragglenote2forwork/arc3-hydra1a` · `notebooks/hydra1a/` · `automation/hydra1/build_hydra1.py --variant 1a` | **CPU-READY** 09-28 (rehearsal w/ AGENTFIX green; bundle dataset uploaded + verified) | EXP-004 | tests ✅ · preflight ✅ · quota 10-03 |
| 3 | Mon 10-05 | **Hydra-1b** (Hydra-1a + M2 facts memory, −52 % prompt tokens) | `kragglenote2forwork/arc3-hydra1b` · `notebooks/hydra1b/` · `--variant 1b` | **CPU-READY** 09-28 | EXP-005 | tests ✅ · preflight ✅ · quota 10-03 |
| 4 | Tue 10-06+ | Hydra-2 = best of {0, 1a, 1b} + EXP-001 serving winner (+ perception hints when B04 lands) | TBD | NOT READY (needs EXP-001 + first scores) | EXP-003 | – |
| – | – | Rollback: best scored version (3.85) | `kragglenote2forwork/duck-qwen3-8-anim-base-35c14d` | READY (re-submit only deliberately) | – | – |

If a smoke commit fails, do NOT submit that version. Serving failure (vLLM OOM/startup with MTP off + 10 GiB KV — shared
by Hydra-0/1a/1b): rebuild with `--kv-gib 7` (Thuitanium-measured MTP-off config) and retry once the same day. Harness
failure (1a/1b only): submit the previous READY entry instead. Log every failure in MISTAKES.md. Rebuild notebooks from their builders before pushing (the build
is deterministic; tests assert it). After pushing a new bundle version (`automation/hydra1/stage_bundle.py`), Hydra-1
notebooks pick up the latest version automatically — rerun `tests/test_hydra1_notebook.py` first.

Statuses: NOT READY → READY (all gates green) → SUBMITTED (date, ref) → SCORED (public).
