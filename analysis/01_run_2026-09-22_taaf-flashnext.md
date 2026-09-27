# Post-mortem: `taaf-flashnext-sheetu12b-0922` (results_4.zip)

Run: 2026-09-22 15:52 → 17:52 (2 h validation cap, public-25 offline), Duck harness fork, `Qwen3.8-Flash-Next-NVFP4` via vLLM.
Local public-25 mean score **10.02** (median 4.76, 0/25 games fully won). Best Kaggle public LB so far: **3.85** (the hidden set is harder than public-25 for this setup).

## 1. Scoring rule, reverse-engineered from this run

`ft09` finished 4/6 levels, each under the human baseline → 47.62 = (1+2+3+4)/(1+…+6) = 10/21.
`ar25` 4/8 all under baseline → 27.78 = 10/36. So:

- **Level weight = level index** (level k weighs k). A 6-level game: L1 is worth 4.8 %, L6 is worth 28.6 %.
- A level solved in ≤ baseline actions earns full credit; above baseline the credit decays (exact curve verified in `research/21_env_and_scoring.md`).
- Unfinished levels earn 0. **The first 2 levels of a 7-level game are worth only 3/28 = 10.7 %**; deep levels are where the points are.

## 2. The #1 defect: serving throughput (not reasoning quality)

| Metric (vLLM final metrics / logs) | Value |
|---|---|
| Checkpoint size | 125.9 GiB (NVFP4 + CPU "PLE offload"), ~82 GiB on GPU |
| KV cache | **5 GiB** (`--kv-cache-memory-bytes 5368709120`) |
| Requests running concurrently | **3** in 596/719 log ticks (never > 6) |
| Requests waiting | **21.6 on average** (28 games in parallel) |
| Preemptions | 342 |
| Prompt tokens prefilled | 20.46 M (prefix caching **disabled**) |
| Generated tokens | 1.80 M → **236 tok/s aggregate** |
| MTP acceptance | mean length 2.7, 55 % draft acceptance |
| LLM calls in 2 h | 1,045 total ≈ 42 per game ≈ **1 call / 2.9 min per game** |
| Actions in 2 h | 2,453 total ≈ 98 per game (4.4 actions per LLM decision) |

The first wave of requests (≈4.6 k prompt tokens) answered in 5-24 s; once 25 games queued behind 3 KV slots the
per-request latency grew to 130-256 s. **Each game spent ≈85 % of its wall-clock waiting in the vLLM queue.**
Every game ended with a `ReadTimeout` at the 2-hour cap mid-level.

Fix direction (see `plans/MASTER_PLAN.md`): a model whose weights leave ≥ 50 GiB for KV (e.g. a ~27-35B FP8/NVFP4
model or an MoE with small active params), prefix caching ON (system prompt is identical for all 25+ games), and
concurrency matched to KV capacity. Target ≥ 10× more LLM decisions per game-hour.

## 3. Per-game outcome (sorted by score)

| game | levels | actions per level (agent/baseline) | score | observation |
|---|---|---|---|---|
| ft09 | 4/6 | 11/43, 7/12, 14/23, 16/28 | 47.6 | click-only pattern game, very efficient |
| lp85 | 5/8 | 5/17, 12/38, 17/31, 22/16, 11/41 | 41.0 | click-only; L6 57/60 unfinished at cap |
| ar25 | 4/8 | 17/32, 21/50, 71/75, 27/37 | 27.8 | mirror/symmetry game |
| sc25 | 3/6 | 82/36, 5/6, 44/32 | 19.4 | |
| vc33 | 3/7 | 16/7, 19/18, 31/44 | 19.4 | |
| re86 | 3/8 | 21/26, 57/42, 47/86 | 15.8 | |
| m0r0 | 2/6 | 24/30, 37/111 | 14.3 | |
| ka59 | 2/7 | 23/28, 78/109 | 10.7 | |
| tu93 | 3/9 | 19/19, 25/16, 40/34 | 8.9 | |
| sb26 | 2/8 | 9/18, 27/28 | 8.3 | |
| cn04, r11l, sp80, tr87 | 1/6 | – | 4.8 | stuck on L2 |
| ls20 | 1/7 | 21/22 | 3.6 | L2 69/123 unfinished |
| tn36, s5i5, dc22, wa30, su15, lf52 | 1 | – | 1.8-2.8 | |
| bp35 | 1/9 | 47/21 | 0.44 | L1 solved at 2.2× baseline → heavy penalty |
| sk48, cd82, g50t | 0 | – | 0 | never cleared L1 in 2 h (41-80 actions) |

Pattern: **agents are efficient when they do solve a level** (most solved levels are under baseline) — the problem
is that they run out of wall-clock after 1-4 levels. More decisions per hour converts directly into more levels.

## 4. Other defects seen in transcripts

1. Queue-bound latency (above) → only ~40 analysis steps per game.
2. Prompt bloat: 20 M prompt tokens for 1.8 M generated (11:1), prefix caching off → prefill dominates GPU time.
3. Games with hard perception (g50t teleport maze, sk48 snake/rope, cd82 rotating painter) consumed the whole budget
   on L1 world-model debugging; there's no fallback explorer to at least brute-force L1.
4. 25 games × 1 GPU all run concurrently; no prioritisation of games that are progressing (ft09/lp85 were
   progressing at ~1 level / 20 min and were cut off at the cap).
5. No cross-game memory/priors (e.g. "click-only games: try every distinct object once", "HUD bar = action budget").

## 5. What to keep

- Duck prompt structure + python tool + segmentation view (works: ft09/lp85/ar25 solved levels under baseline).
- AGENTFIX F1 (image stripping), F2 (action results), F3 (memory parsing), F5 (HUD-aware no-op detection).
- Watchdog + deterministic teardown (but the teardown raised `vLLM teardown did not reach the bounded terminal gate`
  at the end — harmless in validation, but make sure it cannot fail a real submission).
