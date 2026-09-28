# Experiments (write the thesis BEFORE the run)

Format: ID · date · change · expected effect · falsifier · result · decision

## EXP-001 · planned 10-03 · Serving A/B
- **Change**: replay recorded Duck prompts (results_4 `prompts/*.log`) as 28 concurrent multi-turn agents against
  (A) Flash-Next NVFP4 tuned vLLM (KV ≈ 11 GiB, fp8 KV, prefix cache on, 16 k ctx, seqs matched), (B) Qwen3.8-27B-FP8
  (≈ 55 GiB KV, prefix cache on, 28–32 seqs), baseline (0) = current profile.
- **As built (09-27, B02/B03)**: `python serving/launch.py --suite exp001` (notebook `automation/exp001_notebook/`),
  order = (A) `flashnext_tuned` → (B) `qwen27b_fp8` → (0) `flashnext_baseline` → extras if the 100-min budget allows.
  (A) is now **MTP off, 12 GiB bf16 KV, prefix cache (mamba `align`), 16 k ctx, 28 seqs**. fp8 KV is impossible for
  Flash-Next on the keithtyser runtime (vLLM 0.1.dev20073: QSA raises "requires a BF16 main KV cache"), and KV ≥ 11 GiB
  needs MTP off (81.8 → 74.3 GiB weights). (B) = vLLM 0.19 wheelhouse, fp8 KV, prefix cache, 32 seqs, MTP 2 (fallback:
  the proven 08-17 no-MTP recipe). Headline metric = valid `action(` tool calls per agent-hour; see `serving/README.md`.
- **Expected**: (A) ≥ 2× decisions/game-hour vs (0); (B) ≥ 3× vs (0).
- **Falsifier**: neither beats (0) by ≥ 1.5× → the bottleneck is not KV; revisit (prefill cost of images / prompt size).
- **Metrics**: aggregate generated tok/s, median/p90 turn latency at 28 agents, prefix-cache hit %, preemptions,
  tool-call parse validity %.

## EXP-002 · planned 10-03 · Hydra-0 (combined cheap wins)
- **Change**: stock 5.19 notebook (Scott / ours) + MTP off + KV 7–12 GiB + 20–28 seqs + wave-fit per-game budget +
  ACTION7=UNDO executable + scoring/pacing prompt lines + analyzer_timeout 1200 + effective ctx fix.
- **Expected**: LB ≥ 5 (base mean 3.29, sd 0.51; best public 5.19).
- **Falsifier**: LB < 4 on two submissions → one of the combined changes hurts; bisect via CPU/mock + one GPU smoke.

## EXP-000 · datapoint · base notebook (user's 09-27 submission)
- **What**: `taaf-flashnext-sheetu12b-0922` v1 — code-identical to scottlegrand's 5.19 public notebook and to the
  Hydra-0 base (`automation/hydra0/base_*.ipynb`).
- **Result (09-28)**: **public 3.18**. Together with the 12-submission cluster of the stock base (mean 3.29, sd 0.51) and
  Scott's 5.19, the base family sits at ~3–5 with large per-submission noise. This is the baseline for EXP-002/004/005.

## EXP-004 · planned 10-04 · Hydra-1a (our Duck bundle: S1 + v9 prompt)
- **Change vs Hydra-0**: attach `kragglenote2forwork/arc3-duck-bundle` (vendored Duck, byte-identical serving_setup /
  identity) instead of keithtyser's bundle; enable S1 (per-game state/log isolation) + v9 prompt P1 (true scoring rule +
  per-turn game clock), P2 (UNDO described + executable), P3 (reset discipline, "verify why you won", no video-game
  mapping, visible world-model notes). AGENTFIX still supplies F1/F3/F6/F9/F13/F19/ARM P; Hydra-0 scoring cell off.
  CPU mock: median prompt 20.0k tokens/request (+1.5 % vs Hydra-0's 19.7k).
- **Expected**: ≥ Hydra-0 (+0.5 to +1.5 LB): better goal commitment/verification, fewer wasted RESETs.
- **Falsifier**: ≥ 1 LB below Hydra-0 on the same serving profile → a v9 line hurts; ablate P3 first (largest text).

## EXP-005 · planned 10-05 · Hydra-1b (Hydra-1a + M2 verified-facts memory)
- **Change vs 1a**: `DUCK_PATCH_M2_FACTS=1`: verified-facts block (harness-verified action effects, levels vs baseline,
  model notes; ≤ 2,100 chars) replaces blind oldest-message eviction; trimming budget 12k.
  CPU mock with AGENTFIX layered: median prompt **9.5k vs 20.0k tokens/request (−52 %)**.
- **Expected**: prefill was ~92 % of processed tokens on 09-22 → ≈ 1.5–2× more LLM turns per game-hour → +1 to +3 LB
  over 1a, unless the shorter context loses information the model needs.
- **Falsifier**: LB ≤ 1a on two submissions, or the commit-run log shows no drop in prompt tokens/request → raise
  `DUCK_PATCH_M2_BUDGET` to 14000–16000 (M2 keeps working, more verbatim history) before abandoning M2.
