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
