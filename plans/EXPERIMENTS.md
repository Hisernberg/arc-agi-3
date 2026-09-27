# Experiments (write the thesis BEFORE the run)

Format: ID · date · change · expected effect · falsifier · result · decision

## EXP-001 · planned 10-03 · Serving A/B
- **Change**: replay recorded Duck prompts (results_4 `prompts/*.log`) as 28 concurrent multi-turn agents against
  (A) Flash-Next NVFP4 tuned vLLM (KV ≈ 11 GiB, fp8 KV, prefix cache on, 16 k ctx, seqs matched), (B) Qwen3.8-27B-FP8
  (≈ 55 GiB KV, prefix cache on, 28–32 seqs), baseline (0) = current profile.
- **Expected**: (A) ≥ 2× decisions/game-hour vs (0); (B) ≥ 3× vs (0).
- **Falsifier**: neither beats (0) by ≥ 1.5× → the bottleneck is not KV; revisit (prefill cost of images / prompt size).
- **Metrics**: aggregate generated tok/s, median/p90 turn latency at 28 agents, prefix-cache hit %, preemptions,
  tool-call parse validity %.
