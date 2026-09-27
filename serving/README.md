# Serving kit: profiles, launcher and the Duck replay benchmark (B02 + B03, EXP-001)

The question for GPU day 1 (Sat 10-03) is: **which serving profile gives each game the most LLM decisions per hour** on
1× RTX PRO 6000 (96 GB, SM120) inside an offline Kaggle notebook? The 09-22 run spent 87 % of every request's life in
vLLM's queue (analysis/01), so this is the highest-leverage lever (MASTER_PLAN §3 M1).

| file | what |
|---|---|
| `automation/serving_bench.py` | replay-pack builder, async multi-agent replay client, metrics scraper, mock server, report/compare |
| `automation/serving_pack/duck_0922.jsonl.gz` | replay pack from the 09-22 run: 25 games, 1,026 turns, 1,045 requests (2.2 MB) |
| `serving/profiles/*.json` | serving profiles as data (argv template, env, runtime kind, client settings, memory budget, evidence, fallback) |
| `serving/launch.py` | render/start/wait/stop a profile; `--suite` = one-command A/B |
| `automation/exp001_notebook/` | Kaggle notebook dir (`kragglenote2forwork/arc3-exp001-serving-ab`, private, no internet) + `build_kit.py` |
| `tests/test_serving_bench.py` | CPU tests (pack parsing, Duck context logic, mock end-to-end, profile invariants, suite + fallback) |

## Quick start

```bash
# CPU (any machine): tests and a mock end-to-end run
python -m pytest tests/test_serving_bench.py -q
python automation/serving_bench.py run --mock --agents 28 --duration 20 --warmup 3 --out /tmp/mock
python serving/launch.py --suite mock_cpu --duration 20 --warmup 3 --agents 8 --out /tmp/suite --baseline mock_cpu

# Inspect a profile (no side effects)
python serving/launch.py --list
python serving/launch.py --profile flashnext_tuned --print-argv

# GPU (Kaggle notebook or any box with the datasets mounted): the whole EXP-001 in one command
python serving/launch.py --suite exp001 --duration 720 --warmup 90 --agents 28 --budget-min 100 --out /kaggle/working/exp001

# Against an already running server
python automation/serving_bench.py run --base-url http://127.0.0.1:1234/v1 --label my_server --context-window 16384 --out /tmp/r
python automation/serving_bench.py compare /tmp/exp001/*/report.json --baseline flashnext_baseline

# Rebuild the pack from a results dir (transcripts/, prompts/, artifacts/, vllm-metrics-final.prom)
python automation/serving_bench.py build-pack --results $SCRATCH/results --out $SCRATCH/serving_pack/duck_0922.jsonl.gz \
       --copy-to automation/serving_pack/duck_0922.jsonl.gz
```

### Running EXP-001 on Kaggle

1. Pass the run gate first (CLAUDE.md): thesis in `plans/EXPERIMENTS.md` (already there), CPU tests green, quota check,
   and log about **1.7 GPU-h** in `plans/GPU_LEDGER.md`. The suite budget is 100 min, with about 73 min for the three
   main profiles; the extras only run if time remains.
2. `python automation/exp001_notebook/build_kit.py --out /tmp/arc3-serving-kit`, then upload it once as the **private**
   dataset `kragglenote2forwork/arc3-serving-kit` (`kaggle datasets create -p /tmp/arc3-serving-kit --dir-mode zip`).
3. Push `automation/exp001_notebook/` (the title `arc3-exp001-serving-ab` equals the slug, see MISTAKES #1). Nothing else
   runs on the GPU that day (MISTAKES #5).
4. Read `/kaggle/working/exp001/exp001_summary.md` (see "How to read the results").

Suite order (`--suite exp001`). The most likely winner goes first, so a short session still answers the question:

| # | profile | why | fallback on failure |
|---|---|---|---|
| 1 | `flashnext_tuned` | arm A: Flash-Next, MTP off, 12 GiB bf16 KV, prefix cache (align), 16k ctx, 28 seqs | `flashnext_nomtp_kv7_seqs20` (proven by Thuitanium) |
| 2 | `qwen27b_fp8` | arm B: 27B-FP8, vLLM 0.19, fp8 KV, prefix cache, 32 seqs, MTP 2 | `qwen27b_fp8_nomtp` (proven 08-17 recipe) |
| 3 | `flashnext_baseline` | the exact 09-22 server; reference for the ratios | – |
| 4+ | `flashnext_nomtp_kv16_seqs28_pc`, `flashnext_tuned_mtp2`, `flashnext_nomtp_kv12_seqs28_pc_ctx32k` | extras, run only if the budget allows | kv16 → kv12 |

Each profile runs these steps: runtime prep → start → wait for `/v1/models` → publish `harness_env.json` → 90 s
warm-up (excluded) → 12 min measured replay → teardown. Teardown kills the process group and its descendants, waits for
the port to close and GPU memory to free, and never raises. A profile that fails to start, or whose bench has more than
50 % errors or kills the server, triggers its fallback.

## Profiles

| profile | runtime | weights GiB | KV | seqs | ctx | prefix cache | spec | headroom GiB |
|---|---|---|---|---|---|---|---|---|
| `flashnext_baseline` | flashnext_dev | 81.8 | 5 GiB bf16 (105k tok) | 16 | 32k | off | MTP3 | 7.6 |
| `flashnext_tuned` (= `flashnext_nomtp_kv12_seqs28_pc`) | flashnext_dev | 74.3 | 12 GiB bf16 (~450k tok) | 28 | 16k | align | – | 8.1 |
| `flashnext_nomtp_kv{7,12,16}_seqs{20,28}[_pc]` | flashnext_dev | 74.3 | 7/12/16 GiB bf16 | 20/28 | 16k | off / align | – | 13.1 / 8.1 / 4.1 |
| `flashnext_nomtp_kv12_seqs28_pc_ctx32k` | flashnext_dev | 74.3 | 12 GiB bf16 | 28 | 32k | align | – | 8.1 |
| `flashnext_tuned_mtp2`, `_mtp3` | flashnext_dev | 81.8 | 5 GiB bf16 | 12 | 16k | align | MTP2/3 | 7.6 |
| `flashnext_tuned_ngram` | flashnext_dev | 74.3 | 12 GiB bf16 | 28 | 16k | align | ngram_gpu 3 | 8.1 |
| `qwen27b_fp8` | vllm019_wheelhouse | 28.5 | util 0.93 → ~52 GiB fp8 | 32 | 32k | align | MTP2 | – |
| `qwen27b_fp8_nomtp` | vllm019_wheelhouse | 28.5 | util 0.90 → 48.8 GiB fp8 (398k tok) | default | 32k | align | – | – |
| `mock_cpu` | mock | – | – | – | 32k | – | – | – |

Notes on the table:
- Headroom = 94.43 GiB free before load − weights − KV. The baseline's 7.6 GiB was already tight: FlashInfer MoE
  autotune hit a transient OOM and recovered. For that reason the kv16 profiles cut `--max-num-batched-tokens` to 4096.
- Weights are 81.8 GiB with MTP and 74.3 GiB without (research/10 §5.3, Thuitanium). KV of 11 GiB or more with MTP on
  is not possible: 16 GiB + MTP failed setup (amanatar v40).
- Each profile's `client` block is what the bench (and later the harness) uses: context window, safety margin 6144,
  and `max_tokens` = server default, the same as Duck.

### The context window is set in the place that wins (the "ctx16k" bug)

On 09-22 the notebook set `LOCAL_ANALYZER_CONTEXT_WINDOW=16384`. Then keithtyser's `serving_setup.py` persisted `32768`
into `taaf_setup_env.json`, and the notebook re-applied that file after setup, so 16k never took effect.

Now each profile carries a `harness_env` block, and `launch.py` handles it in four ways:
- It never calls `persist_analyzer_environment`. It borrows only the runtime-extraction, PLE-patch and env functions,
  and restores the caller's `TAAF_KAGGLE_*` variables afterwards.
- After the server is ready, it writes `harness_env` last: into the caller's `$TAAF_KAGGLE_SETUP_ENV` (if one was
  defined before launch), into `os.environ`, and into `<work>/harness_env.json`.
- The server's `--max-model-len` equals `client.context_window` equals `LOCAL_ANALYZER_CONTEXT_WINDOW`. A test enforces
  this.
- A Hydra notebook must apply `harness_env.json` **after** its setup loop.

## Runtime support: what was checked and how

**Flash-Next runtime** (`keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`). This is the docker image
`vllm/vllm-openai:qwen38-flash-next`, vLLM **0.1.dev20073+g8e685d198**, torch 2.13 cu130, FlashInfer 0.6.17. The dataset
holds 6 layer tarballs (7.9 GB). I read `runtime-manifest.json` and the vLLM install layer (layer-24, 310 MB) under
`$SCRATCH/serving_research/runtime/`.

| capability | status | evidence |
|---|---|---|
| `Qwen3_5ForConditionalGeneration` (27B) | registered, and `Qwen3_5MTP` too | `vllm/model_executor/models/registry.py` |
| fp8 KV cache for Flash-Next | **no**: hard `NotImplementedError("Qwen3.8-Flash-Next QSA requires a BF16 main KV cache")` | `vllm/models/qwen3_8_flash_next/nvidia/qsa.py` (supported dtypes: auto, bfloat16) |
| prefix caching for hybrid (mamba-state) Flash-Next | **yes, `--mamba-cache-mode align` only** (`all` raises); upstream labels it experimental; needs chunked prefill; block size 1600 ≤ batched tokens | `nvidia/model.py`, `nvidia/mtp.py`, `config/cache.py`, `model_executor/models/config.py` |
| speculative methods for Flash-Next | `mtp`, `ngram`, `ngram_gpu` only; `--async-scheduling` rejects plain `ngram` | `model_executor/models/config.py`, `config/vllm.py` |
| MTP for 27B | code path present (`qwen3_5` → `qwen3_5_mtp`), and `dflash` exists in this build | `config/speculative.py` |
| extras | `--prefix-match-unit` (finer prefix hits), ReplaySSM, `--kv-cache-dtype-skip-layers` | `config/cache.py` |

**27B runtime** (`driessmit1/arc3-vllm-h100-wheelhouse-v3`: vLLM **0.19.0**, torch 2.10 cu128, FlashInfer 0.6.6,
transformers 4.57.6, which is overridden to 5.8.0 by `keithtyser/taaf-duck-qwen38-serving-v1/serving_wheels`).
- It is **proven on this GPU**: keithtyser's 08-17 run served Qwen3.8-27B-FP8 with fp8 KV and prefix caching (align,
  set automatically).
- Results of that run: 28.51 GiB weights, 48.8 GiB KV = 398,272 tokens, 25 running / 0 waiting all run, prefix hit
  about 42 %, 439 tok/s mean generation (up to 766), no preemptions.
- `Qwen3_5MTP` and the `mtp` alias are in the v0.19.0 source, with no assertion against MTP combined with mamba `align`.
  MTP on this GPU is **untested**, hence the fallback.
- DFlash is not in 0.19.

**27B weights on Kaggle** (all revision 017b9c7a, about 30.89 GB = 28.8 GiB; `mtp.*` tensors included):

| copy | notes |
|---|---|
| `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1` | used here and in keithtyser's 08-17 run: 16 shards of 1.52 GB, plus `outside` (6.0 GB) and `mtp` (0.48 GB) |
| `michaelpoluektov/qwen3-8-27b-fp8` | 30,890,049,597 B |
| `mikedan7/qwen3-8-27b-fp8-official` | 30,890,068,612 B |
| `dangkhoa2016/qwen-qwen3-8-27b-fp8` | 30,890,049,471 B |
| `justnoone2026/qwen3-8-27b-fp8` v2 | 30,890,083,182 B (merged shards) |

Related copies:
- DFlash2 drafter: `dangkhoa2016/z-lab-qwen3-8-27b-dflash2`, 3.85 GB. It needs the Flash-Next dev runtime, since vLLM
  0.19 has no dflash.
- NVFP4 27B builds: `michaelpoluektov/qwen3-8-27b-nvfp4`, `impactganyu/qwen38-27b-*`.

**Pennyroyal (SGLang fork, `jpezzulli/sglang-rtxpro6000`)**:
- License: Apache-2.0.
- Why it matters: it serves Flash-Next with **FP8 KV** (824k-token pool) and radix + Mamba prefix caching. That is
  exactly what vLLM's QSA path cannot do.
- It is not usable offline today. `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1/src/sglang-rtxpro6000` holds only the
  Python source (68 MB, no `sgl-kernel`/`.so`, no wheels).
- The route would be to package the prebuilt image `ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.2` as layer tarballs in a
  private dataset and extract it to /tmp, the same way keithtyser packaged the vLLM image. That needs a download and
  upload of several GB plus one validation run, so it is a W2 candidate, not a day-1 option.

## The benchmark

### Replay pack (`build-pack`)
The pack is built from these inputs:
- `transcripts/*.txt`: every request's reasoning, content, raw tool calls and tool results, plus the user prompt and
  analyzer status of every turn.
- `prompts/*.log`: the python tool description.
- `artifacts/*_events.jsonl`: the board each analysis turn saw. It is rendered as a 4× PNG (256×256), exactly like
  `MULTIMODAL_CONTEXT=current_grid`, and de-duplicated per game.
- `vllm-metrics-final.prom`: the recorded reference numbers.

What the pack contains:
- The system prompt is identical across the 25 games, so it is stored once and all games share that prefix.
- Recorded run: 11.6 decisions/game-hour, 20.9 requests/game-hour, mean prompt 19,582 tokens, mean generation 1,721
  tokens, 3.36 images per request, queue share 87 %.

### Agent model (mirrors ARC3-Inference `tool_agent.py` as shipped on 09-22)
- There are K agents (default 28). Games 1-25 are distinct. Lanes 26-28 replay games 1-3 from mid-game with a lane tag,
  so they never share prefixes with their twin.
- A turn starts with the recorded user prompt and the board image. The agent then calls the model until either:
  - the model's own `python` tool call contains `action(` (Duck: `step_executed`), or
  - 60 s have passed (Duck: `turn_time_budget` yield).

  A reply without a tool call gets Duck's follow-up prompt.
- **Context growth** (default `--history actual`): the model's actual reply (`reasoning`, `content`, `tool_calls`) is
  appended, and the recorded tool results fill the tool messages.
- **Trimming follows Duck**:
  - estimator: `(json chars + 2) // 3`, plus 120 tokens per image (AGENTFIX F1);
  - budget: `ctx − 512 − 6144`;
  - the oldest history blocks are dropped first;
  - at most 30 assistant turns are kept;
  - images stay on the newest 2 history user turns plus the current one.
- `--history recorded` appends the recorded assistant turns instead. Every server then sees the same prompt stream, and
  the number of requests per turn follows the recording.
- **Request**: T 0.6, top_p 0.95, top_k 20, `enable_thinking`, tools=[python], tool_choice auto, non-streaming.
  `max_tokens` is the server default, like Duck's `LOCAL_ANALYZER_MAX_OUTPUT=0`.
- **Transport**: aiohttp (`trust_env` except for loopback), else httpx, else stdlib urllib in threads.
- **Metrics**: `/metrics` is sampled every 5 s (running, waiting, KV usage), plus counter snapshots at the window
  start and end. The snapshots cover tokens, preemptions, prefix and MM cache, spec decode, and the queue, e2e, prefill
  and decode histograms.

### Output per profile (`<out>/<profile>/`)
`report.json` (summary + config + profile + metric snapshots), `summary.md`, `requests.jsonl.gz` (one row per request:
latency, category, tokens), `turns.jsonl.gz`, `metrics_samples.jsonl`, `metrics_final.prom`, `server.log`,
`server_identity.json` (argv, env changes without secrets, GPU memory), `harness_env.json`, `profile_result.json`.
The suite writes `exp001_summary.md|json` and `suite.json` (status and timing of every profile, including fallbacks
and skips).

## How to read the results

1. **Headline: `decisions/game-h`** = completed requests whose reply is a valid `python` call containing `action(`,
   per agent per hour of the measured window. The `× baseline` column divides by the replayed `flashnext_baseline`.
   - EXP-001 thesis: A ≥ 2×, B ≥ 3×.
   - Falsifier: no profile reaches 1.5×. `compare` prints the falsifier line automatically; if it fires, the
     bottleneck is prefill and prompt size, not KV.
2. **Validate the replay.** The replayed baseline should land near the recorded 09-22 run: 11.6 decisions/game-h,
   20.9 requests/game-h, queue share about 87 %, and many waiting requests. These numbers are in every `summary.md`.
   The replay has 28 agents against 25 games, so expect it to be a bit lower. A large mismatch means the replay is not
   representative; trust ratios more than absolute numbers.
3. **Is it still queue-bound?** Look at `waiting` (mean), `queue mean s` / `queue_share_of_e2e`, and `preempt`.
   - The goal: waiting ≈ 0, running ≈ number of agents, 0 preemptions.
   - Once the queue is gone, per-sequence decode speed (`gen tok/s` ÷ running) sets latency, and speculative decoding
     (`spec acc`) matters.
4. **Prefix cache.** `prefix hit` is the fraction of prompt tokens served from cache.
   - Expected: 0 with the cache off; about 0.3-0.7 with align mode, which only reuses at block boundaries (1,600 tokens
     for Flash-Next, 1,568 for 27B).
   - A hit rate near 0 with the cache on means eviction or history trimming kills reuse. Try a larger KV, fewer seqs
     or `--prefix-match-unit`.
5. **Quality guards.** `tool valid %` should be ≥ 95 (09-22: 99.1 %) and `err %` about 0.
   - A profile below 90 % validity is flagged: a parser or template problem would cost real games.
   - `action %` shows how often the model acts rather than probes. The 27B may differ from Flash-Next, and that is a
     behaviour difference, not a serving one.
6. **Confounders.**
   - The 16k profiles send shorter prompts (`prompt p50`); `flashnext_nomtp_kv12_seqs28_pc_ctx32k` isolates that
     effect.
   - `gen p50` differs by model: the 27B thinks longer or shorter per call.
   - Decisions/game-hour measures **throughput at equal harness**, not solve quality. Levels cleared per hour need a
     public-25 run (EXP-002+).
7. **Decision rule.**
   - Pick the profile with the most decisions/game-h among those with validity ≥ 95 % and errors < 5 %.
   - If the 27B wins by less than 1.5× over tuned Flash-Next, prefer Flash-Next: it is the stronger model per decision
     (09-22 local 10.02 vs 27B 3.88 on 08-17).
   - Record the result in `plans/EXPERIMENTS.md` and keep the loser as a fallback profile.

## Known risks / open items
- **Prefix caching is untested** on this exact Flash-Next build (the PLE/QSA path) and for 27B with MTP. Both have
  fallbacks.
- **kv16** leaves 4.1 GiB headroom; it is an extra, not a main arm.
- **Startup time**: Flash-Next takes about 10 min the first time (136 s runtime extraction plus 475 s to ready on
  09-22); the 27B takes about 6-8 min (pip install plus about 4 min load).
- **Fallbacks cost budget**: if a startup fails, the fallback eats that profile's slot and the extras get skipped.
- **Not measured**: CPU-side tool execution time and env stepping between turns (sub-second in the recorded run).
