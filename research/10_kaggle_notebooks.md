# 10 — Public Kaggle notebooks for `arc-prize-2026-arc-agi-3` (survey of 2026-09-27)

`SCRATCH` = `/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad`.
Machine-readable index: `SCRATCH/kaggle/notebooks/index.json` (and `index.csv` for all 1,132 notebooks).

## TL;DR

1. **No public notebook is close to the top of the leaderboard.** The best public notebook scores **5.19**. Only **7 of the 456 scored notebooks reach 4 or more**, and every one of them runs the same stack: the Tufa Labs **Duck harness** with Keith Tyser's **Qwen3.8-Flash-Next-NVFP4 + MTP-3 vLLM** serving.
   - None of the top-10 teams (27.3 to 9.96) has published a current notebook. I checked every member of the top-80 teams.
   - Their only public ARC-AGI-3 code is from the June milestone and scored between 0.1 and 1.3 (§7).
2. **The public top-10 is mostly noise around one codebase.**
   - The byte-identical keithtyser base was submitted independently **12 times**: mean **3.29**, sd **0.51**, range **2.38–4.33**.
   - The #3 notebook (4.33, "Duck Qwen3.8 Anim Base") is that exact code. It is also the best of 13 daily resubmissions.
   - #2 (4.50) differs from the base only by `analyzer_timeout = 1200`. The same code also scored 4.08 and 2.81.
   - #5 (4.09) and #7 (4.08) only add dead or diagnostic code, or turn prefix caching on.
3. **The one variant that seems genuinely better is the #1 notebook, `scottlegrand/taaf-flashnext-sheetu12b-0922` (5.19).** Its 5.19 is about 3.7 sd above the base mean.
   - It is **code-identical to the user's current notebook** (`SCRATCH/nb/user_nb_code.py`; only the outputs differ).
   - It layers runtime patches on the stock bundle:
     - "AGENTFIX": trims old history images, tolerant world-model parsing, and does not wipe memory on game over;
     - **animation frames exposed in the Python sandbox (F13)**;
     - **an all-frames contact-sheet image (F19)**;
     - the **board image upscaled 4→12** (768 px);
     - 16 vLLM sequences with a 16k analyzer context.
   - Its own comments claim that **F13 is "the only change that has ever moved the hidden leaderboard (3.20 → 3.71)"**, and that a *text* narration of the same frames **dropped it to 2.57** ("frames yes, narration no").
4. **The model and serving choice decide most of the score. Prompt and harness tweaks are all inside the noise.** Median scores by family:

   | Family | Median | Best |
   |---|---|---|
   | Qwen3.6-27B-FP8 Duck (Tufa June) | 0.92 | 1.61 |
   | Qwen3.8-27B-FP8 Duck | 1.56 | 3.11 |
   | Qwen3.8-Flash-Next-NVFP4 Duck | 3.21 | 5.19 |
   | Gemma-4-31B | 0.42 | 0.86 |
   | Non-LLM (BFS, random, heuristics) | 0.16 | 0.54 |

5. **A local public-25 score does not predict the hidden leaderboard.** Across 40 notebooks with a full 25-game commit run, the correlation between local mean and public LB is **r = 0.11**. The base code alone ranges 5.8–8.7 locally.
   - **Time allocation matters.** The hidden run has about **110 games** (55 public-LB and 55 private, all played in one 9-hour run). Duck plays 28 concurrently with 7,920 s per game, so 4 waves need about 31.7k s but only about 28k s is available. The last wave is cut roughly in half.
   - One fork set `max_runtime_s_per_game = 27000`. It scored **13.3 locally but only 2.58 on the LB**, because it starved about 80 hidden games.
6. **Techniques worth keeping** (details in §9):
   - the Flash-Next NVFP4 + MTP serving stack;
   - animation-frame access (F13 and F19);
   - a larger board image;
   - HUD / no-op awareness;
   - world-model persistence across game overs;
   - a per-game budget fitted to 4 waves;
   - running many seeds, because the leaderboard is noisy at about ±0.5.

   The notebooks give **no evidence** that any of these gets to 18–20. The top teams are doing something different that is not public: for example, Tong Hui Kang's imitation-trained per-cell policy, NVIDIA's executable world model, or the PLDM/JEPA world-model search seen in UNK's June notebooks (see `13_hf_github_assets.md`).

---

## 1. Method (how scores were obtained)

- **Listing.** `kaggle kernels list --competition arc-prize-2026-arc-agi-3` returns at most 1,100 rows per sort order (11 pages × 100). The union over 7 sort orders is **1,132 notebooks**. The CLI/public API does **not** expose scores.
- **Scores.** Kaggle's internal endpoint `POST https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels`, called with the KAGGLE_API_TOKEN as a Bearer token, returns `bestPublicScore`, `scriptVersionId`, `dataSources`, `medal` and other fields for each notebook.
  - **456 notebooks have a public score.**
  - `bestPublicScore` is the **best score over all submissions of that notebook**, so notebooks that were resubmitted daily are biased upward. For example, wuliao0 has 13 versions, keithtyser 14 and muhibullahansir 34.
- **Meta Kaggle** `Submissions.csv` (2.6 GB) contains **no rows for this competition** because it is still running. `KernelVersionCompetitionSources.csv` lists 13,104 kernel versions for competition 133468; those IDs are kept in `SCRATCH/kaggle/metakaggle/kernel_versions_133468.txt`. Both large CSVs were deleted after checking to save disk.
- **Exact scored code.** The listed `scriptVersionId` is the version that is shown or scored. It can differ from the latest version: amanatar v38 vs v40, cassowaryloraforge v1 vs v17, woguoat v45 vs v50, and cotrd-enhanced v16 vs v20.
  - I downloaded the **scored version of every one of the 456 scored notebooks** from `https://www.kaggle.com/kernels/scriptcontent/<scriptVersionId>/download`. This URL is not rate-limited. Version-specific `kaggle kernels pull owner/slug/N` returns 403.
  - The files are in `SCRATCH/kaggle/notebooks/_scored_sources/`.
  - Each one was normalised (comments and blank lines stripped), hashed and line-diffed against the keithtyser base: `scored_clusters.json`, plus `_diffs/<score>__<ref>.diff` for every distinct variant scoring ≥ 2.0.
- **Outputs.** For 41 target notebooks I ran `kaggle kernels pull -m` and `kaggle kernels output`, skipping the per-game `*_events.jsonl` / `*_viewer_data.json` / `solver_analysis/*.html` bulk.
  - Commit ("Save & Run") runs of Duck notebooks play the **25 public games offline at the competition clock**. Their `score.json` and `benchmark.json` hold per-game local scores, so these are local-eval data, not LB data.
  - I also fetched `score.json` alone for about 50 further Duck-family notebooks: `_scoreprobe/` and `probe_local_scores.json`.
- **Team mapping.** The internal `competitions.LeaderboardService/GetLeaderboard` endpoint lists team members.
  - Top-10 usernames: Tufa Labs = driessmit1, jeroencottaar, dlorah, pressman1, stefano1283, infinitecreativity. Lord Han Solo = lordhansolo. Tong Hui Kang = huikang. Yi-Chia Chen = threerabbits. Daniel Franzen = dfranzen. NVARC3 = cpmpml, darraghdog, sorokin, eladsarafian, galkaplun, yeyinzhu. the last dance = gklambauer, fses91, lukasaichberger, … Third Intelligence = seele1917, joshkun, yukiokumura1, takahashinaoki. Matija/Zhongwei/Fususu = matijaludvig, phuongncn, jonathanwang2022, … rellik13 = sirikilohit.
  - The only ARC-AGI-3 notebooks from top-10 members are Tufa's June releases (jeroencottaar) and one NVARC member's unrelated `gpt-oss-math-search`.

## 2. Landscape

| Family (by attached model and serving) | # scored | Best | Median | Top-5 |
|---|---|---|---|---|
| **A: Qwen3.8-Flash-Next-NVFP4 (keithtyser runtime) + Duck** | 53 | **5.19** | **3.21** | 5.19, 4.50, 4.33, 4.16, 4.09 |
| B: Qwen3.8-27B-FP8 + Duck | 52 | 3.11 | 1.56 | 3.11, 2.98, 2.37, 2.31, 2.23 |
| C: Qwen3.6-27B-FP8 + Duck (Tufa June 30 stack) | 95 | 1.61 | 0.92 | 1.61, 1.47, 1.46, 1.42, 1.39 |
| D: Gemma-4-31B-it (VLM, JSON actions) | 9 | 0.86 | 0.42 | |
| F: other Qwen (2.5, 3-4B, 3.6-35B, …) | 5 | 0.39 | 0.08 | |
| G: no model (random, BFS, graph explorer, heuristics) | 205 | 0.54 | 0.16 | |
| H: other (JEPA, PLDM, LoRA, …) | 37 | 0.79 | 0.11 | |

Nearly every notebook scoring ≥ 2 is a Duck fork. The model progression Qwen3.6-27B → Qwen3.8-27B → **Qwen3.8-Flash-Next** (about 6.5B-active MoE, NVFP4, native MTP head) roughly **tripled** the score. At fixed GPU time, throughput and model quality dominate everything else.

## 3. Ranked table (public score ≥ 3.4, plus lineage and different-approach notebooks)

"Local" is the notebook's own commit run on the 25 public games, taken from the latest output. It is one noisy sample; "–" means no usable output. "≡ base" means the scored code is byte-identical to the keithtyser base after normalisation.

| # | Notebook (scored version) | Author | Public | Date | Approach / diff vs base | Model / serving | Local 25 |
|---|---|---|---|---|---|---|---|
| 1 | [scottlegrand/taaf-flashnext-sheetu12b-0922](https://www.kaggle.com/code/scottlegrand/taaf-flashnext-sheetu12b-0922) v1 | Scott Le Grand | **5.19** | 09-22 | Duck + **AGENTFIX** runtime patches: F1 image-history trim, F3 tolerant memory with no wipe on game over, F9 timing, **F13 anim frames in sandbox**, **F19 frame contact-sheet image**. **ARM P board image ×12**. c16 / ctx16k. **= user's current notebook.** | Flash-Next NVFP4, MTP3, KV 5 GiB, `max_num_seqs=16`, analyzer ctx 16384 | 10.02 (2 h cap) |
| 2 | [chiakazirim/duck-qwen3-8-tuned](https://www.kaggle.com/code/chiakazirim/duck-qwen3-8-tuned) v1 | Akagha Chimgozirim | 4.50 | 09-04 | base + `analyzer_timeout = 1200` (same code: 4.08 and 2.81 elsewhere) | Flash-Next, c8 | 5.16 |
| 3 | [wuliao0/duck-qwen3-8-anim-base](https://www.kaggle.com/code/wuliao0/duck-qwen3-8-anim-base) v13 | wuliao_0 | 4.33 | 09-18 | **≡ base** (13 daily versions; best-of) | Flash-Next, c8 | 5.78 |
| 4 | [muhibullahansir/duck-qwen3-8-anim-base](https://www.kaggle.com/code/muhibullahansir/duck-qwen3-8-anim-base) v34 | Muhibullah Ansir | 4.16 | 09-26 | base + **ACTION7→"UNDO" mapping** + 7,600 s per game + private `/tmp` job dir + audit logger. MTP A/B arm switch. | Flash-Next, c8 | 8.28 |
| 5 | [ghazarosghazaros/duck-qwen3-8-anim-base](https://www.kaggle.com/code/ghazarosghazaros/duck-qwen3-8-anim-base) v5 | Ghazaros | 4.09 | 09-09 | base + inert scaffolding: model catalog, GEPA/few-shot hooks with no artifacts attached | Flash-Next, c8 | 8.74 |
| 6 | [amanatar/arc-agi-3-hybrid-repl-agent](https://www.kaggle.com/code/amanatar/arc-agi-3-hybrid-repl-agent) **v38** | Aman Atar | 4.08 | 09-12 | base + **prefix caching ON**. `n_passes=3` is overridden back to 1. v40 (KV 16 GB) failed vLLM setup. | Flash-Next, c8, prefix cache | – |
| 7 | [matthewblakeward/cassowaryloraforge](https://www.kaggle.com/code/matthewblakeward/cassowaryloraforge) **v1** | M. B. Ward | 4.08 | 09-23 | ≡ #2 "Tuned". The latest v17 is an unrelated Qwen3.8-27B LoRA experiment. | Flash-Next | – |
| 8 | [juliancamilovilla/arc-agi3-nvfp4](https://www.kaggle.com/code/juliancamilovilla/arc-agi3-nvfp4) | J. C. Villa | 3.98 | 09-10 | ≡ base (plus an offline-only 25-min cut) | Flash-Next | 1.28 (25 min) |
| 9 | [tantan0327/arc3-flashnext-asis](https://www.kaggle.com/code/tantan0327/arc3-flashnext-asis) | red-down | 3.92 | 09-09 | ≡ base | Flash-Next | 7.20 |
| 10 | [yocybercode/thui-animfast-b71-full25-r1](https://www.kaggle.com/code/yocybercode/thui-animfast-b71-full25-r1) | Thuitanium (LB 5.36) | 3.74 | 09-11 | **jakobbrggen anim bundle** (animation summaries, `animation()` tool, hard no-op guard) + seed 20260825 + yield 180 s. Same code: 3.49. | Flash-Next | 9.56 |
| 11 | [yanggod/arc-agi-3-duck-flash-next-mtp](https://www.kaggle.com/code/yanggod/arc-agi-3-duck-flash-next-mtp) | Boopathi Raja | 3.71 | 09-05 | ≡ base | Flash-Next | 8.56 |
| 12 | [woguoat/arc3-keith-repro-qwen38next](https://www.kaggle.com/code/woguoat/arc3-keith-repro-qwen38next) **v45** | woguoat | 3.70 | 09-26 | base + 2 user-prompt lines: explicit quadratic RHAE formula, and "use `time_remaining_seconds`; change hypothesis class after failures" | Flash-Next | – |
| 13 | [yanggod/arc-agi-3-duck-flash-next-wavefit-b](https://www.kaggle.com/code/yanggod/arc-agi-3-duck-flash-next-wavefit-b) | Boopathi Raja | 3.64 | 09-12 | base + **wave-fit per-game budget**: `(window / ceil(N/28)) × 0.97`, applied only in the rerun | Flash-Next | 7.12 |
| 14 | [samrishb/just-seeing-what-can-i-change-to-make-it-better](https://www.kaggle.com/code/samrishb/just-seeing-what-can-i-change-to-make-it-better) | Samrish B | 3.62 | 09-02 | base + read-only preflight/metrics (gameplay ≡ base; cluster of 4: 3.62/3.48/2.76/2.71) | Flash-Next | 6.76 |
| 15 | [yocybercode/thui-l1-v0-full25-r1](https://www.kaggle.com/code/yocybercode/thui-l1-v0-full25-r1) | Thuitanium | 3.60 | 09-12 | base + **HUD-band learner**: rows changing on ≥ 90 % of actions become the band; band-only changes give `board_changed=False, no_impact=True` (Son Pham's lever) | Flash-Next | **10.93** |
| 16 | [nihilisticneuralnet/arc-agi-3-retained-reasoning](https://www.kaggle.com/code/nihilisticneuralnet/arc-agi-3-retained-reasoning) v8 | parthenos | 3.57 | 09-11 | **Rewritten harness in a private dataset**: persistent `memory` dict, raw `.grid`, `.components(4/8)`, `diff_frames`, `tried_actions`, `expect` assertions that cancel a batch, audited predictions, `animation_frames` | Flash-Next | 5.83 |
| 17 | [yocybercode/thui-fast-b78-mtp0-full25-r1](https://www.kaggle.com/code/yocybercode/thui-fast-b78-mtp0-full25-r1) | Thuitanium | 3.49 | 09-10 | base with **MTP off** | Flash-Next, no spec-decode | 6.96 |
| 18 | [keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp](https://www.kaggle.com/code/keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp) v14 | ktyser | 3.38 | 09-01 | **THE BASE**: Tufa June-30 Duck source unchanged + new serving | Flash-Next NVFP4, MTP3, c8, KV 5 GiB, ctx 32k | 6.76 |
| – | [ayodejiibrahimlateef/arc3-v124-observed-probe-memory-rtx6000](https://www.kaggle.com/code/ayodejiibrahimlateef/arc3-v124-observed-probe-memory-rtx6000) | Ayodeji | 3.27 | 09-07 | base + bounded 16 KB JSON `memory` + host "probe ledger" (warns after 3 unchanged identical probes; clears plan keys on level change) | Flash-Next | 8.43 |
| – | [iamjasonfeng/cotrd-enhanced](https://www.kaggle.com/code/iamjasonfeng/cotrd-enhanced) **v16** | Jason Feng | 3.16 | 09-19 | "CoTRD": logit-bias contrast (reasoning vs no-reasoning probe); MTP off; per-level-block 2 h budgets | Flash-Next | – |
| – | [anhadmahajan06/arc-agi-3-fluid-intelligence-agent](https://www.kaggle.com/code/anhadmahajan06/arc-agi-3-fluid-intelligence-agent) | Anhad Mahajan | 3.11 | 09-11 | Best of family B: Duck on the keithtyser Qwen3.8-27B-FP8 serving bundle, plus a standalone fallback agent | Qwen3.8-27B-FP8 | – |
| – | [analyticaobscura/i-guess-it-s-just-luck-thanks-foysal](https://www.kaggle.com/code/analyticaobscura/i-guess-it-s-just-luck-thanks-foysal) / [foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b](https://www.kaggle.com/code/foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b) (289 votes) | OzanM. / FOYSAL | 2.98 / 2.23 | 08-19 | Same code, cluster of 6: mean 2.08, sd 0.45. Duck + anim bundle on Qwen3.8-27B-FP8. | Qwen3.8-27B-FP8 | – |
| – | [lucifer19/duck-qwen3-8-efficient-agent-c17](https://www.kaggle.com/code/lucifer19/duck-qwen3-8-efficient-agent-c17) | Agent Trainer | 2.84 | 09-06 | base + an "efficient agent" paragraph appended to the system prompt (c18 = the same plus an audit: 2.67) | Flash-Next | 7.11 |
| – | [juliancamilovilla/arc-agi3-nvfp4-batch-long](https://www.kaggle.com/code/juliancamilovilla/arc-agi3-nvfp4-batch-long) / carry-long | J. C. Villa | 2.95 / 2.66 | 09-16 | base + a user-prompt note: "batch predictable moves" when actions per turn < 3; or a note on "winning transitions" of past levels | Flash-Next | 4.62 / 4.39 |
| – | [defiaudit/agi3-duck-qwen38-anim-v2](https://www.kaggle.com/code/defiaudit/agi3-duck-qwen38-anim-v2) | DefiAudit | 2.58 | 09-21 | Anti-stall bundle + **`max_runtime_s_per_game = 27000`**. **Local 13.31, LB 2.58**: it starves the hidden set. | Flash-Next | 13.31 |
| – | [jakobbrggen/taaf-anim-arc-agi-3-solver](https://www.kaggle.com/code/jakobbrggen/taaf-anim-arc-agi-3-solver) | Jakob Brüggen | 1.61 | 08-07 | Origin of the **animation-awareness + hard no-op guard** bundle (on Qwen3.6-27B) | Qwen3.6-27B-FP8 | – |
| – | [jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner) (338 votes) / [taaf-duck-harness-kaggle](https://www.kaggle.com/code/jeroencottaar/taaf-duck-harness-kaggle) | Jeroen Cottaar (Tufa) | 1.25 / 1.30 | 06-30 | **Origin of everything above**. Readable notebook; solver in the `jeroencottaar/taaf-kaggle-source-share` dataset. Same-code cluster of 9: mean 1.02, sd 0.28. | Qwen3.6-27B-FP8 (vrfai), vLLM 0.19, ctx 32k of 64k, prefix cache | – |

## 4. How big is the noise? (the most important table)

Scored-version code clusters (normalised hash):

| Cluster | n | Mean | sd | Min | Max | What it is |
|---|---|---|---|---|---|---|
| `060b1cdfc4` ≡ keithtyser base | 12 | **3.29** | **0.51** | 2.38 | 4.33 | wuliao0 4.33, tantan 3.92, yanggod 3.71, akhileshgodugu 3.44, dantelok 3.40, keithtyser 3.38, nitish 3.21, samrishb 3.00, kaiwalya 2.97, uninhibited 2.93, kashyap 2.80, zoli800 2.38 |
| `9fb71dfafc` base + timeout 1200 | 3 | 3.80 | 0.72 | 2.81 | 4.50 | chiakazirim, cassowary v1, kunaldesale (yanggod's timeout-1200 arm also scored 2.96) |
| `0a3d0b4a66` base + read-only preflight | 4 | 3.14 | 0.41 | 2.71 | 3.62 | |
| `2bb3d3dfd8` thui-animfast | 2 | 3.62 | 0.12 | 3.49 | 3.74 | |
| `d147a7e9e4` Tufa June readable notebook | 9 | 1.02 | 0.28 | 0.59 | 1.39 | Tufa's own writeup: "best submission got scores as low as 0.77" |
| `dece485ac2` foysal Qwen3.8-27B v12 | 6 | 2.08 | 0.45 | 1.59 | 2.98 | |

Consequences:
- **One submission cannot resolve an effect smaller than about ±1.0** (2 sd) on this stack. The user's 3.85 and scottlegrand's 5.19 may be the same code on different draws. 5.19 is about 3.7 sd above the base mean, so the scottlegrand patches are *probably* worth about +1 to +1.5, but that is not certain.
- **Local public-25 runs are almost uninformative about the LB.**
  - Base code locally: 5.78, 6.42, 6.74, 6.76, 7.10, 7.18, 7.20, 7.24, 7.46, 8.56, 8.61, 8.74. That is sd ≈ 0.9 on a mean of about 7.3.
  - Correlation between local and LB over 40 notebooks: r = 0.11.
  - The public games are also easier: a local mean of about 7 corresponds to an LB of about 3.3.
- Per-game local profile of the Duck family (39 runs × 25 games, `SCRATCH/kaggle/notebooks/probe_local_scores.json`):
  - **Strong:** ft09 38, lp85 23, vc33 18, re86 16, ar25 12.
  - **Mid:** r11l, tu93, tr87, m0r0, ka59, sc25 at 6–7.
  - **Near-zero:** sk48 0.1, bp35 0.5, g50t 0.8, sp80 1.3, lf52 1.8, tn36 1.8.
  - The near-zero set matches AGENTFIX's note that "bp35 / sp80 / tn36 [are] not understandable without the animation".

## 5. The common base, deconstructed

### 5.1 Tufa Labs "Duck" harness (ARC3-Inference, unchanged in every Flash-Next fork)

Source: `SCRATCH/kaggle/datasets/keithtyser__duck-qwen38-nvfp4-mtp-vllm-smoke-v1/src/ARC3-Inference/inference/`. `diff -r` against `jeroencottaar__taaf-kaggle-source-share` is empty, so it is the June-30 code.

- **Loop:** one analyzer "step" per turn. The model gets a system prompt, persistent history (up to 30 assistant turns, trimmed to fit a 32k context with a 512-token safety margin), and a user prompt. The user prompt contains:
  - the previous executed actions;
  - level-transition flags;
  - the valid actions;
  - the carried **"Working world model"** block, with fields World / Goal / Action model, Recent findings, Open questions, Plan and Cross-level notes. These are parsed from labelled lines in the assistant text. The stock parser needs an exact `Label:` prefix and **wipes everything except Cross-level notes on level transition *and* on game over**.
- **One tool: `python`** in a sandbox:
  - fresh namespace each call;
  - stdlib whitelist (collections, heapq, itertools, math, re, …);
  - 30 s limit;
  - output capped at about 1,024 tokens.

  The model **acts from inside Python** by calling `action([...])`, which takes a list of `UP/DOWN/LEFT/RIGHT/SPACE`, `{'action':'MOUSE','row':r,'col':c}` or `RESET` and can be batched or looped. Runtime variables: `current_frame` (`.ascii`, `.segmentation`, `.level`, `.step`, `.shape`), `previous_frame`, `history`, `transitions`, `last_transition`, `last_action_result`, `valid_actions`.
- **Board representation:**
  - **Letter-coded ASCII** (16 letters).
  - A **segmentation** into 4-connected same-colour objects with position-independent `hash`, `boundary` polygon, `children` (containment) and an `adjacency_list`. This is presented as the *primary* view.
  - The **raw numeric grid is deliberately hidden**.
  - **Plus one PNG of the current board** (`MULTIMODAL_CONTEXT=current_grid`, upscale 4 = 256 px).
- **Prompt content:** "treat each turn as observe–plan–act"; "write BFS / search when the objective is understood"; "HUD / timer bars are not puzzle pieces"; "batch reliable sequences"; "optimise for few actions".
- **Clicks (ACTION6):** exposed as `MOUSE` with integer row and col. The model picks coordinates itself, usually from segmentation boundaries. There is no harness-side click-target generator.
- **ACTION7 is not mapped** in `action_names.py` (only ACTION1-6 and RESET), so undo is unusable in stock Duck. Patches: muhibullahansir (ACTION7↔"UNDO"); AGENTFIX F6 (off by default in the #1 notebook).
- **Reset:** `ONLY_RESET_LEVELS=true`, so RESET restarts only the current level. On `GAME_OVER` the solver issues an **automatic RESET** and keeps going. There is no give-up: a game runs until it is won or its per-game clock expires.
- **Sampling:** temperature 0.6, top-p 0.95, top-k 20, thinking on (`preserve_thinking`), seed −1.
  - `LOCAL_ANALYZER_TOOL_STEPS=0`: unlimited tool calls per step.
  - `YIELD_SECONDS=60`: after 60 s the step yields back to the solver loop and continues.
- **Scheduling:** `concurrency=28` games at once, `max_runtime_s_per_game=7920`, `analyzer_timeout=900`. The 32,400 s notebook budget minus a 600 s teardown reserve gives the soft end.

### 5.2 keithtyser Flash-Next serving (what took Duck from about 1 to about 3.3)

Files: `serving_setup.py` (123 KB), `vllm-patches/`, `vllm_server_watchdog.py` in the same dataset.

- **Model:** `RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision 7b71922. vLLM reports the architecture as `Qwen4ExpForConditionalGeneration`.
  - It is a Kaggle model asset of 135 GB / 419 files: `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`.
  - Weights take **81.8 GiB** of the 96 GB card. There is a 51B-parameter n-gram "PLE" embedding table, loaded as FP8 through a **custom vLLM loader patch** (`radixark_nvfp4_ple_fp8.patch`). About 6.5B parameters are active (per ghazaros's notes).
- **Runtime:** a pinned offline vLLM (`0.1.dev20073+g8e685d198`, from the `vllm/vllm-openai:qwen38-flash-next` image). It ships as layer blobs in `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1` (about 7.9 GB).
- **Launch command:** `--quantization modelopt_fp4 --dtype bfloat16 --kv-cache-memory-bytes 5368709120 (5 GiB) --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 --async-scheduling --enable-chunked-prefill --max-cudagraph-capture-size 32 --no-enable-prefix-caching --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config {"method":"mtp","num_speculative_tokens":3}`.
- **Watchdog:** polls `/v1/models` every 15 s and restarts after 4 failures (at most 2 restarts). Setup (queue, runtime, 135 GB load) takes **about 63 min** of the 9 h according to yanggod.
- **Measured throughput:** about **230–240 generated tok/s aggregate**, about 1.8M generated tokens per 2.2 h over 25 concurrent games. The GPU is saturated: there are 26–50 `analyzer request failed … Read timed out` events per run.
- **KV cache is the binding constraint.** With 5 GiB, only 8 × 32k sequences fit, while 28 games compete for them. scottlegrand's fix: `max_num_seqs=16` with the analyzer context halved to 16k in the same 5 GiB. amanatar's attempt at 16 GB KV **failed to start** (vllm-setup-failure.json).

## 6. Deep dives

### 6.1 #1 scottlegrand/taaf-flashnext-sheetu12b-0922 (5.19). The user's notebook.

Files: `SCRATCH/kaggle/notebooks/scottlegrand__taaf-flashnext-sheetu12b-0922/`. A flat copy is in `taaf-flashnext-sheetu12b-0922.py`; cell 10 holds about 2,700 lines of patches. A diff against `SCRATCH/nb/user_nb_code.py` shows only output lines.

- **Serving profile `kv5-bf16-mtp3-c16-cg32-ctx16k`:** 16 sequences, `LOCAL_ANALYZER_CONTEXT_WINDOW=16384`, KV 5 GiB, MTP3, no prefix cache.
- **AGENTFIX switches** (the "c16 + v8" set). Switches marked on are live:
  - **F1 images (on):** history keeps images only for the newest 2 image-bearing messages. The token estimator charges a flat 120 tokens per image instead of `len(b64)/3`, which over-charged 5–8× and evicted reasoning. Notes in the code: "median 8 obsolete boards per request, ~26 % of budget"; stripping *all* images caused "+51 % reasoning and 29 % fewer actions per 2 h".
  - **F3 memory (on):** a tolerant parser for `World model (revised):` and similar labels. **Game over no longer wipes the world model**, since a game over is a retry of the same level.
  - **F9 timing (on):** a per-request timing log.
  - **F13 anim (on):** the engine returns a list of frames per action, and stock Duck only ever showed the last one. F13 stores up to 6 intermediate frames (first, last-before-final and evenly spaced, with the true count kept) for the 3 most recent history entries, and exposes them as `last_transition.frames` and `.frame_count` with `.segmentation` working on them. The claim in the code: **"the only change that has ever moved the hidden leaderboard (3.20 → 3.71)"**.
  - **F13b summary (off):** a harness-written one-line animation narration. It **"did not transfer (2.57)"**.
  - **F19 sheet (on):** when the last action animated, one extra image is attached. It is a labelled contact sheet of up to 8 frames (scale 4, at most 1 MP), placed before the current-board image, with a single legend line and no narration. Only the newest sheet is kept. Measured cost: about 541–1,218 estimated tokens against a 21.5k-token typical prompt.
  - **ARM P:** board image upscale **4 → 12**, so the 64×64 board is sent as 768×768 px.
- **Off by default:**
  - F2 (always return the action result);
  - F4 (loop guard);
  - F5 (HUD-aware no-op tagging);
  - F6 (ACTION7);
  - F7 (per-level effect ledger);
  - F10 (stall throttling);
  - F11 (dedup of about 36 % repeated prompt text).

  The docstrings record a "v3 regression hunt": **refusing known no-ops misled the model** on cn04 and g50t, so refusal is off.
- **Settings:** 7,920 s per game, concurrency 28, `analyzer_timeout 900`. The local validation run is capped at 2 h: mean **10.02**, 2,453 actions, 43 levels, 1.70M tokens, 236 tok/s. Best games: ft09 47.6, lp85 41.0, ar25 27.8.
- **Why it is probably the best public notebook:**
  - It is the only public variant that gives the model *new information*, the animation frames, in a form it actually uses: pixels and sandbox data rather than narration.
  - It also frees context and KV for more concurrent requests: c16 and image trimming.

### 6.2 keithtyser base (3.38; cluster mean 3.29) and its pure-luck forks (#3, #8, #9, #11, and others)

Described in §5.2.

- The "Anim Base" name used by wuliao0, muhibullahansir, ghazaros and samrishb is misleading: the code attaches **no** animation bundle.
- wuliao0 resubmitted the identical notebook daily, 13 versions from 09-04 to 09-18, and the best was 4.33. The same trick explains many 3.9–4.3 entries.

### 6.3 #2 / #7 "Tuned" (4.50 and 4.08; also 2.81)

- The only gameplay-relevant change is `analyzer_timeout 900 → 1200`, plus two gateway-wait edits.
- The four submissions of timeout-1200 code (4.50, 4.08, 2.96, 2.81) have mean 3.59, against 3.29 for the base. That is not significant.
- Rationale: under saturation, a 900 s analyzer timeout abandons long requests, and the logs show 30–50 read timeouts per run.

### 6.4 #4 muhibullahansir (4.16, 34 versions)

- Maps **ACTION7 → "UNDO"** into the Duck action namespace before the solver imports. The patch is guarded by the sha256 of `action_names.py`.
- `max_runtime_s_per_game = 7600`.
- Sends the job dir and recordings to a private `/tmp` path during the real rerun.
- Adds an MTP on/off A/B arm switch and a large read-only audit logger (off in the rerun).
- Local 8.28. The notebook refers to a "6.16 baseline run". Otherwise gameplay equals the base.

### 6.5 #6 amanatar "Hybrid REPL Agent" (v38 = 4.08)

- The name is marketing: it is the base with `TAAF_VLLM_ENABLE_PREFIX_CACHING=1`.
- `bm.n_passes=3` is set and then reset to 1 before the run.
- The later v40, with KV 16 GB and 16k batched tokens, **failed at vLLM setup** because the Flash-Next weights leave about 9 GB.
- Prefix caching is the only serving change here that plausibly speeds up the prefill-heavy workload: about 20k-token prompts that are mostly repeated history.

### 6.6 Thuitanium team (yocybercode + sahasawatt, LB 5.36): `thui-*` A/B notebooks

- They publish **arm/control pairs over the full public 25** (`-v0` vs `-ctl`, `full25-r1`), each with a permutation-test rationale. Examples: thui-l1 (HUD band), thui-wm (world model), thui-db, thui-af, thui-rs, thui-m0-s20, thui-a7 (ACTION7), thui-gemma, thui-compact, thui-reflect, thui-rank.
- Local results (single runs):
  - thui-l1-v0 **10.93**;
  - thui-animfast **9.56**;
  - thui-fast-v0 9.32 (≡ base);
  - thui-fast-b78-mtp0 6.96 (MTP off).
- LB results: 3.60 / 3.74 / 3.21 / 3.49. None separates from the base on the LB.
- Useful mechanisms:
  - **L1 HUD-band learner**: rows changing on ≥ 90 % of the first ≥ 20 actions, at most 4 rows, set `no_impact`.
  - **animfast**: jakobbrggen's anim bundle. It adds `last_action_result['animation']` with `frames`, `unique_frames`, `board_unchanged` and `transient_pixels`/`transient_bbox`, an `animation(frame=k, region=…)` retrieval tool, a proactive hint after ≥ 6 turns without progress on type-1 animations, and a **hard no-op guard** keyed on (level, board-hash, action).
  - Seed pinning and a 180 s yield.

### 6.7 #12 woguoat v45 (3.70): two prompt lines

The notes are in Chinese. The lines are ported (CC0) from `gedouluhui/taaf-p3-bundle` "P3.1", which the author says reached "hidden 1.43, +28 % over P1" on an older model. Appended to every user prompt:

> - Scoring punishes long solutions quadratically: each level scores roughly (baseline_actions / your_actions)^2 * 100, so a path twice as long as necessary scores about four times less. Think before acting, then commit to the shortest reliable sequence and batch confirmed steps in one `action(...)` call.
> - Action results report `time_remaining_seconds`. Use it to pace yourself: probe until the mechanic is understood, then execute decisively. When several distinct hypotheses have failed on a level, stop repeating near-identical attempts and change the hypothesis class instead.

### 6.8 #13 yanggod wavefit-b (3.64): per-game budget sized to waves

- `per_game = min(7920, (remaining_window / ceil(N_games/28)) × 0.97)`, with a floor of half the stock value. It only takes effect in the rerun.
- The author's reasoning: about 110 hidden games at concurrency 28 make 4 waves needing 31,680 s. After setup about 28,000 s remain, so the fourth wave (about a quarter of the games) is cut to roughly half its time.
- One LB sample (3.64) cannot confirm the effect. The mechanism is sound, and the defiaudit 27,000 s counter-example (LB 2.58) shows the budget matters.

### 6.9 #16 retained_reasoning (3.57): the most "engineered" public harness

The code is in a private dataset (`nihilisticneuralnet/120926arc`); only the prompts are visible in its outputs. It extends Duck with:
- a persistent `memory` dict with `remember()` and `forget()`, surviving context trimming and game overs; the model can store simulator source and `exec` it later;
- the exact `.grid` exposed, plus `.components(4|8)` and `.color_counts`;
- `state_key()`, `diff_frames()` and `tried_actions()` (previous actions from the same observed state);
- **declared expectations** per action (`{'action':'LEFT','expect':{'board_changed':True,'max_changed_cells':8,'cells':[[4,7,2]]}}`). A failed expectation records a counterexample and **cancels the rest of the batch**;
- `audited_predictions` and `prediction_stats()`;
- `.animation_frames` on every history entry;
- a "store an executable transition model, replay it against all observed transitions, reject it on mismatch, then search" instruction.

Result: LB 3.57 and local 5.83. There is no measurable gain on this model, which suggests the 6.5B-active model cannot exploit the extra machinery within the time budget.

### 6.10 Other variants (all within noise, listed so they are not re-tried blindly)

| Variant | LB | Local |
|---|---|---|
| MTP off (thui b78) | 3.49 | 6.96 |
| v124 bounded memory + probe ledger | 3.27 | 8.43 |
| CoTRD logit contrast | 3.16 / 2.78; Full-CAD 1.85 | – |
| c17 "efficient agent" system-prompt paragraph | 2.84 | – |
| c18 (c17 + audit) | 2.67 | – |
| Action-batching budget note | 2.95 | – |
| Winning-transition carry note | 2.66 | – |
| elminimle: world model kept across levels + mandatory 2-line note + reflection every 10 steps + skill ledger | 2.33 | 7.28 |
| defiaudit anti-stall supervisor | 2.56–2.58 | – |
| saurabhkumar v11.1 (+680 lines) | 2.47 | 9.17 |

## 7. What differentiates 20+ teams from 3–5 notebooks (inferred; they publish nothing current)

- **Public top-10 artifacts are all pre-July, and none is Duck-like at scale:**
  - Tufa's Duck (1.2–1.3 in June, now 27.3 privately);
  - UNK (#18 now): hippolytepilchen's **PLDM/HWM latent world model with expert iteration** (best-first search → fine-tune world model online per game → reward-MPC planner), 0.41 in June; and thomasferraz's **TD-JEPA + BFS**, 0.11;
  - face-of-agi (#22): Qwen3.6-35B-FP8;
  - Milestone #2 and #3 candidates (ruichardliu, mbmmurad): **Gemma-4-31B VLM with JSON actions**, 0.86.
- Evidence from other sources (see `13_hf_github_assets.md`):
  - Tong Hui Kang (#3) publishes a 0.4M-parameter **imitation-trained per-cell policy**;
  - NVIDIA/NVARC uses an **executable world-model multi-agent** approach;
  - Tufa's current 27.3 is not public, but their writeup notes that the same submission ranged 0.77–1.30.
- **Inference:** the jump from about 5 to 20+ does not come from prompt or harness tweaks on the same 6.5B-active LLM. Every such tweak in the public record lands in 2.3–4.5. It comes from one or more of:
  1. a much stronger or faster policy per GPU-second, such as a small learned policy or world-model search that runs many more actions per game;
  2. better exploitation of the per-level action-efficiency metric;
  3. far better time and throughput allocation across about 110 games.

## 8. Differentiators *within* the public field (3–5 vs 1–2)

1. **The model and serving stack.** Flash-Next NVFP4 + MTP (median 3.2) vs 27B FP8 (1.6) vs Qwen3.6 (0.9). Speed matters because the score is capped by how many analysis steps each game gets before its clock expires.
2. **Keep the 28-way concurrency and the wave structure intact.** Budgets that starve the hidden set (27,000 s per game) or cut it (offline-only cuts are harmless) are the big failure mode.
3. **Animation access.** The anim bundle and F13/F19 are the only mechanism with a claimed hidden-LB delta (3.20 → 3.71), and the #1 notebook has it.
4. **Everything else** (prompt lines, memory dicts, probe ledgers, CoTRD, efficiency directives) is inside the ±0.5 noise.

## 9. Lessons / what to copy

- **Keep:** the scottlegrand/AGENTFIX stack as the baseline. It is already the user's notebook and the best public evidence. The F13 frames, F19 sheet and ×12 board image are the parts with some hidden-LB support.
- **Cheap and plausibly positive, not yet combined in one notebook:**
  1. `ACTION7→UNDO` mapping (muhibullahansir) or AGENTFIX F6. Undo is currently unusable in Duck, and some public games expose it.
  2. **Wave-fit** per-game budget for about 110 games (yanggod). Also consider giving games that clear levels more time than games that are stalled: AGENTFIX F10 exists but is off.
  3. `analyzer_timeout 1200`, because saturated runs time out 30–50 times.
  4. **Prefix caching ON** (amanatar v38). The workload is about 20k-token, mostly repeated prompts.
  5. HUD-band `no_impact` tagging (thui-l1: local 10.93, best single local run), as information only and never as refusal.
  6. The two quadratic-scoring / pacing prompt lines (woguoat).
- **Do not:**
  - refuse or block known no-ops (AGENTFIX v3 regression);
  - narrate animations in text (2.57);
  - raise the KV cache above about 9 GB (setup fails);
  - raise `max_runtime_s_per_game` far above the wave-fit value (2.58);
  - trust a single local 25-game run (r = 0.11) or a single LB submission (sd 0.5).
- **Methodology:** judge a change by **several LB submissions** or by **a large local sample** (≥ 3 seeds × 25 games), as Thuitanium tries to. The public field's history is mostly best-of-N luck.
- **Strategic:** reaching 18–20 needs a qualitatively different ingredient (§7), not another Duck prompt patch.

## 10. Downloaded paths

All paths are under `SCRATCH/kaggle/`.

- `notebooks/index.csv` / `notebooks/index.json`: all 1,132 notebooks with score, votes, dates, data sources and URL. `index.json` adds curated fields for the analysed ones.
- `notebooks/raw_lists/`: raw CLI and internal-API listings (`union_internal.json`, `internal_*.json`), and `target_versions.json` (per-notebook version history).
- `notebooks/_scored_sources/`: **scored-version source of all 456 scored notebooks** (`<owner>__<slug>__<scriptVersionId>.ipynb|.py`).
- `notebooks/scored_clusters.json`: normalised-code hash, and lines added and removed vs base, for each scored notebook.
- `notebooks/_diffs/`: code diff vs the keithtyser base for 41 distinct variants scoring ≥ 2.0.
- `notebooks/_base/keithtyser_v14.ipynb`, `_base/keith.py`: the base.
- `notebooks/<owner>__<slug>/`: latest source (`.ipynb` plus a flattened `.py` for the ones read), `kernel-metadata.json`, `output_files.txt` and `output/`. The output holds `score.json`, `benchmark.json`, `transcripts/`, `prompts/`, logs and vLLM logs; bulk per-game event and viewer JSON is skipped. For amanatar, cassowaryloraforge, woguoat and cotrd-enhanced, the scored version is in `listed_version_vN/`.
- `notebooks/_scoreprobe/<owner>__<slug>/score.json` and `probe_local_scores.json`: local public-25 scores for about 50 Duck-family notebooks. `local_results.json` covers the fully pulled ones.
- `notebooks/local_results.json`: per-game local scores for the pulled notebooks.
- `datasets/` (solver code; small):
  - `keithtyser__duck-qwen38-nvfp4-mtp-vllm-smoke-v1` (70 MB unzipped): Duck source, serving setup, vLLM PLE patch, watchdog, pickled benchmark;
  - `jeroencottaar__taaf-kaggle-source-share` and `jeroencottaar__taaf-kaggle-source`: Tufa June;
  - `jakobbrggen__taaf-kaggle-source-anim-20260807-anim`: `animation.py` and `noop_guard.py`;
  - `jakobbrggen__taaf-kaggle-source` (09-01, adds `avo/`);
  - `keithtyser__taaf-duck-qwen38-serving-v1` (Qwen3.8-27B-FP8 serving);
  - `thtennant__taaf-kaggle-source-share-fork`;
  - `defiaudit__agi3-taaf-bundle-v2` and `-v3` (anti-stall / supervisor);
  - `raist321__taaf-polyphony-v25-bundle`.
- **Model and runtime datasets not downloaded** (recorded only):
  - `keithtyser/qwen3-8-flash-next-nvfp4` (Kaggle model, 135.25 GB, 419 files, HF `RadixArk/Qwen3.8-Flash-Next-NVFP4` @7b719225);
  - `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1` (6 layer blobs, about 7.9 GB total);
  - `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`;
  - `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` and `driessmit1/arc3-vllm-h100-wheelhouse-v3` (Tufa June);
  - `google/gemma-4/…/gemma-4-31b-it`.
- `lb/lb_internal.json`: full public LB (3,368 teams) with member usernames.
