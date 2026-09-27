# 12 — Web SOTA survey: ARC-AGI-3 (as of 2026-09-27)

Everything cited here was fetched on 2026-09-27. Raw copies are in `SCRATCH/web/`, where
SCRATCH=`/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad`.
The layout is listed at the end. Tags: **[V]** means I verified it against code or primary data,
**[Q]** means quoted from an official source, **[C]** means a community claim, **[S]** means my speculation.

---

## 0. TL;DR (the ten facts that matter most)

1. **Per-level score = min(1.15, (human/agent)²).** A game's score is the level-index-weighted mean of
   its level scores, capped at the weighted fraction of levels completed. The total is the plain
   mean over games, and an unplayed game counts as 0. **Every** ACTION1-7 counts as one action, dead
   clicks included, and so does every RESET after the level start. There is **no 5× action cap on
   Kaggle**. Actions spent on a level you never finish cost **nothing**, because that level just
   scores 0. [V: `arc_agi/scorecard.py` 0.9.8; Q: docs + technical report; Q: Greg Kamradt on Kaggle]
2. Kaggle hidden set: **110 games**. **55 score the public LB and 55 score the private LB**, and both
   are scored in the same run, at submission time. Competition mode allows **one `make` per game**,
   and a **game reset is turned into a level reset**. [Q]
3. The Kaggle LB on 09-27: Tufa 27.29, Lord Han Solo 20.80, Tong Hui Kang 20.53, Yi-Chia Chen 18.80,
   Daniel Franzen 16.68, NVARC3 16.07. Almost every top team jumped between 09-06 and 09-26. Before
   that, the whole board was below 5 (history in §2.3).
4. **No top-10 method is public.** The only open, top-tier lineage is the Tufa "Duck" REPL harness
   (June milestone; 1.21 LB at the time on Qwen3.6-27B). Tufa **will not** open-source for Milestone 2
   (09-30). Tong Hui Kang (#3) and Lord Han Solo (#2) will share only under certain placements.
5. All public notebooks are Duck forks on **Qwen3.8-Flash-Next NVFP4 (125B-A6B MoE + 51B n-gram PLE)**,
   served by vLLM with MTP. The best public notebook scores about 5.2 LB (Scott Le Grand). Everyone
   reports huge variance: the same notebook ranges 2.5–5.5, and one notebook scored 0.55–1.29 over 11
   submissions.
6. Frontier API results on the semi-private set (the ARC verified leaderboard): **GPT-6 Astra 62.7%
   (standard harness) and 99.9% ("Provider Adapter" = retained reasoning + compaction)**. Claude Opus
   5 scored 30.2%, Gemini 3.8 Flash scored 35.0% (PA). OpenAI reported that **retained reasoning +
   compaction tripled GPT-5.6 Sol (13.3→38.3%) with 6× fewer output tokens.**
7. On the public 25 games, many harnesses reach 95–100 RHAE with frontier models: Tycho, Retrodict,
   AVO, VISTA, Kepler, Twin and baseline1. Their common mechanisms: a logged history, verified
   predictions before acting (retrodiction), executable world models, plans that halt on the first
   mispredicted frame, and memory that survives compaction.
8. Pure algorithmic agents (graph exploration, CNN action-effect models) topped the 2025 preview
   (StochasticGoose 12.58%, Blind Squirrel 6.71%). On the 2026 hidden set they plateau at about
   **0.27 LB**, because the environments were calibrated so that random play solves a level less than
   once in 10,000 tries. Use them as **tools inside the LLM loop**, not as the driver.
9. The serving stack is probably the cheapest big lever. Our own 09-22 public run (vLLM, 5 GiB KV,
   16 seqs, prefix caching off) spent **87% of request latency queued**, had 342 preemptions, and
   decoded about 250 tok/s aggregate, which is about 42 LLM turns per game. A tuned SGLang build for
   one RTX PRO 6000 (Pennyroyal) reports 446 tok/s at C4 and 632 tok/s at C8, with an 824K-token KV
   pool and radix prefix caching.
10. Human baselines for the 25 public games (183 levels, 17,135 actions) are in
    `SCRATCH/web/replays/public_baselines_from_replays.json`. They match `metadata.json`.

---

## 1. Scoring formula (verified)

### 1.1 Official text
- Docs, "ARC-AGI-3 Scoring Methodology" (https://docs.arcprize.org/methodology) [Q]:
  - "`level_score = (human_baseline_actions / ai_actions) ^ 2`"
  - "The maximum score per level is capped at **1.15x** human baseline."
  - "The game score is the **weighted average** of all per-level scores, using the 1-indexed level
    number as the weight … it is capped based on how many levels the AI actually completed …
    `max_game_score = (1+2+3+4)/(1+2+3+4+5) = 66.7%`"
  - "Total score is the **average of all game scores**."
  - Baseline: "the **upper median human** (by fewest actions) per level … if four players complete a
    level, third place is the baseline; if five players complete it, third place is still the baseline."
- Technical report, §4.1 (https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf, arXiv 2603.24621) [Q]:
  - `S_{l,e} = min(1.15, (h_{l,e}/a_{l,e})^2)`
  - `E_e = min( Σ_{l≤k} w_l / Σ_l w_l ,  Σ_l w_l·S_{l,e} / Σ_l w_l )`, with `w_l = l`, and S=0 for
    uncompleted levels
  - `T = (1/|D|) Σ_e E_e`. The design is inspired by SPL (Success weighted by Path Length).
  - The 5× human-baseline action budget applies **only to the verified leaderboard**, as an API cost cap.
- Changelog, 2026-04-14 (https://docs.arcprize.org/changelog): the baseline changed from the
  2nd-best human to the median human, and the cap changed from 1.0× to **1.15×**.
- Note: the Kaggle "Data" and "Evaluation" pages still say `min(h/a, 1.0)²` and "capped at 100%".
  They are stale; the scorer that ships uses 115 (see below).

### 1.2 The code that actually runs (arc_agi 0.9.8, from the competition wheel) [V]
`SCRATCH/web/wheels/arc_agi/arc_agi/scorecard.py`:
```python
# EnvironmentScoreCalculator.add_level
if completed:
    score = ((baseline_actions / actions_taken) ** 2) * 100
    score = min(score, 115.0)          # Cap at 115
else:
    score = 0.0                        # uncompleted level (any number of actions spent) -> 0
# to_score
weight = level_index (1-based); score = Σ w·s / Σ w ; max_score = Σ_{s>0} w / Σ w * 100 ; score = min(score, max_score)
```
- `level_actions` for level i is the cumulative action counter when level i completed, minus the
  counter when level i-1 completed. It therefore includes failed attempts, RESETs and GAME_OVER
  retries on that level.
- **Action counting** (`Card.update_scorecard`): ids 1–7 call `take_action` (+1 each). RESET (id 0)
  either calls `inc_reset_count` (+1 action **and** +1 reset) or, on a full reset, starts a **new
  play**. Nothing checks whether the frame changed.
  - Greg Kamradt confirmed that the scored server matches this: a no-op ACTION6 dead click counts in
    both the RHAE tally and the action budget
    (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/718638).
- **Engine reset semantics** (`arcengine/base_game.py handle_reset`): if the level action count is 0
  or the state is WIN, a RESET is a `full_reset` (back to level 1). Otherwise it is a `level_reset`.
  `set_level()` zeroes the counter, so **a RESET at the very start of a level, or two RESETs in a row,
  restarts the whole game** in offline mode. With `ONLY_RESET_LEVELS=true` (the public Duck notebooks
  set it) and in COMPETITION mode, "Game Resets are not allowed and become Level Resets"
  (https://docs.arcprize.org/toolkit/competition_mode).
- **Multiple plays** (offline and online modes only): the environment score is the play with the
  **max levels_completed**. Competition mode allows only one `make` per game, so on Kaggle there is
  exactly one play. Keep local eval in COMPETITION mode, or best-of-plays will inflate it.
- **Unplayed games** score 0: "Scoring is against all available environments, even if you choose not
  to interact with them" (competition_mode doc).

### 1.3 Consequences worth internalising
- Actions to score: 1× baseline gives 100%, 1.5× gives 44%, 2× gives 25%, 3× gives 11%, 5× gives 4%,
  10× gives 1%. Beating the baseline pays at most 115%.
- **Depth dominates.** In a 7-level game, clearing L1–L3 at exactly the baseline scores 6/28 = 21.4%.
  Clearing all 7 levels at 2× the baseline scores 25%. Level k carries weight k. For a 6-level game,
  L1 alone is worth at most 100/21 = 4.76; for a 9-level game, 100/45 = 2.22
  (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728299 and /743060).
- **Continuing to act on a level you never finish costs 0 score.** Stopping early only saves wall-clock
  for other games. Stop and resume rules are a compute-allocation problem, not a score problem.
- Kaggle: "We don't have the 5x action cap during the kaggle competition … Kaggle has another
  mechanism … the limited compute" (Greg Kamradt,
  https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713921).
  "Private scores are calc'd at original run time. They aren't rerun … the submission plays both
  datasets" (/729985). "Submissions run on all 110 tasks" (/684852).
- Games are seeded with stable seeds, not procedurally generated. The one known exception is the lf52
  noise animation (/694153). "A game's available actions are the same throughout" (/702079). Partial
  observability exists (ls20's last level, /686532).
- `environment_info.baseline_actions`: one early post claimed it was readable during submission
  (/687655). A later report says it comes back empty in submission mode (/739801). **Don't rely on it.**
- Do **not** use the local "null-coordinate" bug. `ACTION6 {x:None,y:None}` raises a TypeError that
  the 0.9.8 wrapper reports as a WIN on 18/25 public games (arXiv 2605.25931 §5.8). It is a local
  artifact. It is also a **validation hazard**: any malformed click in a local eval can produce a
  phantom win, so reject any "win" that doesn't come with a valid action.

### 1.4 Action semantics [Q/V]
- https://docs.arcprize.org/actions: RESET; ACTION1–4 map semantically to up/down/left/right;
  ACTION5 = interact/select/rotate/attach/execute…; **ACTION6 = click at (x, y), 0–63**;
  **ACTION7 = undo** ("ACTION7 will always be an undo action for games that support it",
  llms.txt API reference). Human keys: WASD/arrows, Space or F for ACTION5, mouse for ACTION6, Z for
  ACTION7.
- In GAME_OVER only RESET is valid. Anything else returns HTTP 400.
- ACTION6 coordinates are **display** coordinates on the 64×64 frame, with x = column and y = row.
  Smaller level grids are integer-upscaled and letterboxed (`arcengine/camera.py`,
  `display_to_grid` returns None inside the letterbox). So every pixel inside one upscaled cell is the
  same click. Letterbox clicks do nothing but are still counted.
- Public games that expose ACTION7/undo: ar25, bp35, lf52, sb26, sk48, su15. It pops a state stack,
  e.g. su15's `vczehveskr` [V]. Undo costs +1 action, but it can replace a RESET plus a replay.
- Frames can be **sequences**: animation frames come before the settled frame. 13 of 25 public games
  return multi-frame responses; sp80's pour returns 22 frames (Taaf-Anim writeup, /734369).

---

## 2. Top teams & methods

### 2.1 Milestone 1 (June 30) winners — https://arcprize.org/blog/arc-prize-2026-milestone-1
| # | Team | Method | Model | LB then |
|---|---|---|---|---|
| 1 | **Tufa Labs, "The Duck"** (Bessis, Cottaar, Pressman, Smit, Tešnar, Viel) | LLM writes and runs Python in a REPL. Game state is exposed as variables (`current_frame.ascii`, `.segmentation`, `history`, `transitions`, `valid_actions`, `action()`). A 4×-upscaled image goes in every turn. A world-model note is carried across turns. **Eviction** keeps about a 32k input context (64k max). UNDO is hidden from the model. Hand-built tools *hurt*; the gains came from better base models plus multimodality. | Qwen3.6-27B FP8, vLLM | 1.21 (public-25 mean 1.60 ± 0.45 over 20 tries) |
| 2 | Reki | A VLM as the policy: labelled frame images in, one JSON (what changed, plan, 1–4 actions) out; reflection memory about every 10 steps. numpy click heuristics prefer small, rare-colour, button-like shapes. A "dead-signature" rule stops clicking object types that never change anything. | Gemma-4-31B | ~0.9 |
| 3 | Md B. M. Murad, "forge" | Same as Reki inside a profile framework (generator + arbiter + confidence prompt). The best run had all extras **off**. | Gemma-4-31B | 0.86 |

Sources: Tufa writeup https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133
(saved as `SCRATCH/web/kaggle_disc/717133_*.md`), https://tufalabs.ai/research/duck-harness/ ,
https://github.com/Tufalabs/duck-harness (code, viewer, a 25×20 example run; MIT dataset,
Apache-2.0 notebook), and the MLST episode https://www.youtube.com/watch?v=Vg6FBKTlfOw.
Duck's ancestors: the Symbolica Agentica/ARCgentica recursive agents (36.08% public on day 1,
https://www.symbolica.ai/blog/arc-agi-3) and the RGB (Read-Grep-Bash) agent built on OpenCode
(https://blog.alexisfox.dev/arcagi3).

### 2.2 2025 Preview competition — https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
| Agent | Score / levels | Core idea | Code |
|---|---|---|---|
| **StochasticGoose** (Dries Smit, Tufa; advisor Jack Cole) | 12.58%, 18 levels, 256k actions | A CNN (4 conv layers on a 16-channel one-hot 64×64 input) predicts P(frame changes \| state, action) for ACTION1–5 plus a **conv coordinate head** for ACTION6. It samples by sigmoid probability, keeps a 200k hash-deduplicated buffer, and resets the model and buffer on each new level. | https://github.com/DriesSmit/ARC3-solution |
| **Blind Squirrel** (Will Dick) | 6.71%, 13 levels | A state graph built from frames. It prunes loop and no-op actions. On a score increase it back-labels distances and retrains a ResNet18 value model to rank (state, action). | https://github.com/wd13ca/ARC-AGI-3-Agents |
| Explore It Till You Solve It (E. Rudakov) | 3.64% (bug); 3rd by levels | Training-free graph exploration. Single-colour connected-component segmentation, status-bar masking, 5 **priority tiers** of click targets (size, shape, colour salience), state hash, and BFS to the nearest untested (state, action) at the current priority. It escalates the tier when exhausted. Median 30/52 levels in 8 h; a reset-loop bug cost it on the private set. | arXiv 2512.24156, https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore |
| Fluxonian | 8.04% | DSL + LLM (GPT-4.1) | https://github.com/FluxonApps/arc-prize |

### 2.3 2026 Kaggle leaderboard and trajectories
Kaggle API LB on 09-27, plus minute-level history from Tong Hui Kang's monitor
(https://arc3.huikang.dev/leaderboard, JSON at
`https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run?comp=arc3`, saved as
`SCRATCH/web/lb_history_arc3.json`). 3,366 teams, 39k submissions.

| Rank | Team | Score | Subs | Trajectory (date:score) | What is known |
|---|---|---|---|---|---|
| 1 | Tufa Labs | 27.29 | 147 | 06-03 1.21 → 08-23 4.58 → **09-06 11.04 → 09-13 18.81 → 09-26 27.29** | Duck lineage. "We will not be open-sourcing for the second milestone" (/742801); they will release after the end. Community speculation: fine-tuning plus better token throughput/allocation. They have large GPU access. [C] |
| 2 | Lord Han Solo | 20.80 | 75 | 08-22 3.36 → 09-18 11.54 → 09-19 15.60 → 09-20 18.42 → 09-26 20.80 | Solo. Will share "if I make the top 3 in Milestone 2 among participants who have indicated their willingness to share". Nothing else is public. |
| 3 | Tong Hui Kang | 20.53 | 85 | 08-23 2.24 → 09-16 8.72 → 09-21 10.78 → 09-24 15.02 → 09-25 20.53 | Won the Nemotron reasoning progress prize with LoRA via Tinker (his Kaggle notebooks: "Tinker submission", "Adapter validation"). His May autoresearch CNN policy on "ideal gameplay traces" of modified games did **not** beat random (blog: https://blog.huikang.dev/2026/05/31/autoresearch-hackathon.html). Intends to share only if he finishes 1st. |
| 4 | Yi-Chia Chen | 18.80 | **13** | 09-17 4.79 → **09-19 12.88** → 09-25 18.80 | Nothing public. Very few submissions, so heavy offline iteration. |
| 5 | Daniel Franzen | 16.68 | 83 | 09-05 7.63 → 09-16 11.59 → 09-24 16.68 | Of "the ARChitects" (ARC Prize 2024 winner: test-time training, product-of-experts). No ARC-3 writeup found. |
| 6 | NVARC3 | 16.07 | 22 | 09-06 3.32 → 09-16 11.04 → 09-19 16.07 | NVIDIA KGMoN (CPMP / J-F Puget; rfbr). They won ARC Prize 2025 as NVARC. "We tried AVO style ideas and haven't seen them beat our current harness" (/737617). They open-source only at the end, and only with a gold medal. |
| 7 | the last dance | 13.70 | 64 | 09-05 3.54 → 09-24 13.70 | — |
| 8 | Third Intelligence | 12.31 | 58 | 09-02 3.97 → 09-26 12.31 | — |
| 9 | Matija L & Zhongwei W & Fususu | 11.64 | 176 | 09-17 6.18 → 09-18 11.49 | Fususu's shared negatives are summarised in §2.4 (/739938). |
| 10 | rellik13 | 9.96 | 38 | — | — |
| — | mostik.ai (12-PhD startup) | 7.51 | 47 | held #1 around 09-03, stopped 09-06 | Team-size controversy (/739186) |

Best **public** notebooks: scottlegrand/taaf-flashnext-sheetu12b-0922 (5.19, "AGENTFIX" fixes, the
source of our fork), keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp, wuliao0/duck-qwen3-8-anim-base,
jakobbrggen/taaf-anim-arc-agi-3-solver. Sources are pulled into `SCRATCH/web/kaggle_nb/*.py`.

### 2.4 What the community measured (mostly negatives, all on Duck + Qwen)
- **Model upgrades moved the LB most.** Qwen3.6-27B → Qwen3.8-27B gave "a consistent 2x" on the
  public 25 (/735243). Qwen3.8-27B → Flash-Next NVFP4 is now the standard. DeepSeek-V4-Flash needs
  heavy quantisation and expert pruning to fit, has worse token efficiency and slow decode, and its
  vision doesn't work (/742788).
- **Tokens are the currency, not actions.** Every Taaf-Anim run hit the 132-min per-game wall-clock
  cap. +17% tokens per action gave proportionally fewer actions (/734369).
- The no-op guard (block repeating an exact (level, state, action) that changed nothing) saved 12–20%
  of actions. **Exposing animation frames** helps, but only via compact metadata; the small model
  rarely calls the animation tool at the right time (/734369).
- gedouluhui P3.1, hidden 1.12 → 1.43 (+28%): (a) deterministic connected-component **click-candidate
  hints** (centroids ranked by salience), and (b) putting the **true scoring rule plus
  `time_remaining_seconds` pacing** in the prompt. **Rejected**: 64k context (throughput collapse, 283
  → 195 tok/s), **T=0.3** (games with any progress fell from 12 to 8; keep T≈0.6), and an archetype
  playbook (premature commitment) (/743060).
- Serguei Makarov's controlled study: consensus sampling, a reasoning-discipline block, higher-res
  images, and an executable world model with replay validation plus search planning **all ≤ baseline**.
  The small model ignored the world-model scaffold, and its simulators didn't beat "nothing changes".
  Stuck and successful runs diverge within the first few moves of a level: early goal judgement is the
  failure, the "lottery effect" (/743723).
- Fususu (#9 team): Gemma-4-31B sees better but codes much worse than Qwen. A deterministic scene
  decoder lets gpt-oss play but adds bias. Multi-role Observer/Theorist/Actor/Advisor improves
  reasoning at 5× the calls. Thinking on vs off makes little difference. Beyond **16 concurrent
  requests** throughput saturates. Imitation from human play doesn't beat plain Qwen. A strategy
  library adds game-type bias (/739938).
- Rakha: pinned, validated serving gave more than prompt work. Recovery should be single-step and
  verified, not an unconditional recovery agent. Memory should hold **verified transition facts** only.
  Multi-role only when gated. A progress-weighted budget allocator gave about 13.6% counterfactual
  action savings. Treat RL as online bandit-style control (/739938).
- Fine-tuning:
  - STaR / rejection sampling on the harness's **own winning trajectories**, LoRA on Qwen3.6-27B: LB
    1.25 → 1.94. CC0 data at `justforgags/arc3-sft-trajectories`. Scaling the data naively hurt
    (/739047).
  - Scott Le Grand: behaviour cloning on same-model reasoning traces helped (public 11.0 → 13.0).
  - Flash-Next training needs an H200-class node (/742835).
- Public-to-hidden correlation is weak and noisy. For example, public 22.3 → LB 5.37 while public
  17.3 → LB 6.91. The hidden games are harder and out of distribution (tech report §3.6).

---

## 3. Frontier model results & traces

### 3.1 ARC verified leaderboard, ARC-AGI-3 semi-private (55 games)
Data: https://arcprize.org/media/data/leaderboard/v3.json (generated 2026-09-24), saved as
`SCRATCH/web/lb_v3.json`.

| Model / harness | Score | Cost |
|---|---|---|
| GPT-6 Astra — Provider Adapter (High) | **99.95%** | $18.8k |
| GPT-6 Astra — Provider Adapter (Max) | 98.6% | $17.3k |
| GPT-6 Astra (Max), Standard harness | **62.7%** | $26.1k |
| Gemini 3.8 Flash — Provider Adapter (High) | 35.0% | $4.5k |
| Claude Opus 5 (High) | 30.2% | $20.7k |
| Gemini 3.8 Flash (High) | 10.4% | $4.4k |
| GPT-5.6 Sol (Max) | 7.8% | $25.1k |
| Grok 4.6 (XHigh) | 2.1% | — |
| Claude Opus 4.8 (High) | 1.5% | — |
| GPT-6 Luna — PA (Max) | 0.59% | $237 |
| GPT-5.5 (High) / Opus 4.7 / Gemini 3.1 Pro / GPT-5.4 | 0.43 / 0.18 / 0.42 / 0.21% | — |

- The Standard harness is minimal. Its system prompt is literally "You are playing a game. Your goal
  is to win. Include any context you want to carry forward in your reply…". The model carries its own
  notes (https://github.com/arcprize/arc-agi-3-benchmarking, `benchmarking/agent.py`).
- The Provider Adapter harness preserves opaque reasoning state and compacts
  (https://arcprize.org/blog/astra). It was 3.66× faster and used 49% fewer tokens. Astra used fewer
  actions than the median human on 96% of levels, and 51.7% fewer per level.
- OpenAI, "How enabling two settings tripled our scores"
  (https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/ ; summary at
  https://officechai.com/ai/openai-says-gpt-5-6s-score-on-arc-agi-3-tripled-after-turning-on-two-api-settings/):
  **retained reasoning** plus **compaction** (summarise instead of truncate) took GPT-5.6 Sol from
  13.3% to 38.3% on the public set, with **~6× fewer output tokens per game**.
- ARC's GPT-5.5/Opus 4.7 failure analysis (https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis)
  names three failure modes: (1) a true local effect wrapped in a false world model; (2) mapping the
  game onto a known one (Tetris, Sokoban, Frogger, Breakout…); (3) "solved the level, didn't learn the
  game". The fix is to verify *why* a level was won before level 2.

### 3.2 Public-demo (25 games, 183 levels) replays, from ARC's session API [V]
Mean RHAE over the 25 public games, recomputed from 475 session records
(`SCRATCH/web/replays/sessions_summary.json`):

| Config | Mean | Levels | Actions |
|---|---|---|---|
| Astra PA Max / XHigh / Medium | 100.0 | 183 | 6,485 / 6,817 / 7,021 |
| Astra Standard Max | 68.3 | 137 | 10,904 |
| Gemini 3.8 Flash PA High | 57.3 | 131 | 17,215 |
| Claude Opus 5 High (Standard) | 40.7 | 95 | 10,557 |
| Gemini 3.8 Flash Standard High | 19.3 | 65 | 11,848 |

The human baseline total is 17,135 actions over 183 levels. Per-game baselines are in
`SCRATCH/web/replays/public_baselines_from_replays.json`.

### 3.3 Community leaderboard, public 25 (https://arcprize.org/leaderboard/community)
Scores are self-reported.

| Harness | RHAE | Cost | Mechanism |
|---|---|---|---|
| Tycho (NIMI) | 100 (Opus 5 / GPT-5.6 Sol) | $2,986 | Deterministic Moore-machine framing. The actor delegates an executable model (State/transition/render/outcome) to a builder. Animation, level-complete and game-over frames are separated from actionable frames (arXiv 2607.28287) |
| Retrodict (R. Brown) | 99.9 | $654 | Every frame is logged. Hypotheses must **retrodict the log** before a live action. Each planned action carries an **expected-cells check**, and the runner halts on the first mismatch. playbook.md survives context resets. Escalation after 300 actions or 2 RESETs on one level. A vision model "primes" the first move. HUD/timer guidance comes from Duck (https://github.com/ryanbbrown/Retrodict) |
| Kepler 1.0 | 100 (Opus 5) | $778 | `world_model.py` certified against the full history. It plans inside the model; a guarded channel voids the plan on the first misprediction (HF `cveinnt/kepler-arc-agi-3-traces`) |
| baseline1 (Rodionov) | 58.1 → 99 | $400–2.7k | Executable Python world model, verifier, MDL-style simplification. GPT-5.6-sol textual-only also completes everything at max effort (arXiv 2605.05138, 2607.15439) |
| Twin | 93.3 | — | "Digital twin". No action until the program reproduces every observed transition; counterexample repair. "Building a usable world model is simpler than anticipated; **the harder problem is inferring the right goal**" (arXiv 2608.14490) |
| NVIDIA AVO | 100 (Opus 5, 6,624 actions) | — | A general long-horizon coding agent with persistent memory and a supervisor that intervenes on stagnation. **Text-only exact 64×64 grid** input (https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-…) |
| VISTA (MIT, Kaiming He) | 100 (Opus 5, 7,542 actions) | — | 512×512 PNG (8× nearest plus grid lines), a zoom-region tool, lossless visual memory, free-form language reasoning (https://vista-research.github.io/) |
| OPINE-World | 78.4 | $1,040 | CEGIS: an actor plus a model-writer, with "ontology-error" guided exploration (arXiv 2607.01531) |
| NOOA (NVIDIA) | 85.1 | $332 | CodeAct plus NumPy world-model helpers (arXiv 2607.20709) |
| TELL (rednote) | 43.9 | — | A single conversation that compounds confirmed knowledge in MEMORY.md |
| DreamTeam (NVIDIA) | 38.1 | $18k | 6 roles and "workspace optimisation" (arXiv 2605.09650) |
| Polyphony (Mininglamp) | 19.8 | $115 | A **27B** coding agent grows a per-game heuristic system as Python files |
| Prime Agent (PrimeIntellect) | 95.5 | ~$944 | RLM-style persistent IPython plus a continual harness (arXiv 2608.23552) |

### 3.4 Where to download traces [V]
- **ARC replay API (no key needed):**
  - `GET https://arcprize.org/api/sessions/{guid}` returns the scorecard: per-level actions and
    **per-level baselines**, resets and tags.
  - `GET https://arcprize.org/api/recordings/{game_id}/{guid}` returns JSONL of every step: frame(s),
    action, and `reasoning` holding the model's visible output/notes plus token usage. Hidden CoT is
    not included.
  - The GUIDs are the `/replay/<guid>` links on https://arcprize.org/results/openai-gpt-6-astra (also
    `/results/anthropic-claude-opus-5`, `/results/google-gemini-3-8-flash`). Parsed index:
    `SCRATCH/web/replays/replay_index.json`.
  - Downloaded: all 25 Astra-PA-Max recordings (340 MB) in
    `SCRATCH/web/replays/recordings_astra_pa_max/`.
  - Human top-10 replays per public game are linked from https://arcprize.org/blog/arc-agi-3-human-dataset .
- Hugging Face:
  - `schema-harness/arc-agi-3-schema-traces`: 50 trajectories (GPT-5.6 Sol, Opus 4.8/Fable 5) plus a
    scorer.
  - `guanning/arc-agi-3-schema-traces-gpt56` and `-opus48`.
  - `AgentNativeResearchLab/arc-agi3-{codex-gpt5.5,codex-gpt5.6sol,cc-opus4.8,cc-fable5,kimi-k2.7,cc-glm5.2,agy-gemini3.1pro,grok-4.5}-{game}`:
    many per-game agent logs.
  - `cveinnt/kepler-arc-agi-3-traces`.
  - `magic-sword/arc_agi_3_public_demo_human_testing`: 340 human sessions.
  - `fredericowieser/arc-agi-3-wm-traces`.
  - Kaggle: `justforgags/arc3-sft-trajectories` (Duck + Qwen winning trajectories, CC0) and
    `yw8837/arc-agi-3-run-history-300-game-diagnostics`.
  - Caveat: training on closed-model outputs may conflict with provider terms, and the ARC rules
    require open models and weights.
- Extra games for held-out local validation (public-25 overfits):
  - https://github.com/sonpham-org/arc-3 (arc3.sonpham.net): 927 games, including a reviewed
    "arena" set of 50 and 29 custom games.
  - https://github.com/theredbluepill/arc-interactive (252 games).
  - The ARC3.Games player (/693213).

---

## 4. Model choices for 96 GB offline (RTX PRO 6000 Blackwell, SM120, g4-standard-48; 9 h; no internet)

| Model | Size / active | Modality | 96 GB fit | Notes and evidence |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next** (08-26) | 125B MoE, 6B active + **51B n-gram PLE** + 4B MTP; hybrid Gated-DeltaNet + sparse attention; 262K ctx | text+image+video | NVFP4 checkpoint about 135 GB total. Weights take about 82 GiB on the GPU, with the **PLE (~47.7 GiB) offloaded to host RAM** (`VLLM_PLE_CPU_OFFLOAD=1`) | The current Kaggle meta. Model cards: https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 ; vLLM recipe https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next ; community NVFP4 builds RadixArk/…, keithtyser Kaggle model. Thinking mode by default; `reasoning_effort` ∈ {xhigh, medium, low}; `preserve_thinking=True` by default. Recommended sampling: T=1.0, top_p 0.95, top_k 20 (thinking). |
| **Qwen3.8-27B** (08-14) | 27.8B dense (48 GDN + 16 full-attention layers) | text+image | FP8 about 31 GB, leaving lots of KV | 2× over Qwen3.6-27B on public (/735243). ~283 tok/s decode at 4k, prefill-bound at 32k (/743060). Pennyroyal FP8 + **DFlash2** speculation: 108 tok/s C1, 375 tok/s C4. The **only practical fine-tuning target** on one card. |
| Qwen3.6-27B | 27B dense | text+image | FP8 | The June milestone model; superseded. |
| Gemma-4-31B / 26B-A4B | 30.7B dense / 25B-A3.8B | text+image | BF16/FP8 fits | Better vision, much worse at code and logic than Qwen (/739938). Used by M1 #2/#3. |
| gpt-oss-120b / 20b | 117B-A5B MXFP4 (~63 GB) | text only | fits | Official ARC template (https://www.kaggle.com/code/gregkamradt/arc-agi-3-gpt-oss-120b). Needs a scene decoder; harmony-parser 500s about 8% (/738599). |
| Nemotron 3 Nano 30B-A3B | 30B-A3B | text | fits | Being tested by one team as an ablation (/739938). |
| DeepSeek-V4(.1)-Flash | ~284B MoE | text (+exp. vision) | only with pruning | Not competitive under these constraints (/742788, /703879). |
| GLM-5.3-Flash | 320B-A18B | text | too big | Summer's best value model (https://fastino.ai/blog/best-open-weight-models-2026), but too large here. |
| Qwen 4 27B | announced 09-22, **no weights yet** | — | — | Watch for it. A drop before 11-02 would change the meta (https://www.orcarouter.ai/blog/qwen-4-max-lineup-announced-apsara-2026). |

**Serving facts that matter:**
- Public notebook profile: vLLM 0.19 wheelhouse, NVFP4, MTP=3, 16–32k context, KV **5 GiB**, 8–16
  seqs, **prefix caching OFF**, 28 concurrent games, 7,920 s per game.
- Our 09-22 run under that profile (`SCRATCH/results/vllm-*`):
  - 1,045 requests, 20.5M prompt tokens vs 1.8M generated (11:1, so prefill-heavy).
  - **Queue time 152k of 175.5k s total latency (87%)**; 342 preemptions.
  - MTP acceptance 59.7%; about 250 tok/s aggregate decode.
- Pennyroyal (SGLang fork for SM120) on one RTX PRO 6000:
  - Flash-Next NVFP4: 182 tok/s C1, **446 tok/s C4**, 632 tok/s C8, 14.8k tok/s cold prefill at 64K.
  - **824K-token KV pool** with FP8 KV and radix/Mamba prefix caching; PLE in host RAM.
  - Optional online FP8: +28% decode.
  - Sources: https://github.com/jpezzulli/sglang-rtxpro6000 (branch `pennyroyal-main-sm120-final`),
    https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722 ,
    https://github.com/keplerzip/Qwen3.8-Flash-Next-NVFP4-RTX-PRO-6000-Single .
  - Caveats: needs an offline wheel or container build and a license check. The text-only profile
    drops the vision tower. The qwen3_coder tool parser has a false-tool-call bug (fix:
    sgl-project/sglang#38624).
- vLLM recipe: on H100 the MTP acceptance was about 36% and MTP made throughput *worse*. Measure on
  our own traffic.

---

## 5. Algorithmic (non-LLM) approaches — what they buy and where they fail

| Technique | Evidence | Use for us |
|---|---|---|
| **Frame hashing plus state graph with BFS to the frontier** | Rudakov (2512.24156), Blind Squirrel, and ARC's own validator (hash nodes; P(random win) ≤ 1e-4 per level, tech report §3.5) | Deterministic `goto(state)` / "replay the path back to a known state" tool. Loop detection. Recovery after GAME_OVER by replaying the prefix of the successful path. |
| **Volatility-masked state key** (mask cells or rows changing in ≥20–40% of frames: timers, energy bars) | BDR-Pro (https://github.com/BDR-Pro/arc-prize-2026-arc-agi-3) | Needed so hashing and no-op detection ignore HUD bars. Duck prompts already treat edge strips as timers. |
| **Connected-component segmentation, then click targets** (centroids, priority tiers by size, shape, rare colour; one click per cell given the camera scale) | Rudakov tiers, Reki's rare-colour button heuristic plus "dead-signature" suppression, gedouluhui hints (+28% hidden) | Give the LLM a ranked candidate list. Dedupe clicks by object and cell. Auto-suppress object types whose clicks never change anything. |
| **Action-effect learning** (P(change \| s, a); avatar detection by consistent (colour, action) → (dx, dy)) | StochasticGoose CNN; BDR-Pro avatar model | Cheap online stats: which actions move what, which are no-ops. Feed them to the model as verified facts. |
| No-op / death memory | Taaf-Anim guard (−12–20% actions), BDR-Pro | Block the exact repeat for free (no env action). Ban transitions known to kill. |
| Novelty / Go-Explore (return to a frontier state, then explore) | Rudakov (BFS to the nearest untested edge), Blind Squirrel | Useful for navigation levels with small state spaces. Useless on long-horizon logic levels. |
| Level replay / macro reuse | Retrodict and Tycho plan-and-verify; Kepler guarded channel | After GAME_OVER or a level reset, re-execute the verified prefix automatically instead of re-deriving it with LLM turns. |
| Program-synthesis world models | Tycho, Kepler, Twin, baseline1, OPINE | They work with frontier models. With 27B/6B-active models the scaffold was ignored or unhelpful (/743723). At most, keep it as an optional tool. |
| Pure programmatic agent | BDR-Pro plateaued at **0.26–0.27 LB**; the official random agent scores about 0.27 | Not a driver on the hidden set. |

---

## 6. Ideas ranked by expected impact for our submission
Context: an AGENTFIX Duck fork on Flash-Next NVFP4, LB 3.85, target 18–20.

1. **Fix serving throughput and queueing (highest EV, low research risk).** We are compute-bound: 87%
   of request time is queued, with preemptions and about 42 turns per game.
   - (a) Right-size KV and seqs: PLE offload is already on, so raise `--kv-cache-memory-bytes` from
     5 GiB as far as VRAM allows, try an FP8 KV cache, and align max-num-seqs with game concurrency.
   - (b) Enable prefix caching if vLLM supports it for this hybrid-attention model, or else evaluate
     the Pennyroyal SGLang build (radix + Mamba-state cache, 446–632 tok/s aggregate).
   - (c) Tune MTP tokens (2 vs 3) on real traffic.
   - (d) Cut prompt tokens per turn: we average about 19.6k prompt tokens per request.
   - Expect 2–3× more turns per game. The community hints the top jump came from "squeezing more
     tokens out of the hardware and allocating them properly" (/742801). [S]
2. **Retained reasoning plus compaction instead of blind eviction.** This gave 3× for GPT-5.6 Sol and
   62.7 → 99.9 for Astra.
   - Keep `<think>` from the last K assistant turns (Flash-Next supports `preserve_thinking`).
   - Replace oldest-message eviction with a model-written, **verified-facts** summary: controls, rules
     confirmed by evidence, goal hypothesis, the current plan.
   - Needs item 1 for KV headroom.
3. **Plan-with-expectations action channel.** The model submits a batch of actions plus predicted
   effects (cells or objects). The runner executes step by step and **halts on the first mismatch**,
   returning a diff (Retrodict, Kepler, Tycho).
   - Together with the no-op guard and death bans, this cuts wasted actions (quadratic payoff) and
     wasted LLM turns.
   - Also auto-replay the verified prefix after GAME_OVER or level reset.
4. **Deterministic perception pack in every observation.**
   - A per-step diff line; a changed-object list; the avatar and its moves; ranked click candidates
     (one per cell at the camera scale); dead-signature suppression; HUD/timer masking.
   - Compact animation metadata: frame count, transient bbox.
   - Evidence: +28% hidden (/743060); ARC failure mode #1.
5. **Prompt hygiene with measured wins.**
   - The true scoring rule plus `time_remaining` pacing (+28%).
   - ACTION7 named as UNDO and exposed where available (/742477).
   - RESET discipline: never reset at the start of a level.
   - "Verify *why* you won before the next level" (ARC failure mode #3).
   - "Don't map to known video games" (failure mode #2).
   - Keep T≈0.6–1.0; keep high `reasoning_effort` (xhigh beat low, /735243).
6. **Compute allocation across the 110 games.** Unfinished-level actions are free, so the only question
   is wall-clock. Use a progress-aware scheduler: extra time for games that just cleared a level, less
   for games stagnant for N turns. Mind the risk that this overfits public (/740812).
7. **Tools, not drivers, from algorithmic methods.** Offer `goto`/BFS over the hashed state graph for
   navigation games, `try_clickables()` over ranked unseen objects, and a no-op-free exploration sweep
   for the first few actions of a level (budget about 0.3–0.5× baseline).
8. **Variance management.**
   - Seed per request from (game, level, step).
   - Measure configs as the mean of ≥3 public passes plus held-out extra games (§3.4).
   - Choose the 2 final submissions for robustness: both are already scored on the 55 private games at
     submission time.
9. **Fine-tuning (only if items 1–5 plateau).** STaR/rejection-sampling LoRA on our own winning
   Flash-Next or 27B trajectories (/739047: 1.25 → 1.94). Flash-Next is not trainable on one RTX PRO
   6000. Switching to a fine-tuned 27B trades base capability for adaptation. High cost, uncertain.
10. **Watch for Qwen 4 27B weights.** An open 27B from the Flash-Next architecture before 11-02 would
    reset the meta.

Not worth doing, per community evidence: a 64k context, T=0.3, archetype or strategy playbooks,
consensus voting on every step, always-on multi-agent roles, higher-resolution images at more tokens,
a mandatory executable world model for small models, pure BFS/CNN agents, training on public-game
solutions.

---

## 7. Raw files saved (SCRATCH/web/)
- `docs/*.md`: 21 docs.arcprize.org pages (actions, scorecards, recordings, competition_mode,
  changelog, arc-prize-2026, …). `docs_methodology.*`, `docs_llms.txt`.
- `ARC_AGI_3_Technical_Report.{pdf,txt}`, `blog_*.{html,txt}` (launch, milestone-1, astra,
  gpt55/opus47, human dataset, preview 30-day), `comp_2026*.txt`, `leaderboard*.txt`, `lb_v3.json`,
  `leaderboard_data.js`, `results_*.txt`.
- `papers/`: 16 arXiv PDFs with text (2512.24156, 2601.10904, 2603.17683, 2605.05138, 2605.09650,
  2605.13037, 2605.25931, 2607.01531, 2607.15439, 2607.20064, 2607.20709, 2607.28287, 2608.04066,
  2608.14490, 2608.23552, 2609.21032).
- `kaggle_disc/`: all 225 competition discussion topics with comments (`<id>_<title>.md`), plus the
  competition pages (`page_rules.md`, `page_Evaluation.md`, `page_data-description.md`,
  `page_Prizes.md`, `page_Timeline.md`, …).
- `kaggle_nb/`: pulled public notebooks (Scott Le Grand AGENTFIX, keithtyser, wuliao0, thui,
  yanggod). `kaggle_lb.json`, `lb_history_arc3.json`, `kernels_*.json`.
- `replays/`: `sessions/` (475 frontier session records), `sessions_summary.json`, `replay_index.json`,
  `public_baselines_from_replays.json`, `recordings_astra_pa_max/` (25 full JSONL recordings).
- `models/`: Flash-Next and 27B cards, the vLLM recipe, the Pennyroyal and keplerzip READMEs, the
  NVIDIA forum thread, the model landscape pages. `hf/`: dataset cards.
- `repos/`: arcprize/arc-agi-3-benchmarking (clone), tonghuikang/arc3 (clone), READMEs of Retrodict,
  Tycho, Kepler, BDR-Pro, leejianrong.
- `wheels/`: unpacked arc_agi 0.9.8 and arcengine 0.9.3 (scorer and engine source).
- Symbolica, VISTA, NVIDIA AVO, Tufa blog, StochasticGoose and Blind Squirrel READMEs,
  huikang_autoresearch.
