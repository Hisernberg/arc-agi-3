# 11 — Kaggle forum intel: ARC Prize 2026 ARC-AGI-3 (plus relevant ARC-AGI-2 / Paper Track threads)

Collected 2026-09-27 (~08:30 UTC). Coverage: **all 225 topics / 1,144 messages** (topic starters + every comment and nested reply) of the `arc-prize-2026-arc-agi-3` forum (forumId 10403401), plus all 79 topics / 464 messages of the ARC-AGI-2 forum and 14 topics / 27 messages of the Paper Track forum, which were screened for ARC-AGI-3 content. The completeness check passed: for every topic the number of comments fetched equals Kaggle's `totalMessages`.

**Where the data came from.** Meta Kaggle (snapshot 2026-09-26) does **not** contain this forum: `ForumTopics.csv` has 0 rows for forumId 10403401, because active-competition forums are excluded. Every message was therefore pulled through Kaggle's internal web API, authenticated with a Bearer token:
- `discussions.DiscussionsService/GetTopicListByForumId` with `{forumId, page}`, 20 topics per page.
- `GetForumTopicById` with `{forumTopicId, includeComments:true}`. The response includes `rawMarkdown` for the post and for every nested reply.

Raw data lives in `SCRATCH/kaggle/discussions/`:
- `topics.jsonl` and `messages.jsonl`: every message with author, tier, LB rank at posting time, date, votes, markdown and html.
- `raw_<forum>/<topic_id>.json`: the raw API responses.
- `dump_arc-agi-3.txt`: a chronological readable dump of the whole forum.
- `comp_rules_arc3.txt` and `comp_overview_arc3.md`: the rules and overview text, taken from Meta Kaggle `Competitions.csv`.
- `huikang_history.json` and `lb_progression_top40.txt`: the full public-LB history with per-day runtime, pulled from Tong Hui Kang's community tracker `https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run`.
- `linked_nb/scott/`: the public 5.19 notebook by Scott Le Grand. The user's notebook descends from it (same AGENTFIX block and the same `kv5-bf16-mtp3-c16-cg32-ctx16k` profile).

Conventions used below:
- Scores are **percentage points**. LB "5.19" means 5.19%, and the maximum is 100.
- "Host" means ARC Prize staff: Greg Kamradt (President, ARC Prize Foundation) and fchollet.
- "Kaggle staff" means María Cruz, Addison Howard, inversion (Walter Reade), Dustin, LucyHe2 and herbison.

---

## 0. TL;DR — the 15 most decision-relevant facts

1. **Test set structure.**
   - One submission plays **all 110 hidden games**. 55 are semi-private (public LB) and 55 are private.
   - The **private score is computed during that same run and never re-run**. At the end you pick 2 submissions whose private scores already exist (Greg, 729985; CPMP, 697944).
   - Public demo games are **easier by design** than semi-private, and private is "significantly more difficult … intentionally out-of-distribution" (Greg, 703990/693128).
2. **Exact metric (from `arc_agi/scorecard.py`, verified to 1e-9 by two independent posters, 728299/743060):**
   - `level = min(100*(baseline/actions)^2, 115)`.
   - `game = min(Σ i·level_i / Σ i, 100·Σ_{completed} i / Σ i)`. The weight is the 1-indexed level number, and an unfinished game is capped at its completed weight share.
   - LB = plain mean over games.
   - So **depth beats efficiency**: clearing only L1 of a 6-level game is worth at most 4.76, and a later level's weight counts twice (once in the weighted mean, once via the cap).
3. **Every action counts, and counting never resets.**
   - Every ACTION1–7 counts, **including no-op/dead clicks** (confirmed by Greg: the scored server matches offline competition mode, 718638).
   - **RESET counts** in the scorecard.
   - Actions accumulate across all attempts of a level. GAME_OVER then RESET continues the same level, and the action count never resets.
   - Resetting twice to restart the whole game is **disabled** in competition mode.
   - `make()` may be called only once per game, and a game cannot be played in parallel instances.
4. **Budgets.**
   - There is **no action cap on Kaggle**. The 5×-human cut-off exists only on ARC's verified leaderboard (Greg, 713921).
   - There are no rate limits.
   - The only budgets are 9 h of wall clock (32,400 s) on 1× RTX PRO 6000 96 GB (g4-standard-48) with internet off, and a **15-minute inactivity kill**.
   - "Kaggle Error" usually means the notebook crashed or took more than about 15–20 min before its first game action.
5. **Variance is enormous.**
   - Tufa: identical notebook scored 0.77 to 1.30 on the LB; public set 1.60 ± 0.45.
   - yw8837: 11 identical submissions scored 0.55 to 1.29.
   - Fususu: "2.5–5.5 same code".
   - Two identical passes on the 25 public games scored 2.85 vs 4.75.
   - Per-game outcomes are bimodal (a mechanic is either understood early or the game yields about 0). 84.7% of the variance comes from whether a level is cleared at all (borro1980).
   - **Local-to-LB conversion is about 3–4× down.** Reported pairs (public local → LB): 17.3→5.2 or 6.9, 22.3→5.4, 15.7→7.4, 11.0→2.7, 12→3.2, 10.2→3.1, 7.6→3.4, 9.9→6.2.
   - Tuning purely to raise public-set depth (e.g. more time for games already at L2+) **did not transfer** (Scott, 740812).
6. **Model timeline, which drove all the big LB jumps.**
   - Qwen3.6-27B-FP8 (Duck, Milestone 1, 1.21).
   - Gemma-4-31B (Milestone 1 2nd and 3rd places, about 0.8–0.9).
   - **Qwen3.8-27B FP8, released 2026-08-14.** This roughly doubled local scores and moved the top from 1.86 to 2.x within 12 h.
   - **Qwen3.8-Flash-Next NVFP4** (125B MoE, about 6B active, 262k ctx, fits the card only with FP4 plus host-offloaded embeddings; first forum mention 2026-09-07). This is what current public notebooks and most top teams use.
   - Reasoning effort **xhigh beats low**, and temperature 0.6 beats 0.3.
   - DeepSeek-V4-Flash was too slow once quantised. GPT-OSS-120B works (config below). Fine-tuning Flash-Next needs H200-class nodes; the RTX PRO 6000 cannot train a ~100B model.
7. **The Duck harness (Tufa, Milestone 1 winner) is the ancestor of every high-scoring public notebook.**
   - Python REPL tool, game exposed as variables, a 4× upscaled image each turn, an ASCII grid, and a connected-component segmentation tool.
   - A carried-over "World model:" note.
   - Oldest-message eviction at 32k input (64k max).
   - Details in §2.1.
8. **Documented harness wins, all small and noisy:**
   - Put the true scoring formula plus `time_remaining` pacing into the prompt: 1.12 → 1.43.
   - Deterministic connected-component **click candidates**.
   - A repeated-no-op guard.
   - Mapping ACTION7 as "UNDO" (the docs call it "Simple undo action").
   - Forcing world-model updates into *visible* output. 66.8% of Qwen tool-call turns had hidden reasoning only, so the updates were lost.
   - The AGENTFIX set (images only in the current prompt, always return the action result, tolerant memory parsing, no wipe on game-over, HUD-band masking).
   - vLLM concurrency of about 16 sequences, with 28 games in parallel.
9. **Documented losers:**
   - 64k context (throughput collapse, 1.20).
   - Temperature 0.3 (0.95).
   - Archetype/strategy playbooks (1.54, premature commitment).
   - Multi-role LLM pipelines (5× tokens).
   - Consensus sampling.
   - Executable world-model scaffolds (the model ignores them).
   - Higher-res images, if they cost too many tokens.
   - Unconditional recovery agents.
   - Removing the soft deadline (1.33 vs 1.66).
   - **Tokens per action are the real currency.** Every run is wall-clock bound, not action bound.
10. **Fine-tuning evidence.**
    - STaR-style LoRA on the harness's *own* winning trajectories (Qwen3.6) took LB 1.25 → 1.94 (manas joshi). Curation beats quantity, and the gain is base-model specific.
    - Scott: behaviour cloning on the same model's own reasoning traces took local 11.04 → 12.97.
    - Training on demo-game *solutions* backfires. Chollet: the 25 are a "demonstration set", not training data.
    - Selective down-weighting of reasoning tokens produced a "no tool call" collapse.
11. **Leaderboard dynamics.**
    - Top on 2026-09-26: Tufa Labs **27.29**, Lord Han Solo 20.80, Tong Hui Kang 20.53, Yi-Chia Chen 18.80, Daniel Franzen 16.68, NVARC3 (NVIDIA: CPMP, Darragh …) 16.07.
    - Tufa went 4.71 (08-30) → 11.04 (09-06) → 18.81 (09-13) → 27.29 (09-26).
    - All top teams run the full ~9 h (runtimes 530–550 min).
    - Only 9 teams are ≥10, 24 are ≥7, 54 are ≥5 and 187 are ≥4. The user's 3.85 is roughly rank 190–240.
12. **Milestone 2 (2026-09-30 23:59 UTC).** It is based on the **public LB** snapshot, and the prize walks down the ladder to the highest team that open-sources in time.
    - **Tufa will not open-source** for Milestone 2; they will release at the end.
    - **NVIDIA will not** either; they release at the end if they finish in gold.
    - Tong Hui Kang will share only if 1st.
    - Lord Han Solo will share if top-3 among teams willing to share.
    - Scott Le Grand already published his 5.19 notebook (`scottlegrand/taaf-flashnext-sheetu12b-0922`).
13. **Rules and infrastructure.**
    - 1 submission per day. Failed "Error" submissions reportedly do not consume the daily slot.
    - Submission GPU time does not use your weekly quota, but you need quota to create the version.
    - Team-merge and entry deadline is 2026-10-26, and notebook publishing is disabled after 2026-10-26.
    - Winner licence is CC-BY 4.0, and the rules also require an "open source model + weights" per OSAID. Whether Qwen qualifies was asked but is still unanswered.
    - Coding assistants are allowed. Self-generated synthetic data has not historically counted as "external data".
14. **Hidden-set API facts.**
    - The grid is always 64×64.
    - `available_actions` is constant within a game.
    - Games are deterministic with fixed seeds (whether seeds are identical across *submissions* is unanswered).
    - A response can contain **multiple animation frames**; 13 of the 25 public games return multi-frame responses (sp80: 22 frames).
    - Game source files are **not** available during the rerun. Only the 25 public games sit in `environment_files`.
    - `environment_info.baseline_actions` was visible in April (Jeroen), but in submission mode it "can come back empty" (Sept).
15. **The frontier "solved" ARC-AGI-3 with money, not with this box.**
    - Claude Opus 5: 30% (2026-07-24).
    - GPT-5.6 Sol: 7.8%.
    - VISTA / Schema harness with Opus 5: 99%. The model writes a Python world model and a sandbox verifies it, and the 64→512 px upscale is key.
    - NVIDIA AVO: 100%.
    - OpenAI Astra: 99% on the 110-game set at $17.3k of compute.
    - None of this runs on one RTX PRO 6000 in 9 h. The Kaggle top is 27%.

---

## 1. Official / organizer statements (rules, scoring, runtime, hardware, resets, ACTION7)

### 1.1 Scoring formula and scale
- **Scale.** Scores are percentages; 0.25 on the LB means 0.25%. The Kaggle page never says so explicitly, but Jeroen (Tufa) and Greg confirm it: "The high score is 100, not 1" (716696) and "Per level you can score up to 1.15x. This gets capped at 100% per game" (Greg, 705022).
- **Per-level cap moved from 1.0 to 1.15, and the human baseline changed.** Methodology page and blog `arcprize.org/blog/arc-agi-3-human-dataset`. The update also moved the human baseline from "second-best" human to (upper) median (Hendrik Nowak, 684625). The Kaggle data page still states `min(h/a,1)^2`, but the shipped scorer uses the 115 cap.
- **Exact shipped scorer** (Akmal Xodarev 728299; gedouluhui 743060, verified on real run logs):
  - `level_score = min((baseline_actions/actions_taken)^2 * 100, 115)`; an uncompleted level scores 0.
  - `weighted = Σ level_score[i]*i / Σ i`, `cap = Σ_{completed} i / Σ i * 100`, `game = min(weighted, cap)`.
  - Worked examples: cd82 (6 levels, baselines [55,8,41,21,23,23]). Matching every baseline gives 100. Twice the baseline everywhere gives 25. Clearing 4/6 perfectly gives **47.62, not 66.7**. su15 with only L1 of 9 at 100 gives 2.22. tu93 with 69 actions vs a baseline of 19 gives ≈0.17.
  - Consequence: "One cracked level ≈ 2–5 points per game; a game is either understood or worth zero."
- **What counts as an action.**
  - Internal operations such as tool calls, reasoning and retries are free.
  - Every ACTION1–7 costs one action, including a **frame-unchanged ACTION6 dead click**. Alvaro Camacho De Pass measured 100 dead clicks = 100 actions on lp85, and Greg confirmed "Yes, they match" for the scored server (718638).
  - RESET counts in the scorecard. The engine's internal *budget* counter excludes it, but the scorecard counts it.
  - The scorecard sums across runs, so there is no way to null out earlier actions (vialactea, 692135).
- **Viktor Korsun's scenario (738766)** was "explore, reset, then play perfectly" scoring 0.25 then 0.09 on a byte-identical rerun. It does **not** earn 100: exploration actions on that level are counted.

### 1.2 Test set, runs, leaderboards
- "Submissions run on all 110 tasks" (inversion, 684852). "Yes, the submission plays both datasets. Only the 50% of public tasks are shown" (Greg, 729985). `leaderboardPercentage=50` per the competition API.
- "Private scores are calc'd at original run time. They aren't rerun" (Greg, 729985). You choose up to 2 final submissions.
- **Milestones** use the **public** LB (Greg, 703056). Milestone 1 was a snapshot at June 30 23:59 UTC, with the notebook made public under an OSS licence by the same deadline (María, 713634).
- The 25 public games are in `environment_files` during the rerun as well, but the agent plays **110 new semi-private/private games through the gateway** (inversion, 684898).
- Public demo games "were made to be easier than semi private"; a drop from 1.56 locally to 0.05 on the LB is "in line with expectations" (Greg, 703990). The private set is "significantly more difficult for both humans and AI, and intentionally OOD relative to the public set" (tech report, quoted in 693128).
- Games were designed to resist random play: no level can be beaten by accident in 50k random steps, and non-tutorial levels survive 1M random steps (arXiv 2603.24621, quoted in 693128).

### 1.3 Game / API mechanics (host answers)
- **Grid.** Always 64×64 ("safe to assume", Greg, 707717 and 692316).
- **`available_actions`** "are the same throughout" a game. "Whether or not they are valid is a different story" (Greg, 702079).
- **Determinism.** "A seed is used for each one. Games weren't designed with randomness or proc gen" (Greg, 694153). The lf52 black-noise transition animation is unseeded (Gijs Schenk). Whether the 110 environments are identical across scored runs, and whether the 55/55 split is fixed, is **unanswered** (738762).
- **Animations.** "N frames are returned back, so the entire animation sequence is shown. Same thing which is shown to humans is given to AI" (Greg, 687449). `frame` is a *list*.
- **RESET semantics** (Greg, 693123):
  - "It restarts the game if you're on action 0 of a new level … If you take an action, then click reset, it'll send you back to the beginning of that level."
  - In competition mode, "restart game" (reset twice) is **disabled**.
  - "Humans/AI can reset a level at any time."
  - You can reset as often as you like, and the action count keeps incrementing (687187).
  - The env var `ONLY_RESET_LEVELS=true` pins level-only resets (Scott's and the user's notebooks set it).
- **GAME_OVER.** "Even after a game over you'll continue on the same level you left off at" (Greg, 693307). After GAME_OVER the only valid action is RESET.
- **One environment per game.** "You only get 1 run per env in competition mode. `make`'ing the game a 2nd time is not valid" (Greg, 693307). No parallel instances of the same game (Jeroen, 688135). Calling `env.make()` inside each worker process opens a separate scorecard per process and scores 0. Call `make` on the main thread and then parallelise (Gabriel Mirea, 686416).
- **`frame.score` vs `levels_completed`.** `score` is legacy from the preview; use `levels_completed` and `win_levels` (Greg, 692809). There is no visible "level complete" frame; detect the increment of `levels_completed` (716504).
- **`get_pixels()`** returns the level's native viewport (e.g. 8×8 for bp35), not the 64×64 player frame. Use `env.step(...).frame` (Jeroen/drozza, 707862).
- **Action semantics.** ACTION1–4 are "semantically mapped to" up/down/left/right (docs `/actions`), and the human UI draws arrows. **ACTION7** is "Additional simple action" in the overview but "**Simple undo action**" at `docs.arcprize.org/actions` (Van-Phuc Huynh, 742477). Duck forks crash with an index error on ACTION7. Muurur got his best score by naming it "UNDO", and AGENTFIX F6 says it is UNDO in 3–5 of the 6 public games that expose it. The Duck deliberately did not give the model UNDO, because it undid big batches and wasted energy (717133).
- **Partial observability exists.** ls20 level 7 has fog of war (686532).

### 1.4 Runtime, hardware, submission mechanics
- **Hardware history.**
  - At launch: P100 / 2×T4.
  - 2026-04-28: H100s were added.
  - About 2026-05-05: H100 stock-out, **replaced by RTX PRO 6000 (g4-standard-48, 96 GB)** (María, 697720).
  - The RTX instances are for ARC-AGI-3 notebooks only; using them for other work risks a ban.
  - Runtime was raised from 6 h to **9 h** on 2026-05-07 (697944). A bespoke ARC setting still capped runs at 6 h until inversion fixed it on 2026-05-19 (699208).
- **Termination / scoring trigger** (Greg, 694750): scoring happens when there were interactions **and** the notebook terminates. It terminates when (A) all games are played, (B) the runtime is used up, or (C) **the agent stops interacting for 15 minutes**. The `submission.parquet` is a dummy Kaggle requirement and is not used.
- **"Kaggle Error"** "means that your submission runs into an error before starting the games" (Jeroen, 741608). In general it is an exception, or more than 15 minutes before the first move (Jeroen, 699859). Other causes reported:
  - Heavy setup.
  - A stale kernel. A **brand-new kernel slug** fixed 7 straight failures (733697).
  - `resource.setrlimit(RLIMIT_AS)` (724841).
  - The GPU not selected (about 20% of 500 failed submissions; Greg, 727119).
  - About a third of failures had no visible error: deadlocks, infinite loops, wrong endpoint.
- **Notebook Timeout.** A random agent that ran the full 9 h got "Notebook Timeout" and no score. Adding a soft deadline at 8 h 30 m plus a 0.1 s delay per thread fixed it (Gabriel Mirea, 712719). Build in a ≥15–30 min buffer. Removing the Duck's rerun soft-deadline override **scored worse**, 1.33 vs 1.66 (739801).
- **Infrastructure limits from the Kaggle team (Greg, 724841):**
  - Exit code 139 (SIGSEGV) is captured, but details are not exposed.
  - Docker logs are capped at 10 MB. Exceeding it does not kill the run; output is just no longer captured.
  - `/kaggle/working` is capped at 20 GB and exceeding it kills the kernel. Other scratch space is about 60 GB.
  - No `RLIMIT_NPROC` and no `RLIMIT_AS`. Memory is enforced by cgroup (30 GB for CPU notebooks).
  - "spawning many threads on a 4-core CPU allocation will lead to contention".
- **Frame logs.** "We don't store the frames in memory. The frames get logged to a jsonl that is append only" (Greg, 697423). The Swarm hard-codes `record=True`, which can produce about 1 GB per heavy game; disable it (733697).
- **Swarm defaults.** `Agent.MAX_ACTIONS=80` (the loop actually runs 81) silently stops agents that don't override it (734054). The Swarm starts one thread per game, and you may re-orchestrate it freely (716295).
- **No rate limits** on Kaggle. The random agent did about 6.5 M actions when maxed (Greg, 689492 and 687187).
- **Quotas.** Submission GPU time "is not subject to user quota" (CPMP, ARC-2 694745), but you need free quota to create the version (734585). Linking Colab Pro adds about 15 GPU-hours per week.
- **Queues.** RTX queues reached 8–12 h in September, worst around 02:00–09:00 UTC. Kaggle says capacity was restored on 2026-09-22. Sometimes a finished run still shows as "queued" (a display bug).
- **Submission limit.** 1 per day. Between May 27 and June 8, 5 per day was allowed by mistake; the surplus submissions were invalidated, including Tufa's 1.30 (705405). Error submissions reportedly don't consume the daily slot (733697). The day boundary is UTC.
- **Deadlines** (from the competition API):
  - Entry and team merger: 2026-10-26.
  - Kernel publishing disabled after 2026-10-26 23:59.
  - Final submissions: 2026-11-02 23:59 UTC.
  - Winners announced: 2026-12-04.
  - Max team size: 8.
- The arc_agi used in notebooks is 0.9.8 with arcengine 0.9.3. 0.9.9 is on PyPI, and one user saw "significantly different agent performance" between the versions (728220, unanswered).

### 1.5 Rules / eligibility points raised
- **Models.** Any offline open model is allowed ("Yes, this is fine to use", Greg on Qwen3.5, 688481). Solutions need not be open-sourced until a prize is claimed. Winner licence is CC-BY 4.0.
- **Rules §2.5.a** requires an "open source system, open source model, and open source weights/parameters" per the OSAID checklist, and then gives a carve-out for "pretrained models with an incompatible license". Whether Qwen (Apache weights, closed data) is prize-eligible was asked (ARC-2 740268) and is **unanswered**.
- **Coding assistants** are allowed for development (Greg, ARC-2 701528).
- **Self-generated synthetic data** has never counted as "external data" in past competitions; NVARC published theirs after the deadline (CPMP, 742940).
- **External compute** must pass a "Reasonableness Standard". Renting an H200 node or 8×H100 for fine-tuning is common, but the question was debated (740812, 742835).
- **Team size.** Mostik.AI (12 PhDs) was accused of exceeding the 8-person limit (739186). They stopped submitting after Sep 6 and are now #17 at 7.51. NVIDIA (CPMP) stresses it uses only public NVIDIA tech.
- **Using the public game `.py` files.** Using the 25 public files is fine ("already open source"). Notebooks that deep-copy game classes gain nothing on hidden games because the source is not available there (Greg, 699900 and 707925). Benedicte claims obfuscation was added after early source-reading exploits.

---

## 2. Deep summaries of high-value threads

### 2.1 Tufa Labs — Duck harness, Milestone 1 winner (717133, 716696; 71 and 38 votes)
Authors: Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Dries Smit, Michal Tešnar, Stefano Viel.
- Code: GitHub `Tufalabs/duck-harness`. Notebooks: `jeroencottaar/taaf-duck-harness-kaggle-share` (recommended) and `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`.
- The solver ships as a dataset; the notebook is Apache 2.0 and the code dataset is MIT.

**Scores.** Kaggle 1.21. The 1.30 was voided as a surplus submission, and the same notebook ranges 0.77–1.30. The public set scored 1.60 ± 0.45 over 25 games × 20 tries. Performance is very uneven: some games clear more than 40% of levels, and some never clear L1.

**Design principles.** Use open models that fit 96 GB, chiefly Qwen3.6-27B-FP8 served with vLLM. No game-specific knowledge. Keep the harness lightweight and "the model in the driver seat". Inspired by RGB Agent (OpenCode plus a log file; near-human efficiency on the preview) and Symbolica ARCgentica (recursive sub-agents; 36% with frontier models).

**Architecture:**
- **Python REPL tool.** Each call has a 30 s limit and returns at most 4,096 characters of output, and the REPL is reset between calls.
- **Pre-loaded variables:**
  - `current_frame` (with `.ascii`, `.segmentation`, level and step), `previous_frame`, and `history` (`.action`/`.frame` pairs).
  - `transitions` (state–action–state), `result` objects (action count, level, reward, state, valid actions), `last_transition`, `last_action` and `last_action_result`.
  - `valid_actions`, plus `action(...)`, which can execute one or many actions programmatically.
  - Actions are renamed `UP/DOWN/LEFT/RIGHT/SPACE/MOUSE/RESET`, and **UNDO is not exposed**.
- **World model.** The model writes `World model:`-tagged notes, which are copied into the next user message until overwritten.
- **Context.**
  - The full system prompt is always kept.
  - Each turn is reasoning plus a python tool call.
  - After actions, a new user message carries level transitions, valid actions, whether a reset occurred, a reminder of the tool API and world-model instructions, and "collect evidence before acting".
  - **Max context 64k; the harness keeps about 32k of input by evicting the oldest user message and the assistant turns after it.**
  - Prefix caching was not used optimally.
- **Perception.**
  - A **4× upscaled image** of the board is injected at the start of each turn. Qwen's encoder uses 16×16 patches, and 4× gave the best understanding.
  - More frames or video hurt small models, so there is **no animation feedback**, which matters for sb26 and tn36.
  - A **segmentation tool** returns 4-connected components with adjacency and parent–child relations, which stops the model from dumping the full grid.
- **System prompt** carries many ARC-3-generic hints. Examples: the energy/time bar is not the goal; don't invent goals like "move block to a fixed position"; don't hallucinate classic sprites or Atari games; exact variable formats.
- **Lessons.**
  - Gains came from **better base models and multimodality**. "Hand-crafting specific tools … did not help, as it seems to hinder the creative abilities of the model."
  - Future work: compaction/memory, better perception (coding models don't reason well over ASCII crops), and post-training.

**Comments.**
- The world-model persistence bug (Jason Feng, 734843; details in §3).
- Duck + GPT-5-class models were tried on a few games only, because of cost (see the blog `tufalabs.ai/research/duck-harness`).
- Borro1980 (732932): the harness default `n_passes: 20` was produced on 2× B200 in a 13 h window. On one RTX 6000, one pass of 25 concurrent games takes about **124 min of play plus 8 min of model load at about 193 tok/s**, i.e. about 4 passes in 9 h.

### 2.2 Milestone 1 results (host post 725002)
- **1st: Tufa Duck** (above).
- **2nd: Reki** (`ruichardliu/milestone1-2nd-solution`), "vision-LLM-as-policy" with **Gemma-4-31B**:
  - Each turn renders the recent frames as labelled images and asks for one JSON object: what changed, a short plan, and the next **1–4 actions**.
  - Reflection memory refreshed about every 10 steps.
  - **Numpy click heuristic:** fallback and exploratory clicks prefer small, rare-coloured, button-like shapes over random pixels.
  - **"Dead-signature":** if clicking a *type* of object never changes anything, stop clicking that type for the rest of the level.
  - Reki says his later 0.00 re-run was an infrastructure or timeout failure, not agent randomness.
- **3rd: Md Boktiar Mahbub Murad, "forge"** (LB 0.86): also Gemma-4-31B. It adds a profile framework with a candidate generator, an arbiter and a confidence gate for safe/reversible moves. Built on `ko0kip/arc-agi-3-gemma-4-31b-reflection-agent`. **The best run had all the extra machinery switched off.**
- Also: Akhil Tolani, 0.79 (716711), Gemma-4-31B QAT with pruned vocab plus a LeWM/JEPA dynamics model; scores 0.6–0.8 "depending on luck".

### 2.3 Tufa will not open-source for Milestone 2 (742801, 31 votes) and the open-sourcing standoff (742935, 743624)
- Tufa: "we worry it may not be possible to beat the best of 4000 copies of that solution … winner simply being the luckiest." They will release at the end of the competition.
- CPMP (NVIDIA / NVARC3): will not share either, and will open-source at the end if they finish in gold. The ARC-2 experience: public notebooks became tiny variations of their solution, with a massive leaderboard cluster.
- Tong Hui Kang (currently #3, 20.53): will share only if 1st; would not share from 2nd or 3rd.
- Lord Han Solo (#2, 20.80): will share if in the top 3 among teams willing to share.
- Scott Le Grand published his notebook at **5.19** (`scottlegrand/taaf-flashnext-sheetu12b-0922`), "descended from the duck with extensive agentic and a little hand-tuning".
- Milestone prize mechanics: it moves down the ladder (Nick Pellegrin). Top teams publish 1–2 h before the deadline so copies can't finish scoring in time. A fine-tuned model must be published as a dependency, but whether the training data and script are also required is unanswered. There is also a licence question about dependencies with "License: Unknown" (743753).
- xz on Tufa's 27.29: "a lot of the improvements simply came from squeezing more tokens out of the hardware and allocating them properly. But I am not sure if 27+ is doable with the base model." Scott guesses "part of that is some fine tuning".

### 2.4 Public-LB progression and runtimes (Tong Hui Kang's tracker; 709355, 23 votes)
Each entry below is date: best score (runtime in minutes of that day's submission).

| Team (subs) | Progression |
|---|---|
| Tufa Labs (147) | 06-03 1.21 → 08-19 2.07 (Qwen3.8 week) → 08-23 4.58 → 08-30 4.71 → **09-06 11.04 → 09-13 18.81 → 09-26 27.29**, all about 540 min |
| Lord Han Solo (75, first submission 08-22) | 3.36 → 09-13 8.44 → 09-19 15.60 → 09-20 18.42 → 09-26 20.80 |
| Tong Hui Kang (85) | 08-25 3.39 → 09-08 5.13 → 09-21 10.78 → 09-24 15.02 → 09-25 20.53 |
| Yi-Chia Chen (13) | 09-12 3.24 → 09-19 12.88 → 09-24 18.63 → 18.80. **Often runs only 90–500 min.** |
| Daniel Franzen (83) | 09-05 7.63 → 09-16 11.59 → 09-24 16.68 |
| NVARC3 (22, from 09-06) | 3.32 → 09-10 7.69 → 09-16 11.04 → 09-19 16.07 |

Takeaways:
- A large capability step happened in early/mid September, consistent with the Flash-Next model plus better token allocation.
- Nearly everyone at the top uses the full 9 h. Yi-Chia Chen is the exception: 18.8 in about 430–540 min, with several runs of 90–300 min, which suggests a more token-efficient or early-exit design.
- Distribution: 9 teams ≥10, 24 ≥7, 54 ≥5, 187 ≥4, 641 ≥3, 1,284 ≥1. 3,369 teams in total.

### 2.5 "What are your agents scoring on the 25 public games?" (732854, 48 messages)
Local public score → LB:

| Who | Local public | LB |
|---|---|---|
| Scott Le Grand | 3.8 | 0.9 |
| Scott Le Grand (Qwen3.8 Flash-Next NVFP4, game-independent harness changes) | 11.04 | 2.71 |
| Scott Le Grand (best) | 12 | 3.2 |
| Fususu | 3.8 | 0.9–1.8 |
| Fususu | 17.34 | 5.19 / 6.91 |
| Fususu | 22.26 | 5.37 |
| daoviet | 6.8 | 1.19 |
| mikelou1 | 2.8 | 2.4 |
| OverfitOracle | "5.0+ stable" | 1.6 |
| Nick Pellegrin | 5.0–5.4 | 1.4–1.8 |
| Nick Pellegrin | 7.6 / 10.2 / 7.7 | 3.35 / 3.05 / 4.42 |
| Ya Xu | 9.91 (runs 9–14) | 6.23, stable 4.0+ |
| Giang Rita | 15.7 | 7.37 |
| @s3werst | 11.95 | 3.99 |

Other points from the thread:
- **The public Flash-Next notebook** (`wuliao0/duck-qwen3-8-anim-base`) runs 6–12 locally and 1.96–4.33 on the LB (Ya Xu).
- Nick Pellegrin: "It almost feels like doing better on the public set (even without any game-specific stuff) correlates to weaker private scores."
- Van-Phuc Huynh: public 24.x max, 16.x median, not yet submitted.
- Scott: game-specific solvers built with Claude "Fable" reached about 38% RHAE, 116/183 levels, but took 5 days and are impractical. A general solver distilled from them overfit.
- Scott: "most of what I have done is just take the stock duck client for nvfp4 and increase the sequence concurrency to 16 … will get you to the brink of the top 10%. All of my harness engineering beyond that has been utterly useless."
- Scott: behaviour cloning with "reasoning traces from the same model" gives local 11.04 → 12.97. Using traces from another model is a regression.
- Ya Xu: SFT on Flash-Next NVFP4 is "nearly impossible with 96 GB". OverfitOracle: RL is possible for Qwen3.8-27B after heavy quantisation.

### 2.6 Harness-improvement write-ups by participants
**Taaf Anim Agent — Jakob Brüggen (734369; code `jakobbrggen/taaf-anim-arc-agi-3-solver`)**
1. **Hard no-op guard.** The harness remembers every (level, board, action) that changed nothing and blocks exact repeats before they reach the environment. This costs no game action, only an LLM turn. Measured -12–20% actions, but runs are **wall-clock bound**, so the benefit is unclear.
2. **Animation access.**
   - The API returns a frame list, and the Duck discards all but the last frame.
   - 13 of the 25 public games return multi-frame responses. sp80's ACTION5 returns 22 distinct frames, and 624 pixels exist only mid-animation. In bp35 every action animates.
   - Three stages were implemented: (a) always attach animation metadata (number of frames, distinct frames, whether the board is unchanged, transient pixel count and bbox); (b) an `animation()` tool returning a compact diff timeline; (c) a hint when stuck.
   - Results: public A/B +1.4% (p = 0.92), though the private set "went well". Tokens per action rose 384 → 449 (+17%), and **"tokens are the real currency, not actions"**: more tokens per action means fewer actions under the 132-min per-game cap. The model called the tool in 21 of 24 runs, but only 2 of 181 calls landed on informative animations.
   - A 30 s sandbox timeout killed the single most useful call.
   - Xuan: frames are vital for frontier Sol (tn36) but blow up Qwen's context. VISTA (vista-research.github.io) upscales 64 → **512 px** for vision.

**yw8837, LB 1.17 (731522):**
- Duck + Qwen3.6-27B-FP8, concurrency 28, 7,920 s per game.
- Patches: stop a batch if an identical cardinal direction leaves the board unchanged, and return control for re-observation; the analyzer yield window was raised from 60 to 90 s.
- 11 submissions scored 1.29, 1.05, 0.71, 0.73, 0.68, 0.75, 0.55, 1.11, 1.17, 0.90, 0.94 (a 2.3× spread).
- One public run: 15 levels, 3,908 actions, 1.77 M tokens. The ledger dataset is `yw8837/arc-agi-3-run-history-300-game-diagnostics`.

**gedouluhui, "P3.1 Perception-Augmented Duck" (743060), hidden 1.43.** Setup: Duck plus Qwen3.8-27B FP8; T=0.6, top-p 0.95, top-k 20; 256 px frames; 32k context with eviction; history of 30 turns; per-game budget 132 min; vLLM 0.19.0.
- Throughput: about **283 tok/s decode at 4k context, about 30 tok/s at 32k** (prefill-dominated).
- Kept:
  - The Qwen3.6 → 3.8 model swap (1.12).
  - **Perception hints:** when MOUSE is available, same-colour 4-connected component centroids are sorted by salience and given as click candidates (about 120 LOC).
  - **Two prompt lines**, giving 1.43 (+28%):
    - (i) "Scoring punishes long solutions quadratically: each level scores roughly (baseline_actions / your_actions)^2 × 100 … commit to the shortest reliable sequence and batch confirmed steps in one `action(...)` call."
    - (ii) "Action results report `time_remaining_seconds`. Use it to pace yourself: probe until the mechanic is understood, then execute decisively. When several distinct hypotheses have failed on a level … change the hypothesis class instead."
  - An ACTION7 label fix.
- Rejected:
  - Context 64k plus a 4,096 output cap: 1.20. Throughput fell from 283 to 195 tok/s, and bp35 burned 28k tokens with zero actions.
  - T=0.3: 0.95. Games with any progress fell from 12 to 8: "the exploration in the tail of the sampling distribution *is* the problem-solving capability".
  - Archetype playbook: 1.54. It caused premature classification.
- The "lottery effect": 17 of 25 games were cracked at least once, but only 5–8 in any single run (sb26 scored 27.78 in one run and 2.78 in another). Cracked levels usually score about 100, so **the gap is entirely in never-cracked games** and depends on whether the opening minutes find the right mechanic.
- Public single-pass σ ≈ 0.8–0.9 points, so a single run resolves only effects larger than about ±1.5.

**Serguei Makarov, "Harness or model?" (743723):**
- Isolated single-layer A/Bs. At or below baseline: consensus between samples, a reasoning-discipline block, **higher-resolution board images**, and an **executable world model with replay validation plus search planning**. The model mostly ignored the world-model scaffold, and the simulators it wrote rarely beat a "nothing changes" baseline on unseen transitions.
- Aligning stuck and successful runs of the same game showed that they diverge in the first few moves of a level. The stuck run had usually already tried the decisive action type, so **the failure is early goal judgement, then defending it**. Stall signals (repeats, move counts) detect it far too late.
- Conclusion: once the harness is tuned, the lever is the model or training.

**Rakha Abid Bangsawan (739938):** Duck + Qwen went from 1.65 to 2.86 (and 3.48 on resubmission).
- Deterministic, auditable serving: pinned config, MTP and concurrency, startup validation, a real tool-call preflight, and fail-closed on the wrong model or parser.
- Single-step verified recovery (one action, observe, continue only on evidence, otherwise cool down), triggered by symptoms such as repeated no-ops or stagnation.
- Memory of *verified transition facts* scoped by game and level, instead of free-form summaries.
- Roles gated rather than always on.
- A progress-weighted compute allocator (about 13.6% counterfactual action savings; the 4 proposed stops never later progressed).
- RL used as an online bandit controller rather than for weight updates.
- Variance hypothesis: **continuous batching and request scheduling change trajectories**. Proposes per-request seeds from (base_seed, game, level, step) and optimising E[score] − λ·Std.

**Fususu, "What have you tried" (739938, 18 votes):**
- Gemma-4-31B has better vision but worse logic and coding than Qwen3.6-27B.
- A system decoder (objects/movers) lets GPT-OSS 20B/120B play, but introduces roundabout paths and game bias.
- Observer / theory / actor / advisor roles cost 5× the calls.
- Thinking on gives better code but is slow; off is fast and careless.
- **Concurrency above 16 saturates**, because KV and prefill/decode interfere.
- Playing all games by hand and training on the data was no better than Qwen3.8 27B or Flash-Next alone.
- A strategy library causes game-specific bias.

**Scott Le Grand's public notebook (from its markdown; the user's notebook descends from it):**
- Duck prompts and loop are unchanged. Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (ModelOpt NVFP4 weights, BF16 compute).
- vLLM settings: async scheduling, chunked prefill, 3-token NEXTN MTP speculative decoding, 32k/16k context, 8k batched-token cap, 8–16 sequences, CUDA graphs, prefix caching off.
- Owned-server watchdog.
- AGENTFIX:
  - F1: only the current prompt keeps its image. A median of 8 obsolete boards per request had eaten about 26% of the budget, and the estimator over-charged images 5–8×.
  - F2: always return the structured action result. 81% of acting calls previously returned none.
  - F3: tolerant world-model label parsing, and no wipe on GAME_OVER.
  - F4: loop guards against re-running an identical snippet.
  - F5: HUD-band-aware change detection and no-op tagging. All 25 public games draw a 1–2-cell-per-action budget bar.
  - F7: per-action effect ledger.
  - F6: ACTION7 executable, described as UNDO.
- Settings: `max_runtime_s_per_game=7920`, `concurrency=28`, `analyzer_timeout=900`.

### 2.7 Fine-tuning threads
- **manas joshi (739047):** LoRA SFT with Qwen3.6 on the Duck's own *winning* trajectories (STaR / rejection sampling) took the LB from **1.25 to 1.94**.
  - Datasets (CC0): `justforgags/arc3-sft-trajectories`, `justforgags/arc3-duck-lora-sft`, `justforgags/duck-eval-results`.
  - Gotchas: train on assistant tokens only; flatten OpenAI-format tool calls; the architecture is multimodal, so merge and serve as `Qwen3_5ForConditionalGeneration` and copy the processor configs.
  - Scaling the data naively hurt. The effect is base-model specific.
- **Ya Xu:** LoRA "only backfired". Chollet: "the set of public ARC 3 games is called *demonstration set* not *eval set* nor *training set*. It is not meant to be used as training data."
- **wkdrbwnd1 (743319):**
  - Setup: 31 trajectories from 20 games, evaluated on 5 held-out games.
  - Uniform reasoning weight 0.3 scored an average of 25.1. Sentence-labelled weights (most reasoning at 0.1) scored 5.0, because of a **"no tool call" collapse**: after one turn hit the generation limit, 1,231 consecutive turns ended with no action.
  - Lessons: tell the model plainly when its previous turn had no action or was cut off. Listing the runtime objects' attributes in the prompt cut attribute errors from about 7.5% to 2.5%.
- **Nick Pellegrin / AAAAAtjc (742835):** an RL pipeline for Qwen3.8 Flash-Next was built on an H200 node, with no gain yet. The RTX PRO 6000 cannot train a ~100B model.

### 2.8 Other notable threads
- **Qwen 3.8 release (735243):**
  - 2× local for Qwen3.8-27B 8-bit versus 3.6. DeepSeek-V4-Flash is only slightly better than 3.6.
  - Three teams jumped past 2.0 within 12 h. Top 3 moved; the 1.86 plateau broke.
  - Kaggle model mirror: `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` (fewer shards, faster load; public notebook scored 2.23).
  - `xhigh` reasoning beats `low` (Ya Xu); Simon Willison argued Qwen3.8 overthinks at xhigh.
  - Some saw nightly submissions 30–50% lower.
- **GPT-OSS-120B on RTX PRO 6000 (738599):**
  - Setup: `danielhanchen/gpt-oss-120b` MXFP4 (66 GB); wheelhouse `driessmit1/arc3-vllm-h100-wheelhouse-v3` (vLLM 0.19.0 + torch 2.10.0 + flashinfer 0.6.6); tiktoken encodings attached offline.
  - Flags: `--tool-call-parser openai --kv-cache-dtype fp8 --max-model-len 65536 --enable-prefix-caching`, with no `--reasoning-parser`.
  - Performance: about 830 tok/s at 25 concurrent requests; 462k KV tokens; about 90% prefix-cache hits.
  - Bug: an 8.1% harmony-parser HTTP 500 rate, fixed by wrapping `parser.process` in try/except.
- **Offline vLLM checklist (738600, 738602):**
  - Wheelhouse install with `--target`, `--only-binary`, and stamp files.
  - Probe the multiple dataset mount paths.
  - Assert the GPU name (Kaggle once handed out a P100).
  - Verify safetensors shard counts.
  - Set `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `VLLM_NO_USAGE_STATS`.
  - Smoke-test `/v1/models` plus one completion.
  - Soft-deadline exit about 10 min before the hard limit.
  - vLLM 0.21 defaults FlashInfer on, which breaks on RTX; use 0.20.2 or 0.19 (699859). GPT-OSS Triton `sm_120a` ptxas error (703506).
  - vLLM "silently hangs on RTX Pro 6000 … after 15–20 minutes with 8 or 25 concurrent sessions" (Scott, 684625). Hence watchdogs.
- **Hidden reasoning drops memory (734843):** 66.8% of Qwen tool-call responses had hidden reasoning with **zero visible content**, so the Duck's world-model updates were never captured. Fix: `iamjasonfeng/tufa-duck-visible-updates`; Qwen3.8 needs "yelling" to comply. Result: DeepSeek V4 Flash went 9% → 11% on the public set.
- **Human data and training corpora:**
  - Human trajectories: HF `magic-sword/arc_agi_3_public_demo_human_testing` (705424).
  - VCGT video + LLM annotations (715611).
  - Game rule texts: `karnakbaevarthur/arc-agi-3-all-tasks-explanation` (695903) and `magicsword001/arc-agi-3-game-rules`.
  - arc3.games (30 games plus a PushWorld port; 693213).
  - `theredbluepill/arc-interactive` and `NVIDIA/dream-team` (25 generated games; 736540).
  - Offline atlas of all 25 games plus a scorer: `busyaprime/arc-agi-3-offline-atlas-and-scoring`.
  - Unofficial game names (738294): ar25 axis-reflections, bp35, cd82 color-drop, cn04 connector-network, dc22 drawbridge-controls, ft09 flip-tiles, g50t gate-teamwork, ka59 knockback-alignment, lf52 leap-frog, lp85 loop-placement, ls20 lock-smith, m0r0 mirror-reunion, r11l rigging-links, re86 realign-elements, s5i5 stretch-insert, sb26 sequence-builder, sc25 spell-casting, sk48 sliding-kebab, sp80 spill-planning, su15 square-up, tn36 toggle-navigation, tr87 transform-runes, tu93 traverse-unharmed, vc33 volume-control, wa30 warehouse-agents.
  - cn04 is nearly the same logic as ARC-AGI-2 task cbebaa4b (734046).
- **fchollet on difficulty (690656):** Exploration 20%, **Modeling 65%**, Goal-setting 10%, Planning and execution 5%. Planning mostly kicks in on later levels.
- **ls20 anatomy (692883):**
  - A 42-step budget per attempt, batteries refill it, and there are 3 lives. Locks need shape (6), colour (4) and rotation (4) to match; modifiers cycle forward only.
  - Pushers, NPC wanderers, and fog in L7.
  - One new concept per level: a "tutorial" L1, a "mastery test" L5.
- **Frontier references:**
  - Claude Opus 5: 30% (728934).
  - GPT-5.6 Sol: 7.8% (726340).
  - OpenAI Astra ("GPT Astra"): 99% on 110 games for $17,332 (739707).
  - NVIDIA AVO: 100% (737617).
  - VISTA/Schema harness with Opus 5: 99% (code world model plus sandbox verify; ARC-2 733501).
  - Polyphony Agent (Ruiyang Yu et al.) on the community leaderboard uses a 27B model under similar constraints (737617).
  - The hardware gap stands: "Solving with frontier model is not equal to solving 'this' competition" (Fususu/CPMP).

---

## 3. Secret intel / hypotheses — every concrete trick or claim found (with source; treat as unverified unless noted)

**Models and serving**
- Qwen3.8-Flash-Next NVFP4 is the current workhorse (public notebooks and most top teams). Qwen3.8-27B-FP8 is the fallback. Gemma-4-31B has better vision but worse coding and logic. DeepSeek-V4-Flash is too slow after quantisation and its "vision doesn't work". GPT-OSS-120B is text-only but fast (830 tok/s at 25 concurrency). Nemotron Nano 30B-A3B was being tested. [735243, 739938, 742788, 738599]
- Qwen3.8-Flash-Next: 125B MoE, about 6B active, 262k native context; fits the RTX 6000 via FP4 plus host-offloaded N-gram embeddings. [743060]
- Recipe that gets to "the brink of the top 10%": stock Duck with Flash-Next NVFP4, **16 vLLM sequences**, and 28 concurrent games. [Scott, 732854]
- More than 16 concurrent requests saturates. [Fususu]
- MTP (3-token NEXTN) speculative decoding is used by top public notebooks; ablate it for variance. [Scott nb; Rakha]
- Prefix caching is off in Scott's profile (on for GPT-OSS). Tufa said they "do not optimally use prefix caching". Enabling it with a stable prompt prefix may buy throughput. [hypothesis]
- Reasoning effort: xhigh beats low for completing multiple levels. [Ya Xu]
- Temperature 0.6 beats 0.3. [gedouluhui]
- Long context hurts: 32k → 64k dropped throughput about 30% and scored worse. [gedouluhui] At 32k, decode falls to about 30 tok/s because prefill dominates, so **keeping the input context short directly buys actions**. [gedouluhui, Anim write-up]
- Images: Duck uses 4×; Qwen3.8 forks use 8× ("the arm that moved the score"); the user's notebook uses 12×; VISTA uses 512 px. **No clean A/B was published** (739801). Each extra image costs tokens: only one board image per request (AGENTFIX F1).
- The model load time is about 5 min (gpt-oss-20b, qwen3-vl-8b) and about 8 min (Qwen 27B). Budget for it, but note the 15-min inactivity window. [699859, 732932]

**Prompting and harness**
- Put the exact scoring formula (quadratic penalty) and `time_remaining` pacing in the prompt: +28%. [743060]
- The biggest failure mode is early wrong goal commitment. Prompting to switch *hypothesis class* after repeated failures helps a bit, and archetype classification hurts. [743060, 743723]
- The model must be kept from treating the energy/timer bar as the goal and from hallucinating classic-game sprites. The level-up animation can be hallucinated as a "snake". [717133, 728350]
- HUD/budget bar: every public game draws a 1–2-cell-per-action bar at an edge row or column. Mask it when detecting "no-op". [AGENTFIX F5]
- No-op guard: track (level, board, action) → no change. Refusing repeats can mislead the model; tagging `repeat_of_known_no_op` worked better in Scott's regression hunt. [734369, Scott nb]
- Dead clicks count as actions, so click-candidate generation matters: connected-component centroids, and small rare-coloured button-like shapes first. Stop clicking object types that never change anything. [743060, Reki]
- Batch confirmed steps into one `action([...])` call to save LLM turns; stop a batch when the board is unchanged. [743060, 731522]
- Put world-model updates in **visible** output, not reasoning. [734843]
- Always return the structured action result after acting; 81% of acting calls previously got none. Don't wipe memory on GAME_OVER, since it is a level retry. [Scott nb]
- Tell the model explicitly when its previous turn had no tool call or was truncated, to avoid degenerate loops. [743319]
- Listing the runtime object attributes in the prompt cut attribute errors from about 7.5% to 2.5%. [743319]
- Animation frames matter in some games (sp80, bp35, sb26, tn36). Give cheap metadata always and details on demand, but watch the token cost. [734369]
- ACTION7: label it as UNDO (it costs an action). [742477, Scott nb]
- Use RESET deliberately when a level is dead-ended (the shortest path to a solution) rather than burning moves until the level resets itself. A reset at step 0 of a new level is a no-op or disabled in competition mode. [692135, 693123]
- Multi-role pipelines, consensus sampling, strategy libraries, executable world-model scaffolds and higher-res images did not beat the baseline under the 9 h token budget. [739938, 743723]
- Single-step verified recovery works better than unconditional recovery agents. [Rakha]
- Adaptive per-game compute: stop games that stagnate, give time to games making progress. Caveat: giving more time to games already at L2+ helped the public set, not the hidden one. [Rakha, Scott 740812]
- The random agent including *invalid* actions scored 0.27 against 0.04–0.13 with valid-only. Almost certainly noise; one reading is that a no-op "wait" lets dynamics advance. [693146]

**Evaluation and variance**
- Expect a 3–4× drop from public-25 to LB. Judge by the mean of 3+ runs. Public-set σ is about 0.8–0.9 per single pass. Two games carry about 65% of Tufa's public variance, and half of all runs score 0. [732932, 743060]
- Variance sources: sampling, continuous-batching order, and game-level bimodality. The option of seeding each request is untested. [Rakha]
- Some report nightly submissions scoring lower; possibly GPU/PCIe contention on shared hosts. [Scott, Drona Bajaj; unverified]
- Offline eval does not predict the LB, so "only submit to measure". [manas joshi]

**Submission hygiene**
- Write the dummy parquet only when not in a competition rerun (`KAGGLE_IS_COMPETITION_RERUN`). Check the Output File dropdown. Make the first game action within about 10–15 min. Use a soft deadline of about 8 h 30 m – 8 h 45 m. Assert the RTX PRO 6000 in cell 1. A fresh kernel slug can fix persistent Kaggle Errors. [multiple]

---

## 4. Relevant items from the ARC-AGI-2 / Paper Track forums
- NVIDIA (CPMP / Darragh / Ivan Sorokin) leads ARC-AGI-2 and runs NVARC3 on ARC-AGI-3 (22 submissions, 16.07).
- A NVIDIA SFT dataset `nvidia/Nemotron-SFT-ARC-AGI-v1` exists. Its tools (arc-python-executor, commit, augmenter) hint at code-execution agents (cm391, ARC-2 724647).
- Program synthesis with sandbox verification amplifies strong models only (Dominic, ARC-2 733501). The same pattern powered Schema/VISTA to 99% on ARC-AGI-3 with Opus 5.
- A CPMP run on ARC-2 timed out abnormally once, and an identical resubmission ran in 9 h 10 m, so infrastructure noise exists (738313). Kaggle sometimes returns the daily slot after a system error.
- Paper Track:
  - A unified paper covering both ARC-AGI-2 and ARC-AGI-3 is allowed; pick one primary notebook (694752).
  - The paper must document a score achieved by the same team, so teams merge for it. Borro1980 is pitching a measurement paper; the merge deadline is 2026-10-26.
  - The Paper Track pays $75k across 3 places and had 118 teams on Aug 5.
- Licence ambiguity: Kaggle says CC-BY 4.0, while arcprize.org says CC0/MIT-0 (ARC-2 743581, unanswered).

---

## 5. Open questions (unanswered or conflicting on the forum)
1. Are the 110 environments seeded identically across submissions, and is the 55/55 public/private split fixed for every submission? (738762, 726552; no host reply.) This decides whether repeated submissions can be averaged.
2. Human baseline used by the Kaggle scorer: "second-best human" (April) or the newer "(upper) median"? Is `environment_info.baseline_actions` exposed in submission mode (visible per Jeroen in April; "can come back empty" per Asan Ashirov in Sept)? If it is available, an agent can know the per-level target action count. Worth probing.
3. Does ACTION7 do anything other than UNDO in the hidden games? The docs say "Simple undo action", but the overview says "Additional simple action".
4. OSAID "open source model" requirement vs Qwen/Gemma licences (rules §2.5.a and its carve-out). No host answer.
5. Milestone 2 mechanics: is the ranking snapshot by submission time or score-display time? Must fine-tune data or scripts be published, or only the weights? What about dependencies with unknown licences (743753)?
6. What exactly does "Notebook Timeout" do to the score? Marius says the scorecard still counts actions; Gabriel got an error with no score. Treat an overrun as fatal.
7. What took Tufa from 18.8 to 27.3 on 09-26, and Lord Han Solo and Tong Hui Kang from about 5 to about 20 in two weeks? Candidates: fine-tuning (STaR / behaviour cloning), better token and throughput allocation, a new model, or harness changes. Yi-Chia Chen gets 18.8 with runs as short as 90–500 min, which suggests a much more token-efficient loop.
8. Will new models appear before 2026-11-02 (e.g. Qwen 4)? Someone asked for a model freeze like AIMO's; there has been no host response.
9. Is the day/night submission score difference real (infra contention), or noise?
10. Does arc_agi 0.9.9 vs 0.9.8 change behaviour? The competition pins 0.9.8.
11. Is there any effective early-stop or reset discriminant for stuck games? Scott and Serguei both failed to find one.

---

## Appendix A — Full index of the 225 ARC-AGI-3 forum topics (sorted by votes; "Msgs" = comments+replies fetched)

| # | Votes | Msgs | Date | Author | Title |
|---|---|---|---|---|---|
| 1 | 71 | 18 | 2026-07-01 | InfiniteCreativity | [Tufa Labs’ Winning Solution for ARC-AGI-3 Milestone 1](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133) |
| 2 | 41 | 3 | 2026-03-31 | Jeroen Cottaar | [Simplified submission framework](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/686416) |
| 3 | 38 | 16 | 2026-06-30 | Jeroen Cottaar | [Code for milestone 1st place open sourced - writeup tomorrow](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716696) |
| 4 | 31 | 27 | 2026-09-23 | Jeroen Cottaar | [We will not be open-sourcing for the second milestone prize](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742801) |
| 5 | 24 | 8 | 2026-04-15 | CPMP | [It is 0.66%](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/691696) |
| 6 | 23 | 10 | 2026-04-03 | Duc-Cuong Le | [ARC3 Offline Agent Evaluation and Recording Viewer](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687648) |
| 7 | 23 | 5 | 2026-06-18 | Tong Hui Kang | [View your run time and the historical leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/709355) |
| 8 | 21 | 7 | 2026-06-08 | Jeroen Cottaar | [Exact timing of milestone prize](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705043) |
| 9 | 20 | 3 | 2026-04-03 | Jeroen Cottaar | [Human scores visible to agents](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687655) |
| 10 | 20 | 2 | 2026-04-28 | María Cruz **(admin)** | [Upgraded accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/695158) |
| 11 | 20 | 22 | 2026-09-20 | Chew Kok Wah | [Long Queue for RTX Pro 6000 (8 hours) partially due to abusive used for non ARC-AGI-3 tasks, Proposal to mitigate](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742148) |
| 12 | 18 | 6 | 2026-09-07 | Fususu | [What Have You Tried So Far?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739938) |
| 13 | 14 | 18 | 2026-03-29 | Pavel Orlov | [What does 0.25 on LB mean?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/685805) |
| 14 | 14 | 10 | 2026-05-03 | Jeroen Cottaar | [H100s effectively unavailable](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/696615) |
| 15 | 14 | 4 | 2026-05-31 | Tong Hui Kang | [I did "autoresearch" in a hackathon and won some GPU credits](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703426) |
| 16 | 14 | 4 | 2026-07-13 | Greg Kamradt **(host)** | [ARC Prize 2026: ARC-AGI-3 Milestone Prize #1](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/725002) |
| 17 | 14 | 48 | 2026-08-04 | Reki | [What are your agents scoring on the 25 public games?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732854) |
| 18 | 13 | 35 | 2026-09-03 | Chan Kha Vu | [Mostik.AI - amazing results, but... 12 members out of 8 allowed?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739186) |
| 19 | 12 | 11 | 2026-04-04 | Komil Parmar | [Is Offline BFS "Cheating the Spirit" of ARC-AGI-3?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687950) |
| 20 | 12 | 9 | 2026-05-13 | Jeroen Cottaar | [Submissions not running 9 hours](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/699208) |
| 21 | 12 | 5 | 2026-07-17 | Greg Kamradt **(host)** | [500 Submissions Analyzed - Common Errors](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119) |
| 22 | 11 | 7 | 2026-05-07 | María Cruz **(admin)** | [Update on Code Requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944) |
| 23 | 10 | 1 | 2026-08-02 | Inexperienced Me | [Minimalistic All-in-One Toolkit for ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732419) |
| 24 | 10 | 2 | 2026-08-31 | Tong Hui Kang | [Unofficial full game names](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738294) |
| 25 | 9 | 7 | 2026-03-25 | Jeroen Cottaar | [Intended compute budget](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/684724) |
| 26 | 9 | 6 | 2026-05-07 | María Cruz **(admin)** | [Update on accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697720) |
| 27 | 9 | 2 | 2026-08-11 | Jakob Brüggen | [Write Up: Taaf Anim Agent](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369) |
| 28 | 8 | 9 | 2026-03-25 | María Cruz **(admin)** | [How to get started + Competition's Official Discord](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/684625) |
| 29 | 8 | 1 | 2026-06-29 | MagicSword | [[Dataset] 🚀 Released Human VCGT Dataset: Human Video + LLM-Annotated Explanations for Imitation Learning](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/715611) |
| 30 | 8 | 1 | 2026-06-30 | Akhil Tolani | [[0.79 public score] Open source code for Milestone 1 - Gemma 4 31B QAT Pruned Vocab + LeWM/JEPA dynamics model](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716711) |
| 31 | 8 | 4 | 2026-09-02 | manas joshi | [Open-sourcing winning-trajectory datasets + a LoRA for the duck harness (STaR fine-tuning)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739047) |
| 32 | 7 | 0 | 2026-04-02 | Greg Kamradt **(host)** | [Welcome to ARC Prize 2026 - ARC-AGI-3 Edition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687384) |
| 33 | 7 | 9 | 2026-04-12 | Bosyj Jakub | [[Question] Leaderboard Scores Clarification: 66% or 0.66%?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/690654) |
| 34 | 7 | 2 | 2026-06-30 | Md Boktiar Mahbub Murad | [Code for milestone 3rd place open sourced [LB 0.86]](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716719) |
| 35 | 7 | 13 | 2026-08-26 | Drona Bajaj | [Sudden increase in top 3 teams?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/737617) |
| 36 | 7 | 7 | 2026-09-24 | Fususu | [Who will open-sourcing for the second milestone prize?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742935) |
| 37 | 6 | 5 | 2026-04-02 | Jack | [Multiple Accounts](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687225) |
| 38 | 6 | 0 | 2026-05-10 | Isaiah Nwukor | [Retro Game Training Set For Visual Classification (Labeled Data)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/698623) |
| 39 | 5 | 1 | 2026-03-26 | narsil (jobs-in-data.com) | [Does using Claude Code/ Codex/ Gemini make you work more or less](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/685007) |
| 40 | 5 | 4 | 2026-03-30 | Bridgeport | [Is it possible to develop and test my code for this competition locally? Any workarounds?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/686155) |
| 41 | 5 | 4 | 2026-04-02 | Kevin Tung Nguyen | [What if AGI isn't about doing everything but about knowing what it can't do?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687134) |
| 42 | 5 | 5 | 2026-04-14 | Ravi Annaswamy | [Stop searching the Action Tree, Discover the Transformation Graph (milestone chain)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/691143) |
| 43 | 5 | 2 | 2026-04-28 | Michael Poluektov | [Better containers](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/695046) |
| 44 | 5 | 5 | 2026-06-24 | María Cruz **(admin)** | [Clarification on deadline for milestone prizes](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713634) |
| 45 | 5 | 3 | 2026-07-24 | Geremie Yeo | [Claude Opus 5 achieves 30% on ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728934) |
| 46 | 5 | 19 | 2026-08-14 | OverfitOracle | [Qwen 3.8 release](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735243) |
| 47 | 4 | 2 | 2026-03-26 | Jeroen Cottaar | [Do we run on public and private test set?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/684852) |
| 48 | 4 | 3 | 2026-04-20 | Gijs Schenk | [Introducing ARC3.Games: Browser-Based ARCEngine Game Player & Gameplay Database](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693213) |
| 49 | 4 | 23 | 2026-04-25 | Froggy McFrogson | [Generating training data ideas?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/694549) |
| 50 | 4 | 2 | 2026-05-03 | Raviksh Singh Dikola | [Long queue for accessing H100 GPUs](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/696616) |
| 51 | 4 | 2 | 2026-05-30 | Yaroslav kholmirzayev | [How I fixed the "Kaggle error" in ARC-AGI-3 submissions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703416) |
| 52 | 4 | 4 | 2026-07-13 | Jakup Ymeraj | [About using GPU RTX PRO 6000](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724890) |
| 53 | 4 | 1 | 2026-07-31 | yw8837 | [[LB 1.17] ARC-AGI-3 Qwen3.6 Duck + 300-Game Diagnostics](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/731522) |
| 54 | 4 | 1 | 2026-08-04 | Jason Feng | [I'm open sourcing two of my solutions.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732823) |
| 55 | 4 | 2 | 2026-08-21 | robenten | [non-official games for training](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/736540) |
| 56 | 4 | 6 | 2026-08-21 | Nick Pellegrin | [Public vs. Private Discrepancy](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/736578) |
| 57 | 3 | 3 | 2026-03-26 | Vladimir Zagorodskikh | [environment_files folder purpose](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/684898) |
| 58 | 3 | 10 | 2026-03-26 | Jason Yoon | [Kaggle Error displayed after submission.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/684969) |
| 59 | 3 | 1 | 2026-03-31 | Hellboy | [setup arc engine agents?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/686501) |
| 60 | 3 | 8 | 2026-04-12 | Froggy McFrogson | [Some ideas on Approaches](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/690656) |
| 61 | 3 | 22 | 2026-04-16 | Froggy McFrogson | [How many tries do you get at each level? (Confused 😖)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/692135) |
| 62 | 3 | 0 | 2026-04-30 | Karnakbayev Artur | [[Resource] ARC-AGI-3 Logic & Action Metadata](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/695903) |
| 63 | 3 | 4 | 2026-05-24 | Van-Phuc Huynh | [Exploring LLM + Reasoning Approaches for ARC-AGI-3 - Anyone Going Deep Into This?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/702478) |
| 64 | 3 | 12 | 2026-06-09 | María Cruz **(admin)** | [Correcting an Error in Competition Submission Limits](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705405) |
| 65 | 3 | 12 | 2026-09-05 | Fususu | [How long do you have to wait to run the notebook?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739674) |
| 66 | 3 | 3 | 2026-09-10 | Matija Ludvig | [Looking for additional teammates](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/740604) |
| 67 | 3 | 12 | 2026-09-11 | Fususu | [Only 6 submissions and jumped straight into the top 3????](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/740812) |
| 68 | 3 | 3 | 2026-09-22 | Van-Phuc Huynh | [About ACTION 7.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742477) |
| 69 | 3 | 11 | 2026-09-26 | Ya Xu | [I suppose the drama surrounding the milestone awards for open-source solutions has just launched.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743624) |
| 70 | 3 | 6 | 2026-09-26 | Serguei Makarov | [Harness or model? Notes from a week of controlled harness experiments](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743723) |
| 71 | 2 | 6 | 2026-03-28 | Kevin Tung Nguyen | [Before Solving ARC, Should We Redefine the Problem Itself?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/685469) |
| 72 | 2 | 3 | 2026-04-02 | Bridgeport | [How long can we play a game.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687187) |
| 73 | 2 | 4 | 2026-04-03 | Hellboy | [Is Possible To Not Use A Agent From Host?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687484) |
| 74 | 2 | 1 | 2026-04-04 | zyw | [Tackling ARC-AGI 3: Leveraging Puzzle Games to Address Data Scarcity](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/688114) |
| 75 | 2 | 3 | 2026-04-06 | Josh | [Clarification on using local open-weight Qwen3.5 in ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/688481) |
| 76 | 2 | 1 | 2026-04-08 | vialactea | [What is the API rate limit?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/689492) |
| 77 | 2 | 5 | 2026-04-11 | Zhengxu Yu | [Provide some error info for competition submission?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/690334) |
| 78 | 2 | 0 | 2026-04-18 | Ravi Annaswamy | [What knowledge has to be discovered by a live playing agent from a novel game? (locksmith ls20 example)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/692883) |
| 79 | 2 | 0 | 2026-05-17 | Isaiah Nwukor | [Diffusion of memory?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/700442) |
| 80 | 2 | 3 | 2026-05-28 | Jakob Brüggen | [Milestone 1 - Evaluation](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703056) |
| 81 | 2 | 10 | 2026-06-08 | Yield Smarter | [How is the 1.30 on the LB possible?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705022) |
| 82 | 2 | 4 | 2026-06-08 | parthenos | [5 submissions/day disabled](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705094) |
| 83 | 2 | 0 | 2026-06-10 | MagicSword | [[Dataset] ARC-AGI-3 Human Trajectory Dataset on Hugging Face – Fueling Top-Down Action Planning?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705424) |
| 84 | 2 | 9 | 2026-06-12 | Yield Smarter | [Is the public game bp35 broken?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/707862) |
| 85 | 2 | 3 | 2026-06-24 | Ivan | [Copied a random notebook, got 0.10 in just 3 hours - what's actually happening under the hood?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713643) |
| 86 | 2 | 6 | 2026-07-03 | Alvaro Camacho De Pass | [Clarification requested: does a no-op ACTION6 (unchanged frame) count as an action on the scored server?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/718638) |
| 87 | 2 | 2 | 2026-07-19 | Vladimir Yakunin | [Constraint Before Control: A Semantic Architecture for ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727505) |
| 88 | 2 | 0 | 2026-07-22 | Imed Magroune | [When will arc-agi 0.9.9 be available in Competition Notebooks?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728220) |
| 89 | 2 | 2 | 2026-08-12 | Jason Feng | [Potential persistent memory issue with the Tufa Duck harness](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734843) |
| 90 | 2 | 7 | 2026-08-14 | Dinesh kumar Thiyagarajan | [Submission stuck in "Queued" (RTX Pro 6000) for 28+ min — anyone else?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735147) |
| 91 | 2 | 0 | 2026-08-16 | FOYSAL | [Qwen 3.8 27B](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735479) |
| 92 | 2 | 0 | 2026-08-24 | ShotaAzuma | [Submission still shows Running >11h](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/737227) |
| 93 | 2 | 1 | 2026-09-01 | Kolenkovskin | [Are the 110 environments identically seeded across scored runs?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738762) |
| 94 | 2 | 0 | 2026-09-06 | Asan Ashirov | [MULTIMODAL_UPSCALE: the writeup says 4 is best, the popular forks all run 8 — has anyone actually A/B'd it?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739801) |
| 95 | 2 | 5 | 2026-09-16 | mayakaripel | [Version 29 submission fails with generic "Kaggle Error" despite successful notebook run](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/741608) |
| 96 | 2 | 15 | 2026-09-17 | xz | [Anyone using Claude Code for this competition? What is your experience?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/741839) |
| 97 | 2 | 5 | 2026-09-23 | Van-Phuc Huynh | [Can a New Model Reach the Top?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742788) |
| 98 | 1 | 1 | 2026-03-29 | Hellboy | [hi friend i have a problem in setup agents package in this competition is you can have solution to this problem,thank yo?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/685939) |
| 99 | 1 | 3 | 2026-03-30 | pom | [Submission limit: only 1 per day?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/686041) |
| 100 | 1 | 0 | 2026-04-03 | Hellboy | [Host?,What Is The Minimum Game requierement for submission best score in this competitition?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687483) |
| 101 | 1 | 1 | 2026-04-03 | Ivan Semenenko | [Just want to share results for public set.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687788) |
| 102 | 1 | 6 | 2026-04-04 | Nirvana Q. M. 明日涅槃 | [in private eval: can agent keep re-playing a game until it hits 100%?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/688135) |
| 103 | 1 | 0 | 2026-04-09 | Hellboy | [submission file to integration with a agent?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/689858) |
| 104 | 1 | 6 | 2026-04-12 | Ravi Annaswamy | [Importance of drastic data reduction via re-presentation and a Six stage approach](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/690676) |
| 105 | 1 | 5 | 2026-04-16 | Froggy McFrogson | [Do all entries run for 6 hours?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/692450) |
| 106 | 1 | 4 | 2026-04-18 | Froggy McFrogson | [What is patched vs standard FrameData?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/692809) |
| 107 | 1 | 3 | 2026-04-19 | Jacquelynn Hall | [RESET Confusion - UI vs API](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693123) |
| 108 | 1 | 3 | 2026-04-19 | Froggy McFrogson | [Weird result - limiting to available actions does WORSE!!](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693146) |
| 109 | 1 | 4 | 2026-04-20 | Van-Phuc Huynh | [Top-ranked users are bots.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693283) |
| 110 | 1 | 3 | 2026-05-05 | AIXM.ai-RyanX-Fufront | [Score discrepancy on submissions despite high local sim — request scoring trace (Human LB #1)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697407) |
| 111 | 1 | 8 | 2026-05-06 | mori42 | [What determines the termination of a game episode during online testing? (WIN / GAME_OVER / MAX_ACTIONS / time limit / custom is_done ?)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697423) |
| 112 | 1 | 0 | 2026-05-06 | biubiubiu~ | [Exploring LLM Integration in Agents: Beyong Traditional DL/ML Methods](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697476) |
| 113 | 1 | 2 | 2026-05-11 | Joep van Opdorp | [competition submission crashes](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/698749) |
| 114 | 1 | 2 | 2026-05-17 | bbarclay6 | [Arc 3 website test dataset. Is anybody getting higher than Average game score: 4.74%.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/700486) |
| 115 | 1 | 7 | 2026-05-31 | Van-Phuc Huynh | [How to run GPT-OSS on RTX 6000?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703506) |
| 116 | 1 | 5 | 2026-06-02 | Akhil Tolani | [Phase B scoring 30× lower than local evaluation](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703990) |
| 117 | 1 | 1 | 2026-06-04 | LEEKANG | [Did the 100 submission limit per team get lifted?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/704278) |
| 118 | 1 | 4 | 2026-06-09 | Sweety Seelam | [Submissions count per day](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705198) |
| 119 | 1 | 2 | 2026-06-10 | Maren Sajdaras | [What is the current_score in this competition?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705499) |
| 120 | 1 | 1 | 2026-06-11 | Nick Pellegrin | [Can we assume grid size?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/707717) |
| 121 | 1 | 3 | 2026-06-20 | Akhil Tolani | [Phase A Testing](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/710054) |
| 122 | 1 | 1 | 2026-06-25 | AAAAAtjc | [Action Cap of 5*human baseline clarification](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713921) |
| 123 | 1 | 1 | 2026-06-26 | Yield Smarter | [Which information is actually available in a submission? API documentation](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/714501) |
| 124 | 1 | 2 | 2026-06-30 | Dipam Chakraborty | [Submission parallelism questions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716295) |
| 125 | 1 | 6 | 2026-06-30 | Samuel Xu | [Insanely Fast Scoring?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716369) |
| 126 | 1 | 2 | 2026-07-03 | OverfitOracle | [looking for teammates.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/718572) |
| 127 | 1 | 0 | 2026-07-06 | Doruk Doğrular | [Bottleneck: Human playable synthetic game generation(assuming game types, need +25000 not 25 games)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/720662) |
| 128 | 1 | 2 | 2026-07-15 | Alvaro Camacho De Pass | [Run-to-run variance in the public score for a fixed agent](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/726552) |
| 129 | 1 | 0 | 2026-07-17 | Doruk Doğrular | [x(-1)/week is scary.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/726903) |
| 130 | 1 | 0 | 2026-07-22 | Akmal Xodarev (Busya PRIME) | [Reading the score exactly: finishing 4 of 6 levels scores 47.6, not 66.7](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728299) |
| 131 | 1 | 4 | 2026-08-04 | Endre Csiki | [Looking for a Python/DSL Developer – Abstract "Architect" Mind (100% on Task r11l)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732706) |
| 132 | 1 | 0 | 2026-08-07 | Antoine Matemane Mahirwe | [Solved (sort of): "system error" persisted across 7 straight submissions, even reproducing the exact code from the only successful run — fix was a brand-new kernel](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/733697) |
| 133 | 1 | 0 | 2026-08-09 | maximo lorenzo y losada | [Agent.MAX_ACTIONS defaults to 80, the loop takes 81, and the README never mentions it](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734054) |
| 134 | 1 | 2 | 2026-08-15 | FOYSAL | [Too many High Score](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735381) |
| 135 | 1 | 3 | 2026-08-17 | Pengyi Peng1 | [ARC-AGI-3 run went backwards on the leaderboard. What are we missing?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735590) |
| 136 | 1 | 2 | 2026-08-17 | Rahul Ray | [qwen3.8-27B model vs dataset](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735662) |
| 137 | 1 | 1 | 2026-09-01 | niulai | [Fail fast or lose hours: guard cells that protect your GPU quota](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738602) |
| 138 | 1 | 4 | 2026-09-05 | CreatZy | [The latest GPT Astra has killed the game?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739707) |
| 139 | 1 | 6 | 2026-09-08 | xz | [Anyone experiencing long queue time?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/740262) |
| 140 | 1 | 1 | 2026-09-24 | Nikolai Nedovodin | [Rules clarification: does self-generated synthetic data count as "External Data"?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742940) |
| 141 | 1 | 0 | 2026-09-24 | gedouluhui | [P3.1: Perception-Augmented Duck — What Prompt Engineering Can and Cannot Buy on ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743060) |
| 142 | 0 | 2 | 2026-03-25 | Andreas | [Ambiguity in prize rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/684682) |
| 143 | 0 | 0 | 2026-03-27 | Md Rakibul Islam Rocky | [How many of you guys are using AutoResearch 🙂](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/685364) |
| 144 | 0 | 1 | 2026-03-31 | Isaiah Nwukor | [5 Minute Commit Run Limit](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/686503) |
| 145 | 0 | 4 | 2026-03-31 | redstr | [Are the games guaranteed to be fully observable?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/686532) |
| 146 | 0 | 1 | 2026-04-03 | redstr | [Animations in the UI](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/687449) |
| 147 | 0 | 1 | 2026-04-09 | Sankalp Upadhyay | [Clarification on private model (non-LLM)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/689597) |
| 148 | 0 | 0 | 2026-04-13 | Isaiah Nwukor | [Has there been a maximum code run adjustment set to 2 minutes?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/690834) |
| 149 | 0 | 0 | 2026-04-13 | Froggy McFrogson | [LLMs and confidence predictions.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/690861) |
| 150 | 0 | 1 | 2026-04-16 | Nirvana Q. M. 明日涅槃 | [in Private Eval set: can we safely assume that at each level of a game, the grid's width & height would stay the same?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/692316) |
| 151 | 0 | 6 | 2026-04-19 | Froggy McFrogson | [Thoughts about why Random does so well.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693128) |
| 152 | 0 | 1 | 2026-04-20 | Bradley Shervheim | [Re-attempting a game later: does the scorecard update?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693307) |
| 153 | 0 | 3 | 2026-04-20 | Avery Scott | [404 Error for Scorecards and Replays](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693366) |
| 154 | 0 | 2 | 2026-04-22 | Ravi Makhija | [Understanding action space](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/693839) |
| 155 | 0 | 2 | 2026-04-23 | Froggy McFrogson | [Is there randomness in the games?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/694153) |
| 156 | 0 | 2 | 2026-04-26 | Ravi Makhija | [Understanding how submission works](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/694750) |
| 157 | 0 | 8 | 2026-04-26 | Ravi Makhija | [Potential web UI advantage](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/694757) |
| 158 | 0 | 6 | 2026-05-06 | Mario Santorelli | [i cant seem to submit my entry](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697564) |
| 159 | 0 | 0 | 2026-05-07 | Joelz | [Keep getting Kaggle error on GPU T4x2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697749) |
| 160 | 0 | 1 | 2026-05-09 | Shreyas | [From Public Chronos to v7: The Struggle of Teaching an Agent to Actually "Remember" Game Mechanics (and seeking advice!)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/698429) |
| 161 | 0 | 2 | 2026-05-10 | Zishuo Dong | [For ARC-AGI3, please confirm](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/698507) |
| 162 | 0 | 2 | 2026-05-14 | Pranav Kumar | [Update python version in Kaggle notebook](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/699517) |
| 163 | 0 | 1 | 2026-05-15 | Michael Poluektov | [Do we need to handle timeout explicitly?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/699817) |
| 164 | 0 | 18 | 2026-05-15 | JK-Piece | [Keep getting Kaggle Error.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/699859) |
| 165 | 0 | 3 | 2026-05-15 | xiayicheng3@gmail.com | [Concerning for competition rule: direct access to game code?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/699900) |
| 166 | 0 | 0 | 2026-05-19 | Hicham Hmidan | [Hicham Cr Style](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/701810) |
| 167 | 0 | 3 | 2026-05-21 | Nirvana Q. M. 明日涅槃 | [not safe to assume available actions stay the same throughout a game play?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/702079) |
| 168 | 0 | 3 | 2026-05-27 | Bernardus | [Parameter size](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/702970) |
| 169 | 0 | 1 | 2026-05-31 | Jakob Brüggen | [Making deterministic solutions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703488) |
| 170 | 0 | 5 | 2026-06-02 | Bernardus | [DeepSeek-V4 Telemetry: Running a 284B MoE at 0.00 GB Active VRAM  for competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703879) |
| 171 | 0 | 2 | 2026-06-08 | MagicSword | [What is FrameDataRaw? I want to know about the data structure that the agent observes.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705068) |
| 172 | 0 | 0 | 2026-06-09 | Marius Heuser | [What's the evaluation metric on kaggle?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705235) |
| 173 | 0 | 3 | 2026-06-12 | Moeed Nazki | [[ARC-AGI-3] Persistent Hidden Test Crashes: Heartbeats, Pydantic Validation, and TTA Overhead](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/707901) |
| 174 | 0 | 3 | 2026-06-12 | paul | [Exploit on .py game file](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/707925) |
| 175 | 0 | 1 | 2026-06-12 | Nick Pellegrin | [Common Approaches?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/707927) |
| 176 | 0 | 1 | 2026-06-13 | JackLX | [How to get rid of: / INFO / Successfully fetched metadata for game](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/708000) |
| 177 | 0 | 1 | 2026-06-16 | Sonam Eyden | [Kaggle error -  will using qwen model affect the submission status?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/708633) |
| 178 | 0 | 1 | 2026-06-19 | Nick Pellegrin | [How I'm Building 'Expert' Datasets](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/709542) |
| 179 | 0 | 1 | 2026-06-20 | Reki | [Is the June 30 Phase 1 ranking based on the public or private leaderboard?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/710898) |
| 180 | 0 | 5 | 2026-06-23 | Gabriel Mirea | [Confused about notebook timeout](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/712719) |
| 181 | 0 | 2 | 2026-06-25 | Alex Paul | [Solver stops after 3 hours despite 8-hour documentation](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/714128) |
| 182 | 0 | 0 | 2026-06-26 | Ocean | [Broken link in data page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/714574) |
| 183 | 0 | 7 | 2026-06-29 | Michal Z Gow | [Huge discrepancy between public and hidden results?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/715721) |
| 184 | 0 | 2 | 2026-06-30 | Dario Copparoni | [Find out if game level or game is concluded](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716504) |
| 185 | 0 | 0 | 2026-06-30 | Thiago Munhoz da Nóbrega | [Suggestion for next ARC Prize competition!](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716691) |
| 186 | 0 | 1 | 2026-07-01 | Smart Manoj | [Bug: Score mismatch in Notebooks Page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717055) |
| 187 | 0 | 1 | 2026-07-02 | JawaZero | [What does it look like if you exceed the 9h limit on computation?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717778) |
| 188 | 0 | 2 | 2026-07-07 | Atik Masud kanak | [Need Team](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/723601) |
| 189 | 0 | 2 | 2026-07-08 | Programmer Bill Ma | [Kaggle Error after submission](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/723792) |
| 190 | 0 | 1 | 2026-07-13 | Programmer Bill Ma | [Submit Error ~30min? Check for resource.setrlimit(RLIMIT_AS, ...)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724841) |
| 191 | 0 | 0 | 2026-07-15 | Thiago Munhoz da Nóbrega | [GPT-5.6 Sol sets a new SOTA on ARC-AGI-3: 7.8%](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/726340) |
| 192 | 0 | 11 | 2026-07-22 | OverfitOracle | [Is 100% Accuracy Realistic With the Available Compute?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728278) |
| 193 | 0 | 3 | 2026-07-27 | Ahmed Mobasher | [Three clarifications on final scoring mechanics](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985) |
| 194 | 0 | 4 | 2026-07-31 | Alex Paul | [Question About Dual Notebook Executions During Competition Submission](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/731290) |
| 195 | 0 | 3 | 2026-08-05 | Jason Feng | [A lot of Kaggle errors](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732974) |
| 196 | 0 | 0 | 2026-08-09 | Doruk Doğrular | [(Question to competition owners) - 1- Continuity of task similarity of arc-agi-2(cbebaa4b) and arc-agi-3(cn04)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734046) |
| 197 | 0 | 0 | 2026-08-09 | Jason Feng | [Chimpanzee-1.1: An RPS-Trained Model for ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734092) |
| 198 | 0 | 0 | 2026-08-10 | مشعل العتيبي | [Five consecutive "Kaggle Error" submissions — anyone else?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734233) |
| 199 | 0 | 0 | 2026-08-11 | mina wailin | [Submission finishes in ~30s with Phase-A dummy parquet (no gateway / no score badge) — Phase B not running?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734414) |
| 200 | 0 | 5 | 2026-08-12 | Jason Feng | [i can't submit to the competition if i have used up my gpu quota, but the competition scoring does not use my gpu quota.](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734585) |
| 201 | 0 | 2 | 2026-08-12 | Singaraj B | [Are the hidden evaluation game files available to the agent?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734677) |
| 202 | 0 | 2 | 2026-08-13 | mina wailin | [All my submissions score 0.00 — even an exact copy of the official Random Agent sample](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734989) |
| 203 | 0 | 0 | 2026-08-13 | Jason Feng | [Dynamic Value Model: 14% on ARC-AGI 3 public eval with DeepSeek V4 Flash (vs 6% baseline)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734994) |
| 204 | 0 | 2 | 2026-08-30 | Son Pham | [Successful run with 25-public set data failed when submitting](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738216) |
| 205 | 0 | 0 | 2026-09-01 | niulai | [GPT-OSS-120B works on the RTX Pro 6000 — offline vLLM 0.19 config + a fix for harmony-parser 500s (8% of requests)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738599) |
| 206 | 0 | 0 | 2026-09-01 | niulai | [Offline vLLM on Kaggle: the wheelhouse pattern, mount-path gotchas, and env vars that bite](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738600) |
| 207 | 0 | 1 | 2026-09-01 | Viktor Korsun | [Is the exploration phase burning score?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/738766) |
| 208 | 0 | 2 | 2026-09-14 | Homoalways | [Efficient skill acquisition in frontier LLMs](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/741248) |
| 209 | 0 | 0 | 2026-09-18 | mayakaripel | [Fresh official-style notebook fails with Kaggle Error during ARC-AGI-3 submission](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742000) |
| 210 | 0 | 0 | 2026-09-21 | Jason Feng | [Chain of Thought Reinforcement Decoding](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742342) |
| 211 | 0 | 3 | 2026-09-23 | Nick Pellegrin | [Finetuning a model (SFT or RL) on the RTX 6000 Pro?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742835) |
| 212 | 0 | 0 | 2026-09-25 | wkdrbwnd1 | [Partial-reasoning supervision made our agent stop acting: "no tool call" collapse in a small fine-tune (offline observations)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743319) |
| 213 | 0 | 0 | 2026-09-26 | Viswesh Suri | [Milestone 2: license of executable-source dataset dependencies](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743753) |
| 214 | 0 | 0 | 2026-09-27 | Vighnesh Hariharan | [ARC-AGI-3 Submission Environment and Minor Eligibility](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743785) |
| 215 | -1 | 0 | 2026-04-28 | Yeshua M. Coker | [Sharing Of Notebooks](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/695055) |
| 216 | -1 | 0 | 2026-07-23 | Doruk Doğrular | [Level pass animation - snake halucination](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728350) |
| 217 | -2 | 4 | 2026-04-09 | ShadowCoder | [Too few submissions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/689621) |
| 218 | -2 | 1 | 2026-07-08 | Programmer Bill Ma | [rerun 中读取 environment_files 自建本地模拟器是否合规](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/723752) |
| 219 | -2 | 7 | 2026-07-15 | Jakob Brüggen | [How long do you think it'll take until we reach AGI?](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/726367) |
| 220 | -3 | 0 | 2026-06-30 | Tsitsino Dzotsenidze | [Milestone Update: Adaptive Strategy Under Changing Conditions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/716637) |
| 221 | -4 | 0 | 2026-04-05 | Hellboy | [SETUP SERVER PROBLEM IS IN HERE IS CAN HELP TO SETUP SERVER API OKAY,THANK YOU? TypeError: Agent.__init__() missing 5 required positional arguments: 'game_id', 'agent_name', 'ROOT_URL', 'record', and 'arc_env'](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/688220) |
| 222 | -5 | 1 | 2026-07-22 | Maren Sajdaras | [A clarification for the input that enters the agent, FOR THE SAKE OF BETTER SCORE](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728210) |
| 223 | -5 | 0 | 2026-07-28 | Hayford Kofi Quaye | [Active Neuro-Symbolic Search Engine via Minimum Description Length for Interactive ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/730225) |
| 224 | -11 | 3 | 2026-08-05 | borro1980 | [Paper Track team-up: I have a finished write-up but no leaderboard score](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732932) |
| 225 | -13 | 1 | 2026-04-24 | PAJAKCHAI THONGCHAI | [The World in 2050 will be dream or dark. We can write it](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/694329) |

## Appendix B — Sister-forum topics screened as ARC-AGI-3-relevant

| Forum | Topic | Why relevant |
|---|---|---|
| ARC-AGI-2 | [709522 Historical leaderboard with runtime information](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/709522) | Same tracker (arc3.huikang.dev) covers ARC-AGI-3 |
| ARC-AGI-2 | [694745 Request to increase daily submissions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/694745) | Submission GPU time is not charged to the user quota; Greg: the semi-private set "shouldn't be used to test against" |
| ARC-AGI-2 | [701528 AI coding assistants](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/701528) | Greg: coding assistants allowed |
| ARC-AGI-2 | [740268 OSAID base-model requirement](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/740268) | Open question on Qwen eligibility (same rules text as ARC-AGI-3) |
| ARC-AGI-2 | [743581 Grand Prize writeup / licence](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/743581) | CC-BY-4.0 vs CC0/MIT-0 ambiguity |
| ARC-AGI-2 | [733501 Program Synthesis beats no reasoning](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/733501) | Schema/VISTA harness 99% on ARC-AGI-3 with Opus 5 (code world model + sandbox) |
| ARC-AGI-2 | [724647 Good to see movement on LB](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/724647) | Hint that NVIDIA uses code-execution SFT data (Nemotron-SFT-ARC-AGI-v1) |
| ARC-AGI-2 | [738313 Weird timeout](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/738313) | Infra noise: identical run timed out once |
| ARC-AGI-2 | [689054 Upgraded Accelerators](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/689054) | Accelerator plan (H100 was intended for ARC-AGI-3) |
| ARC-AGI-2 | [742796 LLM-agent is all you need](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742796) | Qwen3.8 agent reaches only 13.75% on ARC-2; cross-competition context |
| Paper Track | [694752 One unified writeup for ARC-AGI-2 and ARC-AGI-3?](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/discussion/694752) | Unified paper allowed; choose one primary notebook |
| Paper Track | [708161 TranscendPlexity](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/discussion/708161) | Claims offline Claude-written solvers for ft09/vc33 (public games only) |
