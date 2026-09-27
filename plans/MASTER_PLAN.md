# ARC-AGI-3 Master Plan — "Hydra" (v1, 2026-09-27)

Owner: kragglenote2forwork · Target: **public LB 18–20+ (top-3 zone)**; stretch 25 · Today: best 3.85 (rank ~242/3365)
Horizon: 2026-09-27 → 2026-11-02 (36 days; GPU weeks start Sat 10-03, 10-10, 10-17, 10-24, 10-31)
Sources: `research/10–21`, `analysis/01`, campaign repo docs 01–16. Numbers carry their source.

---

## 1. Ground truth (verified)

| Fact | Value | Source |
|---|---|---|
| Hidden set | **110 games per submission**: 55 public LB + 55 private LB, scored in the same run (no final re-run) | research/11 §, host replies |
| Level score | `min(1.15, (human_median_actions / agent_actions)²)`; unfinished level = 0 | `arc_agi/scorecard.py` 0.9.8 (research/12 §1, 21 §5) |
| Game score | level-index-weighted mean (level k weighs k), capped at weighted fraction completed | same |
| LB score | plain mean over the 110 (55) games; unplayed game = 0 | same |
| Actions | every ACTION1-7 = 1 (dead clicks too), every RESET after level start = 1; **no action cap**; actions on a never-finished level cost **nothing** | Greg Kamradt, scorer code |
| Runtime | 9 h wall clock (32,400 s), 1× RTX PRO 6000 96 GB, no internet; kill after 15 min without interaction | rules, research/11 |
| Resets | competition mode: one `make` per game; RESET = level reset | research/12 |
| Quota | Kaggle GPU 30 h/week (shared with other comps) → **ARC budget 15 h/week**; scoring re-runs are **not** charged | `kaggle quota`, campaign doc 14 |
| Submissions | 1/day; failed ("Kaggle Error") submissions reportedly don't consume the slot | research/11 |
| Public-25 vs LB | LB ≈ public-25 ÷ 3–4 for Duck-family agents; **local public-25 barely predicts LB (r = 0.16 over 41 notebooks)** | research/10, 11 |
| LB noise | identical base notebook submitted 12×: mean 3.29, **sd 0.51**, range 2.38–4.33 → one submission can't resolve Δ < ~1 at this level | research/10 |
| Wave-fit bug | 110 games / 28 slots / 7,920 s cap → 4 waves need ~31.7 k s but only ~28 k s remain after setup → **last wave cut short in stock Duck** | research/10 |
| MTP cost | Flash-Next weights 81.8 GiB with MTP-3 vs **74.3 GiB with MTP off** (KV 105 k → 188 k tokens at 5 GiB); MTP off + 7 GiB KV + 28 seqs = 722 vs 359 tok/s (Thuitanium) | research/10 |

**Consequences**
1. **Depth beats efficiency.** A 7-level game: L1 = 3.6 %, L1-L3 = 21 %, L1-L5 = 54 %. Clearing one more level is worth
   far more than shaving actions on an already-cleared one. But once a level is cleared, 2× baseline actions keeps only 25 %.
2. **Wall-clock is the currency.** Every top team uses the full ~540 min. Since unfinished-level actions are free, the
   only cost of exploring is time; the cost of *sloppy play on a level you do finish* is quadratic.
3. **Variance is huge** (per-game all-or-nothing). Evaluate with paired comparisons and repeated submissions; choose
   final 2 submissions for robustness.

## 2. Where we are and why (diagnosis)

| Symptom | Cause | Evidence |
|---|---|---|
| 42 LLM turns per game per 2 h; 87 % of request time queued | **KV starvation**: Flash-Next NVFP4 weights 82 GiB → KV 5 GiB (105 k tokens = 3.2 × 32 k); 28 games share ~3 decode slots; prefix cache off → 20.5 M prompt tokens re-prefilled for 1.8 M generated | analysis/01, research/12 §4 |
| World-model / memory machinery never worked in production (v4 → v7) | state-key bug: every game shared one `unknown` state (fixed in campaign v7.1/v8, **never scored**) | campaign doc 14 update |
| Stuck from move 1 on g50t/sk48/cd82 | model commits to a wrong goal early and defends it; no verified-prediction discipline | research/11 §9, transcripts |
| LB 3.85 ≪ top 20+ | top teams jumped 5 → 20+ between 09-03 and 09-26 with Flash-Next + "squeezing more tokens out of the hardware and allocating them properly" (/742801) | research/11, 12 |

## 3. Strategy: four multipliers

Score ≈ (LLM decisions per game) × (quality per decision) × (action efficiency) × (time allocation). We are ~5× behind the
top; each multiplier below is worth 1.3–3× on its own, and they compound.

### M0 — "Hydra-0": combine the proven cheap wins nobody has combined (first submission, 10-03)
Evidence-backed, config/prompt-level, each individually measured by someone: Flash-Next **MTP off** + KV 7–12 GiB +
20–28 seqs (2× tok/s) · **wave-fit per-game budget** from measured remaining time (fixes the cut-off 4th wave) ·
ACTION7 executable as UNDO · scoring-rule + pacing prompt lines (+28 %) · analyzer_timeout 1200 · context budget
applied where it actually takes effect (MISTAKES #10) · keep Scott's animation-frame patch (the only patch that moved
the hidden LB, 3.20 → 3.71) · keep AGENTFIX F1–F3 + v7.1 state fix. Expected: 5–8 LB (vs 3.29 base mean).

### M1 — Compute (target ≥ 4× decisions per game). *Highest EV, lowest research risk.*
- **Serving A/B on real traffic** (GPU day 1): replay recorded Duck prompts as 28 concurrent multi-turn agents against
  - **A: Flash-Next NVFP4, tuned vLLM, MTP OFF** (frees 7.5 GiB): KV 5 → 12+ GiB, `--kv-cache-dtype fp8` (~2× tokens),
    context cap 16 k with compaction, max-num-seqs = KV-feasible concurrency, prefix caching ON (test hybrid-model support),
    MTP 2 vs 3.
  - **B: Qwen3.8-27B-FP8** (31 GB, 16 full-attn layers × 4 KV heads ⇒ 64 KiB/token bf16, **≈ 55–60 GiB KV ≈ 0.9–1.8 M tokens**):
    28–32 seqs, prefix caching ON (every turn of a game re-uses its own prefix), MTP/DFlash if available.
  - **C (stretch): Pennyroyal SGLang build** for Flash-Next (824 K-token KV, 446–632 tok/s aggregate) — only if it can be
    packaged offline and the license allows.
  - Metric: *effective decisions per game-hour* at equal quality proxy (tool-call validity, action rate). Pick the winner;
    the loser stays as a fallback profile.
- **Shorter prompts**: one board image per request (AGENTFIX F1), frame-delta instead of full ASCII after turn 1,
  compacted history, cap tool-output printing. Target ≤ 8 k prompt tokens median (today ~19.6 k).
- **Time allocation**: progress-aware scheduler — games that just cleared a level get priority; games stagnant for N turns
  on level 1 get fewer tokens (but never starve below the 15-min interaction kill).

### M2 — Decision quality (retained reasoning + verified memory)
- Keep `<think>` of the last K turns (`preserve_thinking`); replace blind eviction with a **model-written, harness-verified
  facts block** (controls, confirmed rules, goal hypothesis + evidence, plan). Frontier evidence: 3× for GPT-5.6 Sol,
  62.7 → 99.9 for GPT-6 Astra (research/12 §3).
- **Deterministic perception pack** in every observation: frame diff (raw ints), changed-object list with the avatar and its
  displacement, ranked click candidates (one per object/cell, dead-signature suppression), HUD/timer masking, animation
  summary. Community: +28 % hidden from click hints + scoring line (/743060).
- Prompt hygiene with measured wins: true scoring rule + `time_remaining` line; ACTION7 = UNDO where exposed; never RESET at
  level start; "verify *why* you won before the next level"; visible world-model updates (67 % of turns hid them).
- Carry the campaign's v7.1/v8 fixes (state key, typed memory, predict-verify, macro replay) — reimplemented cleanly.

### M3 — Action efficiency (quadratic payoff on cleared levels)
- **Plan-with-expectations action channel**: the model submits a batch + predicted effects; the runner executes step by
  step and halts on the first mismatch, returning the diff (Retrodict/Kepler/Tycho pattern).
- Repeat-no-op guard (free: blocked without an env action), death-transition bans, **auto-replay of the verified prefix**
  after GAME_OVER / level reset.
- Exploration budget: cheap deterministic probe of untested actions/click candidates capped at ~0.3–0.5 × expected
  baseline on a fresh level, only when the model has no confident plan.

### M4 — Adaptation (week 3+, only if M1–M3 plateau and the 27B path wins A/B)
- Behaviour-cloning / STaR LoRA of Qwen3.8-27B on **our own winning trajectories** from public-25 runs (never on
  hand-made public solutions — it backfires). Evidence: 1.25 → 1.94 LB (/739047). Flash-Next can't be trained on one card.
- Imitation action-prior (Tong Hui Kang's 0.4 M-param policy idea) as a *tool* ranking candidate actions.

**Not doing** (community-measured neutral/negative): 64 k context, T=0.3, genre playbooks, consensus voting per step,
multi-role pipelines, forcing an executable world model on small models, pure BFS/CNN driver agents (plateau ~0.27 LB),
hi-res images, training on public-game solutions.

## 4. Target architecture ("Hydra" = Duck core + four multipliers)

```
Kaggle notebook (thin) ── setup: vLLM/SGLang server (profile chosen by A/B) + watchdog + soft deadline 8h35m
   └─ scheduler: 110 games, K concurrent, progress-aware priority, 15-min-kill guard, per-game time floor
        └─ per game: Duck REPL agent (python tool, segmentation view)
             ├─ perception pack  (agent/perception.py)   frame delta · objects/avatar · click candidates · HUD mask · anim
             ├─ memory           (agent/memory.py)       retained think(K) + verified-facts compaction + per-game state
             ├─ action channel   (agent/actions.py)      plan+expectations · halt-on-mismatch · no-op/death guard · replay
             └─ prompt           (agent/prompts.py)      scoring + time line · UNDO · reset discipline · visible updates
```
Code lives in `agent/` as a normal package (no runtime monkey-patching), shipped as a private Kaggle dataset
`kragglenote2forwork/arc3-hydra-src` (versioned per build), plus the chosen model + runtime datasets.

## 5. Calendar and GPU budget (≤ 15 h/week; submissions 1/day)

Submission commits use **smoke mode** (model load + 3 public games × 8 min ≈ 0.4 h GPU) — the real 9 h play happens in the
free Kaggle re-run. Full public-25 validation runs (≈ 2.5 h) are reserved for decisions that need them.

| Week | Dates | GPU plan (h) | Goals / decisions | Submissions |
|---|---|---|---|---|
| **W0** | 09-27 → 10-02 | 0 (quota empty) | CPU only: build Hydra package on the Duck source; perception pack + action channel + memory tested on all 25 games offline; serving load-test kit; notebook builder + preflight; HF datasets; per-game mechanics study | none (per user) |
| **W1** | 10-03 → 10-09 | 1.5 serving A/B · 2 × 2.5 public-25 · 7 × 0.4 smoke ≈ 9.3 (+5.7 spare) | Day 1: serving A/B → choose model/profile. Ship Hydra-1 (M1 only), then Hydra-2 (M1+perception+prompt). Re-submit best once for variance. | 7 |
| **W2** | 10-10 → 10-16 | ≈ 12 | M2 memory/compaction + M3 action channel; paired A/B via submissions; start collecting winning trajectories | 7 |
| **W3** | 10-17 → 10-23 | ≈ 15 (incl. ≤ 5 h LoRA if 27B path) | M4 LoRA experiment (27B) OR scheduler/time-allocation tuning (Flash-Next); ablate losers away | 7 |
| **W4** | 10-24 → 10-30 | ≈ 12 | **Freeze by 10-26** (publishing/team cut-off); robustness: crash-proofing, deadline safety, 2–3 re-submissions of the top candidates | 7 |
| **W5** | 10-31 → 11-02 | ≈ 3 | Final re-submissions; select final 2 (best public mean over repeats, different seeds/configs) | 3 |

Interim targets (public LB): W1 ≥ 6 · W2 ≥ 10 · W3 ≥ 14 · W4 ≥ 18. If a week misses its target by > 30 %, the Saturday
review must change the plan (not just tune): e.g. swap serving profile, drop a multiplier that shows no gain.

## 6. Experiment discipline
- One thesis per submission; log it in `plans/EXPERIMENTS.md` before pushing (ID, change, expected Δ, falsifier).
- Compare configs on **(a)** LB — the only signal that matters; sd ≈ 0.5 at LB 3 → submit a candidate ≥ 2× before
  concluding when Δ < 1, **(b)** public-25 levels cleared + decisions/game-hour (sanity + throughput only; r = 0.16 with LB),
  **(c)** CPU replay metrics (prompt tokens, invalid tool calls, wasted actions).
- Keep a rollback candidate (the best-scoring version) re-submittable at all times.
- After every run: post-mortem in `analysis/NN_*.md`, update `plans/STATUS.md`, `plans/MISTAKES.md`.

## 7. Automation
- **Daily (06:4x UTC)** — `automation/routines/daily.md`: sync LB/submissions/quota → update STATUS → if a validated
  candidate is queued in `plans/SUBMISSION_QUEUE.md` and gates pass: build → push (smoke commit) → wait → submit →
  log; else continue the top BACKLOG item on CPU. Never before 2026-10-03.
- **Weekly (Saturday)** — `automation/routines/weekly.md`: research sweep (new Kaggle discussions/notebooks, LB deltas,
  HF models/datasets, web), write `research/weekly/<date>.md`, grade the week vs targets, rewrite next week's section of
  this plan, refresh the GPU ledger, and queue the week's submissions.

## 8. Risk register
| Risk | Mitigation |
|---|---|
| Serving change breaks tool parsing / multimodal | A/B includes a correctness check (valid tool calls %, parse errors) on replayed prompts |
| Silent vLLM hang, 15-min interaction kill | watchdog (restart ≤ 2), per-game heartbeat, fallback "safe action" is NOT used (actions cost) — instead keep games alive by scheduling |
| Kernel title ≠ slug → push renames kernel | preflight asserts title == slug title |
| GPU slots blocked by other comps (max 2 batch sessions) | daily routine checks running kernels and retries later; never schedule ARC + other GPU work the same day |
| Deadline overrun (9 h) | soft deadline 8 h 35 m; games stop cleanly; submission.parquet written by gateway regardless |
| LB variance hides real gains | repeat submissions of top 2 candidates; decide on ≥ 2 datapoints when Δ < 20 % |
| Qwen 4 27B / new model drops | weekly sweep watches; the serving kit makes a swap a 1-day job |
