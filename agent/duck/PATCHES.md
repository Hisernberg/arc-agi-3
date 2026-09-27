# Source patches to the vendored Duck harness

Each fix is an edit in the source itself (no runtime monkey-patching), tagged `[DUCK-PATCH <ID>]`
and guarded by `inference/agent/patches.py`. Switches are read at call time.

| Switch | Effect |
|---|---|
| `DUCK_PATCHES=0` | every fix off: the bundle's code paths byte-for-byte |
| `DUCK_PATCH_<ID>=0` | one fix off (`F1_IMAGES`, `F2_RESULT`, `F3_MEMORY`, `S1_STATE_KEY`, `P1_SCORING`, `P2_UNDO`, `P3_DISCIPLINE`) |
| `DUCK_PATCHES_V9=0` | the B06/B07 "v9" group (P1-P3, M1, M2) off = exactly the B01 set (pinned by `tests/golden/`) |
| `DUCK_PATCH_M1_THINK=1`, `DUCK_PATCH_M2_FACTS=1` | opt-in: retained reasoning (M1), verified facts + compaction (M2), below |
| `DUCK_PATCH_B08_SCHEDULER=1` | opt-in: progress-aware turn scheduler (B08, below) |

`agent/duck_offline.py --profile` bundles them: `patched` (every default: F1/F2/F3/S1 + P1/P2/P3),
`v9` (patched + M1 + M2), `pre_v9` (F1/F2/F3/S1 only, the code before B06/B07), `shipped` (F1+F3 only =
the AGENTFIX set live in the 09-22 notebook `taaf-flashnext-sheetu12b-0922`), `stock` (`DUCK_PATCHES=0`).

## F1 — images in history (`DUCK_PATCH_F1_IMAGES`, AGENTFIX F1)

*Where*: `tool_agent.py` `_persistent_history_messages` → `_strip_history_images`; `_estimate_tokens`
→ `_flat_image_charge`; `ToolAgent.__init__` → `_request_safety_margin_tokens()`.

- Carried history keeps the board image only in the newest `DUCK_PATCH_F1_IMAGE_KEEP` (default **2**,
  as shipped) image-bearing user messages; older ones keep their text. A request therefore holds at
  most 3 images (2 history + current). `F1_IMAGE_KEEP=0` gives one image per request.
- The trimming estimator charges each image a flat `DUCK_PATCH_F1_IMAGE_TOKENS` (120) instead of
  `len(base64)//3` (~1.5 k per 768 px board, 5-8x too high), so history is no longer evicted to make
  room for phantom image tokens.
- With accurate image estimates the reply needs explicit room: safety margin 512 → 6144
  (`DUCK_PATCH_F1_SAFETY_MARGIN`), i.e. context budget 32768 − 512 − 6144 = **26112** estimator
  tokens — exactly what the 09-22 production transcripts report.

Evidence (AGENTFIX notes): median 8 obsolete boards per request, ~26 % of the budget. Mock: stock sends
4.3 images/request (max 6) vs 2.8 (max 3) patched.

## F2 — action result always returned (`DUCK_PATCH_F2_RESULT`, AGENTFIX F2)

*Where*: `tool_agent.py` `_run_python_tool` (attach `last_action_result`), `_build_user_prompt`
(outcome line).

- A python call that executed an action always carries `last_action_result` (executed, action_num,
  level, score, board_changed, level_completed, game_over, run_complete, stop_reason, stop_detail,
  executed/requested counts). Stock dropped the result whenever the snippet printed anything (81 % of
  acting calls in production; 33/33 acting calls in the mock).
- The next user prompt gets the harness's own one-line outcome (`_describe_last_outcome`, previously
  dead code), e.g. `Actions 4-6 (3 total) produced a board change; verify ...`.
- **Not in the 09-22 notebook** (`AGENTFIX_RESULT=0` there); enabled here because the fix is
  model-visible information only. Use `--profile shipped` for exact 09-22 parity. The HUD-only variant
  of the outcome line depended on AGENTFIX F5 and is not ported.

## F3 — memory parsing (`DUCK_PATCH_F3_MEMORY`, AGENTFIX F3)

*Where*: `tool_agent.py` `_extract_labeled_blocks` → `_extract_labeled_blocks_tolerant`;
`_update_summarized_knowledge_from_step_summary`.

- World-model labels may carry a short qualifier: `World model (revised):`, `World model update:`,
  `Plan (next):`, `Plan v2:`. The stock exact-prefix match silently dropped these, so the carried
  "working world model" was often empty. Prose like `Plan for the next move:` is still not a label.
- GAME_OVER no longer wipes the world model (the runner auto-resets: it is a retry of the same level);
  a level transition or run completion still does.

## S1 — per-game state key (`DUCK_PATCH_S1_STATE_KEY`, campaign v7.1 fix, MISTAKES #2)

*Where*: `runtime_state.py` `parse_state_path` / `game_key_from_state_path` (the single parser for
`<game_id>_p<N>_tool_runtime_state.json`); `tool_agent.py` `_resolve_run_artifact_location`,
`_ensure_session`, `ToolAgent.game_key`.

- v4-v7 derived the game key from the state path with a heuristic that skipped `*.json` components →
  every game shared one `unknown` state. The stock agent has no per-game store, but the same weakness
  exists in two places, both fixed:
  1. `_ensure_session` keyed a session by the state file's **directory** — shared by every game of a
     run — so an analyzer reused across games would carry one game's history into another. Now the
     session is the state **file**; `ToolAgent.game_key` exposes the parsed game id (for B06 memory).
  2. `_resolve_run_artifact_location` wrote the shared `prompts/prompt.log` whenever ≤ 1 state file
     existed at that instant (the first game, or the last game still running). Reproduced in the mock:
     the stock profile writes `prompts/prompt.log` next to the per-game logs; patched never does.
- Anything keyed per game (B06 `agent/memory.py`) must use `game_key_from_state_path`; the test
  `test_state_key_parses_the_production_state_path` pins the production path format.

## B08 — progress-aware scheduler (`DUCK_PATCH_B08_SCHEDULER`, **opt-in**, default off)

*Where*: `framework/solver.py` — `_b08_module` / `_b08_make_gate` / `_b08_write_summary` (module level),
`HarnessSolver._run_games` (one worker thread per game, gate built from the wall clock left *now*),
`HarnessSolver._play_one` (finally: `gate.finish`, split into `_play_one_inner`), `_HarnessGameSession`
(`scheduler_gate` field; `acquire` before / `release` after every `analyze()` call; `should_stop` sees a
retired game; `runtime_limit_reached` defers to the gate; `timing_payload` reports the gate's remaining time).
Policy lives in `agent/scheduler.py` (loaded by file path, so no `sys.path` change is needed).

- Off (default): the stock loop, byte-for-byte behaviour (semaphore of `concurrency`, fixed
  `max_runtime_s_per_game` per game, gateway order).
- On: `concurrency` = turn slots; per-game floor/cap from `plan_budget(n_games, soft_deadline - now)`;
  priority hot > fresh > normal > stagnant (weighted fair queuing), keepalive at 7 min idle (no active game
  waits > 10 min), start curve so every game starts by half the window, no turn after `deadline - 90 s`.
  Knobs: `DUCK_PATCH_B08_SCHEDULER_CONFIG` = JSON of `SchedulerConfig` fields (+ `window_s`);
  summary JSON in `<job_dir>/hydra_scheduler.json` (or `DUCK_PATCH_B08_SCHEDULER_SUMMARY`); log lines
  `HYDRA_SCHED start|admit|retire|status|final`.
- Evidence (simulation, `tests/test_scheduler.py`, 110 games / 28 slots / 28 k s, 10 seeds): +4-6 % levels vs
  wave-fit and +6-8 % vs stock TAAF at the 09-22 throughput (0.163 turns/s), +4 % at 4x throughput, winning 9-10
  of 10 seeds; stock's fourth wave gets ~55 % of its head's service, the scheduler's tail ~85 %. With 16 slots or a
  20 k s window stock never starts 26-46 games; the scheduler starts all. Synthetic games only: the real gain
  needs a GPU A/B. Real-loop test: 4 games / 2 slots through the
  mock LLM, never > 2 concurrent `analyze()` calls, turns interleaved, per-game state isolated.

## P1 — true scoring rule + game clock (`DUCK_PATCH_P1_SCORING`, B07, default on)

*Where*: `prompts.py` `SCORING_RULE_V9` (replaces the `GAME_OVERVIEW_ADDENDUM` line "Optimize for as few
in-game actions as possible", which contradicts depth > efficiency); `tool_agent.py` `_build_system_prompt`,
`_game_clock_line`, `_build_user_prompt`, `ToolAgent.set_game_status`; `framework/solver.py`
`_HarnessGameSession.game_status` + the call before every `analyze()`.

- System prompt: per-level `min(1.15, (human/agent)^2)`, uncleared level = 0, level k weighs k, every
  action on a cleared level counts (RESET/UNDO/dead clicks too), actions on a never-cleared level cost
  nothing, depth beats efficiency; pace with the clock, switch hypothesis class after repeated failures.
- Every turn prompt (after the state line): `Game clock: 21 min used, 111 min left (84% of this game's
  time). Levels cleared: 1 of 7.` Remaining = the sooner of the per-game budget and the run's soft
  deadline (and, with B08 on, the scheduler's own remaining time via `timing_payload`).
- Evidence: gedouluhui (743060) measured +28 % hidden (1.12 -> 1.43) for the scoring line + `time_remaining`
  pacing together with click hints.

## P2 — ACTION7 = UNDO, executable (`DUCK_PATCH_P2_UNDO`, B07, default on, MISTAKES #11)

*Where*: `action_names.py` (`_P2_*` maps, `_maps()`), `prompts.py` `UNDO_LINE_V9`, `tool_agent.py`
`_build_user_prompt`.

- Stock advertised `ACTION7` (ar25, bp35, lf52, sb26, sk48, su15 expose it; all implement it as undo) but
  `to_engine_action("ACTION7")` returned None -> `Unknown action` on every call. Now it is shown as `UNDO`
  in `Valid actions right now`, executes under both names, and the turn prompt adds (only when UNDO is
  valid): "UNDO reverts your previous action and costs one action; prefer it over RESET ...".
- Test `test_every_advertised_action_is_executable`: every advertised action of all 25 games round-trips.

## P3 — play discipline + visible notes (`DUCK_PATCH_P3_DISCIPLINE`, B07, default on)

*Where*: `prompts.py` `PLAY_DISCIPLINE_ADDENDUM_V9`, `VISIBLE_NOTES_LINE_V9`, `LEVEL_CLEARED_LINE_V9`;
`tool_agent.py` `_build_system_prompt`, `_build_user_prompt`.

- System prompt "Play discipline": never RESET at a level's start (RESET costs an action); after clearing
  a level state which action/effect cleared it and verify on the new board; no mapping to known video
  games; only labelled notes in the *visible* reply are carried forward (66.8 % of Qwen tool-call turns
  had hidden reasoning only, research/11 734843).
- Turn prompt: the optional-prefix line becomes "Before each tool call, write your revised notes as
  visible text ..."; on the first turn after a level transition: "first write `Recent findings:` saying
  which action and effect cleared it ...".

System prompt size (P1+P3): **12,681 -> 13,675 chars (+994)** = +284 tok at chars/3.5; Qwen3.6-family
tokenizer (`tokenizer.json` from the HF mirror in scratch): 2,855 -> 3,079 (+224). Test cap: +300 x 3.5 chars.

## M1 — retained reasoning of the last K turns (`DUCK_PATCH_M1_THINK`, B06, **opt-in**)

*Where*: `tool_agent.py` `_strip_history_reasoning`, `_persistent_history_messages`, `_chat_completion`;
`utils/openai_compat.py` `build_chat_payload(extra_chat_template_kwargs=...)`.

How stock handles reasoning (checked while porting): `_extract_reasoning_text` reads `reasoning`, then
`reasoning_content`, strips `<think>` tags/blank lines and stores it on the history message as
`reasoning` only. **Every** carried assistant turn keeps it until the whole turn is evicted; the
trimming estimate counts it (7.3 k of 21.1 k tokens/request in the mock). The request sends only
`chat_template_kwargs={"enable_thinking": true}`; whether old reasoning is *rendered* depends on vLLM
mapping `reasoning` -> the template's `reasoning_content` and on the template's `preserve_thinking`
default (Qwen3.8 README: on by default; the Qwen3.6-family template renders history thinking only when
`preserve_thinking` is true). Production's 19.6 k tokens/request is close to the mock's with-reasoning
figure (21.1 k) rather than without (13.1 k), so it was probably rendered — **unverified; check
`usage.prompt_tokens` M1 on/off on GPU (B02)**.

- Carried history keeps `reasoning` on the newest `DUCK_PATCH_M1_THINK_KEEP` (default **2**) assistant
  messages; older ones lose it *before* trimming, so the budget buys visible history instead. The
  in-flight turn always keeps its own reasoning.
- On the wire (after trimming, so estimates are unchanged) each retained `reasoning` is mirrored into
  `reasoning_content` (the field Qwen templates read) and `chat_template_kwargs.preserve_thinking=true`
  is sent, so the kept reasoning is rendered regardless of server defaults.
- Alone, M1 does not shrink requests at the 26 k budget (the freed room fills with ~2x more visible
  turns); combined with M2 it bounds the reasoning share (~2.2 k of 10.5 k).

## M2 — verified-facts block + compaction + lower budget (`DUCK_PATCH_M2_FACTS`, B06, **opt-in**)

*Where*: new module `inference/agent/facts_memory.py` (`FactsLedger`); `tool_agent.py`
`_summarized_knowledge_lines`, `_update_summarized_knowledge_from_step_summary`, `_run_python_tool`,
`analyze` (ingest + protected turn prompt), `_trim_messages_for_context`, `_effective_budget_tokens`,
`_compact_history_prompts`, `_persistent_history_messages`, `set_game_status`/`_ensure_session`.

- **Facts block** (replaces "Working world model carried from earlier turns"), rebuilt every turn:
  1. harness-verified effects: every executed action is read back from the runtime-state history the
     solver writes (real before/after frames) and tallied per (level, action key): `changed` (cells
     outside a 3-cell border band, ~cells + last bbox), `HUD/edge-only` (<= 4 band cells: the per-action
     step bar — ls20 rows 61-62, ar25 col 63), `no change`, `level up`, `GAME_OVER` (from the harness's
     action results). MOUSE is keyed by the clicked cell's colour (`MOUSE@R`) with sample hit/dead
     coordinates; current level in detail, earlier levels merged, `Cleared by: L1 at action 23 (last:
     ...)` per cleared level;
  2. level history from the solver: levels cleared, actions used per level `/human baseline` when the
     environment exposes baselines (None in submission mode -> omitted);
  3. the model's latest labelled notes + each cleared level's last notes (archived instead of wiped).
  Capped at `DUCK_PATCH_M2_FACTS_TOKENS` (default 700) enforced as chars = tokens x 3 (2,100 chars,
  ~600 tok at chars/3.5); only model notes are truncated — verified facts switch to a compact
  rendering but every (level, action) key stays (`test_facts_ledger_classifies_caps_and_never_drops...`).
- **Compaction**: carried (old) turn prompts keep only their per-turn header (outcome lines + `Current
  state:`); follow-ups keep their first sentence. The static instructions and the facts block are sent
  once, in the newest prompt.
- **Budget**: trimming uses `min(window budget, DUCK_PATCH_M2_BUDGET)` (default **12,000** estimator
  tokens; the shipped effective budget was 26,112). The current turn's prompt (facts + board) is never
  evicted by in-flight growth (`protect=`). Transcripts print the effective `context_budget_tokens`.
  Note for serving (B03): with `LOCAL_ANALYZER_CONTEXT_WINDOW=16384` the window budget is 9,728
  (16384 - 512 - 6144) and wins over 12,000.
- Per-game isolation: the ledger and the solver status belong to the session and are reset in
  `_ensure_session` whenever the state file changes (S1 key); tests assert ledger key == game id, action
  counts per game, and no cross-game notes.

Measured (mock, ls20/vc33/ar25 x 40 steps, chars/3.5 + images by pixels; `analysis/03_prompt_memory_v9.md`):
median prompt tokens/request **21,093 (pre_v9) -> 21,224 (P1-P3) -> 10,486 (v9 = P + M1 + M2)**, max
22,941 -> 11,179; total prompt tokens for the 120 actions (3 x 40) 2.60 M -> 1.29 M (-50 %).

## Not ported (and why)

| Notebook / campaign piece | Status |
|---|---|
| AGENTFIX F4 loop guard, F5 HUD no-op, F6 ACTION7, F7 ledger, F10 stall, F11 dedup | were OFF in the 09-22 run; F5/F6/F7 ideas go to B04/B05 as clean modules. **F6**: ported as P2 (ACTION7 was advertised but unmappable: `Unknown action 'ACTION7'` on ar25); F7's per-action effect ledger is the verified half of M2. |
| AGENTFIX F9 timing lines | logging only; the mock records per-request tokens instead |
| agentfix_anim F13 (animation frames in the sandbox + one system-prompt line), agentfix_sheet F19 (frame-sheet image) | ON in the 09-22 run, not ported: they add prompt content/images. Our system prompt equals the production one except the F13 line (`Some actions animate: ...`). |
| ARM P (`MULTIMODAL_UPSCALE` 4 → 12) | env only: `SHIPPED_ANALYZER_ENV` in `duck_offline.py` sets 12 |
| campaign v8 (`apex_v8_patch.py`: world_model block, per-game notes/baselines, macro replay, predict-verify) | not ported: B04-B06 rebuild these as `agent/` modules; its per-public-game notes/baselines must not ship (CLAUDE.md: no public-game hard-coding) |

## Config findings made while vendoring

- The 09-22 notebook's "ctx16k" profile never reached the analyzer: cell 3 sets
  `LOCAL_ANALYZER_CONTEXT_WINDOW=16384`, but cell 9 re-applies the setup env (`32768` from
  `serving_setup.py`) before `tool_agent` is imported; production transcripts show
  `context_budget_tokens: 26112` (32 k window). vLLM also ran `--max-model-len 32768`.
- Transcripts and `prompts/*.log` render tool results with `_render_tool_result_display`, which shows
  only stdout/result/error: extra keys the model does receive (`last_action_result`, `truncated`) are
  invisible in the logs. Read `mock_metrics.json` → `tool_results` for the exact model input.
- The sandbox receives the full history of 64x64 grids on every `action()` (O(n²) per game) and the
  runtime-state file grows the same way; fine at ~100 actions, worth capping for long games.

## Mock token baseline for B06/B07 (2026-09-27)

`duck_offline.py --compare-profiles --steps 40` (ls20, vc33, ar25; seed 0; 131 requests per profile).
Tokens = chars / 3.5 (+16 chars per message); images = one token per 32x32 px (+2), i.e. 578 per
768x768 board at upscale 12 — **unverified for Qwen3.8's vision tower; measure on GPU (B02)**. The mock
writes 4000 chars of reasoning + ~1.4 k chars of code per reply (≈ 1.55 k completion tokens, production
average 1.72 k). Old-turn reasoning is sent in the `reasoning` field; whether the chat template renders
it is also unverified, so both totals are shown.

(`patched` in this table = the B01 patch set, today's `pre_v9` profile; the B06/B07 profiles are
measured in `analysis/03_prompt_memory_v9.md`: v9 median 10 486, max 11 179.)

| profile | prompt tok/request median (p90, max) | excl. old reasoning median | images/request mean (max) |
|---|---|---|---|
| patched | 21 093 (22 436, 22 941) | 13 102 | 2.8 (3) |
| shipped | 21 556 (22 584, 22 945) | 12 797 | 2.8 (3) |
| stock | 19 243 (21 496, 22 344) | 13 185 | 4.3 (6) |

Fixed, harness-authored parts (exact): system prompt **12 681 chars ≈ 3 623 tok**, tool schema 1 127
chars ≈ 322, user prompt 2 770-2 971 chars ≈ 790-850 (+ carried world model), current board ≈ 578.
First request of a game ≈ 5.3 k. Steady-state mean composition (patched): old reasoning 7.3 k,
history user prompts 3.8 k, system 3.6 k, images 1.6 k, tool-call code 1.5 k, current user prompt
1.0 k, tool results 0.5 k (F2 adds ~0.24 k), tool schema 0.3 k.

**Takeaway**: prompts grow 5.3 k → 22 k over the first ~8 requests (~5 analysis turns), then sit at the
trimming budget (26 112 estimator tokens ≈ 22.5-23 k by chars/3.5), so the steady-state size is set by the budget and
the 30-assistant-turn cap, not by content. Production measured ~19.6 k/request (20.46 M / 1 045 calls).
Shrinking prompts (B06/B07, target ≤ 8 k) therefore needs a smaller budget or compaction, not only
shorter prompts. The system prompt alone is ~18 % of a steady-state request.
