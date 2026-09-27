# Source patches to the vendored Duck harness

Each fix is an edit in the source itself (no runtime monkey-patching), tagged `[DUCK-PATCH <ID>]`
and guarded by `inference/agent/patches.py`. Switches are read at call time.

| Switch | Effect |
|---|---|
| `DUCK_PATCHES=0` | every fix off: the bundle's code paths byte-for-byte |
| `DUCK_PATCH_<ID>=0` | one fix off (`F1_IMAGES`, `F2_RESULT`, `F3_MEMORY`, `S1_STATE_KEY`) |

`agent/duck_offline.py --profile` bundles them: `patched` (all on, default), `shipped` (F1+F3 only =
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

## Not ported (and why)

| Notebook / campaign piece | Status |
|---|---|
| AGENTFIX F4 loop guard, F5 HUD no-op, F6 ACTION7, F7 ledger, F10 stall, F11 dedup | were OFF in the 09-22 run; F5/F6/F7 ideas go to B04/B05 as clean modules. **F6 note**: ACTION7 is advertised in `valid_actions` but `action_names.py` cannot map it (`Unknown action 'ACTION7'`, seen on ar25 in the mock). |
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
