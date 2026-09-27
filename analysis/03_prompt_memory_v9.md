# 03 — Prompt v9 (B07) + memory/compaction (B06): what changed and what it costs per turn

Date 2026-09-27 · CPU only (mock LLM) · code: `agent/duck/src/ARC3-Inference/inference/agent/`
(`[DUCK-PATCH P1/P2/P3/M1/M2]`), docs: `agent/duck/PATCHES.md`, tests: `tests/test_duck_prompt_memory.py`.

## TL;DR

- **Prompt v9 (P1-P3, default on)**: the true scoring rule + a per-turn game clock, ACTION7 exposed and
  **executable** as UNDO (it was advertised but rejected on 6/25 public games, MISTAKES #11), reset
  discipline, "verify why you won", no video-game mapping, visible world-model notes. System prompt
  **12,681 → 13,675 chars (+994; +284 tok at chars/3.5; +224 with a Qwen3.6-family tokenizer)**, within +300.
- **Memory (M1 + M2, opt-in)**: retained reasoning of the last K=2 assistant turns, a harness-verified
  facts block (≤ 2,100 chars), compacted history prompts and a 12,000-token trimming budget (was 26,112).
- **Prompt tokens/request (mock, 3 games × 40 steps): median 21,093 → 10,486 (−50 %), max 22,941 → 11,179;
  total 2.60 M → 1.29 M for the same 120 actions.** Since the 09-22 serving was prefill-bound at ~20 k
  context, this should buy up to ~2× requests per GPU-hour before any serving change (expected, not yet
  measured; B02/B03).
- Every v9 switch off reproduces the pre-v9 requests **byte-for-byte** (golden fingerprints captured
  before the change); 27/27 Duck tests green.

## 1. What changed

| ID (switch) | Default | Change |
|---|---|---|
| P1 `DUCK_PATCH_P1_SCORING` | on | System prompt: `min(1.15,(human/agent)^2)` per cleared level, uncleared = 0, level k weighs k, every action on a cleared level counts (RESET/UNDO/dead clicks too), actions on a never-cleared level cost nothing, depth beats efficiency, pace with the clock, switch hypothesis class after repeated failures (replaces "optimize for as few actions as possible"). Every turn: `Game clock: 21 min used, 111 min left (84% of this game's time). Levels cleared: 1 of 7.` (solver → `ToolAgent.set_game_status`; remaining = min(per-game budget, soft deadline); B08's gate time when on). |
| P2 `DUCK_PATCH_P2_UNDO` | on | `ACTION7 ↔ UNDO` in `action_names.py`; shown in `Valid actions`, executable under both names; one turn-prompt line only where UNDO is valid ("reverts your previous action and costs one action"). |
| P3 `DUCK_PATCH_P3_DISCIPLINE` | on | System "Play discipline" (never RESET at a level's start; state what cleared a level and verify on the new board; no video-game mapping; only visible labelled notes are carried). Turn prompt: notes must be visible text; after a level transition, write `Recent findings:` on why it was cleared first. |
| M1 `DUCK_PATCH_M1_THINK` (`_KEEP`=2) | **off** | Carried history keeps `reasoning` on the newest K assistant messages only (stripped *before* trimming); the in-flight turn keeps its own. On the wire: `reasoning_content` mirror + `preserve_thinking: true`. |
| M2 `DUCK_PATCH_M2_FACTS` (`_BUDGET`=12000, `_FACTS_TOKENS`=700) | **off** | `FactsLedger` facts block replaces the free-form carried world model; old turn prompts compacted to their header; trimming budget min(window, 12,000); current turn prompt protected from eviction. |

Why M1/M2 are opt-in: they change what the model remembers and whether old thinking is rendered; the
mock can prove mechanics and token counts, not decision quality. They need a GPU A/B (proposal in §6).
P1-P3 are text + a bug fix with community evidence (+28 % hidden for the scoring/time lines, /743060).

## 2. Reasoning handling in the shipped harness (task check)

- `_extract_reasoning_text` reads `reasoning`, then `reasoning_content`, strips `<think>` tags and blank
  lines; the assistant history message stores it as **`reasoning` only**.
- Nothing drops reasoning selectively: every carried assistant turn keeps it until its whole turn is
  evicted, and the estimator counts it (mock: 7.3 k of 21.1 k tokens/request).
- `openai_compat.build_chat_payload` sends `chat_template_kwargs={"enable_thinking": true}` only. Whether
  history thinking is *rendered* depends on vLLM passing `reasoning` to the template's `reasoning_content`
  and on the template's `preserve_thinking` (Qwen3.8 README: on by default; the Qwen3.6-family template in
  our HF mirror renders history thinking only when `preserve_thinking` is true). `serving_setup.py`
  (our path) uses the model's `chat_template.jinja` without `--default-chat-template-kwargs`;
  TAAF's `kaggle.py` (unused) passes `preserve_thinking: true`.
- Production 19.6 k tokens/request is close to the mock's with-reasoning 21.1 k, not the without 13.1 k,
  so old reasoning was **probably rendered** — unverified. M1 makes it explicit either way.

## 3. Measurement

`duck_offline.run_offline(("ls20","vc33","ar25"), steps=40, seed=0)` per variant (same scripted policy:
4,000-char reasoning + ~1.4 k chars code per reply, the 16-request opening script, then weighted random
scenarios). Tokens = chars/3.5 (+16 chars/message) + images at 1 token per 32×32 px (+2), as in
PATCHES.md; "Qwen tok" = the text parts through a Qwen3.6-family `tokenizer.json` (vocab assumed close
to Qwen3.8's) + the same image estimate. Script: `SCRATCH/b06/measure2.py`; raw: `SCRATCH/b06/measure40_final/measure.json`.

| variant | requests | median | p90 | max | turn-start median | excl. old reasoning | Qwen tok median | images/req | total prompt tok |
|---|---|---|---|---|---|---|---|---|---|
| **pre_v9** (before) | 131 | **21,093** | 22,436 | 22,941 | 21,041 | 13,102 | 16,606 | 2.8 | 2.60 M |
| P1-P3 (new default) | 130 | 21,224 | 22,794 | 22,901 | 21,355 | 13,405 | 16,827 | 2.8 | 2.58 M |
| P + M1 (K=2) | 130 | 21,274 | 21,924 | 22,462 | 21,341 | 18,353 | 18,483 | 2.8 | 2.47 M |
| P + M2 (12 k) | 130 | 9,610 | 10,472 | 10,872 | 9,643 | 7,271 | 7,825 | 2.0 | 1.21 M |
| **v9 = P + M1 + M2** | 130 | **10,486** | 10,872 | 11,179 | 10,697 | 8,090 | 8,647 | 2.3 | **1.29 M** |

The pre_v9 row equals the B01 baseline in PATCHES.md to the token (same seed/code path).
Budget calibration (v9, same runs): M2_BUDGET 12,000 → median 10.5 k; 13,000 → 11.3 k; 14,000 → 12.1 k.

Mean composition per request (tokens):

| section | pre_v9 | v9 |
|---|---|---|
| system prompt | 3,623 | 3,907 |
| tool schema | 322 | 322 |
| current turn prompt (incl. facts block) | 959 | 1,126 |
| history turn prompts | 3,839 | 153 |
| old + in-flight reasoning | 7,346 | 2,215 |
| tool-call code | 1,454 | 583 |
| tool results | 509 | 203 |
| images | 1,641 | 1,356 |

Reading it: P1-P3 cost ~+0.3 k/request (system +284, turn prompt ~+45). M1 alone does **not** shrink
requests — at the 26 k budget the room freed from old reasoning is refilled with ~2× more visible turns
(history prompts 3.8 k → 6.3 k); it changes *what* is carried, not how much. The size reduction comes from
M2 (compaction + budget); with M1 on top, ~2.2 k of each request is the last two turns' thinking. At
12 k the carried history is ~2 previous assistant turns; everything older survives only as verified
facts + notes. The fixed part (system + schema + current prompt + board) is now ~60 % of a request, so
prefix caching of the 4.2 k system+schema prefix (B03) matters more than before.

Facts block, real sample (ar25, request 42; 1,157 chars):

```
Verified facts (harness-checked from real frames; kept when old history is trimmed):
- Levels cleared: 0 of 8. Now L1: 37 actions on this level, 37 in total.
- L1 effects: LEFT 4x: changed 4 (~108 cells, last r15-23 c18-44); RIGHT 7x: changed 7 (...); UNDO 1x:
  changed 1 (~108 cells, ...); SPACE 5x: HUD/edge-only 5; MOUSE@B 3x: no change 3 [dead r0c63 r14c63 r63c0]; ...
Model notes (latest; not verified by the harness):
- World model: ... - Goal model: ... - Plan: ... - Recent findings: ...
```
On ls20 the ledger separates blocked moves (only the step bar on rows 61-62 ticks → `HUD/edge-only`)
from real moves (~50 cells): `DOWN 9x: changed 6 (~50 cells, ...), HUD/edge-only 3`.

## 4. Tests (`tests/test_duck_prompt_memory.py`, 14 tests, ~45 s; with `test_duck_offline.py` 27/27 green)

- ACTION7: every advertised action of all 25 games round-trips through the action map; e2e on all six
  ACTION7 games (ar25, bp35, lf52, sb26, sk48, su15): `ACTION7` reaches the engine, `UNDO` listed and
  explained each turn; `pre_v9` still rejects it (`Unknown action`).
- Scoring rule in every system prompt, a monotone game clock in every turn prompt.
- Facts block in every turn prompt, ≤ 2,100 chars, its action total == executed actions even after
  trimming; unit test with 3 levels / >6 MOUSE colours / 12 k-char notes: capped, every (level, action)
  key kept, long notes never displace verified lines; real ls20 frames: step bar → `HUD/edge-only`.
- Harness estimate ≤ 12,000 for every request (and binding), transcripts show `context_budget_tokens:
  12000`, old turn prompts carry no facts/instructions, steady-state median ≤ 12 k.
- M1: carried reasoning == min(K, available) on the newest messages (K=2 and K=1), in-flight turn keeps
  its own, `reasoning_content` mirror + `preserve_thinking`; M1 off keeps every turn's reasoning as before.
- Isolation: ledger key == game id per agent, per-game action counts, no cross-game notes; switching
  state files resets status and ledger.
- **Switches off == before**: per-switch off, group switch off (`DUCK_PATCHES_V9=0`) and `stock` reproduce
  `tests/golden/duck_requests_pre_v9.json` (normalized sha256 of every request, captured pre-change).

## 5. Caveats

- The mock measures size and mechanics, not play quality. Its reasoning (4 k chars) and tool outputs are
  synthetic; real turns have longer tool results (≤ 4 k chars) and variable thinking.
- Image tokens (578 per 768 px board) and reasoning rendering are unverified for Qwen3.8 (B02 GPU run:
  compare `usage.prompt_tokens` for M1 on/off at equal messages).
- A 12 k budget keeps ~2 previous turns verbatim. If the A/B shows lost context (repeated probes,
  forgotten plans), raise `DUCK_PATCH_M2_BUDGET` (14 k ≈ 12 k/request) before dropping M2.
- `LOCAL_ANALYZER_CONTEXT_WINDOW=16384` would give a 9,728 window budget, which then wins over 12,000
  (MISTAKES #10: check `context_budget_tokens` in transcripts, not notebook cells).
- Edge-band heuristic (3 cells, ≤ 4 cells/action) fits all public HUD bars probed (ls20 rows 61-62,
  ar25 col 63, g50t row 63); a hidden game with gameplay inside the band gets `changed` only when > 4
  cells move there.

## 6. Proposed GPU experiments (for plans/EXPERIMENTS.md)

1. **Hydra-2 prompt** = `patched` (P1-P3 on, M off) vs `pre_v9`: expected small positive (scoring/time
   lines +28 % in the community on 27B; UNDO on 6/25 public games). Falsified if public-25 levels drop
   and LB ≤ baseline over 2 submissions.
2. **Memory** = `v9` (`DUCK_PATCH_M1_THINK=1 DUCK_PATCH_M2_FACTS=1`) vs `patched`: expected ~2× requests
   per game-hour from −50 % prompt tokens at equal serving, plus fewer wrong-goal repeats. Measure
   decisions/game-hour, invalid tool calls, repeated identical probes, levels cleared. Falsified if
   decisions/game-hour rise < 1.3× or levels cleared fall. If throughput rises but quality falls, try
   M2_BUDGET 14000 and K=3.
3. With M2 on, re-evaluate the serving profile (B03): shorter prompts allow more concurrent sequences
   and possibly `--max-model-len 24576` (12 k prompt + reasoning headroom).

## 7. Reproduce

```
SCRATCH/venv/bin/python -m pytest tests/test_duck_offline.py tests/test_duck_prompt_memory.py -q
SCRATCH/venv/bin/python agent/duck_offline.py --compare-profiles --steps 40   # pre_v9 / patched / v9 rows
```
The M1-only / M2-only rows and the Qwen-tokenizer column came from a scratch script that adds temporary
profiles (`SCRATCH/b06/measure2.py`; `duck_offline.patch_profile` resets the `DUCK_PATCH_*` switch
variables, so per-switch overrides must go through `PROFILES`, not the shell).
