# BACKLOG (prioritised; take the top unblocked item; mark DONE with date + commit)

Sources of code:
- Shipped Duck bundle (exactly what our 3.85/10.02 runs used): `Hisernberg/arc-agi-3-campaign` → `data/duck_bundle/src/`
  (`ARC3-Inference/inference/agent/{tool_agent,prompts,python_tool_sandbox,runtime_state,vision_context}.py`,
  `tufa-arc-agi-framework/src/taaf/*`), plus `serving_setup.py`, `vllm_server_watchdog.py`.
- Upstream: github.com/Tufalabs/duck-harness (incl. `example-run/`: 25 games × 20 passes, Qwen3.6-27B-FP8).
- Our runtime patches: AGENTFIX (user notebook cell 10), campaign `kernels/apex_v8_patch.py` (v7.1 state fix + v8).
- Offline env: `agent/arcenv.py` (25 games, exact clone, official scorer).

## W0 — CPU build week (09-27 → 10-02)

| P | ID | Item | Output | Done-when |
|---|---|---|---|---|
| 0 | B01 | **Vendor the Duck source** into `agent/duck/` (from the shipped bundle), make it run end-to-end offline on CPU against `arcenv` with a **mock LLM** (scripted tool calls) | `agent/duck/`, `tests/test_duck_offline.py` | 3 games × 20 steps run through the real tool_agent loop with the mock; per-game state isolated (MISTAKES #2) |
| 0 | B02 | **Serving load-test kit**: parse `results/prompts/*.log` into multi-turn conversations; async client that replays 28 concurrent agents against an OpenAI-compatible server; reports tok/s, latency p50/p90, prefix-hit %, preemptions (scrape `/metrics`) | `automation/serving_bench.py` + recorded-prompt pack (HF dataset) | runs against a CPU mock server; one-command GPU run for EXP-001 |
| 0 | B03 | **Serving profiles** as data: `serving/profiles/{flashnext_baseline,flashnext_tuned,qwen27b_fp8}.json` → launch args; check the vLLM runtime in the keithtyser dataset supports Qwen3_5 (27B) + fp8 KV + prefix caching for hybrid models | profiles + notes | profile renders valid `vllm serve` argv; docs cite flags |
| 0 | B04 | **Perception pack** `agent/perception.py`: raw-int frame delta, HUD/timer mask (edge bars), object diff w/ avatar detection, ranked click candidates (1 per object, dead-signature suppression), animation summary; compact text ≤ 150 tokens | module + tests on all 25 games | deterministic outputs on replays; avatar found on keyboard games; click candidates include every winning click in the frontier/human replays (recall check) |
| 0 | B05 | **Action channel** `agent/actions.py`: plan-with-expectations batches, halt-on-mismatch, repeat-no-op guard (no env action), death-transition ban, verified-prefix auto-replay after GAME_OVER/level reset | module + tests | on replays: replaying a won level's action list after RESET reproduces the win; guard blocks exact repeats |
| 1 | B06 | **Memory/compaction** `agent/memory.py`: retained-think(K) + verified-facts block (controls/rules/goal/plan with evidence refs), token-capped; per-game state keyed by game id | module + tests using results_4 transcripts | block ≤ 700 tokens; never loses a harness-verified fact |
| 1 | B07 | **Prompt v9** `agent/prompts_v9.py`: true scoring rule + time_remaining line, UNDO where ACTION7 exposed, reset discipline, visible world-model updates, "verify why you won", no video-game mapping | prompt + diff vs Duck | token count ≤ Duck system prompt + 300 |
| 1 | B08 | **Scheduler**: progress-aware game scheduling for 110 games / K slots / 9 h, 15-min-kill guard, soft deadline 8h35m | `agent/scheduler.py` + simulation test | simulated 110-game run respects all deadlines |
| 1 | B09 | **Notebook builder + preflight**: thin notebook (install → start server → run Hydra → teardown never raises); smoke mode for commits (3 games × 8 min); dataset upload of `agent/` as `kragglenote2forwork/arc3-hydra-src`; preflight asserts title==slug, attachments, accelerator, deadline, no secrets | `automation/build_notebook.py`, `automation/preflight.py` | dry-run builds a notebook dir that passes preflight |
| 2 | B10 | **Trace mining for generic priors**: GPT-6 Astra 25 recordings, Kepler/Opus runs, AgentNative frontier traces, human sessions, Tufa example-run → per-game time-to-level, action-type mix, first-N-action patterns, which perception features precede wins | `analysis/10_trace_mining.md` | ≥ 5 concrete, game-agnostic heuristics with evidence |
| 2 | B11 | **Oracle study (CPU)**: BFS/beam with state cloning on each public level → min actions vs human baseline; classify levels by search depth/branching; shows where cheap probing is safe | `analysis/11_oracle_study.md` | table for all 183 levels (or timeout marks) |
| 2 | B12 | **Diff the 4.68 Team-AIRIS bundle** (`urad-duck-r10-targeted`) vs our notebook; port anything with evidence | `analysis/12_airis_diff.md` | list of deltas + verdicts |
| 3 | B13 | (PARTLY DONE 09-27: traces, discussions, frontier-replays, research) HF datasets: `arc-agi-3-kaggle-notebooks`, `arc-agi-3-kaggle-discussions`, `arc-agi-3-research`, prompt pack | private datasets | uploaded with cards |

## W1+ (GPU weeks) — see MASTER_PLAN §5; refined each Saturday by the weekly routine.
