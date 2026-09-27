# Mistakes ledger — read before every GPU run / submission

| # | Date | Mistake | Cost | Rule now |
|---|---|---|---|---|
| 1 | 09-26 | Pushed a notebook whose title didn't match the kernel slug → Kaggle renamed/forked the kernel | 1 extra 2.6 h GPU run | Preflight asserts title == slug title |
| 2 | 09-19→09-26 | Game-id parsed from `..._tool_runtime_state.json` skipped `.json` names → **all games shared one "unknown" state**; memory/world-model machinery dead in every production run (v4 3.85 included) | ~a week of misleading A/Bs | Unit test on the *production* state path; assert per-game isolation in smoke |
| 3 | 09-04→09-27 | Serving profile: 5 GiB KV for 28 concurrent games, prefix cache off → 87 % of time queued, ~42 LLM turns/game | Most of the gap to the top | Serving A/B on real traffic before any harness tuning; log `Running/Waiting` from vLLM |
| 4 | 09-xx | Every submission commit ran a 2–2.6 h public-25 validation on GPU | ~2.6 h quota per submission | Commit in smoke mode (≤ 0.4 h); full validation only when a decision needs it |
| 5 | 09-26 | Scheduled an ARC push while 2 other-competition GPU kernels were running ("Maximum batch GPU session count of 2") | 2 h delay | Check running kernels first; don't mix ARC and other GPU jobs on ARC days |
| 6 | 09-23/24 | Re-submitted old versions without a thesis | 2 daily slots (only variance info gained) | Re-submit only the top-2 candidates, deliberately, for variance |
| 7 | 09-26 | Code-comp submit via CLI failed without `-f submission.parquet` | retries | `kaggle_pipeline.py submit` always passes `-k owner/slug -v N -f submission.parquet` |
| 8 | 09-26/27 | Uploaded files (ipynb, zip) silently did not land in an earlier session | analysis delayed | Verify uploads exist (size + hash) before planning around them |
| 9 | 09-22 | Validation run ended with `vLLM teardown did not reach the bounded terminal gate` (RuntimeError in teardown) | none yet | Teardown must never raise in competition mode; wrap in try/except, log only |
