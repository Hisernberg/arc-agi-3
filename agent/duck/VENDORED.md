# Vendored Duck harness (B01, 2026-09-27)

This directory is the **exact Kaggle source bundle our runs shipped** (the dataset
`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`, the one behind our 3.85 LB / 10.02 public-25 runs),
copied file-for-file from the campaign repo, with the layout of the Kaggle dataset kept so this
directory can be uploaded as a dataset again (the notebook's `_source_path_entries(bundle_dir)`
puts every `src/<repo>/src` or `src/<repo>` on `sys.path`).

| | |
|---|---|
| Copied from | `Hisernberg/arc-agi-3-campaign` → `data/duck_bundle/` (81 files, `__pycache__` dropped) |
| Bundle manifest | `SOURCE_IDENTITY.json`, `taaf-kaggle-bundle.json` (unchanged) |
| Source snapshot | `git_status.txt`: ARC3-Inference `aa69123` (DIRTY, branch add-kaggle-share-flag), tufa-arc-agi-framework `fe9f7c4` (clean), re-arc-3 `57e46d619d` (pinned, not bundled) |
| Upstream compared | github.com/Tufalabs/duck-harness `main` @ `7652836056c5` (cloned 2026-09-27) |
| Modified here | only the source patches in `PATCHES.md` (tagged `[DUCK-PATCH ...]`); everything else is byte-identical to the bundle |

## Layout

```
agent/duck/
  src/ARC3-Inference/            Duck agent (inference/agent/*), TAAF adapter (inference/framework/solver.py),
                                 scorer (inference/tools/eval.py), viewer, configs, Makefile, uv.lock
  src/tufa-arc-agi-framework/    TAAF: Benchmark, Game/GameAPI, Solver, diagnostics, deploy_kaggle
  serving_setup.py               vLLM launch for Qwen3.8-Flash-Next NVFP4 (+ persisted analyzer env)
  serving_teardown.py, vllm_server_watchdog.py, setup_commands.json, teardown_commands.json
  vllm-patches/                  RadixArk NVFP4 PLE-FP8 patch for the pinned vLLM runtime
  benchmark_initial.pkl          pickled taaf Benchmark (25 public GameAPI, HarnessSolver config)
  deploy_target.pkl              pickled taaf KaggleTarget (RTX Pro 6000, 32400 s, no internet)
  VENDORED.md, PATCHES.md        this provenance note, the source patches
```

The pickles are kept as reference data only; `agent/duck_offline.py` rebuilds the same configuration
in code (`SHIPPED_SOLVER_FIELDS`, `SHIPPED_BENCHMARK_FIELDS`, `SHIPPED_TARGET_FIELDS`,
`PUBLIC_GAME_IDS`, `SHIPPED_ANALYZER_ENV`) and `tests/test_duck_offline.py` asserts field equality
against the pickles. What the pickles contain:

- `Benchmark(label="duck-harness-kaggle", n_passes=1, game_weights=None)`, 25 `GameAPI` with
  `ArcadeSpec(OFFLINE, environments_dir="__auto__")`; the notebook replaces `bm.games` at run time
  (live gateway in a real rerun, `environment_files` offline otherwise) and sets `bm.job_dir`.
- `HarnessSolver(model="local", analyzer_timeout=900, max_runtime_s_per_game=7920, concurrency=28,
  max_actions_per_game=None, save_request_logs=False, start_local_server=False)`. The `kaggle_*`
  fields (Qwen3.6-27B wheelhouse/model) are only used by TAAF's deploy path, not at run time.
  `model="local"` makes `ToolAgent` read `LOCAL_ANALYZER_BASE_URL` / `LOCAL_ANALYZER_MODEL_ID`,
  which `serving_setup.py` persists (`persist_analyzer_environment`).

## Diff vs upstream GitHub (Tufalabs/duck-harness @ 7652836)

The agent core and the TAAF solver adapter are **identical** to upstream:
`inference/agent/{tool_agent,prompts,python_tool_sandbox,runtime_state,vision_context,action_names}.py`,
`inference/framework/{solver,kaggle}.py`, `inference/utils/{segmentation,grid_utils,openai_compat,...}.py`,
`taaf/{benchmark,game,solver,diagnostics,...}.py` (before our patches).

| File | bundle-only / upstream-only lines | Nature |
|---|---|---|
| `inference/framework/run.py` | 70 / 22 | bundle imports `re_arc.EnvSampler` (re-arc-3 dataset/tag game selection); upstream removed it (official ids only). Not imported by the solver path. |
| `inference/tools/significance.py` | 52 / 0 | bundle has extra paired-significance helpers |
| `inference/tools/traces.py` | 3 / 23 | minor |
| `inference/tools/eval.py` | 2 / 0 | bundle records `re_arc_commit` (reads `uv.lock`) |
| `inference/utils/rearc_{baselines,version}.py` | bundle only | re-arc-3 helpers |
| `taaf/game_api.py` | 14 / 15 | `environments_dir="__auto__"` resolves via `re_arc` in the bundle; upstream raises. We always pass an explicit dir. |
| `taaf/standard_benchmarks.py`, `competition_arcade.py`, `deploy*.py` | small | re-arc-3 references / wording |
| `configs/inference.json`, `pyproject.toml`, `uv.lock`, `Makefile`, READMEs | small | cluster paths, `re-arc-3` source repo, editable TAAF |

The bundle does **not** contain the notebook's runtime patches (AGENTFIX cell 10, ARM P upscale,
anim/sheet features) nor the campaign's v7.1/v8 world-model patch; see `PATCHES.md` for what was
ported into source and what was not.

## Pristine hashes (before patches)

| File | sha256 (bundle) |
|---|---|
| `inference/agent/tool_agent.py` | `535ee88b81b262fa5aedb785466ade9f3183a6417656fb6733427242baac7c9d` |
| `inference/agent/runtime_state.py` | `c32cbc4352b2e632efef33ebcb252af4f54867088a934d63fe1df5465934c5af` |
| `inference/agent/prompts.py` | `a8e203fbe288116a24f29f6880f0657873df51fee76b0af9ef2b44cc3d0186d3` |
| `inference/agent/python_tool_sandbox.py` | `765e90cf8d141f912f6cfbba498426c9b0de9453e6e44c97d3bc8df45cf9aa91` |
| `inference/framework/solver.py` | `65ddfcdbe8c24fdc31246ff018b51469827b21c38995243a36a818adbed33370` |

Compare any file with `diff -u ../arc-agi-3-campaign/data/duck_bundle/<path> agent/duck/<path>`.

Shipping note (B09): at setup time `serving_setup.py` → `source_identity()` checks only its **own**
sha256 (`serving_setup_sha256`) plus the model/runtime manifest ids in `SOURCE_IDENTITY.json`; the
`src/` tree is not hashed, so the source patches ship as-is. Editing `serving_setup.py` requires
updating `SOURCE_IDENTITY.json` or the Kaggle run aborts before vLLM starts.

## Running it on CPU

`agent/duck_offline.py` (see its docstring) plays N offline games through the real loop against a
scripted OpenAI-compatible mock; `tests/test_duck_offline.py` is the regression suite. Python 3.12
with `arc_agi==0.9.8`, `arcengine==0.9.3`, `requests`, `pillow`, `matplotlib`, `numpy`, `scipy`,
`imageio`, `pytest` (the scratch venv has them). No GPU, no network, ~10 s for 3 games x 20 actions.
