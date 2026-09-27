# automation/ — build, preflight and push a Kaggle notebook

Everything here runs on CPU. Nothing touches Kaggle except `kaggle_pipeline.py` (push/submit) and
`preflight.py --kaggle-check` (read-only); both need `KAGGLE_API_TOKEN` in the environment and never write it
anywhere. Rules that apply to every push: `CLAUDE.md` (≤ 1 submission per UTC day, GPU ledger first, run gate).

| File | What |
|---|---|
| `package_source.py` | stage `agent/` + `serving/` + the serving bench as the Kaggle dataset `kragglenote2forwork/arc3-hydra-src` (versioned, secret-scanned). **Does not upload.** |
| `build_notebook.py` | candidate JSON → notebook dir (`kernel-metadata.json` + `<slug>.ipynb` + `build_info.json`); `exec-dry` rehearses the built notebook on CPU |
| `preflight.py` | offline checks (title/slug, attachments, accelerator, internet, secrets, notebook JSON, smoke wiring, time budget, source version); `--kaggle-check` verifies attachments and the existing kernel title online |
| `candidates/*.json` | one file per experiment; `example.json` documents every field |
| `kaggle_pipeline.py` | `quota`, `sync`, `push`, `wait`, `submit`, `daily` (1/day guard) |
| `serving_bench.py`, `exp001_notebook/` | serving load test (B02/EXP-001) |
| `hydra0/` | the Hydra-0 patch notebook (separate, built on the user's 09-22 notebook) |

## What the built notebook does

1. `HYDRA_CONFIG` (the candidate, embedded) and the mode: **rerun** when `KAGGLE_IS_COMPETITION_RERUN` is set,
   otherwise **smoke** (the commit run).
2. GPU check: fails fast unless an RTX PRO 6000 is present.
3. `pip install arc-agi` from the competition wheelhouse.
4. Finds the source dataset by its `HYDRA_SOURCE.json` marker, imports `agent/hydra_run.py`, checks the pinned
   source version (strict in smoke).
5. Serving: profile name → start command `python serving/launch.py --profile <name> --work-dir …` (B03 interface:
   starts the server, waits for `/v1/models`, prints `{base_url, pid, harness_env, log}`), plus a log-only health
   monitor.
6. Analyzer env layered 09-22 baseline → profile `harness_env` → candidate `analyzer_env`, DUCK_PATCH switches and
   the scheduler config set, then asserted **before** the Duck agent is imported (MISTAKES #10).
7. Run inside `try/finally: teardown` (teardown never raises, MISTAKES #9):
   - rerun: every gateway game, soft deadline `time.soft_deadline_s` (8h35m) after notebook start; with
     `DUCK_PATCH_B08_SCHEDULER=1` the progress-aware scheduler (`agent/scheduler.py`) plans per-game budgets from
     the wall clock actually left after setup;
   - smoke: 3 public games × 8 min from the offline environment files, strict checks, then the placeholder
     `submission.parquet`. Pipeline sanity only: public-25 scores barely predict the LB (r = 0.16).
8. Effective config read back from transcripts / server log / `hydra_scheduler.json` (`EFFECTIVE_CONFIG`,
   `EFFECTIVE_CONFIG_CHECK`); a mismatch fails the smoke commit, so that version cannot be submitted.

## Workflow

```bash
PY=$SCRATCH/venv/bin/python        # python 3.12 + arc_agi 0.9.8 + arcengine 0.9.3 (CPU tests need it)

# 0. CPU tests green
$PY -m pytest tests/ -q

# 1. stage the source dataset (prints the version and the upload command; nothing is uploaded)
$PY automation/package_source.py                 # -> SCRATCH/kaggle_datasets/arc3-hydra-src/<version>/
#    upload (user / orchestrator), then wait until `kaggle datasets status kragglenote2forwork/arc3-hydra-src` is ready:
#    kaggle datasets version -p <staged dir> --dir-mode zip -m "<version>"     (first time: datasets create)

# 2. write the candidate (copy candidates/example.json; one thesis per submission, plans/EXPERIMENTS.md)
#    kernel.title must slugify to kernel.id's slug; for an existing kernel use its exact title (MISTAKES #1)

# 3. build, pinned to the staged version
$PY automation/build_notebook.py build --candidate automation/candidates/<cand>.json --source-dir <staged dir>
#    -> notebooks/<slug>/ ; add --existing-title "<title>" if the kernel already exists

# 4. rehearse on CPU (mock LLM; commit branch, then the rerun branch)
$PY automation/build_notebook.py exec-dry --dir notebooks/<slug> --source-dir <staged dir>
$PY automation/build_notebook.py exec-dry --dir notebooks/<slug> --source-dir <staged dir> --rerun

# 5. preflight (offline, then online once KAGGLE_API_TOKEN is set)
$PY automation/preflight.py --dir notebooks/<slug> --source-dir <staged dir>
$PY automation/preflight.py --dir notebooks/<slug> --kaggle-check

# 6. run gate (CLAUDE.md): thesis written, tests + preflight green, `kaggle_pipeline.py quota`, GPU_LEDGER entry,
#    no other GPU batch kernel running (MISTAKES #5). Then push (smoke commit ≈ 0.2-0.4 GPU-h) and submit:
python automation/kaggle_pipeline.py daily --dir notebooks/<slug> -m "<EXP-id>: <thesis>"
#    (or push / wait / submit separately; submit always passes -f submission.parquet, MISTAKES #7)
```

`preflight.py` exit code is 0 when nothing FAILs (WARN is allowed: e.g. first action > 11 min after start). The
`time_budget` check prints the rerun plan (fair share / floor / cap / max active games) and FAILs a stock-loop
notebook whose waves do not fit the window (the "last wave cut short" failure).

## Dry-run knobs (`exec-dry` sets them)

`HYDRA_DRY_RUN=1` (skip GPU check + pip; scripted mock LLM; GPU serving profiles are only rendered with
`--print-argv`, `mock` profiles are launched for real), `HYDRA_KAGGLE_INPUT`, `HYDRA_WORKING_DIR`,
`HYDRA_DRY_SMOKE_SECONDS` (30), `HYDRA_DRY_MAX_ACTIONS` (10), `HYDRA_DRY_GAMES` (rerun rehearsal games),
`HYDRA_DRY_DEADLINE_S` (120). The stdout of a rehearsal is in `<work>/notebook_stdout.log`.

## Known gaps

- The serving health monitor only logs; it does not restart a hung vLLM (the vendored watchdog is tied to the
  09-22 `serving_setup.py` argv). Follow-up if EXP-001 shows hangs.
- `--dir-mode zip` uploads: the notebook tolerates one extra directory level when locating `agent/hydra_run.py`,
  but the first real push should confirm the mounted layout (`HYDRA_SOURCE` log line).
