#!/usr/bin/env python3
"""Build a Kaggle notebook dir (kernel-metadata.json + .ipynb) for a Hydra candidate (backlog B09).

    python automation/build_notebook.py build --candidate automation/candidates/example.json \
        [--out DIR] [--source-version V | --source-dir STAGED] [--existing-title T]
    python automation/build_notebook.py exec-dry --dir NB_DIR --source-dir STAGED [--rerun] [--games a,b,c]

The notebook is thin: it embeds the candidate config (`HYDRA_CONFIG = {...}`), checks the GPU, installs the ARC
runtime from the competition wheelhouse, finds the source dataset (`kragglenote2forwork/arc3-hydra-src`, marker
`HYDRA_SOURCE.json`) and calls `agent/hydra_run.py`: serving profile -> start command (`serving/launch.py`), analyzer
env asserted before import, then

* competition rerun (`KAGGLE_IS_COMPETITION_RERUN`): every gateway game until the soft deadline (8h35m);
* commit (Save & Run): SMOKE = 3 public games x 8 min, strict effective-config checks, placeholder
  `submission.parquet`;

teardown in a `finally` that never raises, and the effective config read back from the logs. Attachments come
from the candidate + serving profile: competition, the source dataset, the profile's runtime datasets and model.

`exec-dry` rehearses the built notebook on CPU: a fake /kaggle/input (source dataset symlink + the competition's
offline environment files), `HYDRA_DRY_RUN=1` (no GPU check, no pip, games against the scripted mock LLM; a
`mock` serving profile is still launched through launch.py), every code cell executed in order in one namespace
in a fresh interpreter.

Candidate JSON (see automation/candidates/example.json):
    name, kernel{id,title,is_private}, source_dataset, source_version|null, serving_profile, analyzer_env{},
    duck_patches{}, scheduler{}, solver{}, smoke{games[3],minutes}, time{soft_deadline_s,...},
    extra_dataset_sources[], extra_model_sources[]
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import os
import pprint
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "automation"))
from preflight import COMPETITION, MACHINE_SHAPE, kaggle_slug  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad")
DEFAULT_SOURCE_DATASET = "kragglenote2forwork/arc3-hydra-src"
BASE_URL_KEYS = {"LOCAL_ANALYZER_BASE_URL", "OPENAI_BASE_URL"}
PIP_ESTIMATE_S = 90.0
BOOTSTRAP_ESTIMATE_S = 60.0


# ----------------------------------------------------------------------------------------------------------------
# Candidate + serving profile -> config
# ----------------------------------------------------------------------------------------------------------------


def load_profile(name: str) -> dict[str, Any]:
    """Read serving/profiles/<name>.json through serving/launch.py's own loader (schema check); read-only use."""
    launch = REPO / "serving" / "launch.py"
    if launch.is_file():
        spec = importlib.util.spec_from_file_location("hydra_serving_launch", launch)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module.load_profile(name)
    return json.loads((REPO / "serving" / "profiles" / f"{name}.json").read_text())


def resolve_serving(profile_name: str) -> dict[str, Any]:
    prof = load_profile(profile_name)
    runtime = prof.get("runtime") or {}
    model = prof.get("model") or {}
    startup = prof.get("startup") or {}
    return {
        "profile": prof["name"],
        "runtime_kind": runtime.get("kind", ""),
        "datasets": list(runtime.get("datasets") or []),
        "models": [model["kaggle_model"]] if model.get("kaggle_model") else [],
        "served_model_name": model.get("served_model_name", ""),
        "harness_env": {str(k): str(v) for k, v in (prof.get("harness_env") or {}).items()},
        "client": prof.get("client") or {},
        "startup": startup,
        # launch.py prepares the runtime (extract + patch) before its own readiness timeout starts
        "start_timeout_s": float(startup.get("timeout_s", 1500)) + 900.0,
        "health_interval_s": 60.0,
    }


def load_candidate(path: Path) -> dict[str, Any]:
    cand = json.loads(Path(path).read_text())
    for key in ("name", "kernel", "serving_profile", "smoke"):
        if key not in cand:
            raise ValueError(f"candidate {path}: missing {key!r}")
    for key in ("id", "title"):
        if key not in cand["kernel"]:
            raise ValueError(f"candidate {path}: kernel.{key} missing")
    return cand


def build_config(cand: dict[str, Any], *, source_version: str | None, built: str) -> dict[str, Any]:
    serving = resolve_serving(cand["serving_profile"])
    analyzer_env = {str(k): str(v) for k, v in (cand.get("analyzer_env") or {}).items()}
    expected = {k: v for k, v in {**serving["harness_env"], **analyzer_env}.items() if k not in BASE_URL_KEYS}
    t = dict(cand.get("time") or {})
    setup = t.get("setup_estimate_s")
    if setup is None:
        setup = PIP_ESTIMATE_S + BOOTSTRAP_ESTIMATE_S + float(serving["startup"].get("expected_s", 600))
    time_cfg = {
        "soft_deadline_s": float(t.get("soft_deadline_s", 8 * 3600 + 35 * 60)),
        "notebook_limit_s": float(t.get("notebook_limit_s", 32_400)),
        "setup_estimate_s": float(setup),
        "first_turn_estimate_s": float(t.get("first_turn_estimate_s", 90)),
        "teardown_estimate_s": float(t.get("teardown_estimate_s", 120)),
        "expected_games": int(t.get("expected_games", 110)),
        "turn_estimate_s": t.get("turn_estimate_s"),
    }
    serving["datasets"] = serving["datasets"] + list(cand.get("extra_dataset_sources") or [])
    serving["models"] = serving["models"] + list(cand.get("extra_model_sources") or [])
    return {
        "name": cand["name"],
        "kernel_id": cand["kernel"]["id"],
        "title": cand["kernel"]["title"],
        "source_dataset": cand.get("source_dataset", DEFAULT_SOURCE_DATASET),
        "source_version": source_version or cand.get("source_version"),
        "serving": serving,
        "analyzer_env": analyzer_env,
        "expected_analyzer_env": expected,
        "duck_patches": {str(k): str(v) for k, v in (cand.get("duck_patches") or {}).items()},
        "scheduler": dict(cand.get("scheduler") or {}),
        "solver": dict(cand.get("solver") or {}),
        "smoke": {"games": list(cand["smoke"].get("games") or []), "minutes": float(cand["smoke"].get("minutes", 8))},
        "time": time_cfg,
        "teardown_commands": list(cand.get("teardown_commands") or []),
        "built_utc": built,
        "generator": "automation/build_notebook.py",
    }


def kernel_metadata(cand: dict[str, Any], config: dict[str, Any], code_file: str) -> dict[str, Any]:
    kernel = cand["kernel"]
    datasets = [config["source_dataset"]] + [d for d in config["serving"]["datasets"] if d != config["source_dataset"]]
    return {
        "id": kernel["id"],
        "title": kernel["title"],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(kernel.get("is_private", True)),
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": list(dict.fromkeys(datasets)),
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": list(dict.fromkeys(config["serving"]["models"])),
        "keywords": [],
    }


# ----------------------------------------------------------------------------------------------------------------
# Notebook template
# ----------------------------------------------------------------------------------------------------------------

CELL_CONFIG = '''# Hydra candidate config (generated by automation/build_notebook.py: edit the candidate JSON and rebuild).
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOK_START_EPOCH = time.time()
HYDRA_CONFIG = __HYDRA_CONFIG__
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
DRY_RUN = os.environ.get("HYDRA_DRY_RUN", "").strip().lower() in {"1", "true"}  # CPU rehearsal only
KAGGLE_INPUT = Path(os.environ.get("HYDRA_KAGGLE_INPUT", "/kaggle/input"))
WORKING_DIR = Path(os.environ.get("HYDRA_WORKING_DIR", "/kaggle/working"))
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print("HYDRA_MODE " + json.dumps({"candidate": HYDRA_CONFIG["name"], "mode": "rerun" if TRUE_SUBMISSION else "smoke",
                                  "dry_run": DRY_RUN, "soft_deadline_s": HYDRA_CONFIG["time"]["soft_deadline_s"]}),
      flush=True)
'''

CELL_GPU = '''# GPU check before anything else: a missing RTX PRO 6000 is the most common Kaggle submission error.
def _gpu_names():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


if DRY_RUN:
    print("GPU_CHECK " + json.dumps({"skipped": "dry run"}), flush=True)
else:
    _gpus = _gpu_names()
    print("GPU_CHECK " + json.dumps({"gpus": _gpus}), flush=True)
    if not any("RTX PRO 6000" in name.upper() for name in _gpus):
        raise RuntimeError(f"Expected an NVIDIA RTX PRO 6000, found {_gpus or 'no GPU'}: select that accelerator.")
'''

CELL_PIP = '''# ARC runtime from the competition wheelhouse (the rerun has no internet).
_WHEELS = KAGGLE_INPUT / "competitions" / "arc-prize-2026-arc-agi-3" / "arc_agi_3_wheels"
if DRY_RUN:
    print("HYDRA_PIP " + json.dumps({"skipped": "dry run"}), flush=True)
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-warn-conflicts",
                           "--disable-pip-version-check", "--find-links", str(_WHEELS), "arc-agi"],
                          stdout=subprocess.DEVNULL)
    print("HYDRA_PIP " + json.dumps({"wheels": str(_WHEELS)}), flush=True)
'''

CELL_SOURCE = '''# Locate the Hydra source dataset by its marker file and import the notebook runtime (agent/hydra_run.py).
def _find_source_root():
    owner, slug = HYDRA_CONFIG["source_dataset"].split("/", 1)
    for cand in (KAGGLE_INPUT / slug, KAGGLE_INPUT / "datasets" / owner / slug):
        if (cand / "HYDRA_SOURCE.json").is_file():
            return cand
    for marker in sorted(KAGGLE_INPUT.rglob("HYDRA_SOURCE.json")):
        return marker.parent
    raise RuntimeError(f"Hydra source dataset {HYDRA_CONFIG['source_dataset']} not found under {KAGGLE_INPUT}.")


SOURCE_ROOT = _find_source_root()
_RUNTIME = SOURCE_ROOT / "agent" / "hydra_run.py"
if not _RUNTIME.is_file():  # tolerate one extra directory level from the dataset's zip upload
    _RUNTIME = next(SOURCE_ROOT.rglob("agent/hydra_run.py"))
sys.path.insert(0, str(_RUNTIME.parent))
import hydra_run as H  # noqa: E402

CTX = H.RunContext.create(HYDRA_CONFIG, agent_dir=_RUNTIME.parent, kaggle_input=KAGGLE_INPUT,
                          working_dir=WORKING_DIR, start_epoch=NOTEBOOK_START_EPOCH,
                          true_submission=TRUE_SUBMISSION, dry_run=DRY_RUN)
SOURCE_MANIFEST = H.bootstrap(CTX)
'''

CELL_SERVING = '''# Serving profile -> start command (serving/launch.py): start, wait for /v1/models, collect the harness env.
SERVING = H.start_serving(CTX)
'''

CELL_ENV = '''# Analyzer env (09-22 baseline -> serving profile -> candidate) and DUCK_PATCH switches, asserted BEFORE the
# Duck agent is imported (MISTAKES #10: a later cell re-applying setup env silently undid a notebook override).
HARNESS_ENV = H.apply_harness_env(CTX, SERVING)
'''

CELL_RUN = '''# Play. Rerun: every gateway game until the soft deadline. Commit: SMOKE + placeholder submission.parquet.
RESULT = None
try:
    if TRUE_SUBMISSION:
        RESULT = H.run_rerun(CTX, SERVING)
    else:
        RESULT = H.run_smoke(CTX, SERVING)
finally:
    H.teardown(CTX, SERVING)  # never raises (MISTAKES #9)
'''

CELL_EFFECTIVE = '''# What actually ran, read back from transcripts / server log / scheduler summary (MISTAKES #10).
# Strict in the commit (a mismatch fails the version, so it cannot be submitted); log-only in the rerun.
EFFECTIVE_CONFIG = H.effective_config_report(CTX, SERVING, RESULT)
'''


def _md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str, idx: int) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "id": f"hydra-{idx:02d}", "metadata": {}, "outputs": [],
            "source": text}


def render_notebook(config: dict[str, Any], candidate_rel: str) -> dict[str, Any]:
    sched_on = config["duck_patches"].get("DUCK_PATCH_B08_SCHEDULER", "0").lower() in {"1", "true"}
    solver = config["solver"]
    loop = ("progress-aware scheduler (B08): budgets from the wall clock left after setup, "
            f"{solver.get('concurrency', 28)} turn slots" if sched_on else
            f"stock TAAF loop: {solver.get('concurrency', 28)} games at a time, "
            f"{solver.get('max_runtime_s_per_game', 7920)} s each")
    pinned = f", pinned version `{config['source_version']}`" if config.get("source_version") else ""
    header = (
        f"# {config['title']}\n\n"
        f"Hydra candidate `{config['name']}`, built by `automation/build_notebook.py` ({config['built_utc']}) from "
        f"`{candidate_rel}`.\n\n"
        f"* **Commit / Save & Run**: SMOKE: serving profile `{config['serving']['profile']}` + "
        f"{len(config['smoke']['games'])} public games ({', '.join(config['smoke']['games'])}) x "
        f"{config['smoke']['minutes']:g} min from the offline environment files, strict effective-config checks, then "
        "the placeholder `submission.parquet`. Pipeline sanity only (public scores barely predict the LB).\n"
        f"* **Competition rerun** (`KAGGLE_IS_COMPETITION_RERUN`): every gateway game, {loop}; soft deadline "
        f"{config['time']['soft_deadline_s']:.0f} s after the notebook starts.\n"
        f"* Source: dataset `{config['source_dataset']}` (marker `HYDRA_SOURCE.json`{pinned}). "
        "No credentials anywhere in this notebook.\n"
    )
    config_literal = pprint.pformat(config, indent=1, width=110, sort_dicts=True)
    cells = [
        _md(header),
        _code(CELL_CONFIG.replace("__HYDRA_CONFIG__", config_literal), 1),
        _md("## 1. GPU check"),
        _code(CELL_GPU, 2),
        _md("## 2. ARC runtime"),
        _code(CELL_PIP, 3),
        _md("## 3. Source dataset"),
        _code(CELL_SOURCE, 4),
        _md("## 4. Serving"),
        _code(CELL_SERVING, 5),
        _code(CELL_ENV, 6),
        _md("## 5. Run (rerun: all games / commit: smoke) and teardown"),
        _code(CELL_RUN, 7),
        _md("## 6. Effective configuration"),
        _code(CELL_EFFECTIVE, 8),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "nvidiaRtxPro6000", "isInternetEnabled": False, "isGpuEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(candidate_path: Path, out: Path | None = None, *, source_version: str | None = None,
          source_dir: Path | None = None, existing_title: str | None = None) -> dict[str, Any]:
    candidate_path = Path(candidate_path)
    cand = load_candidate(candidate_path)
    title, kid = cand["kernel"]["title"], cand["kernel"]["id"]
    slug = kid.split("/", 1)[1]
    if kaggle_slug(title) != slug:  # MISTAKES #1, fail at build time already
        raise ValueError(f"kernel title {title!r} slugifies to {kaggle_slug(title)!r} but the id slug is {slug!r}")
    if source_dir is not None and source_version is None:
        source_version = json.loads((Path(source_dir) / "HYDRA_SOURCE.json").read_text())["version"]
    built = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    config = build_config(cand, source_version=source_version, built=built)
    out = Path(out) if out else REPO / "notebooks" / slug
    out.mkdir(parents=True, exist_ok=True)
    code_file = f"{slug}.ipynb"
    try:
        candidate_rel = str(candidate_path.resolve().relative_to(REPO))
    except ValueError:
        candidate_rel = candidate_path.name
    nb = render_notebook(config, candidate_rel)
    (out / code_file).write_text(json.dumps(nb, indent=1) + "\n")
    meta = kernel_metadata(cand, config, code_file)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    info = {
        "candidate": candidate_rel,
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "built_utc": built,
        "source_version": config["source_version"],
        "serving_profile": config["serving"]["profile"],
        "existing_title": existing_title,
        "next": [f"python automation/preflight.py --dir {out} --kaggle-check",
                 f"python automation/kaggle_pipeline.py push --dir {out}"],
    }
    (out / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return {"dir": str(out), "code_file": code_file, "kernel": kid, "config": config, "metadata": meta}


# ----------------------------------------------------------------------------------------------------------------
# CPU rehearsal
# ----------------------------------------------------------------------------------------------------------------


def default_env_dir() -> str:
    for cand in (os.environ.get("ARC_ENV_DIR"), str(SCRATCH / "kaggle" / "comp" / "environment_files"),
                 str(REPO / "environment_files"), f"/kaggle/input/competitions/{COMPETITION}/environment_files"):
        if cand and Path(cand).is_dir():
            return cand
    raise FileNotFoundError("offline environment_files not found (set ARC_ENV_DIR)")


def exec_dry(nb_dir: Path, source_dir: Path, *, rerun: bool = False, games: str | None = None, seconds: float = 30,
             max_actions: int = 10, work: Path | None = None, python: str | None = None, env_dir: str | None = None,
             timeout: float = 900, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    nb_dir, source_dir = Path(nb_dir), Path(source_dir).resolve()
    meta = json.loads((nb_dir / "kernel-metadata.json").read_text())
    nb_path = nb_dir / meta["code_file"]
    nb = json.loads(nb_path.read_text())
    from preflight import extract_hydra_config

    config = extract_hydra_config(nb) or {}
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="hydra_dry_", dir=str(SCRATCH if SCRATCH.is_dir() else None)))
    kaggle_input, working = work / "input", work / "working"
    owner, slug = str(config.get("source_dataset", DEFAULT_SOURCE_DATASET)).split("/", 1)
    (kaggle_input / "datasets" / owner).mkdir(parents=True, exist_ok=True)
    link = kaggle_input / "datasets" / owner / slug
    if not link.exists():
        link.symlink_to(source_dir, target_is_directory=True)
    comp = kaggle_input / "competitions" / COMPETITION
    comp.mkdir(parents=True, exist_ok=True)
    if not (comp / "environment_files").exists():
        (comp / "environment_files").symlink_to(Path(env_dir or default_env_dir()).resolve(), target_is_directory=True)
    working.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DUCK_PATCH", "LOCAL_ANALYZER", "OPENAI_", "HYDRA_", "KAGGLE_IS_", "TAAF_"))}
    env.update({"HYDRA_DRY_RUN": "1", "HYDRA_KAGGLE_INPUT": str(kaggle_input), "HYDRA_WORKING_DIR": str(working),
                "HYDRA_DRY_SMOKE_SECONDS": str(seconds), "HYDRA_DRY_MAX_ACTIONS": str(max_actions),
                "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
    if games:
        env["HYDRA_DRY_GAMES"] = games
    if rerun:
        env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    env.update(extra_env or {})
    log_path = work / "notebook_stdout.log"
    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run([python or sys.executable, str(Path(__file__).resolve()), "_exec-cells", "--nb", str(nb_path)],
                              cwd=str(work), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return {"rc": proc.returncode, "work": str(work), "working": str(working), "log": str(log_path),
            "stdout": log_path.read_text(encoding="utf-8", errors="replace"), "seconds": time.monotonic() - t0}


def _exec_cells(nb_path: Path) -> int:
    """Execute every code cell of a notebook in order, in one namespace (like a kernel), top-level await allowed."""
    nb = json.loads(Path(nb_path).read_text())
    namespace: dict[str, Any] = {"__name__": "__main__"}
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        print(f"=== HYDRA_DRY cell {i + 1}/{len(code_cells)} ({cell.get('id', '')}) ===", flush=True)
        code = compile(src, f"<cell {i + 1}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)
        if code.co_flags & inspect.CO_COROUTINE:
            asyncio.run(eval(code, namespace))
        else:
            exec(code, namespace)
    print("=== HYDRA_DRY done ===", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a notebook dir from a candidate JSON")
    b.add_argument("--candidate", required=True)
    b.add_argument("--out", default=None, help="output dir (default notebooks/<slug>)")
    b.add_argument("--source-version", default=None, help="pin the source dataset version (HYDRA_SOURCE.json)")
    b.add_argument("--source-dir", default=None, help="staged dataset dir: pin its version")
    b.add_argument("--existing-title", default=None, help="title of the existing kernel (preflight compares)")
    d = sub.add_parser("exec-dry", help="execute the built notebook's cells on CPU (mock LLM)")
    d.add_argument("--dir", required=True)
    d.add_argument("--source-dir", required=True)
    d.add_argument("--rerun", action="store_true", help="rehearse the competition-rerun branch")
    d.add_argument("--games", default=None, help="rerun rehearsal games (comma list)")
    d.add_argument("--seconds", type=float, default=30)
    d.add_argument("--max-actions", type=int, default=10)
    d.add_argument("--work", default=None)
    e = sub.add_parser("_exec-cells")
    e.add_argument("--nb", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        res = build(Path(a.candidate), Path(a.out) if a.out else None, source_version=a.source_version,
                    source_dir=Path(a.source_dir) if a.source_dir else None, existing_title=a.existing_title)
        print(json.dumps({k: res[k] for k in ("dir", "code_file", "kernel")}, indent=1))
        print(json.dumps(res["metadata"], indent=1))
        print(f"next: python automation/preflight.py --dir {res['dir']} --kaggle-check")
        return 0
    if a.cmd == "exec-dry":
        res = exec_dry(Path(a.dir), Path(a.source_dir), rerun=a.rerun, games=a.games, seconds=a.seconds,
                       max_actions=a.max_actions, work=Path(a.work) if a.work else None)
        tail = "\n".join(line for line in res["stdout"].splitlines()
                         if line.startswith(("HYDRA_", "GPU_CHECK", "EFFECTIVE_CONFIG", "===", "Traceback")))
        print(tail)
        print(json.dumps({k: res[k] for k in ("rc", "work", "log")}, indent=1))
        return int(res["rc"] != 0)
    if a.cmd == "_exec-cells":
        return _exec_cells(Path(a.nb))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
