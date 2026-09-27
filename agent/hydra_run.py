"""Hydra notebook runtime (backlog B09): what a notebook built by `automation/build_notebook.py` does after it has
found the source dataset (`kragglenote2forwork/arc3-hydra-src`). Shipped inside that dataset; the notebook only
holds the candidate config and a few calls, so every step here is unit-testable on CPU.

Flow (one call per notebook cell):
    ctx = RunContext.create(config, agent_dir=..., kaggle_input=..., working_dir=..., start_epoch=...)
    bootstrap(ctx)                 # sys.path, source-version check, run-wide env
    serving = start_serving(ctx)   # profile name -> start command (serving/launch.py) -> ready -> harness env
    apply_harness_env(ctx, serving)  # analyzer env layering + DUCK_PATCH switches, asserted BEFORE the import
    run_rerun(ctx, serving)  |  run_smoke(ctx, serving)    # competition gateway  |  3 public games x 8 min
    teardown(ctx, serving)         # never raises (MISTAKES #9)
    effective_config_report(ctx, serving, result)   # what actually ran, from the logs (MISTAKES #10)

Modes: a real competition rerun (`KAGGLE_IS_COMPETITION_RERUN`) plays every gateway game with the full budget and
never raises on a config mismatch (logs only); a commit ("Save & Run") is the SMOKE run: 3 public games x 8 min
from the competition's offline environment files, strict checks, then the placeholder `submission.parquet`
Kaggle requires. `HYDRA_DRY_RUN=1` runs either path on CPU: no GPU check, no pip, games against the scripted mock
LLM of `duck_offline.py` (a `mock`-kind serving profile is still launched for real through `serving/launch.py`).

Log lines worth grepping: HYDRA_MODE, GPU_CHECK, HYDRA_SOURCE, HYDRA_SERVING, HYDRA_ENV, HYDRA_GAME,
HYDRA_RUN_SUMMARY, HYDRA_SCHED, EFFECTIVE_CONFIG, EFFECTIVE_CONFIG_CHECK, HYDRA_TEARDOWN.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

COMPETITION = "arc-prize-2026-arc-agi-3"
GPU_NAME = "RTX PRO 6000"
SOURCE_MARKER = "HYDRA_SOURCE.json"
# The Kaggle gateway's documented local dummy key (Tufa notebook); not a credential.
GATEWAY_DUMMY_KEY = "test-key-123"
GATEWAY_URL = "http://gateway:8001/"
TRUTHY = {"1", "true", "yes", "on"}


def log(tag: str, **fields: Any) -> None:
    print(f"{tag} {json.dumps(fields, sort_keys=True, default=str)}", flush=True)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY


# ----------------------------------------------------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class RunContext:
    config: dict[str, Any]
    agent_dir: Path
    kaggle_input: Path
    working_dir: Path
    start_epoch: float
    true_submission: bool
    dry_run: bool
    notes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, config: dict[str, Any], *, agent_dir: str | Path, kaggle_input: str | Path,
               working_dir: str | Path, start_epoch: float, true_submission: bool | None = None,
               dry_run: bool | None = None) -> "RunContext":
        ctx = cls(
            config=config,
            agent_dir=Path(agent_dir).resolve(),
            kaggle_input=Path(kaggle_input),
            working_dir=Path(working_dir),
            start_epoch=float(start_epoch),
            true_submission=_env_flag("KAGGLE_IS_COMPETITION_RERUN") if true_submission is None else true_submission,
            dry_run=_env_flag("HYDRA_DRY_RUN") if dry_run is None else dry_run,
        )
        ctx.working_dir.mkdir(parents=True, exist_ok=True)
        return ctx

    @property
    def src_root(self) -> Path:
        return self.agent_dir.parent

    @property
    def mode(self) -> str:
        return "rerun" if self.true_submission else "smoke"

    @property
    def strict(self) -> bool:
        """Commit (smoke) runs fail loudly; a competition rerun never raises on a check (logs only)."""
        return not self.true_submission

    @property
    def comp_dir(self) -> Path:
        return self.kaggle_input / "competitions" / COMPETITION

    @property
    def soft_deadline_epoch(self) -> float:
        return self.start_epoch + float(self.config["time"]["soft_deadline_s"])

    def check(self, ok: bool, message: str) -> None:
        if ok:
            return
        self.notes.setdefault("problems", []).append(message)
        log("HYDRA_CHECK_FAILED", mode=self.mode, message=message)
        if self.strict:
            raise RuntimeError(message)


# ----------------------------------------------------------------------------------------------------------------
# Environment checks and bootstrap
# ----------------------------------------------------------------------------------------------------------------


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def gpu_check(dry_run: bool) -> dict[str, Any]:
    if dry_run:
        log("GPU_CHECK", skipped="dry run")
        return {"skipped": True}
    names = gpu_names()
    log("GPU_CHECK", gpus=names)
    if not any(GPU_NAME in name.upper() for name in names):
        raise RuntimeError(f"expected an NVIDIA {GPU_NAME}, found {names or 'no GPU'}: select the RTX PRO 6000 accelerator")
    return {"gpus": names}


def install_arc_runtime(comp_dir: Path, dry_run: bool) -> dict[str, Any]:
    wheels = comp_dir / "arc_agi_3_wheels"
    if dry_run:
        log("HYDRA_PIP", skipped="dry run", wheels=str(wheels))
        return {"skipped": True}
    t0 = time.monotonic()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-warn-conflicts",
         "--disable-pip-version-check", "--find-links", str(wheels), "arc-agi"],
        stdout=subprocess.DEVNULL,
    )
    log("HYDRA_PIP", seconds=round(time.monotonic() - t0, 1), wheels=str(wheels))
    return {"seconds": time.monotonic() - t0}


def duck_source_roots(agent_dir: Path) -> list[Path]:
    return [agent_dir / "duck" / "src" / "ARC3-Inference", agent_dir / "duck" / "src" / "tufa-arc-agi-framework" / "src"]


def bootstrap(ctx: RunContext) -> dict[str, Any]:
    """Import paths, source manifest check, run-wide env (before anything imports the Duck modules)."""
    for entry in [ctx.agent_dir, *duck_source_roots(ctx.agent_dir)]:
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in [ctx.agent_dir, *duck_source_roots(ctx.agent_dir)]]
        + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    manifest_path = ctx.src_root / SOURCE_MARKER
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    want = ctx.config.get("source_version")
    have = manifest.get("version")
    log("HYDRA_SOURCE", root=str(ctx.src_root), version=have, pinned=want, dataset=manifest.get("dataset"))
    if want:
        ctx.check(have == want, f"source dataset version {have!r} != pinned {want!r} (attach the right version)")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"
    os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if ctx.true_submission else "0"
    os.environ["ONLY_RESET_LEVELS"] = "true"      # RESET keeps the level (arc_agi caches this at client build)
    os.environ.setdefault("RECORDINGS_DIR", str(ctx.working_dir / "server_recording"))
    lib = "/usr/local/nvidia/lib64"
    os.environ["LIBRARY_PATH"] = os.pathsep.join([lib] + [p for p in os.environ.get("LIBRARY_PATH", "").split(os.pathsep) if p and p != lib])
    return manifest


# ----------------------------------------------------------------------------------------------------------------
# Serving: profile name -> start command
# ----------------------------------------------------------------------------------------------------------------


def serving_start_command(profile: str, *, python: str, src_root: Path, work_dir: Path,
                          port: int | None = None) -> list[str]:
    """The interface to `serving/launch.py` (B03): start the profile's server, wait for /v1/models, print one JSON
    line {base_url, pid, harness_env, ready_s, log} and exit 0 (server keeps running in its own session)."""
    cmd = [python, str(src_root / "serving" / "launch.py"), "--profile", profile, "--work-dir", str(work_dir)]
    if port is not None:
        cmd += ["--port", str(port)]
    return cmd


@dataclass
class ServingHandle:
    profile: str
    kind: str
    base_url: str = ""
    pid: int | None = None
    log_path: str | None = None
    harness_env: dict[str, str] = field(default_factory=dict)
    ready_s: float | None = None
    mock_server: Any = None          # dry run: duck_offline.MockLLMServer (scripted Duck policy)
    mock_factory: Any = None
    health: "HealthMonitor | None" = None
    command: list[str] = field(default_factory=list)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_launch(cmd: list[str], timeout_s: float, env: dict[str, str]) -> dict[str, Any]:
    """Run the launch command, echo its output, return the last JSON object it printed."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    last: dict[str, Any] = {}
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        tail = (tail + [line])[-40:]
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                last = json.loads(text)
            except json.JSONDecodeError:
                pass
        if time.monotonic() > deadline:
            proc.kill()
            raise TimeoutError(f"serving start exceeded {timeout_s:.0f}s")
    rc = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    if rc != 0 or not last.get("base_url"):
        raise RuntimeError(f"serving start failed rc={rc}: {' | '.join(tail[-8:])}")
    return last


def start_serving(ctx: RunContext) -> ServingHandle:
    serving = ctx.config["serving"]
    profile, kind = serving["profile"], serving.get("runtime_kind", "")
    work_dir = ctx.working_dir / "serving"
    handle = ServingHandle(profile=profile, kind=kind)
    launch_real = not ctx.dry_run or kind == "mock"
    if launch_real:
        port = _free_port() if ctx.dry_run else None
        cmd = serving_start_command(profile, python=sys.executable, src_root=ctx.src_root, work_dir=work_dir, port=port)
        handle.command = cmd
        t0 = time.monotonic()
        log("HYDRA_SERVING", event="start", profile=profile, kind=kind, command=cmd)
        info = _run_launch(cmd, float(serving.get("start_timeout_s", 2400)), dict(os.environ))
        handle.base_url = str(info["base_url"])
        handle.pid = info.get("pid")
        handle.log_path = info.get("log")
        handle.harness_env = {str(k): str(v) for k, v in (info.get("harness_env") or {}).items()}
        handle.ready_s = info.get("ready_s")
        log("HYDRA_SERVING", event="ready", profile=profile, base_url=handle.base_url, pid=handle.pid,
            ready_s=handle.ready_s, wall_s=round(time.monotonic() - t0, 1), log=handle.log_path)
        handle.health = HealthMonitor(handle.base_url, interval_s=float(serving.get("health_interval_s", 60)))
        handle.health.start()
    else:
        cmd = serving_start_command(profile, python=sys.executable, src_root=ctx.src_root, work_dir=work_dir) + ["--print-argv"]
        rendered = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        ctx.check(rendered.returncode == 0, f"profile {profile} does not render: {rendered.stderr[-500:]}")
        handle.command = cmd
        handle.harness_env = {str(k): str(v) for k, v in (serving.get("harness_env") or {}).items()}
        log("HYDRA_SERVING", event="dry_skip", profile=profile, kind=kind, rendered=rendered.returncode == 0)
    if ctx.dry_run:
        import duck_offline  # scripted OpenAI-compatible mock of the Duck policy (agent/duck_offline.py)

        seed = int(os.environ.get("HYDRA_DRY_SEED", "0"))
        server = duck_offline.MockLLMServer(duck_offline.ScriptedPolicy(seed=seed))
        server.__enter__()
        handle.mock_server = server
        handle.mock_factory = duck_offline.make_mock_analyzer_factory(server, {})
        log("HYDRA_SERVING", event="dry_mock", port=server.port)
    return handle


class HealthMonitor(threading.Thread):
    """Logs /v1/models reachability changes (a silent vLLM hang is a known RTX PRO 6000 failure). Log only."""

    def __init__(self, base_url: str, interval_s: float = 60.0):
        super().__init__(name="hydra-health", daemon=True)
        self.url = base_url.rstrip("/") + "/models"
        self.interval_s = max(5.0, interval_s)
        self._stop = threading.Event()
        self.failures = 0
        self.checks = 0

    def run(self) -> None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        healthy = True
        while not self._stop.wait(self.interval_s):
            self.checks += 1
            try:
                with opener.open(self.url, timeout=10) as resp:
                    ok = resp.status < 500
            except Exception:
                ok = False
            if not ok:
                self.failures += 1
            if ok != healthy:
                log("HYDRA_SERVING_HEALTH", healthy=ok, failures=self.failures, checks=self.checks)
                healthy = ok

    def stop(self) -> None:
        self._stop.set()


# ----------------------------------------------------------------------------------------------------------------
# Harness environment (MISTAKES #10: set everything, then assert, then import)
# ----------------------------------------------------------------------------------------------------------------

_ANALYZER_CONSTANTS = {  # env key -> tool_agent module constant (read at import time)
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "_LOCAL_ANALYZER_CONTEXT_WINDOW",
    "LOCAL_ANALYZER_MAX_OUTPUT": "_LOCAL_ANALYZER_MAX_OUTPUT",
    "LOCAL_ANALYZER_TOOL_STEPS": "_LOCAL_ANALYZER_TOOL_STEPS",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "_LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS",
    "LOCAL_ANALYZER_YIELD_SECONDS": "_LOCAL_ANALYZER_YIELD_SECONDS",
    "LOCAL_ANALYZER_TEMPERATURE": "_LOCAL_ANALYZER_TEMPERATURE",
}


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_harness_env(ctx: RunContext, serving: ServingHandle) -> dict[str, Any]:
    """Layer the analyzer env (09-22 baseline -> serving profile -> candidate), set the DUCK_PATCH switches and the
    scheduler config, verify nothing imported tool_agent earlier, import it, and assert its constants."""
    import duck_offline  # noqa: F401  (sets the 09-22 analyzer env with setdefault, puts the Duck sources on sys.path)

    already = "inference.agent.tool_agent" in sys.modules
    env: dict[str, str] = dict(duck_offline.SHIPPED_ANALYZER_ENV)
    env.update(serving.harness_env)
    env.update({str(k): str(v) for k, v in (ctx.config.get("analyzer_env") or {}).items()})
    if serving.base_url:
        env["LOCAL_ANALYZER_BASE_URL"] = serving.base_url
        env["OPENAI_BASE_URL"] = serving.base_url
    os.environ.update(env)
    patches_env = {str(k): str(v) for k, v in (ctx.config.get("duck_patches") or {}).items()}
    os.environ.update(patches_env)
    sched = dict(ctx.config.get("scheduler") or {})
    if sched:
        os.environ["DUCK_PATCH_B08_SCHEDULER_CONFIG"] = json.dumps(sched, sort_keys=True)
    os.environ["DUCK_PATCH_B08_SCHEDULER_SUMMARY"] = str(ctx.working_dir / "hydra_scheduler.json")

    ctx.check(not already, "inference.agent.tool_agent was imported before the analyzer env was set (MISTAKES #10)")
    from inference.agent import patches
    from inference.agent import tool_agent as ta

    expected = {str(k): str(v) for k, v in (ctx.config.get("expected_analyzer_env") or {}).items()}
    mismatches = []
    for key, want in expected.items():
        have = os.environ.get(key)
        if have != want:
            mismatches.append(f"env {key}={have!r} expected {want!r}")
        const = _ANALYZER_CONSTANTS.get(key)
        if const is not None and _as_number(getattr(ta, const, None)) != _as_number(want):
            mismatches.append(f"tool_agent.{const}={getattr(ta, const, None)!r} expected {want!r}")
    patch_report = patches.report()
    for key, want in patches_env.items():
        if key.startswith("DUCK_PATCH_") and key[len("DUCK_PATCH_"):] in patch_report:
            pid = key[len("DUCK_PATCH_"):]
            if patch_report[pid] != (want.strip().lower() in TRUTHY):
                mismatches.append(f"patch {pid} enabled={patch_report[pid]} expected {want!r}")
    probe = ta.ToolAgent(model="local", base_url=env.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1"))
    budget = int(probe._context_budget_tokens)
    ctx.notes["expected_budget"] = budget
    view = {k: env[k] for k in sorted(env) if k.startswith(("LOCAL_ANALYZER", "MULTIMODAL", "INFERENCE"))}
    log("HYDRA_ENV", analyzer=view, patches=patch_report, context_budget_tokens=budget, scheduler=sched or None)
    for problem in mismatches:
        ctx.check(False, problem)
    return {"analyzer": view, "patches": patch_report, "budget": budget, "mismatches": mismatches}


# ----------------------------------------------------------------------------------------------------------------
# Games and runs
# ----------------------------------------------------------------------------------------------------------------


def offline_env_dir(ctx: RunContext) -> str:
    env_dir = ctx.comp_dir / "environment_files"
    if env_dir.is_dir():
        return str(env_dir)
    if ctx.dry_run:
        import duck_offline

        return duck_offline.default_env_dir()
    raise FileNotFoundError(f"competition environment files not found at {env_dir}")


def resolve_offline_ids(names: list[str], env_dir: str) -> list[str]:
    ids = []
    for name in names:
        short = name.split("-", 1)[0]
        metas = sorted(Path(env_dir).glob(f"{short}/*/metadata.json"))
        if not metas:
            raise ValueError(f"game {name!r} not found under {env_dir}")
        ids.append(json.loads(metas[0].read_text())["game_id"])
    return ids


def offline_games(ids: list[str], env_dir: str) -> list[Any]:
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    return [taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec) for gid in ids]


def wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(f"{base_url}api/games", timeout=10) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last}")


def competition_games() -> list[Any]:
    import arc_agi
    import taaf.game_api

    os.environ.setdefault("ARC_API_KEY", GATEWAY_DUMMY_KEY)
    os.environ.setdefault("ARC_BASE_URL", GATEWAY_URL)
    wait_for_gateway(os.environ["ARC_BASE_URL"])
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.COMPETITION,
                                    arc_base_url=os.environ["ARC_BASE_URL"], environments_dir="")
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=spec.arc_base_url,
                            environments_dir="")
    ids = [info.game_id for info in arcade.available_environments]
    if not ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec) for gid in ids]


def build_solver(ctx: RunContext, serving: ServingHandle, overrides: dict[str, Any] | None = None) -> Any:
    import duck_offline
    from inference.framework.solver import HarnessSolver

    fields = dict(duck_offline.SHIPPED_SOLVER_FIELDS)
    fields.update(ctx.config.get("solver") or {})
    fields.update(overrides or {})
    if serving.mock_factory is not None:
        fields["analyzer_factory"] = serving.mock_factory
    return HarnessSolver(**fields)


def _run_coro_in_thread(factory: Any) -> Any:
    """Jupyter already runs an event loop in the main thread; run the benchmark on its own loop in a thread."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = asyncio.run(factory())
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=target, name="hydra-benchmark")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def run_benchmark(ctx: RunContext, serving: ServingHandle, games: list[Any], *, soft_end_epoch: float,
                  solver_overrides: dict[str, Any] | None = None, label: str) -> dict[str, Any]:
    import taaf.benchmark

    job_dir = ctx.working_dir
    if not (job_dir / "git_status.txt").exists():  # TAAF otherwise shells out to git
        (job_dir / "git_status.txt").write_text("hydra notebook run (git status not captured)\n")
    solver = build_solver(ctx, serving, solver_overrides)
    bm = taaf.benchmark.Benchmark(label=label, games=games, solver=solver, n_passes=1, job_dir=job_dir)
    soft_end = datetime.fromtimestamp(soft_end_epoch)
    ctx.notes["setup_s"] = round(time.time() - ctx.start_epoch, 1)
    log("HYDRA_RUN", event="start", mode=ctx.mode, games=len(games), soft_end=soft_end.isoformat(),
        left_s=round(soft_end_epoch - time.time(), 1), concurrency=solver.concurrency,
        max_runtime_s_per_game=solver.max_runtime_s_per_game, max_actions=solver.max_actions_per_game)
    t0 = time.monotonic()
    _run_coro_in_thread(lambda: bm.run(soft_end_time=soft_end, runtime_environment=None, minimal_diagnostics=True))
    try:
        bm._save_json()
    except Exception as exc:
        log("HYDRA_RUN", event="save_json_failed", error=repr(exc))
    runs = []
    for run in bm.game_runs:
        item = {"game_id": run.game_id, "state": run.state, "levels": run.levels_completed,
                "n_levels": run.number_of_levels, "actions": len(run.history), "score": run.final_score}
        runs.append(item)
        log("HYDRA_GAME", **item)
    summary = {
        "mode": ctx.mode, "games": len(runs), "wall_s": round(time.monotonic() - t0, 1),
        "levels": sum(int(r["levels"] or 0) for r in runs), "actions": sum(r["actions"] for r in runs),
        "crashed": [r["game_id"] for r in runs if r["state"] == "crashed"],
        "mean_score": round(sum(float(r["score"] or 0.0) for r in runs) / max(1, len(runs)), 3),
        "runs": runs,
    }
    (ctx.working_dir / "hydra_run_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    log("HYDRA_RUN_SUMMARY", **{k: v for k, v in summary.items() if k != "runs"})
    return summary


def run_rerun(ctx: RunContext, serving: ServingHandle) -> dict[str, Any]:
    """Competition rerun: every gateway game, full budget (the gateway writes the real submission)."""
    if ctx.dry_run:
        names = [g for g in os.environ.get("HYDRA_DRY_GAMES", "ls20,vc33,ft09,ar25,sp80").split(",") if g]
        env_dir = offline_env_dir(ctx)
        games = offline_games(resolve_offline_ids(names, env_dir), env_dir)
        soft_end = min(ctx.soft_deadline_epoch, time.time() + float(os.environ.get("HYDRA_DRY_DEADLINE_S", "40")))
        overrides = {"max_actions_per_game": int(os.environ.get("HYDRA_DRY_MAX_ACTIONS", "10"))}
    else:
        games = competition_games()
        soft_end = ctx.soft_deadline_epoch
        overrides = {}
    log("HYDRA_GATEWAY", games=len(games), setup_s=round(time.time() - ctx.start_epoch, 1),
        window_s=round(soft_end - time.time(), 1))
    return run_benchmark(ctx, serving, games, soft_end_epoch=soft_end, solver_overrides=overrides,
                         label=f"hydra-{ctx.config.get('name', 'candidate')}-rerun")


def run_smoke(ctx: RunContext, serving: ServingHandle) -> dict[str, Any]:
    """Commit run: the pipeline end to end on 3 public games x 8 min, then the placeholder submission.parquet.
    For pipeline sanity only: public-25 scores barely predict the LB (r = 0.16, research/10)."""
    smoke = ctx.config["smoke"]
    seconds = float(smoke["minutes"]) * 60.0
    overrides: dict[str, Any] = {"max_runtime_s_per_game": seconds}
    if ctx.dry_run:
        seconds = float(os.environ.get("HYDRA_DRY_SMOKE_SECONDS", "30"))
        overrides = {"max_runtime_s_per_game": seconds,
                     "max_actions_per_game": int(os.environ.get("HYDRA_DRY_MAX_ACTIONS", "10"))}
    env_dir = offline_env_dir(ctx)
    ids = resolve_offline_ids(list(smoke["games"]), env_dir)
    games = offline_games(ids, env_dir)
    if (ctx.config.get("duck_patches") or {}).get("DUCK_PATCH_B08_SCHEDULER", "0") in TRUTHY:
        # same code path as the rerun, with every game capped at the smoke budget
        sched = dict(ctx.config.get("scheduler") or {})
        sched.update({"cap_s": seconds, "floor_s": min(seconds, float(sched.get("floor_s") or seconds))})
        os.environ["DUCK_PATCH_B08_SCHEDULER_CONFIG"] = json.dumps(sched, sort_keys=True)
    stop_margin = float((ctx.config.get("scheduler") or {}).get("stop_margin_s", 90.0))
    soft_end = min(ctx.soft_deadline_epoch, time.time() + seconds + stop_margin + 60.0)
    summary = run_benchmark(ctx, serving, games, soft_end_epoch=soft_end, solver_overrides=overrides,
                            label=f"hydra-{ctx.config.get('name', 'candidate')}-smoke")
    ctx.check(summary["games"] == len(ids), f"smoke played {summary['games']} games, expected {len(ids)}")
    ctx.check(not summary["crashed"], f"smoke games crashed: {summary['crashed']}")
    idle = [r["game_id"] for r in summary["runs"] if r["actions"] <= 0]
    ctx.check(not idle, f"smoke games without a single action: {idle}")
    summary["submission"] = str(write_placeholder_submission(ctx.working_dir))
    return summary


def write_placeholder_submission(working_dir: Path) -> Path:
    """Kaggle's Save & Run needs a submission.parquet; the real one comes from the gateway in the rerun."""
    path = Path(working_dir) / "submission.parquet"
    rows = {"row_id": ["1_0"], "game_id": ["1"], "end_of_game": [True], "score": [1]}
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table(rows), path)
    log("HYDRA_SUBMISSION", placeholder=str(path), bytes=path.stat().st_size)
    return path


# ----------------------------------------------------------------------------------------------------------------
# Effective config from the logs (MISTAKES #10)
# ----------------------------------------------------------------------------------------------------------------

_STATUS_RE = {
    "model": re.compile(r"^model: (.+)$", re.M),
    "context_budget_tokens": re.compile(r"^context_budget_tokens: (\d+)$", re.M),
    "yield_seconds": re.compile(r"^yield_seconds: (.+)$", re.M),
    "tool_output_tokens": re.compile(r"^tool_output_tokens: (\d+)$", re.M),
}
_SERVER_RE = {
    "max_model_len": re.compile(r"max_model_len['\"]?\s*[:=]\s*(\d+)"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "max_concurrency": re.compile(r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x"),
    "prefix_caching": re.compile(r"enable_prefix_caching['\"]?\s*[:=]\s*(True|False)"),
}


def transcript_statuses(working_dir: Path) -> dict[str, list[str]]:
    seen: dict[str, set[str]] = {key: set() for key in _STATUS_RE}
    for path in sorted((Path(working_dir) / "transcripts").glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for key, rx in _STATUS_RE.items():
            seen[key].update(m.strip() for m in rx.findall(text))
    return {k: sorted(v) for k, v in seen.items()}


def server_log_facts(log_path: str | None) -> dict[str, Any]:
    if not log_path or not Path(log_path).is_file():
        return {}
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    facts: dict[str, Any] = {}
    for key, rx in _SERVER_RE.items():
        hits = rx.findall(text)
        if hits:
            facts[key] = hits[-1].replace(",", "")
    return facts


def effective_config_report(ctx: RunContext, serving: ServingHandle, result: dict[str, Any] | None) -> dict[str, Any]:
    statuses = transcript_statuses(ctx.working_dir)
    server = server_log_facts(serving.log_path)
    sched_path = ctx.working_dir / "hydra_scheduler.json"
    sched = json.loads(sched_path.read_text()) if sched_path.is_file() else None
    expected_model = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
    window = _as_number(os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW"))
    budget = ctx.notes.get("expected_budget")
    mismatches: list[str] = []
    if result and result.get("actions"):
        if statuses["model"] != [expected_model]:
            mismatches.append(f"transcript model {statuses['model']} != {expected_model!r}")
        if budget is not None and statuses["context_budget_tokens"] != [str(budget)]:
            mismatches.append(f"transcript context_budget_tokens {statuses['context_budget_tokens']} != [{budget}]")
    if server.get("max_model_len") and window and float(server["max_model_len"]) < window:
        mismatches.append(f"server max_model_len {server['max_model_len']} < analyzer window {window:.0f}")
    scheduler_on = (ctx.config.get("duck_patches") or {}).get("DUCK_PATCH_B08_SCHEDULER", "0") in TRUTHY
    if scheduler_on and result and result.get("games"):
        if sched is None:
            mismatches.append("scheduler enabled but hydra_scheduler.json missing")
        else:
            started = sum(1 for g in sched["games"].values() if g.get("first_turn_s") is not None)
            if sched["plan"]["n_games"] != result["games"]:
                mismatches.append(f"scheduler planned {sched['plan']['n_games']} games, run had {result['games']}")
            if started != sched["plan"]["n_games"]:
                mismatches.append(f"scheduler started {started}/{sched['plan']['n_games']} games")
    report = {
        "mode": ctx.mode, "dry_run": ctx.dry_run, "profile": serving.profile, "transcripts": statuses,
        "server_log": server, "analyzer_window": window, "expected_budget": budget,
        "scheduler_plan": sched["plan"] if sched else None,
        "setup_s": ctx.notes.get("setup_s"),
    }
    log("EFFECTIVE_CONFIG", **report)
    log("EFFECTIVE_CONFIG_CHECK", ok=not mismatches, mismatches=mismatches)
    (ctx.working_dir / "hydra_effective_config.json").write_text(
        json.dumps({**report, "mismatches": mismatches}, indent=1, default=str) + "\n")
    for problem in mismatches:
        ctx.check(False, problem)
    return {**report, "mismatches": mismatches}


# ----------------------------------------------------------------------------------------------------------------
# Teardown (never raises)
# ----------------------------------------------------------------------------------------------------------------


def _kill_group(pid: int, wait_s: float = 30.0) -> str:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return "gone"
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError):
            return "terminated"
        time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return "killed"


def teardown(ctx: RunContext | None, serving: ServingHandle | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        if serving is not None and serving.health is not None:
            serving.health.stop()
            out["health_failures"] = serving.health.failures
    except Exception as exc:
        out["health_error"] = repr(exc)
    try:
        if serving is not None and serving.mock_server is not None:
            serving.mock_server.__exit__(None, None, None)
            out["mock"] = "stopped"
    except Exception as exc:
        out["mock_error"] = repr(exc)
    try:
        if serving is not None and serving.pid:
            out["server"] = _kill_group(int(serving.pid))
    except Exception as exc:
        out["server_error"] = repr(exc)
    for command in ((ctx.config.get("teardown_commands") if ctx is not None else None) or []):
        try:
            proc = subprocess.run(command, shell=True, check=False, timeout=60, capture_output=True, text=True)
            out.setdefault("commands", []).append({"command": command, "rc": proc.returncode})
        except Exception as exc:
            out.setdefault("commands", []).append({"command": command, "error": repr(exc)})
    try:
        log("HYDRA_TEARDOWN", **out)
    except Exception:
        pass
    return out
