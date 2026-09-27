#!/usr/bin/env python3
"""Start an OpenAI-compatible vLLM server from a serving profile (backlog B03) and run the EXP-001 suite.

Profiles live in `serving/profiles/<name>.json` (data only: argv template, env, runtime kind, client
settings the harness/bench must use, memory budget, evidence, fallback profile).

Usage
-----
  python serving/launch.py --profile flashnext_tuned --print-argv          # render argv/env, no side effects
  python serving/launch.py --profile flashnext_tuned                       # prepare runtime, start, wait for /v1/models
  python serving/launch.py --profile flashnext_tuned --stop-after          # start, wait, then stop (startup smoke)
  python serving/launch.py --suite flashnext_tuned,flashnext_baseline,qwen27b_fp8 \
        --duration 720 --warmup 90 --agents 28 --out /kaggle/working/exp001     # one-command EXP-001
  python serving/launch.py --suite mock_cpu --duration 20 --warmup 3 --agents 8 --out /tmp/x   # CPU dry run

Runtime kinds
-------------
flashnext_dev       keithtyser's pinned vLLM docker image (0.1.dev20073+g8e685d198) extracted from the Kaggle
                    dataset layers to /tmp, PLE FP8 patch applied — reuses the dataset's own serving_setup.py
                    (same code path as the 09-22 run), then launches *our* argv.
vllm019_wheelhouse  vLLM 0.19.0 from driessmit1/arc3-vllm-h100-wheelhouse-v3 pip-installed with --target, plus the
                    transformers 5.8.0 / huggingface_hub 1.5.0 overlay from keithtyser/taaf-duck-qwen38-serving-v1.
system              whatever `python -m vllm` resolves to in the current interpreter.
mock                automation/serving_bench.py mock-server (CPU only).

Teardown never raises (MISTAKES #9). Secrets are never written: env dumps drop *TOKEN*/*KEY*/*SECRET* keys.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PROFILES_DIR = HERE / "profiles"
BENCH_SCRIPT = REPO_ROOT / "automation" / "serving_bench.py"
PRISTINE_ENV = dict(os.environ)
SECRET_RE = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY|_KEY$|^KAGGLE_KEY|CREDENTIAL", re.I)

FLASHNEXT_SOURCE_DATASET = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
FLASHNEXT_RUNTIME_DATASET = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
WHEELHOUSE_DATASET = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
OVERLAY_DATASET = "keithtyser/taaf-duck-qwen38-serving-v1"
VLLM019_STAMP = "vllm==0.19.0 torch==2.10.0 transformers==5.8.0 huggingface-hub==1.5.0\n"
# EXP-001 plan: most likely winner first (in case the session runs short), then arm B, then the 09-22 reference;
# extras only run if the wall-clock budget still allows them.
EXP001_PLAN = ["flashnext_tuned", "qwen27b_fp8", "flashnext_baseline"]
EXP001_EXTRAS = ["flashnext_nomtp_kv16_seqs28_pc", "flashnext_tuned_mtp2", "flashnext_nomtp_kv12_seqs28_pc_ctx32k"]


def log(msg: str) -> None:
    print(f"[launch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------------------------------
def load_profile(name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.suffix:
        p = PROFILES_DIR / f"{name_or_path}.json"
    prof = json.loads(p.read_text())
    if prof.get("schema") != 1:
        raise ValueError(f"{p}: unsupported profile schema {prof.get('schema')}")
    for key in ("name", "runtime", "model", "entrypoint", "args", "client"):
        if key not in prof:
            raise ValueError(f"{p}: missing '{key}'")
    return prof


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def resolve_model_path(prof: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    cands = prof["model"].get("path_candidates") or []
    for c in cands:
        if Path(c).exists():
            return c
    ref = prof["model"].get("kaggle_model")
    if ref and Path("/kaggle/input").is_dir():
        owner, slug = ref.split("/")[:2]
        for root in (Path("/kaggle/input/models") / owner / slug, Path("/kaggle/input") / slug):
            if root.is_dir():
                for cfg in sorted(root.rglob("config.json")):
                    return str(cfg.parent)
    return cands[0] if cands else "MODEL_PATH_UNRESOLVED"


def _fmt_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def render_argv(prof: dict[str, Any], *, python: str | None = None, model_path: str | None = None,
                port: int | None = None) -> list[str]:
    subst = {
        "model_path": model_path or resolve_model_path(prof),
        "served_model_name": prof["model"].get("served_model_name", ""),
        "host": prof.get("host", "127.0.0.1"),
        "port": str(port or prof.get("port", 1234)),
        "bench_script": str(BENCH_SCRIPT),
    }

    def s(x: str) -> str:
        return x.format(**subst) if isinstance(x, str) else x

    argv = [python or sys.executable] + [s(x) for x in prof["entrypoint"]]
    for item in prof["args"]:
        if isinstance(item, str):
            argv.append(s(item))
        else:
            flag, value = item
            argv += [s(flag), s(_fmt_value(value)) if not isinstance(value, (dict, list)) else _fmt_value(value)]
    return argv


def base_url(prof: dict[str, Any], port: int | None = None) -> str:
    return f"http://{prof.get('host', '127.0.0.1')}:{port or prof.get('port', 1234)}/v1"


def safe_env_view(env: dict[str, str], base: dict[str, str] | None = None) -> dict[str, str]:
    """Env keys the runtime added or changed, secrets removed."""
    base = base if base is not None else PRISTINE_ENV
    return {k: v for k, v in sorted(env.items()) if base.get(k) != v and not SECRET_RE.search(k)}


# ----------------------------------------------------------------------------------------------
# Kaggle input discovery
# ----------------------------------------------------------------------------------------------
def dataset_dir(ref: str, marker: str | None = None) -> Path | None:
    owner, slug = ref.split("/", 1)
    for cand in (Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug):
        if cand.is_dir() and (marker is None or (cand / marker).exists()):
            return cand
    root = Path("/kaggle/input")
    if marker and root.is_dir():
        for hit in root.rglob(marker):
            if slug in str(hit.parent):
                return hit.parent
    return None


# ----------------------------------------------------------------------------------------------
# Runtimes
# ----------------------------------------------------------------------------------------------
@contextmanager
def _temp_environ(updates: dict[str, str], keep: tuple[str, ...] = ()):
    """Set env vars for the duration of the block; restore afterwards except keys in `keep`."""
    saved = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, v in saved.items():
            if k in keep:
                continue
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def prepare_flashnext_dev(prof: dict[str, Any], work_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    bundle = dataset_dir(FLASHNEXT_SOURCE_DATASET, "serving_setup.py")
    runtime = dataset_dir(FLASHNEXT_RUNTIME_DATASET, "runtime-manifest.json")
    if bundle is None or runtime is None:
        raise RuntimeError(f"flashnext_dev needs datasets {FLASHNEXT_SOURCE_DATASET} and {FLASHNEXT_RUNTIME_DATASET} attached "
                           f"(found bundle={bundle}, runtime={runtime})")
    work_dir.mkdir(parents=True, exist_ok=True)
    scratch_setup_env = work_dir / "taaf_setup_env.json"
    if not scratch_setup_env.exists():
        scratch_setup_env.write_text("{}\n")
    # keithtyser's serving_setup.py reads these at import time. We only borrow its runtime-extraction, PLE-patch and
    # environment functions (never persist_analyzer_environment, which rewrites LOCAL_ANALYZER_CONTEXT_WINDOW=32768),
    # and restore the caller's values afterwards so a harness notebook keeps its own bundle/setup-env paths.
    updates = {
        "TAAF_KAGGLE_BUNDLE_DIR": str(bundle),
        "TAAF_KAGGLE_WORKING_DIR": os.environ.get("TAAF_KAGGLE_WORKING_DIR") or str(work_dir),
        "TAAF_KAGGLE_SETUP_ENV": os.environ.get("TAAF_KAGGLE_SETUP_ENV") or str(scratch_setup_env),
        "TAAF_KAGGLE_INPUT_PATHS": json.dumps({FLASHNEXT_SOURCE_DATASET: str(bundle), FLASHNEXT_RUNTIME_DATASET: str(runtime)}),
        "TAAF_KAGGLE_FAST_START": "1",
        "TAAF_VLLM_OMP_THREADS": str(prof["runtime"].get("omp_threads", 1)),
    }
    with _temp_environ(updates):
        return _prepare_flashnext_dev_inner(prof, bundle, runtime)


def _prepare_flashnext_dev_inner(prof: dict[str, Any], bundle: Path, runtime: Path) -> tuple[dict[str, str], dict[str, Any]]:
    ss = _import_file("taaf_serving_setup", bundle / "serving_setup.py")
    ss.source_identity()
    info: dict[str, Any] = {"bundle": str(bundle), "runtime_dataset": str(runtime), "vllm_version": ss.VLLM_VERSION}
    marker = ss.RUNTIME_ROOT / ".arc3-extracted.json"
    site = ss.RUNTIME_ROOT / "usr" / "local" / "lib" / "python3.12" / "dist-packages"
    if not marker.exists() or not (site / "vllm" / "__init__.py").exists():
        t0 = time.monotonic()
        ss.validate_runtime_storage()
        extract = ss.verify_and_extract_runtime(runtime, full_layer_hashes=False, scan_extracted_caches=False)
        marker.write_text(json.dumps({"extract_seconds": extract.get("extract_seconds"), "epoch": time.time()}))
        info["extract_seconds"] = time.monotonic() - t0
        log(f"runtime extracted in {info['extract_seconds']:.0f}s -> {ss.RUNTIME_ROOT}")
    else:
        info["extract_seconds"] = 0.0
        log(f"runtime already extracted at {ss.RUNTIME_ROOT}")
    ple = site / "vllm" / "models" / "qwen3_8_flash_next" / "nvidia" / "ple_layer.py"
    sha = ss.sha256_file(ple)
    if sha == ss.VLLM_PLE_STOCK_SHA256:
        ss.patch_ple_layer()
        info["ple_patch"] = "applied"
    elif sha == ss.VLLM_PLE_PATCHED_SHA256:
        info["ple_patch"] = "already"
    else:
        raise RuntimeError(f"unexpected ple_layer.py sha {sha}")
    tuning = ss.resolve_vllm_tuning()
    env, check = ss.runtime_environment(deep_preload_validation=False, tuning=tuning)
    info["environment"] = {k: check.get(k) for k in ("validation_mode", "python", "glibc", "site_packages")}
    env.update({k: str(v) for k, v in (prof.get("env") or {}).items()})
    return env, info


def _pip(args: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    log("pip " + " ".join(args[:3]) + " ...")
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def prepare_vllm019(prof: dict[str, Any], work_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    wheelhouse = dataset_dir(WHEELHOUSE_DATASET, "requirements.lock")
    overlay_root = dataset_dir(OVERLAY_DATASET, "taaf-kaggle-bundle.json") or dataset_dir(OVERLAY_DATASET, "serving_wheels")
    if wheelhouse is None or overlay_root is None:
        raise RuntimeError(f"vllm019_wheelhouse needs {WHEELHOUSE_DATASET} and {OVERLAY_DATASET} attached "
                           f"(found {wheelhouse}, {overlay_root})")
    site = Path(prof["runtime"].get("site_dir", "/tmp/vllm019-site"))
    stamp = site / ".arc3-stamp"
    env = dict(PRISTINE_ENV)
    info: dict[str, Any] = {"site": str(site), "wheelhouse": str(wheelhouse)}
    if not (stamp.exists() and stamp.read_text() == VLLM019_STAMP):
        t0 = time.monotonic()
        shutil.rmtree(site, ignore_errors=True)
        site.mkdir(parents=True)
        _pip(["install", "--no-index", "--find-links", str(wheelhouse), "--requirement", str(wheelhouse / "requirements.lock"),
              "--target", str(site), "--upgrade", "--ignore-installed", "--only-binary", ":all:", "--no-compile",
              "--disable-pip-version-check", "--no-warn-conflicts", "--quiet"], env)
        for pattern in ("transformers", "transformers-*.dist-info", "huggingface_hub", "huggingface_hub-*.dist-info"):
            for path in site.glob(pattern):
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
        wheels = [overlay_root / "serving_wheels" / "transformers-5.8.0-py3-none-any.whl",
                  overlay_root / "serving_wheels" / "huggingface_hub-1.5.0-py3-none-any.whl"]
        missing = [str(w) for w in wheels if not w.is_file()]
        if missing:
            raise FileNotFoundError(f"overlay wheels missing: {missing}")
        _pip(["install", "--no-index", "--no-deps", "--target", str(site), "--upgrade", "--force-reinstall", "--no-compile",
              "--disable-pip-version-check", "--quiet", *map(str, wheels)], env)
        stamp.write_text(VLLM019_STAMP)
        info["install_seconds"] = time.monotonic() - t0
    env["PYTHONPATH"] = os.pathsep.join([str(site)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    lib = "/usr/local/nvidia/lib64"
    env["LIBRARY_PATH"] = os.pathsep.join([lib] + ([env["LIBRARY_PATH"]] if env.get("LIBRARY_PATH") else []))
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    env.update({k: str(v) for k, v in (prof.get("env") or {}).items()})
    check = subprocess.run([sys.executable, "-c", "import importlib.metadata as m; print({n: m.version(n) for n in "
                            "('vllm','torch','transformers','huggingface-hub')})"], env=env, capture_output=True, text=True)
    info["versions"] = check.stdout.strip() or check.stderr.strip()[-300:]
    log(f"vllm019 runtime: {info['versions']}")
    return env, info


def prepare_runtime(prof: dict[str, Any], work_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    kind = prof["runtime"]["kind"]
    if kind == "flashnext_dev":
        return prepare_flashnext_dev(prof, work_dir)
    if kind == "vllm019_wheelhouse":
        return prepare_vllm019(prof, work_dir)
    env = dict(PRISTINE_ENV)
    env.update({k: str(v) for k, v in (prof.get("env") or {}).items()})
    return env, {"kind": kind}


# ----------------------------------------------------------------------------------------------
# Process control
# ----------------------------------------------------------------------------------------------
@dataclass
class ServerHandle:
    profile: str
    argv: list[str]
    base_url: str
    log_path: Path
    proc: subprocess.Popen | None = None
    started: float = 0.0
    ready_s: float | None = None
    runtime_info: dict[str, Any] = field(default_factory=dict)


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_json(url: str, timeout: float = 10.0) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_text(url: str, timeout: float = 10.0) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def tail(path: Path, n: int = 60) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def gpu_memory() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append({"name": parts[0], "used_mib": int(float(parts[1])), "total_mib": int(float(parts[2]))})
    return rows


def wait_gpu_free(max_used_mib: int = 3000, timeout: float = 90.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    rows = gpu_memory()
    while rows and any(r["used_mib"] > max_used_mib for r in rows) and time.monotonic() < deadline:
        time.sleep(3)
        rows = gpu_memory()
    return rows


def start_server(prof: dict[str, Any], *, work_dir: Path, model_path: str | None = None, port: int | None = None) -> ServerHandle:
    work_dir.mkdir(parents=True, exist_ok=True)
    port = port or prof.get("port", 1234)
    host = prof.get("host", "127.0.0.1")
    if port_open(host, port):
        raise RuntimeError(f"port {host}:{port} already in use")
    env, info = prepare_runtime(prof, work_dir)
    argv = render_argv(prof, model_path=model_path, port=port)
    log_path = work_dir / "server.log"
    handle = ServerHandle(prof["name"], argv, base_url(prof, port), log_path, runtime_info=info)
    (work_dir / "server_identity.json").write_text(json.dumps({
        "profile": prof["name"], "argv": argv, "env_changes": safe_env_view(env), "runtime": info,
        "gpu_before": gpu_memory(), "started_epoch": time.time()}, indent=1, default=str))
    log(f"starting {prof['name']}: {' '.join(argv)}")
    fh = open(log_path, "w", encoding="utf-8")
    try:
        handle.proc = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                       cwd=str(work_dir))
    finally:
        fh.close()
    handle.started = time.monotonic()
    return handle


def wait_ready(handle: ServerHandle, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_note = 0.0
    while time.monotonic() < deadline:
        if handle.proc is not None and handle.proc.poll() is not None:
            return {"ready": False, "reason": f"server exited rc={handle.proc.returncode}", "log_tail": tail(handle.log_path)}
        try:
            models = http_json(handle.base_url.rstrip("/") + "/models", timeout=5)
            handle.ready_s = time.monotonic() - handle.started
            log(f"{handle.profile} ready after {handle.ready_s:.0f}s: {[m.get('id') for m in models.get('data', [])]}")
            return {"ready": True, "ready_s": handle.ready_s, "models": models}
        except Exception:
            pass
        if time.monotonic() - last_note > 60:
            last_note = time.monotonic()
            log(f"waiting for {handle.profile} ({time.monotonic() - handle.started:.0f}s): {tail(handle.log_path, 1)[-160:]}")
        time.sleep(5)
    return {"ready": False, "reason": f"timeout after {timeout:.0f}s", "log_tail": tail(handle.log_path)}


def write_harness_env(prof: dict[str, Any], handle: ServerHandle, work_dir: Path) -> dict[str, str]:
    """Publish the env a Duck/Hydra harness must use with this server (context window, model id, URL).

    Written last, after the server is ready, so it wins over serving_setup.py's persisted
    LOCAL_ANALYZER_CONTEXT_WINDOW=32768 (the 09-22 "ctx16k" bug): merged into $TAAF_KAGGLE_SETUP_ENV when the caller
    defined one (the TAAF notebook re-applies that file after its setup loop), exported into os.environ, and saved
    as <work_dir>/harness_env.json.
    """
    env = {k: str(v) for k, v in (prof.get("harness_env") or {}).items()}
    for key in ("LOCAL_ANALYZER_BASE_URL", "OPENAI_BASE_URL"):
        if key in env:
            env[key] = handle.base_url
    if not env:
        return env
    (work_dir / "harness_env.json").write_text(json.dumps(env, indent=1, sort_keys=True) + "\n")
    target = PRISTINE_ENV.get("TAAF_KAGGLE_SETUP_ENV")
    if target:
        try:
            path = Path(target)
            doc = json.loads(path.read_text()) if path.exists() else {}
            if isinstance(doc, dict):
                doc.update(env)
                path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        except (OSError, json.JSONDecodeError) as exc:
            log(f"could not update {target}: {exc!r}")
    os.environ.update(env)
    return env


def _proc_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, sid)"""
    table: dict[int, tuple[int, int]] = {}
    for d in Path("/proc").iterdir() if Path("/proc").is_dir() else []:
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            rest = stat[stat.rindex(")") + 2:].split()
            table[int(d.name)] = (int(rest[1]), int(rest[3]))
        except (OSError, ValueError, IndexError):
            continue
    return table


def _descendants(root: int) -> set[int]:
    table = _proc_table()
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _sid) in table.items():
        kids.setdefault(ppid, []).append(pid)
    out: set[int] = set()
    stack = [root]
    while stack:
        p = stack.pop()
        for c in kids.get(p, []):
            if c not in out:
                out.add(c)
                stack.append(c)
    out |= {pid for pid, (_pp, sid) in table.items() if sid == root}
    out.discard(os.getpid())
    return out


def stop_server(handle: ServerHandle | None, *, term_wait: float = 30.0) -> dict[str, Any]:
    """Stop the server process tree. Never raises."""
    result: dict[str, Any] = {"stopped": True}
    try:
        if handle is None or handle.proc is None:
            return result
        root = handle.proc.pid
        victims = _descendants(root)
        if handle.proc.poll() is None:
            try:
                os.killpg(root, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                handle.proc.wait(timeout=term_wait)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(root, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    handle.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    result["stopped"] = False
        survivors = []
        for pid in sorted(victims | _descendants(root)):
            try:
                os.kill(pid, signal.SIGKILL)
                survivors.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        result["killed_leftovers"] = survivors
        host, port = urlparse(handle.base_url).hostname, urlparse(handle.base_url).port
        deadline = time.monotonic() + 30
        while port and port_open(host, port) and time.monotonic() < deadline:
            time.sleep(1)
        result["port_closed"] = not (port and port_open(host, port))
        result["gpu_after"] = wait_gpu_free()
        log(f"stopped {handle.profile}: {result}")
    except Exception as exc:  # teardown must never raise
        result = {"stopped": False, "error": repr(exc)}
        log(f"stop_server error (ignored): {exc!r}")
    return result


# ----------------------------------------------------------------------------------------------
# Suite (one-command EXP-001)
# ----------------------------------------------------------------------------------------------
def _bench_module():
    sys.path.insert(0, str(BENCH_SCRIPT.parent))
    import serving_bench  # type: ignore
    return serving_bench


def run_profile(prof: dict[str, Any], args: argparse.Namespace, out_root: Path, remaining_s: float) -> dict[str, Any]:
    sb = _bench_module()
    name = prof["name"]
    pdir = out_root / name
    pdir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"profile": name, "status": "started", "t_start": time.time()}
    handle = None
    try:
        res["gpu_before"] = wait_gpu_free()
        handle = start_server(prof, work_dir=pdir, model_path=args.model_path, port=args.port)
        res["argv"] = handle.argv
        res["runtime"] = handle.runtime_info
        timeout = min(float(prof.get("startup", {}).get("timeout_s", 1500)), max(60.0, remaining_s - args.warmup - args.duration))
        ready = wait_ready(handle, timeout)
        res["ready"] = {k: v for k, v in ready.items() if k != "log_tail"}
        if ready["ready"]:
            res["harness_env"] = write_harness_env(prof, handle, pdir)
        if not ready["ready"]:
            res["status"] = "startup_failed"
            res["log_tail"] = ready.get("log_tail", "")[-4000:]
            return res
        client = prof.get("client", {})
        cfg = sb.BenchConfig(
            base_url=handle.base_url, pack=args.pack, label=name, out=str(pdir), agents=args.agents,
            duration_s=args.duration, warmup_s=args.warmup,
            context_window=int(client.get("context_window", 32768)),
            safety_margin=int(client.get("safety_margin", sb.DUCK_SAFETY_MARGIN)),
            reply_reserve=int(client.get("reply_reserve", sb.DUCK_REPLY_RESERVE)),
            max_tokens=client.get("max_tokens"), images_in_history=int(client.get("images_in_history", 2)),
            yield_seconds=float(client.get("yield_seconds", sb.DUCK_YIELD_SECONDS)), history_mode=args.history,
            seed=args.seed, profile=prof,
        )
        report = sb.run_bench(cfg)
        s = report["summary"]
        res["summary_path"] = str(pdir / "report.json")
        res["headline"] = {k: s.get(k) for k in ("decisions_per_game_hour", "requests_per_game_hour", "server_gen_tok_s",
                                                 "latency_p50_s", "tool_call_valid_pct", "error_pct")}
        alive = handle.proc is not None and handle.proc.poll() is None
        res["server_alive_after_bench"] = alive
        if not alive or not s.get("requests_completed") or (s.get("error_pct") or 0) > 50:
            res["status"] = "bench_failed"
            res["log_tail"] = tail(handle.log_path)[-4000:]
        else:
            res["status"] = "ok"
        try:
            res["gpu_loaded"] = gpu_memory()
            (pdir / "metrics_after.prom").write_text(http_text(handle.base_url[:-3] + "/metrics", timeout=15))
        except Exception:
            pass
        return res
    except Exception as exc:
        res["status"] = "error"
        res["error"] = repr(exc)[:2000]
        if handle is not None:
            res["log_tail"] = tail(handle.log_path)[-4000:]
        return res
    finally:
        res["teardown"] = stop_server(handle)
        res["t_end"] = time.time()
        res["elapsed_s"] = res["t_end"] - res["t_start"]
        (pdir / "profile_result.json").write_text(json.dumps(res, indent=1, default=str))
        log(f"profile {name}: {res['status']} in {res['elapsed_s']:.0f}s")


def run_suite(names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    sb = _bench_module()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    budget = args.budget_min * 60.0
    queue = list(names)
    results: list[dict[str, Any]] = []
    tried: set[str] = set()
    while queue:
        name = queue.pop(0)
        if name in tried:
            continue
        tried.add(name)
        prof = load_profile(name)
        remaining = budget - (time.monotonic() - t0)
        need = float(prof.get("startup", {}).get("expected_s", 600)) + args.warmup + args.duration + 90
        if remaining < need:
            log(f"skip {name}: needs ~{need:.0f}s, {remaining:.0f}s left in budget")
            results.append({"profile": name, "status": "skipped_budget", "needed_s": need, "remaining_s": remaining})
            continue
        res = run_profile(prof, args, out_root, remaining)
        results.append(res)
        if res["status"] != "ok" and prof.get("fallback") and prof["fallback"] not in tried:
            log(f"{name} {res['status']} -> fallback {prof['fallback']}")
            queue.insert(0, prof["fallback"])
        (out_root / "suite.json").write_text(json.dumps({"results": results, "elapsed_s": time.monotonic() - t0}, indent=1, default=str))
    reports = [out_root / r["profile"] / "report.json" for r in results if r.get("status") in ("ok", "bench_failed")]
    md, doc = sb.compare_reports([p for p in reports if p.exists()], args.baseline)
    status_lines = ["", "| profile | status | elapsed s | note |", "|---|---|---|---|"]
    for r in results:
        note = r.get("error") or (r.get("ready") or {}).get("reason") or ""
        status_lines.append(f"| {r['profile']} | {r['status']} | {r.get('elapsed_s', 0):.0f} | {str(note)[:120]} |")
    md = md + "\n".join(status_lines) + "\n"
    (out_root / "exp001_summary.md").write_text(md)
    (out_root / "exp001_summary.json").write_text(json.dumps({"compare": doc, "results": results}, indent=1, default=str))
    (out_root / "suite.json").write_text(json.dumps({"results": results, "elapsed_s": time.monotonic() - t0}, indent=1, default=str))
    print(md, flush=True)
    return {"results": results, "summary_md": md}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch a vLLM serving profile / run the EXP-001 suite")
    ap.add_argument("--profile", help="profile name (serving/profiles/<name>.json) or path")
    ap.add_argument("--list", action="store_true", help="list profiles")
    ap.add_argument("--print-argv", action="store_true", help="render argv + client settings and exit (no side effects)")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--work-dir", default="/kaggle/working/serving" if Path("/kaggle/working").is_dir() else "serving_out")
    ap.add_argument("--timeout", type=float, default=None, help="readiness timeout (default from profile)")
    ap.add_argument("--stop-after", action="store_true", help="stop the server once ready (startup smoke test)")
    ap.add_argument("--suite", default=None, help="comma list of profiles to benchmark sequentially; 'exp001' = "
                    "flashnext_tuned,qwen27b_fp8,flashnext_baseline + extras while the budget lasts")
    ap.add_argument("--out", default="/kaggle/working/exp001" if Path("/kaggle/working").is_dir() else "exp001_out")
    ap.add_argument("--pack", default=str(REPO_ROOT / "automation" / "serving_pack" / "duck_0922.jsonl.gz"))
    ap.add_argument("--duration", type=float, default=720.0)
    ap.add_argument("--warmup", type=float, default=90.0)
    ap.add_argument("--agents", type=int, default=28)
    ap.add_argument("--history", choices=["actual", "recorded"], default="actual")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", default="flashnext_baseline")
    ap.add_argument("--budget-min", type=float, default=100.0, help="wall-clock budget for the whole suite")
    a = ap.parse_args(argv)

    if a.list:
        for n in list_profiles():
            p = load_profile(n)
            print(f"{n:24s} {p['runtime']['kind']:20s} {p.get('description', '')[:100]}")
        return 0
    if a.suite:
        names = []
        for n in (x.strip() for x in a.suite.split(",")):
            if n == "exp001":
                names += EXP001_PLAN + EXP001_EXTRAS
            elif n:
                names.append(n)
        for n in names:
            load_profile(n)  # validate early
        res = run_suite(names, a)
        return 0 if any(r.get("status") == "ok" for r in res["results"]) else 1
    if not a.profile:
        ap.error("--profile, --suite or --list required")
    prof = load_profile(a.profile)
    if a.print_argv:
        print(json.dumps({"profile": prof["name"], "runtime": prof["runtime"]["kind"],
                          "argv": render_argv(prof, model_path=a.model_path, port=a.port),
                          "base_url": base_url(prof, a.port), "client": prof.get("client"), "env": prof.get("env", {}),
                          "memory": prof.get("memory"), "fallback": prof.get("fallback")}, indent=1))
        return 0
    handle = None
    try:
        handle = start_server(prof, work_dir=Path(a.work_dir) / prof["name"], model_path=a.model_path, port=a.port)
        ready = wait_ready(handle, a.timeout or float(prof.get("startup", {}).get("timeout_s", 1500)))
        if not ready["ready"]:
            log(f"NOT READY: {ready['reason']}\n{ready.get('log_tail', '')[-3000:]}")
            stop_server(handle)
            return 1
        harness = write_harness_env(prof, handle, Path(a.work_dir) / prof["name"])
        print(json.dumps({"base_url": handle.base_url, "pid": handle.proc.pid if handle.proc else None, "harness_env": harness,
                          "ready_s": ready.get("ready_s"), "log": str(handle.log_path)}), flush=True)
        if a.stop_after:
            stop_server(handle)
        return 0
    except Exception as exc:
        log(f"launch failed: {exc!r}")
        stop_server(handle)
        return 1


if __name__ == "__main__":
    sys.exit(main())
