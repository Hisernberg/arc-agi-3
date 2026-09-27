#!/usr/bin/env python3
"""Preflight for a Kaggle notebook dir (kernel-metadata.json + .ipynb) before `kaggle_pipeline.py push` (B09).

    python automation/preflight.py --dir NB_DIR [--source-dir STAGED_DATASET_DIR] [--kaggle-check] [--json]

Offline checks (always):
  metadata      kernel-metadata.json schema: notebook / python / code_file present / private
  title_slug    title slugifies to the id's slug (MISTAKES #1: a mismatch makes Kaggle rename/fork the kernel);
                equals the existing kernel's title when build_info.json recorded one (or --kaggle-check pulled it)
  accelerator   machine_shape == NvidiaRtxPro6000, enable_gpu true, enable_tpu false
  internet      enable_internet false
  sources       competition attached; the Hydra source dataset + the serving profile's datasets/model attached
  secrets       no credential in any file of NB_DIR (and --source-dir): token patterns + the literal values of
                KAGGLE_API_TOKEN / HF_TOKEN / ARC_API_KEY / GITHUB_TOKEN / GH_TOKEN when set
  notebook_json valid nbformat-4 JSON, code cells parse, no outputs, no shell magics
  smoke_wired   rerun detection via KAGGLE_IS_COMPETITION_RERUN, rerun branch -> run_rerun, commit branch ->
                run_smoke (3 games x 8 min), teardown in a finally, placeholder parquet only in the smoke path
  time_budget   first action within the 15-min inactivity kill, soft deadline + teardown reserve <= 9 h, smoke
                commit <= 0.5 GPU-h (MISTAKES #4), per-game budgets from scheduler.plan_budget; for the stock
                loop: all waves fit the window (the "last wave cut short" failure)
  source        --source-dir: HYDRA_SOURCE.json present, version == the notebook's pinned source_version
Online (--kaggle-check, uses KAGGLE_API_TOKEN if present, never prints it): every attached dataset / model exists,
the competition is reachable, and the existing kernel's title equals ours.

Exit code 0 when no check FAILs (WARN/SKIP are allowed). Checks that are Hydra-specific are SKIPped for notebooks
not built by build_notebook.py, so this also works on hand-made notebook dirs.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
COMPETITION = "arc-prize-2026-arc-agi-3"
MACHINE_SHAPE = "NvidiaRtxPro6000"
NOTEBOOK_LIMIT_S = 32_400.0
INACTIVITY_KILL_S = 900.0
SMOKE_GPU_H_MAX = 0.5

# ----------------------------------------------------------------------------------------------------------------
# Shared helpers (also used by build_notebook.py and package_source.py)
# ----------------------------------------------------------------------------------------------------------------


def kaggle_slug(title: str) -> str:
    """Kaggle derives a kernel slug from its title: lower case, runs of other characters -> '-'."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


SECRET_ENV_VARS = ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "HF_TOKEN", "ARC_API_KEY", "GITHUB_TOKEN", "GH_TOKEN")
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("kaggle_token", re.compile(r"\bKGAT_[0-9A-Za-z]{16,}")),
    ("kaggle_json_key", re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"')),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("api_key_sk", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{32,}")),
    ("credential_assignment", re.compile(
        r"""(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|password|passwd|bearer)\b["']?"""
        r"""\s*[:=]\s*["']([A-Za-z0-9_\-./+=]{20,})["']""")),
]
# Known non-secret placeholders that match the generic assignment pattern.
SECRET_ALLOWLIST = {"offline-kaggle-local-server", "test-key-123", "EMPTY", "sk-no-key-required"}
SCAN_MAX_BYTES = 64 * 1024 * 1024


@dataclass
class SecretHit:
    path: str
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} [{self.kind}]"


def _secret_values() -> list[tuple[str, str]]:
    out = []
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if len(value) >= 8:
            out.append((name, value))
    return out


def _read_text_for_scan(path: Path) -> str | None:
    try:
        if path.stat().st_size > SCAN_MAX_BYTES:
            return None
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)[:SCAN_MAX_BYTES]
    except (OSError, EOFError, gzip.BadGzipFile):
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def scan_secrets(paths: list[Path], root: Path | None = None) -> list[SecretHit]:
    """Token patterns + literal values of the secret env vars. Hits carry file:line and kind only (no value)."""
    values = _secret_values()
    hits: list[SecretHit] = []
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.is_file() and "__pycache__" not in f.parts]
        elif p.is_file():
            files.append(p)
    for f in files:
        text = _read_text_for_scan(f)
        if text is None:
            continue
        rel = str(f.relative_to(root)) if root and f.is_relative_to(root) else str(f)
        for name, value in values:
            if value in text:
                hits.append(SecretHit(rel, text[: text.index(value)].count("\n") + 1, f"env:{name}"))
        for kind, rx in SECRET_PATTERNS:
            for m in rx.finditer(text):
                token = m.group(1) if m.groups() else m.group(0)
                if token in SECRET_ALLOWLIST:
                    continue
                hits.append(SecretHit(rel, text[: m.start()].count("\n") + 1, kind))
    return hits


def notebook_code_cells(nb: dict[str, Any]) -> list[str]:
    return ["".join(c["source"]) if isinstance(c.get("source"), list) else str(c.get("source", ""))
            for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def extract_hydra_config(nb: dict[str, Any]) -> dict[str, Any] | None:
    """The `HYDRA_CONFIG = {...}` literal build_notebook.py embeds in the first code cell."""
    for src in notebook_code_cells(nb):
        if "HYDRA_CONFIG" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HYDRA_CONFIG" for t in node.targets):
                return ast.literal_eval(node.value)
    return None


def load_scheduler_module() -> Any:
    sys.path.insert(0, str(REPO / "agent"))
    import scheduler  # agent/scheduler.py

    return scheduler


def time_budget(config: dict[str, Any], turn_s: float | None = None) -> dict[str, Any]:
    """Rerun + smoke time math for a Hydra notebook config."""
    t = config["time"]
    setup = float(t["setup_estimate_s"])
    soft = float(t["soft_deadline_s"])
    limit = float(t.get("notebook_limit_s", NOTEBOOK_LIMIT_S))
    n_games = int(t.get("expected_games", 110))
    solver = config.get("solver") or {}
    slots = int(solver.get("concurrency", 28))
    first_turn = float(t.get("first_turn_estimate_s", 90.0))
    window = soft - setup
    out: dict[str, Any] = {
        "setup_estimate_s": setup, "first_action_s": setup + first_turn, "soft_deadline_s": soft,
        "teardown_reserve_s": limit - soft, "play_window_s": window, "n_games": n_games, "slots": slots,
    }
    scheduler_on = str((config.get("duck_patches") or {}).get("DUCK_PATCH_B08_SCHEDULER", "0")).lower() in {"1", "true"}
    out["scheduler"] = scheduler_on
    if scheduler_on:
        S = load_scheduler_module()
        cfg = S.SchedulerConfig.from_mapping({"slots": slots, **(config.get("scheduler") or {})})
        plan = S.plan_budget(n_games, window - cfg.stop_margin_s, cfg, turn_s)
        out["plan"] = plan.to_dict()
        out["floors_fit"] = plan.floor_s * n_games <= plan.capacity_s
    else:
        per_game = float(solver.get("max_runtime_s_per_game") or 0.0)
        waves = -(-n_games // max(1, slots))
        out["stock_waves"] = waves
        out["stock_need_s"] = waves * per_game
        out["stock_last_wave_s"] = max(0.0, window - (waves - 1) * per_game)
        out["stock_fits"] = waves * per_game <= window
    smoke = config.get("smoke") or {}
    smoke_s = setup + float(smoke.get("minutes", 8)) * 60.0 + float(t.get("teardown_estimate_s", 120.0))
    out["smoke_gpu_h"] = round(smoke_s / 3600.0, 3)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: str   # PASS / FAIL / WARN / SKIP
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class Preflight:
    def __init__(self, nb_dir: Path, source_dir: Path | None = None):
        self.nb_dir = Path(nb_dir)
        self.source_dir = Path(source_dir) if source_dir else None
        self.checks: list[Check] = []
        self.meta: dict[str, Any] = {}
        self.nb: dict[str, Any] = {}
        self.config: dict[str, Any] | None = None
        self.build_info: dict[str, Any] = {}

    def add(self, name: str, status: str, detail: str = "", **data: Any) -> None:
        self.checks.append(Check(name, status, detail, data))

    @property
    def ok(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)

    # -- loading ----------------------------------------------------------------------------------------------
    def load(self) -> bool:
        meta_path = self.nb_dir / "kernel-metadata.json"
        try:
            self.meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.add("metadata", "FAIL", f"cannot read {meta_path}: {exc}")
            return False
        info = self.nb_dir / "build_info.json"
        if info.is_file():
            try:
                self.build_info = json.loads(info.read_text())
            except json.JSONDecodeError:
                self.add("build_info", "WARN", "build_info.json is not valid JSON")
        code = self.nb_dir / str(self.meta.get("code_file", ""))
        try:
            self.nb = json.loads(code.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.add("notebook_json", "FAIL", f"cannot parse {code.name}: {exc}")
            return False
        try:
            self.config = extract_hydra_config(self.nb)
        except (ValueError, SyntaxError) as exc:
            self.add("notebook_json", "FAIL", f"HYDRA_CONFIG literal does not evaluate: {exc}")
        return True

    # -- offline checks ---------------------------------------------------------------------------------------
    def check_metadata(self) -> None:
        m = self.meta
        missing = [k for k in ("id", "title", "code_file", "language", "kernel_type") if not m.get(k)]
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if m.get("kernel_type") != "notebook":
            problems.append(f"kernel_type={m.get('kernel_type')!r}")
        if m.get("language") != "python":
            problems.append(f"language={m.get('language')!r}")
        if not (self.nb_dir / str(m.get("code_file", ""))).is_file():
            problems.append(f"code_file {m.get('code_file')!r} not in the dir")
        if str(m.get("id", "")).count("/") != 1:
            problems.append(f"id {m.get('id')!r} is not owner/slug")
        self.add("metadata", "FAIL" if problems else "PASS", "; ".join(problems) or f"{m.get('id')} ({m.get('code_file')})")
        if m.get("is_private") is not True and str(m.get("is_private")).lower() != "true":
            self.add("private", "WARN", "is_private is not true: the kernel would be public")

    def check_title(self, existing_title: str | None = None) -> None:
        title, kid = str(self.meta.get("title", "")), str(self.meta.get("id", ""))
        slug = kid.split("/", 1)[1] if "/" in kid else kid
        if kaggle_slug(title) != slug:
            self.add("title_slug", "FAIL", f"title {title!r} slugifies to {kaggle_slug(title)!r}, id slug is {slug!r} "
                     "(MISTAKES #1: Kaggle would rename/fork the kernel)")
            return
        if not 5 <= len(title) <= 50:
            self.add("title_slug", "FAIL", f"title length {len(title)} outside Kaggle's 5-50")
            return
        existing = existing_title if existing_title is not None else self.build_info.get("existing_title")
        if existing and existing != title:
            self.add("title_slug", "FAIL", f"title {title!r} != existing kernel title {existing!r} (MISTAKES #1)")
            return
        self.add("title_slug", "PASS", f"{title!r} -> {slug}" + (f" (== existing {existing!r})" if existing else ""))

    def check_accelerator(self) -> None:
        m = self.meta
        problems = []
        if m.get("machine_shape") != MACHINE_SHAPE:
            problems.append(f"machine_shape={m.get('machine_shape')!r} (want {MACHINE_SHAPE})")
        if str(m.get("enable_gpu")).lower() != "true":
            problems.append("enable_gpu is not true")
        if str(m.get("enable_tpu", "false")).lower() == "true":
            problems.append("enable_tpu is true")
        self.add("accelerator", "FAIL" if problems else "PASS", "; ".join(problems) or MACHINE_SHAPE)

    def check_internet(self) -> None:
        on = str(self.meta.get("enable_internet", "true")).lower() != "false"
        self.add("internet", "FAIL" if on else "PASS", "enable_internet must be false (competition rule)" if on else "off")

    def required_sources(self) -> dict[str, list[str]]:
        cfg = self.config or {}
        serving = cfg.get("serving") or {}
        datasets = [cfg["source_dataset"]] if cfg.get("source_dataset") else []
        datasets += list(serving.get("datasets") or [])
        models = list(serving.get("models") or [])
        return {"competition": [COMPETITION], "datasets": datasets, "models": models}

    def check_sources(self) -> None:
        m = self.meta
        req = self.required_sources()
        attached = {
            "competition": list(m.get("competition_sources") or []),
            "datasets": list(m.get("dataset_sources") or []),
            "models": list(m.get("model_sources") or []),
        }
        problems = []
        for kind, refs in req.items():
            have = {r.lower() for r in attached[kind]}
            for ref in refs:
                if ref.lower() not in have:
                    problems.append(f"{kind[:-1] if kind != 'competition' else kind} {ref} not attached")
        for kind, refs in attached.items():
            if len({r.lower() for r in refs}) != len(refs):
                problems.append(f"duplicate {kind} sources")
        for ref in attached["datasets"]:
            if ref.count("/") not in (1, 2):
                problems.append(f"dataset source {ref!r} is not owner/slug[/version]")
        for ref in attached["models"]:
            if ref.count("/") != 4:
                problems.append(f"model source {ref!r} is not owner/model/framework/variation/version")
        status = "FAIL" if problems else "PASS"
        detail = "; ".join(problems) or (
            f"{len(attached['datasets'])} datasets, {len(attached['models'])} models, competition"
            + ("" if self.config else " (not a Hydra notebook: only the competition was required)"))
        self.add("sources", status, detail, attached=attached, required=req)

    def check_secrets(self) -> None:
        roots = [self.nb_dir] + ([self.source_dir] if self.source_dir else [])
        hits = []
        for root in roots:
            hits += scan_secrets([root], root=root)
        self.add("secrets", "FAIL" if hits else "PASS",
                 "; ".join(str(h) for h in hits[:10]) or f"clean ({', '.join(str(r) for r in roots)})")

    def check_notebook(self) -> None:
        nb = self.nb
        problems = []
        if nb.get("nbformat") != 4:
            problems.append(f"nbformat={nb.get('nbformat')}")
        if not isinstance(nb.get("cells"), list) or not nb["cells"]:
            problems.append("no cells")
        ks = (nb.get("metadata") or {}).get("kernelspec") or {}
        if ks.get("language", "python") != "python" or ks.get("name", "python3") != "python3":
            problems.append(f"kernelspec {ks}")
        for i, cell in enumerate(nb.get("cells") or []):
            if cell.get("cell_type") not in ("code", "markdown", "raw"):
                problems.append(f"cell {i}: bad cell_type")
                continue
            if "source" not in cell or "metadata" not in cell:
                problems.append(f"cell {i}: missing source/metadata")
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs"):
                problems.append(f"cell {i}: has outputs")
            src = "".join(cell["source"]) if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
            if re.search(r"^\s*[!%]", src, re.M):
                problems.append(f"cell {i}: shell/magic line")
            try:
                compile(src, f"<cell {i}>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)
            except SyntaxError as exc:
                problems.append(f"cell {i}: {exc.msg} (line {exc.lineno})")
        self.add("notebook_json", "FAIL" if problems else "PASS", "; ".join(problems) or f"{len(nb.get('cells', []))} cells")

    def check_smoke(self) -> None:
        if self.config is None:
            code = "\n".join(notebook_code_cells(self.nb))
            has = "KAGGLE_IS_COMPETITION_RERUN" in code and "submission.parquet" in code
            self.add("smoke_wired", "WARN" if not has else "SKIP",
                     "not a Hydra notebook; " + ("rerun detection + placeholder parquet present" if has
                                                 else "no KAGGLE_IS_COMPETITION_RERUN / submission.parquet handling"))
            return
        code = "\n".join(notebook_code_cells(self.nb))
        smoke = self.config.get("smoke") or {}
        problems = []
        if len(smoke.get("games") or []) != 3:
            problems.append(f"smoke games {smoke.get('games')} (want 3)")
        if float(smoke.get("minutes", 0)) != 8.0:
            problems.append(f"smoke minutes {smoke.get('minutes')} (want 8)")
        if "KAGGLE_IS_COMPETITION_RERUN" not in code:
            problems.append("no rerun detection")
        run_cell = next((c for c in notebook_code_cells(self.nb) if "run_smoke" in c and "run_rerun" in c), None)
        if run_cell is None:
            problems.append("no cell calling both run_rerun and run_smoke")
        else:
            tree = ast.parse(run_cell)
            ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If) and "TRUE_SUBMISSION" in ast.unparse(n.test)]
            if not ifs:
                problems.append("run branch is not conditioned on TRUE_SUBMISSION")
            else:
                node = ifs[0]
                body, orelse = ast.unparse(ast.Module(node.body, [])), ast.unparse(ast.Module(node.orelse, []))
                if "run_rerun" not in body or "run_smoke" not in orelse:
                    problems.append("TRUE_SUBMISSION branch must call run_rerun, else run_smoke")
            if not any(isinstance(n, ast.Try) and n.finalbody and "teardown" in ast.unparse(ast.Module(n.finalbody, []))
                       for n in ast.walk(tree)):
                problems.append("teardown is not in a finally block")
        if "submission.parquet" in code and "run_smoke" not in code:
            problems.append("placeholder parquet written outside the smoke path")
        self.add("smoke_wired", "FAIL" if problems else "PASS", "; ".join(problems) or
                 f"rerun -> run_rerun, commit -> run_smoke {smoke.get('games')} x {smoke.get('minutes')} min")

    def check_time(self) -> None:
        if self.config is None:
            self.add("time_budget", "SKIP", "not a Hydra notebook")
            return
        tb = time_budget(self.config, (self.config.get("time") or {}).get("turn_estimate_s"))
        problems, warns = [], []
        if tb["first_action_s"] > INACTIVITY_KILL_S - 60:
            problems.append(f"first action at ~{tb['first_action_s']:.0f}s risks the 15-min inactivity kill")
        elif tb["first_action_s"] > 660:
            warns.append(f"first action at ~{tb['first_action_s']:.0f}s (> 11 min)")
        if tb["teardown_reserve_s"] < 900:
            problems.append(f"soft deadline leaves {tb['teardown_reserve_s']:.0f}s < 15 min before the 9 h limit")
        if tb["smoke_gpu_h"] > SMOKE_GPU_H_MAX:
            problems.append(f"smoke commit ~{tb['smoke_gpu_h']} GPU-h > {SMOKE_GPU_H_MAX} (MISTAKES #4)")
        if tb["scheduler"]:
            if not tb["floors_fit"]:
                problems.append("per-game floors do not fit the capacity")
        elif not tb["stock_fits"]:
            problems.append(f"stock loop: {tb['stock_waves']} waves x {tb['stock_need_s'] / tb['stock_waves']:.0f}s = "
                            f"{tb['stock_need_s']:.0f}s > window {tb['play_window_s']:.0f}s; the last wave gets only "
                            f"{tb['stock_last_wave_s']:.0f}s (enable DUCK_PATCH_B08_SCHEDULER or wave-fit the cap)")
        status = "FAIL" if problems else ("WARN" if warns else "PASS")
        summary = (f"setup~{tb['setup_estimate_s']:.0f}s window={tb['play_window_s']:.0f}s smoke~{tb['smoke_gpu_h']} GPU-h"
                   + (f" fair={tb['plan']['fair_share_s']}s floor={tb['plan']['floor_s']}s cap={tb['plan']['cap_s']}s "
                      f"max_active={tb['plan']['max_active']}" if tb["scheduler"] else
                      f" stock {tb['stock_waves']} waves, last wave {tb['stock_last_wave_s']:.0f}s"))
        detail = "; ".join(problems + warns + [summary])
        self.add("time_budget", status, detail, **tb)

    def check_source_dir(self) -> None:
        if self.source_dir is None:
            self.add("source", "SKIP", "no --source-dir given")
            return
        marker = self.source_dir / "HYDRA_SOURCE.json"
        if not marker.is_file():
            self.add("source", "FAIL", f"{marker} missing (stage with automation/package_source.py)")
            return
        manifest = json.loads(marker.read_text())
        problems = []
        want = (self.config or {}).get("source_version")
        if want and manifest.get("version") != want:
            problems.append(f"staged version {manifest.get('version')} != pinned {want}")
        if self.config and manifest.get("dataset") != self.config.get("source_dataset"):
            problems.append(f"staged dataset {manifest.get('dataset')} != {self.config.get('source_dataset')}")
        for rel in ("agent/hydra_run.py", "agent/scheduler.py", "agent/duck_offline.py", "serving/launch.py"):
            if not (self.source_dir / rel).is_file():
                problems.append(f"{rel} missing from the staged dataset")
        profile = ((self.config or {}).get("serving") or {}).get("profile")
        if profile and not (self.source_dir / "serving" / "profiles" / f"{profile}.json").is_file():
            problems.append(f"serving profile {profile} missing from the staged dataset")
        self.add("source", "FAIL" if problems else "PASS", "; ".join(problems) or f"version {manifest.get('version')}")

    def check_profile(self) -> None:
        if self.config is None:
            return
        profile = (self.config.get("serving") or {}).get("profile")
        path = REPO / "serving" / "profiles" / f"{profile}.json"
        if not path.is_file():
            self.add("serving_profile", "FAIL", f"{path} missing")
            return
        self.add("serving_profile", "PASS", f"{profile} ({(self.config.get('serving') or {}).get('runtime_kind')})")

    # -- online -----------------------------------------------------------------------------------------------
    def kaggle_check(self) -> None:
        token = os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle" / "access_token").exists() \
            or (Path.home() / ".kaggle" / "kaggle.json").exists()
        if not token:
            self.add("kaggle_online", "SKIP", "KAGGLE_API_TOKEN not set: attachments/title not verified online")
            return
        if not shutil.which("kaggle"):
            self.add("kaggle_online", "SKIP", "kaggle CLI not installed")
            return

        def run(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=120)

        problems, notes = [], []
        m = self.meta
        for ref in m.get("dataset_sources") or []:
            proc = run("datasets", "files", ref)
            if proc.returncode != 0:
                problems.append(f"dataset {ref}: {(proc.stderr or proc.stdout).strip()[-160:]}")
        for ref in m.get("model_sources") or []:
            proc = run("models", "instances", "versions", "files", ref)
            if proc.returncode != 0:
                problems.append(f"model {ref}: {(proc.stderr or proc.stdout).strip()[-160:]}")
        for ref in m.get("competition_sources") or []:
            proc = run("competitions", "files", ref)
            if proc.returncode != 0:
                problems.append(f"competition {ref}: {(proc.stderr or proc.stdout).strip()[-160:]}")
        with tempfile.TemporaryDirectory() as tmp:
            proc = run("kernels", "pull", str(m.get("id")), "-p", tmp, "-m")
            meta_path = Path(tmp) / "kernel-metadata.json"
            if proc.returncode == 0 and meta_path.is_file():
                existing = json.loads(meta_path.read_text()).get("title")
                notes.append(f"existing kernel title {existing!r}")
                self.check_title(existing_title=existing)
            else:
                notes.append("kernel does not exist yet (first push creates it)")
        self.add("kaggle_online", "FAIL" if problems else "PASS", "; ".join(problems + notes))

    # -- driver -----------------------------------------------------------------------------------------------
    def run(self, kaggle: bool = False) -> list[Check]:
        if not self.load():
            return self.checks
        self.check_metadata()
        self.check_title()
        self.check_accelerator()
        self.check_internet()
        self.check_sources()
        self.check_secrets()
        self.check_notebook()
        self.check_smoke()
        self.check_time()
        self.check_profile()
        self.check_source_dir()
        if kaggle:
            self.kaggle_check()
        return self.checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preflight a Kaggle notebook dir (B09)")
    ap.add_argument("--dir", required=True, help="notebook dir with kernel-metadata.json")
    ap.add_argument("--source-dir", default=None, help="staged source dataset dir (automation/package_source.py)")
    ap.add_argument("--kaggle-check", action="store_true", help="verify attachments + kernel title online")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    pf = Preflight(Path(a.dir), Path(a.source_dir) if a.source_dir else None)
    checks = pf.run(kaggle=a.kaggle_check)
    if a.json:
        print(json.dumps([c.__dict__ for c in checks], indent=1, default=str))
    else:
        for c in checks:
            print(f"{c.status:4s}  {c.name:15s} {c.detail}")
        print("PREFLIGHT " + ("GREEN" if pf.ok else "RED"))
    return 0 if pf.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
