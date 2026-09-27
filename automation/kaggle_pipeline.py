#!/usr/bin/env python3
"""Kaggle automation for arc-prize-2026-arc-agi-3.

All credentials come from the environment (KAGGLE_API_TOKEN). Nothing secret is written to disk here.

Commands
  quota                                   remaining weekly GPU hours (JSON)
  sync                                    snapshot leaderboard + our submissions -> automation/state/*.json,
                                          and regenerate plans/SUBMISSION_LOG.md
  push   --dir NB_DIR [--timeout S]       push a notebook version on the RTX PRO 6000 (commit run)
  wait   --kernel OWNER/SLUG [--max-min M]  poll until the commit run finishes (complete/error/cancel)
  submit --kernel OWNER/SLUG --version N -m MSG
                                          submit a committed notebook version (1 per day on this comp)
  daily  --dir NB_DIR -m MSG [--min-gpu-h H] [--dry-run]
                                          guarded end-to-end: check today's submission not used,
                                          check quota >= H, push, wait, submit, sync.

Safety rails
  * refuses to submit if a submission was already made today (UTC) — the comp allows 1/day
  * refuses to push if remaining GPU quota < --min-gpu-h
  * every action appended to automation/state/actions.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

COMP = "arc-prize-2026-arc-agi-3"
REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "automation" / "state"
STATE.mkdir(parents=True, exist_ok=True)
ACCELERATOR = "NvidiaRtxPro6000"


def _log(event: str, **kw) -> None:
    rec = {"t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "event": event, **kw}
    with open(STATE / "actions.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def _kaggle(*args: str, check: bool = True) -> str:
    if not os.environ.get("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle" / "access_token").exists():
        sys.exit("KAGGLE_API_TOKEN is not set (add it as an environment variable / secret).")
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def _csv(text: str) -> list[dict]:
    import csv
    import io
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("Next Page Token")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def quota() -> dict:
    out = _kaggle("quota")
    res = {}
    for line in out.splitlines():
        m = re.match(r"(GPU|TPU)\s+([\d.]+)h\s+([\d.]+)h\s+([\d.]+)h\s+(\S+)", line.strip())
        if m:
            res[m.group(1)] = {"used_h": float(m.group(2)), "remaining_h": float(m.group(3)),
                               "total_h": float(m.group(4)), "refresh_at": m.group(5)}
    return res


def submissions() -> list[dict]:
    return _csv(_kaggle("competitions", "submissions", COMP, "-v"))


def leaderboard(top: int = 50) -> list[dict]:
    rows = _csv(_kaggle("competitions", "leaderboard", COMP, "-s", "-v"))
    return rows[:top]


def sync() -> None:
    subs, lb, q = submissions(), leaderboard(100), quota()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    (STATE / "submissions.json").write_text(json.dumps(subs, indent=1))
    (STATE / f"leaderboard_{stamp}.json").write_text(json.dumps(lb, indent=1))
    (STATE / "quota.json").write_text(json.dumps(q, indent=1))
    lines = [f"# Submission log (auto-generated {stamp} UTC by automation/kaggle_pipeline.py sync)", "",
             f"GPU quota: {json.dumps(q.get('GPU', {}))}", "",
             "| date | description | status | public |", "|---|---|---|---|"]
    for s in subs:
        lines.append(f"| {s.get('date', '')[:16]} | {s.get('description', '')[:80].replace('|', '/')} | "
                     f"{s.get('status', '').replace('SubmissionStatus.', '')} | {s.get('publicScore', '')} |")
    lines += ["", "## Leaderboard top 15", "", "| # | team | score | last sub |", "|---|---|---|---|"]
    for i, r in enumerate(lb[:15], 1):
        lines.append(f"| {i} | {r.get('teamName')} | {r.get('score')} | {r.get('submissionDate', '')[:10]} |")
    (REPO / "plans" / "SUBMISSION_LOG.md").write_text("\n".join(lines) + "\n")
    _log("sync", n_subs=len(subs), best=max((float(s["publicScore"]) for s in subs if s.get("publicScore")),
                                              default=None), gpu=q.get("GPU"))


def submitted_today() -> bool:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    return any(s.get("date", "").startswith(today) for s in submissions())


def push(nb_dir: str, timeout: int | None = None) -> int:
    args = ["kernels", "push", "-p", nb_dir, "--accelerator", ACCELERATOR]
    if timeout:
        args += ["-t", str(timeout)]
    out = _kaggle(*args)
    m = re.search(r"version\s*(?:number)?\s*(\d+)", out, re.I)
    version = int(m.group(1)) if m else -1
    _log("push", dir=nb_dir, version=version, out=out.strip()[-300:])
    return version


def wait(kernel: str, max_min: float = 600) -> str:
    t0 = time.time()
    while True:
        out = _kaggle("kernels", "status", kernel)
        status = out.strip().split('"')[-2] if '"' in out else out.strip()
        if any(k in status for k in ("COMPLETE", "ERROR", "CANCEL")):
            _log("wait_done", kernel=kernel, status=status, minutes=round((time.time() - t0) / 60, 1))
            return status
        if time.time() - t0 > max_min * 60:
            _log("wait_timeout", kernel=kernel, status=status)
            return status
        time.sleep(60)


def submit(kernel: str, version: int, message: str) -> None:
    if submitted_today():
        _log("submit_refused", reason="already submitted today (UTC)", kernel=kernel, version=version)
        return
    out = _kaggle("competitions", "submit", COMP, "-k", kernel, "-v", str(version), "-f", "submission.parquet",
                  "-m", message)
    _log("submit", kernel=kernel, version=version, message=message, out=out.strip()[-300:])


def daily(nb_dir: str, message: str, min_gpu_h: float, dry_run: bool) -> None:
    meta = json.loads((Path(nb_dir) / "kernel-metadata.json").read_text())
    kernel = meta["id"]
    if submitted_today():
        _log("daily_skip", reason="already submitted today"); return
    q = quota().get("GPU", {})
    if q.get("remaining_h", 0) < min_gpu_h:
        _log("daily_skip", reason=f"GPU quota {q.get('remaining_h')}h < {min_gpu_h}h", quota=q); return
    if dry_run:
        _log("daily_dry_run", kernel=kernel, quota=q, message=message); return
    version = push(nb_dir)
    status = wait(kernel)
    if "COMPLETE" not in status:
        _log("daily_abort", reason=f"commit status {status}", kernel=kernel, version=version); return
    submit(kernel, version, message)
    sync()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("quota")
    sub.add_parser("sync")
    p = sub.add_parser("push"); p.add_argument("--dir", required=True); p.add_argument("--timeout", type=int)
    p = sub.add_parser("wait"); p.add_argument("--kernel", required=True); p.add_argument("--max-min", type=float, default=600)
    p = sub.add_parser("submit"); p.add_argument("--kernel", required=True); p.add_argument("--version", type=int, required=True)
    p.add_argument("-m", "--message", required=True)
    p = sub.add_parser("daily"); p.add_argument("--dir", required=True); p.add_argument("-m", "--message", required=True)
    p.add_argument("--min-gpu-h", type=float, default=0.5); p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd == "quota":
        print(json.dumps(quota(), indent=1))
    elif a.cmd == "sync":
        sync()
    elif a.cmd == "push":
        push(a.dir, a.timeout)
    elif a.cmd == "wait":
        wait(a.kernel, a.max_min)
    elif a.cmd == "submit":
        submit(a.kernel, a.version, a.message)
    elif a.cmd == "daily":
        daily(a.dir, a.message, a.min_gpu_h, a.dry_run)


if __name__ == "__main__":
    main()
