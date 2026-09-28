"""Hydra-1a/1b notebooks (automation/hydra1/) + the private Duck bundle they attach.

    for v in 1a 1b; do python automation/hydra1/build_hydra1.py --variant $v; done
    $SCRATCH/venv/bin/python -m pytest tests/test_hydra1_notebook.py -q

The CPU rehearsal runs the notebook's real AGENTFIX cell + Hydra cells on the vendored Duck with the mock LLM in a
subprocess (AGENTFIX monkeypatches the Duck modules process-wide; isolating it keeps the other test files clean).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "agent"))
VARIANTS = ("1a", "1b")
BUNDLE = "kragglenote2forwork/arc3-duck-bundle"


def _nb(v: str) -> list[str]:
    path = REPO / "notebooks" / f"hydra{v}" / f"arc3-hydra{v}.ipynb"
    if not path.exists():
        pytest.skip(f"build first: python automation/hydra1/build_hydra1.py --variant {v}")
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]]


def _cell3_switches(cells: list[str]) -> dict[str, str]:
    c3 = next(c for c in cells if "PUBLIC25_VLLM_PROFILE_ENV" in c)
    return dict(re.findall(r'^os\.environ\["(DUCK_PATCH[A-Z0-9_]*|HYDRA_SCORING)"\] = "([^"]*)"', c3, re.M))


@pytest.mark.parametrize("v", VARIANTS)
def test_metadata_attaches_our_bundle(v):
    _nb(v)
    meta = json.loads((REPO / "notebooks" / f"hydra{v}" / "kernel-metadata.json").read_text())
    assert meta["id"] == f"kragglenote2forwork/arc3-hydra{v}"
    assert meta["title"].lower().replace(" ", "-") == meta["id"].split("/")[1]          # MISTAKES #1
    assert BUNDLE in meta["dataset_sources"]
    assert "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1" not in meta["dataset_sources"]  # one bundle marker only
    assert "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1" in meta["dataset_sources"]
    assert meta["machine_shape"] == "NvidiaRtxPro6000" and meta["enable_internet"] is False


@pytest.mark.parametrize("v", VARIANTS)
def test_switches_match_the_rehearsed_profile(v):
    import duck_offline
    sw = _cell3_switches(_nb(v))
    assert sw.pop("HYDRA_SCORING") == "0"
    assert sw == duck_offline.PROFILES[f"hydra{v}"]
    c7 = next(c for c in _nb(v) if "DATASET_SOURCES = [" in c)
    assert f'"{BUNDLE}"' in c7 and "DUCK_BUNDLE_VERSION.json" in c7


def test_stage_bundle(tmp_path):
    spec = __import__("importlib.util").util.spec_from_file_location("stage_bundle", REPO / "automation/hydra1/stage_bundle.py")
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = mod.stage(tmp_path / "bundle")
    out = Path(info["out"])
    assert (out / "taaf-kaggle-bundle.json").is_file() and (out / "DUCK_BUNDLE_VERSION.json").is_file()
    assert (out / "dataset-metadata.json").is_file()
    assert not list(out.rglob("__pycache__")) and not list(out.rglob("*.gz")) and not list(out.rglob("*.pyc"))
    assert hashlib.sha256((out / "serving_setup.py").read_bytes()).hexdigest() == mod.SERVING_SETUP_SHA256
    assert (out / "src" / "ARC3-Inference" / "inference" / "agent" / "facts_memory.py").is_file()  # M2 shipped
    assert json.loads((out / "dataset-metadata.json").read_text())["id"] == BUNDLE


REHEARSAL = textwrap.dedent('''
    import json, os, re, sys
    from pathlib import Path
    REPO = Path(sys.argv[1]); v = sys.argv[2]; out = sys.argv[3]
    sys.path.insert(0, str(REPO / "agent"))
    cells = ["".join(c["source"]) for c in json.loads((REPO / f"notebooks/hydra{v}/arc3-hydra{v}.ipynb").read_text())["cells"]]
    c3 = next(c for c in cells if "PUBLIC25_VLLM_PROFILE_ENV" in c)
    for k, val in re.findall(r'^os\\.environ\\["(DUCK_PATCH[A-Z0-9_]*|HYDRA_SCORING)"\\] = "([^"]*)"', c3, re.M):
        os.environ[k] = val                       # exactly the notebook's switches
    import duck_offline
    os.environ["MULTIMODAL_UPSCALE"] = "4"        # the bundle-persisted value ARM P raises to 12
    ns = {"__name__": "__main__"}
    exec(compile(next(c for c in cells if "AGENTFIX -- six measured" in c), "agentfix", "exec"), ns)
    exec(compile(next(c for c in cells if "HYDRA0_PATCH" in c), "hydra0", "exec"), ns)
    res = duck_offline.run_offline(("ls20", "ft09", "ar25"), steps=10, out_dir=out, quiet=True, profile=f"hydra{v}")
    txt = "\\n".join(p.read_text(errors="ignore") for p in res.job_dir.rglob("*.log"))
    print(json.dumps({"hydra0_installed": bool(ns["_hy_report"].get("installed")), "mock_errors": res.mock_errors,
                      "states": {g: r["state"] for g, r in res.games.items()},
                      "undo_on_ar25": "ACTION7" in [r for g, r in res.games.items() if g.startswith("ar25")][0]["action_ids"],
                      "clock": "Game clock:" in txt, "hydra0_line": "Scoring: a level you finish" in txt,
                      "median_prompt": res.token_summary["prompt_tokens_per_request"]["median"]}))
''')


@pytest.mark.parametrize("v", VARIANTS)
def test_cpu_rehearsal_with_agentfix(v, tmp_path):
    pytest.importorskip("arcengine")
    _nb(v)
    proc = subprocess.run([sys.executable, "-c", REHEARSAL, str(REPO), v, str(tmp_path / "run")],
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["mock_errors"] == [] and set(r["states"].values()) <= {"gave_up", "won", "cancelled"}
    assert r["clock"] and r["undo_on_ar25"]
    assert not r["hydra0_installed"] and not r["hydra0_line"]      # v9 P1 supersedes the Hydra-0 scoring cell
    if v == "1b":
        assert r["median_prompt"] < 13_000, r["median_prompt"]      # M2 memory halves the prompt
    else:
        assert r["median_prompt"] > 15_000, r["median_prompt"]
