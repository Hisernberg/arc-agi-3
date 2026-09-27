"""B09: notebook builder + preflight + source packaging, and a CPU rehearsal of the built notebook.

    $SCRATCH/venv/bin/python -m pytest tests/test_build_notebook.py -q

The rehearsal (`build_notebook.py exec-dry`) runs every code cell of the built notebook in a fresh interpreter
with HYDRA_DRY_RUN=1 against a fake /kaggle/input (the staged source dataset + the competition's offline
environment files): GPU check and pip are skipped, the `mock_cpu` serving profile is launched for real through
serving/launch.py, games are played by the real Duck loop against the scripted mock LLM.
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "automation"))
sys.path.insert(0, str(REPO / "agent"))

import build_notebook as B  # noqa: E402
import package_source as P  # noqa: E402
import preflight as PF  # noqa: E402

EXAMPLE = REPO / "automation" / "candidates" / "example.json"


def _candidate(tmp: Path, **changes) -> Path:
    cand = json.loads(EXAMPLE.read_text())
    cand.update(name="dummy", serving_profile="mock_cpu")
    cand["kernel"] = {"id": "kragglenote2forwork/arc3-hydra-dummy", "title": "ARC3 Hydra Dummy", "is_private": True}
    for key, value in changes.items():
        cand[key] = value
    path = tmp / f"cand_{len(list(tmp.glob('cand_*.json')))}.json"
    path.write_text(json.dumps(cand, indent=1))
    return path


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    out = tmp_path_factory.mktemp("stage") / "arc3-hydra-src"
    info = P.stage(out, note="pytest")
    return Path(info["dir"]), info


@pytest.fixture(scope="module")
def built(tmp_path_factory, staged):
    tmp = tmp_path_factory.mktemp("nb")
    src, _ = staged
    res = B.build(_candidate(tmp), tmp / "nb", source_dir=src)
    return Path(res["dir"]), res


def _checks(nb_dir: Path, source_dir: Path | None = None, kaggle: bool = False) -> dict[str, PF.Check]:
    pf = PF.Preflight(nb_dir, source_dir)
    return {c.name: c for c in pf.run(kaggle=kaggle)}


# ------------------------------------------------------------------------------------------------ packaging


def test_package_source_manifest(staged):
    src, info = staged
    manifest = json.loads((src / "HYDRA_SOURCE.json").read_text())
    meta = json.loads((src / "dataset-metadata.json").read_text())
    assert meta["id"] == "kragglenote2forwork/arc3-hydra-src" and meta["title"] == "arc3-hydra-src"
    assert manifest["version"] == info["version"] and manifest["version"].startswith("v")
    assert manifest["version"].endswith(manifest["tree_sha256"][:10])
    for rel in ("agent/hydra_run.py", "agent/scheduler.py", "agent/duck_offline.py", "serving/launch.py",
                "serving/profiles/mock_cpu.json", "automation/serving_bench.py",
                "agent/duck/src/ARC3-Inference/inference/framework/solver.py"):
        assert (src / rel).is_file(), rel
        assert manifest["files"][rel] == P.sha256_file(src / rel)
    assert not list(src.rglob("__pycache__")) and not list(src.rglob("*.pyc"))
    assert PF.scan_secrets([src], root=src) == []
    with pytest.raises(FileExistsError):  # never overwrites a staged version
        P.stage(src)


# ------------------------------------------------------------------------------------------------ build + preflight


def test_build_and_preflight_green(built, staged):
    nb_dir, res = built
    src, info = staged
    meta = json.loads((nb_dir / "kernel-metadata.json").read_text())
    assert meta["machine_shape"] == "NvidiaRtxPro6000" and meta["enable_internet"] is False
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
    assert meta["dataset_sources"][0] == "kragglenote2forwork/arc3-hydra-src"
    assert res["config"]["source_version"] == info["version"]
    checks = _checks(nb_dir, src)
    assert all(c.status != "FAIL" for c in checks.values()), {k: c.detail for k, c in checks.items() if c.status == "FAIL"}
    for name in ("metadata", "title_slug", "accelerator", "internet", "sources", "secrets", "notebook_json",
                 "smoke_wired", "time_budget", "source"):
        assert checks[name].status == "PASS", (name, checks[name].detail)
    assert checks["time_budget"].data["plan"]["n_games"] == 110
    assert PF.main(["--dir", str(nb_dir), "--source-dir", str(src)]) == 0


def test_example_candidate_attaches_the_profile_runtime(tmp_path):
    res = B.build(EXAMPLE, tmp_path / "nb")
    meta = res["metadata"]
    assert "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1" in meta["dataset_sources"]
    assert meta["model_sources"] == ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"]
    checks = _checks(Path(res["dir"]))
    assert checks["sources"].status == "PASS" and checks["title_slug"].status == "PASS"
    assert checks["time_budget"].status in ("PASS", "WARN")


def test_build_refuses_a_title_that_does_not_match_the_slug(tmp_path):
    bad = _candidate(tmp_path)
    cand = json.loads(bad.read_text())
    cand["kernel"]["title"] = "ARC3 Hydra Dummy v2"
    bad.write_text(json.dumps(cand))
    with pytest.raises(ValueError, match="slugifies"):
        B.build(bad, tmp_path / "nb")


def _mutated(built, tmp_path, *, meta=None, cells=None, info=None) -> Path:
    import shutil

    nb_dir, _ = built
    dst = tmp_path / "nb"
    shutil.copytree(nb_dir, dst)
    if meta:
        m = json.loads((dst / "kernel-metadata.json").read_text())
        meta(m)
        (dst / "kernel-metadata.json").write_text(json.dumps(m))
    if cells:
        path = dst / json.loads((dst / "kernel-metadata.json").read_text())["code_file"]
        nb = json.loads(path.read_text())
        cells(nb)
        path.write_text(json.dumps(nb))
    if info:
        i = json.loads((dst / "build_info.json").read_text())
        info(i)
        (dst / "build_info.json").write_text(json.dumps(i))
    return dst


def _set(key, value):
    return lambda m: m.__setitem__(key, value)


@pytest.mark.parametrize("mutation,check", [
    (dict(meta=_set("title", "ARC3 Hydra Dummy Final")), "title_slug"),            # MISTAKES #1
    (dict(info=_set("existing_title", "Arc3 Hydra Dummy")), "title_slug"),         # existing kernel's title differs
    (dict(meta=_set("enable_internet", True)), "internet"),
    (dict(meta=_set("machine_shape", "NvidiaTeslaT4")), "accelerator"),
    (dict(meta=_set("enable_gpu", False)), "accelerator"),
    (dict(meta=_set("dataset_sources", [])), "sources"),
    (dict(meta=_set("competition_sources", [])), "sources"),
    (dict(cells=lambda nb: nb["cells"][1].__setitem__("source", "x = (")), "notebook_json"),
    (dict(cells=lambda nb: nb["cells"][3].__setitem__("outputs", [{"output_type": "stream", "text": "x"}])),
     "notebook_json"),
    (dict(cells=lambda nb: [c.__setitem__("source", c["source"].replace("run_smoke(CTX", "run_rerun(CTX"))
                            for c in nb["cells"]]), "smoke_wired"),
    (dict(cells=lambda nb: [c.__setitem__("source", c["source"].replace("try:\n", "if True:\n").replace(
        "finally:\n    H.teardown", "if True:\n    H.teardown")) for c in nb["cells"]]), "smoke_wired"),
])
def test_preflight_catches(built, tmp_path, mutation, check):
    checks = _checks(_mutated(built, tmp_path, **mutation))
    assert checks[check].status == "FAIL", checks[check].detail


def test_preflight_catches_secrets(built, tmp_path, monkeypatch):
    fake_hf = "hf_" + "Ab3" * 12                     # built at run time: no token-shaped literal in this file
    nb = _mutated(built, tmp_path)
    (nb / "notes.txt").write_text(f"token = {fake_hf}\n")
    checks = _checks(nb)
    assert checks["secrets"].status == "FAIL" and "hf_token" in checks["secrets"].detail
    assert fake_hf not in checks["secrets"].detail  # never echo the value
    (nb / "notes.txt").write_text("nothing here\n")
    value = "kagglesecret-" + "9" * 20
    monkeypatch.setenv("KAGGLE_API_TOKEN", value)
    (nb / "cell_dump.py").write_text(f"x = '{value}'\n")
    checks = _checks(nb)
    assert checks["secrets"].status == "FAIL" and "env:KAGGLE_API_TOKEN" in checks["secrets"].detail
    assert value not in checks["secrets"].detail


def test_preflight_time_budget_flags_the_stock_last_wave(tmp_path):
    cand = _candidate(tmp_path, duck_patches={"DUCK_PATCHES": "1"},
                      solver={"concurrency": 28, "max_runtime_s_per_game": 7920.0})
    res = B.build(cand, tmp_path / "nb")
    tb = _checks(Path(res["dir"]))["time_budget"]
    assert tb.status == "FAIL" and "last wave" in tb.detail
    assert tb.data["stock_waves"] == 4 and tb.data["stock_last_wave_s"] < 7920
    # the wave-fit cap (window x slots / games, 4 waves) fits
    fit = _candidate(tmp_path, duck_patches={"DUCK_PATCHES": "1"},
                     solver={"concurrency": 28, "max_runtime_s_per_game": 7500.0})
    res = B.build(fit, tmp_path / "nb2")
    assert _checks(Path(res["dir"]))["time_budget"].status in ("PASS", "WARN")


def test_preflight_source_version_pin(built, staged, tmp_path):
    src, _ = staged
    nb = _mutated(built, tmp_path, cells=lambda nb: [
        c.__setitem__("source", c["source"].replace("'source_version': 'v", "'source_version': 'vOLD"))
        for c in nb["cells"]])
    checks = _checks(nb, src)
    assert checks["source"].status == "FAIL" and "pinned" in checks["source"].detail


def test_kaggle_check_skips_without_token(built, monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.kaggle credentials either
    checks = _checks(built[0], kaggle=True)
    assert checks["kaggle_online"].status == "SKIP"


def test_time_budget_math():
    cfg = {"time": {"setup_estimate_s": 700, "soft_deadline_s": 30900, "notebook_limit_s": 32400,
                    "expected_games": 110},
           "solver": {"concurrency": 28}, "duck_patches": {"DUCK_PATCH_B08_SCHEDULER": "1"},
           "smoke": {"minutes": 8}}
    tb = PF.time_budget(cfg, turn_s=170)
    assert tb["play_window_s"] == 30200 and tb["teardown_reserve_s"] == 1500
    assert tb["floors_fit"] and tb["plan"]["max_active"] == 34
    assert tb["smoke_gpu_h"] == pytest.approx((700 + 480 + 120) / 3600, abs=1e-3)


# ------------------------------------------------------------------------------------------------ runtime pieces


def test_serving_start_command_is_the_launch_interface(tmp_path):
    import subprocess

    import hydra_run as H

    cmd = H.serving_start_command("flashnext_tuned", python=sys.executable, src_root=REPO, work_dir=tmp_path)
    assert cmd == [sys.executable, str(REPO / "serving" / "launch.py"), "--profile", "flashnext_tuned",
                   "--work-dir", str(tmp_path)]
    rendered = subprocess.run(cmd + ["--print-argv"], capture_output=True, text=True, timeout=120)
    assert rendered.returncode == 0, rendered.stderr
    assert json.loads(rendered.stdout)["profile"] == "flashnext_tuned"


def test_teardown_never_raises(tmp_path):
    import hydra_run as H

    class Boom:
        def __exit__(self, *a):
            raise RuntimeError("mock exit failed")

        def stop(self):
            raise RuntimeError("health stop failed")

    ctx = H.RunContext.create({"teardown_commands": ["exit 3", "definitely-not-a-command-xyz"]},
                              agent_dir=REPO / "agent", kaggle_input=tmp_path, working_dir=tmp_path,
                              start_epoch=0.0, true_submission=True, dry_run=True)
    handle = H.ServingHandle(profile="x", kind="mock", pid=2 ** 22 + 12345, mock_server=Boom())
    handle.health = Boom()
    out = H.teardown(ctx, handle)
    assert "mock_error" in out and "health_error" in out and out["server"] == "gone"
    assert [c.get("rc") for c in out["commands"]] == [3, 127]
    assert H.teardown(None, None) == {}


def test_effective_config_mismatch_is_strict_in_smoke_and_logged_in_rerun(tmp_path, monkeypatch):
    import hydra_run as H

    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "g_p0.txt").write_text(
        "ANALYZER STATUS\nmodel: mock/m\ncontext_budget_tokens: 26112\nyield_seconds: 60.0\n")
    monkeypatch.setenv("LOCAL_ANALYZER_MODEL_ID", "mock/m")
    monkeypatch.setenv("LOCAL_ANALYZER_CONTEXT_WINDOW", "16384")
    serving = H.ServingHandle(profile="p", kind="mock")
    result = {"games": 1, "actions": 5}
    for rerun in (True, False):
        ctx = H.RunContext.create({"time": {"soft_deadline_s": 30900}}, agent_dir=REPO / "agent",
                                  kaggle_input=tmp_path, working_dir=tmp_path, start_epoch=0.0,
                                  true_submission=rerun, dry_run=True)
        ctx.notes["expected_budget"] = 9728  # a 16 k window, but the transcripts show the 32 k budget (MISTAKES #10)
        if rerun:
            rep = H.effective_config_report(ctx, serving, result)
            assert rep["mismatches"] and ctx.notes["problems"]
        else:
            with pytest.raises(RuntimeError, match="context_budget_tokens"):
                H.effective_config_report(ctx, serving, result)


# ------------------------------------------------------------------------------------------------ CPU rehearsal


def _port_open(url: str) -> bool:
    host, port = url.split("//", 1)[1].split("/", 1)[0].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


def _lines(stdout: str, tag: str) -> list[dict]:
    return [json.loads(line.split(" ", 1)[1]) for line in stdout.splitlines() if line.startswith(tag + " ")]


def test_notebook_dry_run_smoke(built, staged, tmp_path):
    pytest.importorskip("arcengine")
    src, _ = staged
    res = B.exec_dry(built[0], src, work=tmp_path / "dry", seconds=30, max_actions=8)
    out = res["stdout"]
    assert res["rc"] == 0, out[-4000:]
    assert _lines(out, "GPU_CHECK") == [{"skipped": "dry run"}]
    ready = [x for x in _lines(out, "HYDRA_SERVING") if x.get("event") == "ready"]
    assert ready and ready[0]["profile"] == "mock_cpu"                 # launched through serving/launch.py
    assert not _port_open(ready[0]["base_url"])                         # ... and torn down
    runs = _lines(out, "HYDRA_GAME")
    assert len(runs) == 3 and all(r["actions"] > 0 and r["state"] != "crashed" for r in runs)
    assert _lines(out, "EFFECTIVE_CONFIG_CHECK") == [{"mismatches": [], "ok": True}]
    working = Path(res["working"])
    import pyarrow.parquet as pq

    table = pq.read_table(working / "submission.parquet")
    assert table.column_names == ["row_id", "game_id", "end_of_game", "score"] and table.num_rows == 1
    sched = json.loads((working / "hydra_scheduler.json").read_text())
    assert sched["plan"]["n_games"] == 3 and all(g["first_turn_s"] is not None for g in sched["games"].values())
    assert _lines(out, "HYDRA_TEARDOWN")[0].get("server") in ("terminated", "killed", "gone")


def test_notebook_dry_run_rerun_with_a_gpu_profile(staged, tmp_path):
    pytest.importorskip("arcengine")
    src, _ = staged
    res = B.build(EXAMPLE, tmp_path / "nb", source_dir=src)   # flashnext_baseline: GPU-only serving is skipped
    run = B.exec_dry(Path(res["dir"]), src, rerun=True, games="ls20,vc33,ft09,ar25", max_actions=5,
                     work=tmp_path / "dry")
    out = run["stdout"]
    assert run["rc"] == 0, out[-4000:]
    assert any(x.get("event") == "dry_skip" and x.get("rendered") for x in _lines(out, "HYDRA_SERVING"))
    runs = _lines(out, "HYDRA_GAME")
    assert len(runs) == 4 and all(r["actions"] == 5 for r in runs)
    assert not (Path(run["working"]) / "submission.parquet").exists()   # the gateway writes the real one
    assert _lines(out, "HYDRA_MODE")[0]["mode"] == "rerun"
    assert _lines(out, "EFFECTIVE_CONFIG_CHECK") == [{"mismatches": [], "ok": True}]
    env = _lines(out, "HYDRA_ENV")[0]
    assert env["patches"]["B08_SCHEDULER"] is True and env["analyzer"]["MULTIMODAL_UPSCALE"] == "12"
