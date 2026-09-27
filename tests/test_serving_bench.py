"""CPU tests for the serving benchmark (B02) and serving profiles / launcher (B03)."""
from __future__ import annotations

import base64
import gzip
import json
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "automation"))
sys.path.insert(0, str(REPO / "serving"))

import serving_bench as sb  # noqa: E402
import launch  # noqa: E402

REAL_PACK = REPO / "automation" / "serving_pack" / "duck_0922.jsonl.gz"

# argv[1:] of SCRATCH/results/vllm-server-identity.json (the 09-22 run), argv[0] is the interpreter.
BASELINE_0922_ARGV = [
    "-m", "vllm.entrypoints.cli.main", "serve",
    "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",
    "--served-model-name", "Qwen/Qwen3.8-Flash-Next-NVFP4", "--host", "127.0.0.1", "--port", "1234",
    "--load-format", "safetensors", "--dtype", "bfloat16", "--quantization", "modelopt_fp4",
    "--tensor-parallel-size", "1", "--distributed-executor-backend", "mp", "--kv-cache-memory-bytes", "5368709120",
    "--max-model-len", "32768", "--max-num-seqs", "16", "--max-num-batched-tokens", "8192", "--async-scheduling",
    "--enable-chunked-prefill", "--max-cudagraph-capture-size", "32", "--no-enable-prefix-caching",
    "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--chat-template",
    "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1/chat_template.jinja",
    "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}', "--no-enable-log-requests",
    "--disable-uvicorn-access-log", "--uvicorn-log-level", "info",
]

SYSTEM = "You are a coding agent solving a grid-based puzzle game.\nRules..."


def _meta(calls: list[tuple[str, str]], finish: str = "tool_calls") -> str:
    raw = [{"id": cid, "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}
           for cid, code in calls]
    return (f"finish_reason: {finish}\ntool_call_count: {len(calls)}\ncontent_chars: 0\nreasoning_chars: 10\n"
            f"tool_call_markup_in_text: no\ntool_calls_recovered_from_markup: no\nraw_tool_calls:\n{json.dumps(raw, indent=2)}")


def _turn(step: int, action: int, clock: str, user: str, requests: list[dict], status: str) -> str:
    parts = [f"--- analysis_step={step} | action={action} | {clock} | tool-agent ---", "[SYSTEM PROMPT]", SYSTEM, "",
             "[USER PROMPT]", user, ""]
    for r in requests:
        parts += ["[MODEL RESPONSE META]", _meta(r["calls"]), "", "[THINKING]", r["thinking"], ""]
        if r.get("content"):
            parts += ["[ASSISTANT]", r["content"], ""]
        for (_cid, code), result in zip(r["calls"], r["results"]):
            parts += ["[TOOL CALL: python]", f"<tool_call>\n<function=python>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>", "",
                      "[TOOL RESULT: python]", result, ""]
    parts += ["[ANALYZER STATUS]", "model: m\n" + status, "", ""]
    return "\n".join(parts)


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    for d in ("transcripts", "prompts", "artifacts"):
        (root / d).mkdir(parents=True)
    for g, game in enumerate(["aa11-00000001", "bb22-00000002"]):
        text = "\n" + _turn(1, 1, "10:00:00", f"State step 1 of {game}", [
            {"calls": [("t1", "print(current_frame.step)")], "thinking": "Let me look.\n[not a section]", "results": ["level 1 step 0"]},
            {"calls": [("t2", "action(['ACTION1'])")], "thinking": "Act now.", "content": "World model: x", "results": ["moved"]},
        ], "step_executed: True\nmessage: Step executed.") + _turn(2, 2, "10:01:00", f"State step 2 of {game}", [
            {"calls": [("t3", "print(1)")], "thinking": "Think more.", "results": ["1"]},
        ], "step_executed: False\nmessage: Yielded control to solver: turn_time_budget.")
        (root / "transcripts" / f"{game}_p0.txt").write_text(text)
        (root / "prompts" / f"{game}_p0.log").write_text(
            "LATEST MODEL CALL SNAPSHOT\n\n[AVAILABLE TOOLS]\n- python: Run one ephemeral Python snippet.\n\n[MODEL INPUT]\n[SYSTEM]\n...")
        board = [[(r + c + g) % 16 for c in range(8)] for r in range(8)]
        events = [
            {"type": "initial", "action_num": 0, "board": board},
            {"type": "analysis", "action_num": 0, "board": board,
             "transcript": f"--- analysis_step=1 | action=1 | 10:00:00 | tool-agent ---\n[SYSTEM PROMPT]"},
            {"type": "action", "action_num": 1, "board": [[5] * 8 for _ in range(8)]},
        ]
        (root / "artifacts" / f"{game}_p0_events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (root / "vllm-metrics-final.prom").write_text(
        'vllm:request_success_total{engine="0",finished_reason="stop"} 6.0\n'
        'vllm:prompt_tokens_total{engine="0"} 60000.0\nvllm:generation_tokens_total{engine="0"} 6000.0\n')
    return root


@pytest.fixture()
def small_pack(results_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "pack.jsonl.gz"
    info = sb.build_pack(results_dir, out)
    assert info["games"] == 2 and info["turns"] == 4 and info["requests"] == 6
    return out


# ---------------------------------------------------------------------------------------------- pack
def test_png_encoder_roundtrip():
    grid = [[0, 5, 9], [14, 15, 1]]
    png = sb.grid_to_png(grid, scale=4)
    assert sb.png_size(png) == (12, 8)
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    raw = zlib.decompress(idat)
    row_len = 1 + 12 * 3
    assert len(raw) == row_len * 8
    assert raw[1 + 4 * 3:1 + 4 * 3 + 3] == bytes(sb.ARC_RGB[5])  # pixel (0, 4) -> grid (0, 1) = black


def test_build_and_load_pack(small_pack: Path):
    header, games = sb.load_pack(small_pack)
    assert len(header["systems"]) == 1 and next(iter(header["systems"].values())) == SYSTEM
    assert header["tools"][0]["function"]["description"] == "Run one ephemeral Python snippet."
    rec = header["recorded_run"]
    assert rec["requests"] == 6.0 and rec["action_requests"] == 2 and rec["step_executed_turns"] == 2
    g = games[0]
    t0, t1 = g.turns
    assert t0["user"].startswith("State step 1") and len(t0["requests"]) == 2
    assert t0["requests"][0]["reasoning"] == "Let me look.\n[not a section]"
    assert t0["requests"][1]["content"] == "World model: x" and t0["requests"][1]["action"] is True
    assert t0["requests"][0]["tool_results"] == ["level 1 step 0"]
    assert t1["status"]["message"].startswith("Yielded")
    url = g.images[t0["image"]]
    assert sb.png_size(base64.b64decode(url.split(",", 1)[1])) == (32, 32)
    assert t1["image"] in g.images  # fallback to latest board by action number
    assert g.pool == ["level 1 step 0", "moved", "1"]


@pytest.mark.skipif(not REAL_PACK.exists(), reason="real pack not built")
def test_real_pack_matches_0922_run():
    header, games = sb.load_pack(REAL_PACK)
    rec = header["recorded_run"]
    assert len(games) == 25 and rec["turns"] == 1026 and rec["requests"] == 1045
    assert len(header["systems"]) == 1  # identical system prompt across games -> shareable prefix
    assert abs(rec["decisions_per_game_hour"] - 11.6) < 0.05
    assert abs(rec["queue_share_of_e2e"] - 0.866) < 0.01
    assert all(t["image"] for g in games for t in g.turns)


# ---------------------------------------------------------------------------------------------- duck context logic
def _msgs(n_turns: int, size: int = 3000) -> list[dict]:
    out = [{"role": "system", "content": "S" * 500}]
    for i in range(n_turns):
        out.append({"role": "user", "content": f"u{i} " + "x" * size})
        out.append({"role": "assistant", "reasoning": "r" * size, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "python", "arguments": "{\"code\": \"1\"}"}}]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": "t" * 200})
    return out


def test_trim_respects_budget_and_user_start():
    est = sb.TokenEstimator([])
    msgs = _msgs(20) + [{"role": "user", "content": "current"}]
    trimmed = sb.trim_messages(msgs, 5000, est)
    assert est(trimmed) <= 5000
    assert trimmed[0]["role"] == "system" and trimmed[1]["role"] == "user"
    assert trimmed[-1]["content"] == "current"


def test_persistent_history_caps_assistant_turns():
    est = sb.TokenEstimator([])
    hist = sb.persistent_history(_msgs(40, size=10), 10 ** 7, est, max_turns=30)
    assert sum(1 for m in hist if m["role"] == "assistant") == 30
    assert hist[0]["role"] == "user"


def test_image_policy_and_estimator():
    url = "data:image/png;base64," + "A" * 5000
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "user", "content": [{"type": "text", "text": f"u{i}"}, {"type": "image_url", "image_url": {"url": url}}]}
        for i in range(5)]
    kept = sb.apply_image_policy(msgs, 3)
    assert sum(sb._n_images(m) for m in kept) == 3
    assert kept[1] == {"role": "user", "content": "u0"}
    est = sb.TokenEstimator([])
    assert est(kept) < 3 * sb.DUCK_IMAGE_TOKENS + 200  # base64 is not charged as text (AGENTFIX F1)


def test_classify_reply():
    def body(msg, finish="tool_calls"):
        return {"choices": [{"message": msg, "finish_reason": finish}]}
    call = lambda args: {"id": "x", "type": "function", "function": {"name": "python", "arguments": args}}  # noqa: E731
    ok = sb.classify_reply(body({"tool_calls": [call(json.dumps({"code": "action(['ACTION1'])"}))]}))
    assert ok["category"] == "valid_tool_call" and ok["action"]
    probe = sb.classify_reply(body({"tool_calls": [call(json.dumps({"code": "print(1)"}))]}))
    assert probe["category"] == "valid_tool_call" and not probe["action"]
    assert sb.classify_reply(body({"tool_calls": [call("{broken")]}))["category"] == "invalid_tool_args"
    assert sb.classify_reply(body({"content": "<tool_call><function=python>"}, "stop"))["category"] == "markup_in_text"
    assert sb.classify_reply(body({"content": "hi"}, "stop"))["category"] == "no_tool_call"
    assert sb.classify_reply(body({"reasoning": "long..."}, "length"))["category"] == "truncated"
    assert sb.classify_reply(None)["category"] == "error"


def test_prometheus_parse_and_histogram_quantile():
    text = ('# HELP x\nvllm:num_requests_waiting{engine="0",model_name="m"} 4.0\n'
            'vllm:request_queue_time_seconds_bucket{le="1.0"} 5\nvllm:request_queue_time_seconds_bucket{le="10.0"} 10\n'
            'vllm:request_queue_time_seconds_bucket{le="+Inf"} 10\nvllm:request_queue_time_seconds_sum 30\n'
            'vllm:request_queue_time_seconds_count 10\n'
            'vllm:cache_config_info{block_size="1600",enable_prefix_caching="False"} 1.0\n')
    snap = sb.snapshot_from_prom(text)
    assert snap["gauges"]["waiting"] == 4.0
    assert snap["info"]["cache_config"]["block_size"] == "1600"
    assert sb.hist_quantile(snap["hist"]["queue_time_s"]["buckets"], 0.5) == pytest.approx(1.0)
    assert sb.hist_quantile(snap["hist"]["queue_time_s"]["buckets"], 0.9) == pytest.approx(8.2)
    empty = sb.snapshot_from_prom("")
    d = sb.diff_snapshots(empty, snap)
    assert d["hist"]["queue_time_s"]["mean"] is None  # missing counters in the first snapshot -> None, no crash


# ---------------------------------------------------------------------------------------------- end-to-end on the mock
@pytest.mark.parametrize("mode", ["actual", "recorded"])
def test_mock_bench_end_to_end(small_pack: Path, tmp_path: Path, mode: str):
    cfg = sb.BenchConfig(pack=str(small_pack), label=f"mock_{mode}", out=str(tmp_path / mode), agents=5, duration_s=3.0,
                         warmup_s=1.0, metrics_interval_s=0.5, history_mode=mode, transport="urllib", grace_s=2.0)
    with sb.MockServerThread(sb.MockConfig(slots=3, decode_tps=5000, gen_mean=60, seed=1)) as srv:
        cfg.base_url = srv.base_url
        report = sb.run_bench(cfg)
    s = report["summary"]
    assert s["requests_completed"] > 5 and s["error_pct"] == 0
    assert s["decisions"] > 0 and s["decisions_per_game_hour"] > 0
    assert 80 <= s["tool_call_valid_pct"] <= 100
    assert s["metrics_available"] and s["server_gen_tok_s"] > 0 and s["prefix_cache_hit_rate"] is not None
    assert s["spec_acceptance_rate"] == pytest.approx(0.6, abs=0.05)
    assert s["latency_p50_s"] is not None and s["turn_latency_p50_s"] is not None
    assert s["images_per_request_mean"] and s["images_per_request_mean"] >= 1
    out = tmp_path / mode
    for f in ("report.json", "summary.md", "requests.jsonl.gz", "turns.jsonl.gz", "metrics_samples.jsonl"):
        assert (out / f).exists(), f
    with gzip.open(out / "requests.jsonl.gz", "rt") as fh:
        rows = [json.loads(line) for line in fh]
    assert rows and {"latency_s", "category", "prompt_tokens"} <= set(rows[0])
    assert "decisions / game-hour" in (out / "summary.md").read_text()


def test_compare_reports(tmp_path: Path):
    def rep(label, dph, valid=99.0):
        p = tmp_path / label / "report.json"
        p.parent.mkdir()
        p.write_text(json.dumps({"summary": {"label": label, "decisions_per_game_hour": dph, "tool_call_valid_pct": valid,
                                             "requests_completed": 10, "error_pct": 0.0}}))
        return p
    paths = [rep("flashnext_baseline", 10.0), rep("flashnext_tuned", 25.0), rep("qwen27b_fp8", 12.0, valid=85.0)]
    md, doc = sb.compare_reports(paths, "flashnext_baseline")
    assert doc["ratios"]["flashnext_tuned"] == pytest.approx(2.5)
    assert "**flashnext_tuned**" in md and "qwen27b_fp8 tool-call validity" in md
    md2, _ = sb.compare_reports([paths[0], rep("weak", 11.0)], "flashnext_baseline")
    assert "falsifier" in md2


# ---------------------------------------------------------------------------------------------- profiles / launcher
def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_baseline_profile_is_exact_0922_argv():
    argv = launch.render_argv(launch.load_profile("flashnext_baseline"), python="/usr/bin/python3")
    assert argv == ["/usr/bin/python3"] + BASELINE_0922_ARGV


@pytest.mark.parametrize("name", launch.list_profiles())
def test_profile_invariants(name: str):
    prof = launch.load_profile(name)
    argv = launch.render_argv(prof, python="py")
    assert "{" not in " ".join(a for a in argv if not a.startswith("{\""))  # all placeholders rendered
    if prof.get("fallback"):
        assert prof["fallback"] in launch.list_profiles()
    if prof["runtime"]["kind"] == "mock":
        return
    ctx = int(_flag(argv, "--max-model-len"))
    assert prof["client"]["context_window"] == ctx
    assert prof["harness_env"]["LOCAL_ANALYZER_CONTEXT_WINDOW"] == str(ctx)  # the 16k-override fix
    if "--enable-prefix-caching" in argv and prof["runtime"]["kind"] == "flashnext_dev":
        assert _flag(argv, "--mamba-cache-mode") == "align"  # Flash-Next rejects mamba_cache_mode 'all'
    if prof["runtime"]["kind"] == "flashnext_dev":
        assert "--kv-cache-dtype" not in argv  # QSA requires a BF16 main KV cache
        kv_gib = int(_flag(argv, "--kv-cache-memory-bytes")) / 1024 ** 3
        weights = 81.8 if "--speculative-config" in argv and "mtp" in _flag(argv, "--speculative-config") else 74.3
        headroom = 94.43 - weights - kv_gib
        assert headroom >= 4.0, f"{name}: only {headroom:.1f} GiB left for activations"
        if headroom < 7.0:
            assert int(_flag(argv, "--max-num-batched-tokens")) <= 4096
        assert int(_flag(argv, "--max-num-seqs")) <= int(_flag(argv, "--max-cudagraph-capture-size"))


def test_qwen27b_profile():
    argv = launch.render_argv(launch.load_profile("qwen27b_fp8"), python="py")
    assert _flag(argv, "--kv-cache-dtype") == "fp8" and "--enable-prefix-caching" in argv
    assert json.loads(_flag(argv, "--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 2}
    assert json.loads(_flag(argv, "--default-chat-template-kwargs")) == {"preserve_thinking": True}
    assert launch.load_profile("qwen27b_fp8")["fallback"] == "qwen27b_fp8_nomtp"


def test_exp001_plan_runs_likely_winner_first():
    assert launch.EXP001_PLAN[0] == "flashnext_tuned"
    t = launch.render_argv(launch.load_profile("flashnext_tuned"), python="py")
    g = launch.render_argv(launch.load_profile("flashnext_nomtp_kv12_seqs28_pc"), python="py")
    assert t == g and "--speculative-config" not in t
    for name in launch.EXP001_PLAN + launch.EXP001_EXTRAS:
        launch.load_profile(name)


def test_print_argv_cli():
    out = subprocess.run([sys.executable, str(REPO / "serving" / "launch.py"), "--profile", "flashnext_tuned", "--print-argv"],
                         capture_output=True, text=True, check=True).stdout
    doc = json.loads(out)
    assert doc["runtime"] == "flashnext_dev" and doc["client"]["context_window"] == 16384


def test_harness_env_wins_over_setup_env(tmp_path: Path, monkeypatch):
    setup_env = tmp_path / "taaf_setup_env.json"
    setup_env.write_text(json.dumps({"LOCAL_ANALYZER_CONTEXT_WINDOW": "32768", "OTHER": "x"}))
    monkeypatch.setitem(launch.PRISTINE_ENV, "TAAF_KAGGLE_SETUP_ENV", str(setup_env))
    prof = launch.load_profile("flashnext_tuned")
    for key in prof["harness_env"]:  # write_harness_env exports into os.environ; let monkeypatch restore it
        monkeypatch.delenv(key, raising=False)
    handle = launch.ServerHandle("flashnext_tuned", [], "http://127.0.0.1:1234/v1", tmp_path / "log")
    env = launch.write_harness_env(prof, handle, tmp_path)
    doc = json.loads(setup_env.read_text())
    assert doc["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "16384" and doc["OTHER"] == "x"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:1234/v1"
    assert json.loads((tmp_path / "harness_env.json").read_text())["LOCAL_ANALYZER_CONTEXT_WINDOW"] == "16384"


def test_env_view_drops_secrets():
    view = launch.safe_env_view({"KAGGLE_API_TOKEN": "t", "HF_TOKEN": "h", "PYTHONPATH": "/x", "ARC_API_KEY": "k"}, base={})
    assert view == {"PYTHONPATH": "/x"}


def test_suite_with_fallback_on_mock(tmp_path: Path, small_pack: Path):
    broken = tmp_path / "broken.json"
    prof = launch.load_profile("mock_cpu")
    prof.update({"name": "broken", "runtime": {"kind": "system"}, "entrypoint": ["-c", "import sys; sys.exit(3)"], "args": [],
                 "fallback": "mock_cpu", "startup": {"timeout_s": 20, "expected_s": 1}})
    broken.write_text(json.dumps(prof))
    out = tmp_path / "suite"
    proc = subprocess.run([sys.executable, str(REPO / "serving" / "launch.py"), "--suite", str(broken), "--duration", "3",
                           "--warmup", "1", "--agents", "4", "--pack", str(small_pack), "--out", str(out),
                           "--baseline", "mock_cpu", "--budget-min", "3"], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    suite = json.loads((out / "suite.json").read_text())
    statuses = [(r["profile"], r["status"]) for r in suite["results"]]
    assert statuses == [("broken", "startup_failed"), ("mock_cpu", "ok")]
    assert (out / "mock_cpu" / "report.json").exists() and (out / "exp001_summary.md").exists()
    assert "mock_cpu" in (out / "exp001_summary.md").read_text()


def test_pack_path_survives_kaggle_gz_decompression(tmp_path):
    """Kaggle datasets auto-decompress `.gz` uploads: `duck_0922.jsonl.gz` arrives as `duck_0922.jsonl`."""
    import gzip
    import shutil
    import serving_bench as sb
    src = sb.DEFAULT_PACK
    plain = tmp_path / "duck_0922.jsonl"
    with gzip.open(src, "rb") as fin, open(plain, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    header, games = sb.load_pack(tmp_path / "duck_0922.jsonl.gz")   # the name the notebook passes
    assert games and header is not None
