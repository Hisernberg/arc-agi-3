# 13 — ARC-AGI-3 assets on Hugging Face and GitHub (survey of 2026-09-27)

`SCRATCH` = `/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad`.
Raw inventories are in `SCRATCH/hf/user_inventory.txt`, `SCRATCH/hf/public_meta.json` and `SCRATCH/hf/public/_cards/` (about 170 model and dataset cards). Clone logs are in `SCRATCH/github/clone.log`.
Relevance is scored from 1 to 5 by how much the item helps us reach an 18–20+ score on the Kaggle leaderboard. The Kaggle setup is 1x RTX PRO 6000 with 96 GB, runs offline and has a 9-hour budget.

## TL;DR

1. **Top teams do not rely on a bigger LLM.**
   - **Tong Hui Kang (#3, 20.53)** publishes an imitation-trained per-coordinate policy of about 0.4M parameters (`tonghuikang/arc3`). It has 384 binary features per cell and action, colour and click heads. It is trained only on won trajectories from *variant* games and validated on the base games. The architecture search was part of an autoresearch hackathon run with Modal.
   - **Tufa Labs (#1)** released the full Duck harness together with a 2.8 GB run: 25 games × 20 passes with Qwen3.6-27B-FP8, mean 1.60 per pass.
   - **NVIDIA** (probably the NVARC3 team, #6) released DreamTeam, a six-agent solver built around an executable world model, plus 25 generated games.
   - **da-fr/arc-agi-3-solution** (Daniel Franzen, #5) was created on 2026-09-25 and is still empty. Watch it.
2. **Trace data exists at scale:**
   - human play: 340 sessions covering all 25 games;
   - Schema-harness frontier traces: 50 runs (guanning's larger GPT-5.6 Sol and Opus 4.8 sets are gated);
   - Kepler Opus-5 traces with a per-game "lab notebook" and a 100.00 scorecard;
   - ARA Labs: about 80 repos with winning replay scripts for each level and model;
   - 14.6M transition rows (`fredericowieser/arc-agi-3-wm-traces`).
3. **Extra environments for training and validation beyond the 25 public games:**
   - 253 community games in `theredbluepill/arc-interactive`;
   - 25 NVIDIA-generated games;
   - sonpham's catalog of 927 games, 571 of them AI-generated;
   - STARGA and ritwika96 procedural corpora.

   This is the missing ingredient for any learned policy and for honest held-out evaluation.
4. **Models:** there is no public checkpoint that is strong for ARC-AGI-3 *and* fits Kaggle.
   - The fine-tunes of Qwen3.6-27B are small LoRAs with no benchmark claims: ericmao ×7, and `star-ga/naestro-agi3-27b`.
   - `Shockem/Qwen3.8-27b-Terse-Coder-NVFP4` (22 GB) cuts chain-of-thought length by about 10× and is the only "drop-in" candidate. It is used by `U4AR/qwen38-arc3-rl`, which does LoRA RL of Qwen3.8-27B with the Duck harness.
5. **The user's own HF account (Nabidnur)** has no ARC-AGI-3 model or dataset. It does hold two Kaggle-account backups that contain ARC-AGI-3 material:
   - Team AIRIS's best ARC-AGI-3 submission, **public 4.68** (local score over 25 games: 7.10);
   - 57 plus 56 ARC-AGI-3 notebooks.

   Everything else in the account is ARC-AGI-2, NASA Space Apps, Kaggriculture, BioHub or unrelated work.

---

## 1. The user's own Hugging Face account (`Nabidnur`, HF_TOKEN, private repos included)

The account has 7 models, 19 datasets and 4 spaces.

| Repo | Type / visibility | Size | Contents | ARC-AGI-3 relevance | Downloaded |
|---|---|---|---|---|---|
| `Nabidnur/checkmpobbyacckraggle` | dataset, **private** | 866 MB (4,813 files) | Backup of Kaggle account **poby7722** (Team AIRIS with krno1legend and koushikrudra). The `arc-prize-2026-arc-agi-3/` folder (428 MB) has 57 notebooks and 3 submission-artifact bundles. The **best submission, 56568844, scored public 4.68** (krno1legend `urad-duck-r10-targeted`, Duck + Qwen3.8-Flash-Next-NVFP4, with the r9/r10 "rescue/budget/gate" patch sources). The **local mean over 25 games was 7.10**. Also includes `duck-qwen3-8-tuned` (local 6.02), transcripts, `*_p0_events.jsonl`, vLLM metrics, `score.json` and `benchmark.json`. The ARC-AGI-2 folder shows **rank #10 in ARC-AGI-2** and #69 of 3,362 in ARC-AGI-3 for that team. `_models/` holds `arc3-generalization-patched` and `arc-jepa`. | **5** (our own best artifacts) | `SCRATCH/hf/user/checkmpobbyacckraggle/` (ARC-AGI-3, ARC-AGI-2, `_manifests`, `_models`, README, ACTIVITY) |
| `Nabidnur/checkmyacckraggle` | dataset, private | 743 MB | Snapshot zip of Kaggle account **koushikrudra** (2026-09-25). For ARC-AGI-3: rank 288, best public 3.68, 159 submissions, 56 notebooks (forge v19/v20, v21–v27 "need-luck" and "ash" variants, omega-planner, `duck-qwen3-8-tuned`). The other 440 MB are unrelated `.npy` files. | 4 | `SCRATCH/hf/user/checkmyacckraggle/` (zip plus `extracted/checkmyacckraggle/arc-prize-2026-arc-agi-3/`) |
| `Nabidnur/arc-agi-2-solver-qwen3-4b` | model, public | 0.1 MB | Code only: NVARC-lineage ARC-AGI-2 TTT pipeline (sorokin qwen3_4b + rank-256 rsLoRA + DFS). No weights. | 1 | `SCRATCH/hf/user/arc-agi-2-solver-qwen3-4b/` |
| `Nabidnur/arc-superbrain-small-models` | model, private | 2.9 MB | ARC-AGI-2 research models: shape_ranker (top-1 79.1% on eval output shapes), tiny ARC-JEPA (0.31M) and TRV (0.2M). The last two did not work (AUC 0.39 and 0.55). | 1 | `SCRATCH/hf/user/arc-superbrain-small-models/` |
| `Nabidnur/kaggriculture-agent` | model, private | 152 MB | Kaggriculture competition agents | 0 | README only |
| `Nabidnur/ignis-*` (4 models) | model, private | 0.3–15 MB | NASA Space Apps wildfire sklearn models | 0 | README only |
| `Nabidnur/arc-agi-2-{grids,cot-sft,vision,synthetic-v1}` | dataset, public | 16–33 MB | ARC-AGI-2 tasks, CoT SFT and synthetic curriculum | 1 | READMEs only |
| `Nabidnur/arc-unified-corpus`, `arc-agi-2-{jepa-episodes,verifier-router,multiformat,task-panels,atlas-images,solution-methods}` | dataset, private | 2–192 MB | ARC-1/2 + RE-ARC + ARC-GEN + ConceptARC corpora and derived JEPA, verifier and vision data | 1 | READMEs and manifests only |
| `Nabidnur/liars-are-information` (+ demo space) | dataset/space | 81 MB | Multi-agent LLM robustness paper (ARC here means AI2-ARC QA) | 0 | README only |
| `Nabidnur/biohub-complete-archive` | dataset, private | 1.95 GB | BioHub cell-tracking archive | 0 | not downloaded |
| `Nabidnur/Sim2Reason-Data`, `spaceapps26-data`, `ignis-fire-calendar` | dataset | <1 MB | unrelated | 0 | README only |
| `Nabidnur/ignis-vault` | dataset, private | tiny | contains `credentials/ignis_keys_encoded.json` | 0 | **deliberately not downloaded.** Stored credentials on HF are a risk; consider rotating them or deleting the file. |
| spaces `nasa`, `verdent_view`, `verdernt_view`, `liars-are-information-demo` | space | small | Streamlit and static demos | 0 | README only |

Related GitHub repos belonging to the user (`Hisernberg`, found through search):
- `Hisernberg/arc-agi-3` is this repo.
- `Hisernberg/arc-agi-3-campaign` is **private**. Its description is "Duck-harness improvements, local CPU agent dev, daily submission automation. Team Kragglenote2 Forwork". The clone failed because this session has no credentials for private repos; it needs `add_repo`.
- `Hisernberg/checkmpobbyacckraggle` and `checkmyacckraggle` are private GitHub mirrors of the two HF backups.
- `Hisernberg/arc-agi-2-kaggle` is private.

**Takeaway:** the most useful internal artifact is the 4.68 `urad-duck-r10-targeted` bundle, found at `.../submissions/best/sub_56568844__pub-4.68/urad_r10_sources_atpyd1b6/` (`r9_driver.py`, `r9_gate.py`, `r9_budget.py`, `r10_rescue.py`, `taaf_ours_patch.py`). It is Team AIRIS's strongest Duck variant, and it beats the 3.85 from `00_CONTEXT.md`, so the team context should be reconciled. It is directly diffable against our current AGENTFIX notebook. Its components:

- **Base:** the public anim bundle `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`.
- **`taaf_ours_patch.py`** (1,208 lines): exact-match text patches over the base bundle.
- **`r9_*` files:** a supervised worker with a lifecycle bound, a weighted waiting gate, and wall-clock budget accounting.
- **`r10_rescue.py`:** first-level stall recovery from observed actions. It uses no extra model calls and no game-ID rules, and it disarms after the first level is solved.

---

## 2. Public HF: trace, replay and transition datasets

| Name | Link | Size | What it contains | Rel. | Downloaded path |
|---|---|---|---|---|---|
| ARC-AGI-3 human play (mirror of the ARC Prize human dataset) | [magic-sword/arc_agi_3_public_demo_human_testing](https://huggingface.co/datasets/magic-sword/arc_agi_3_public_demo_human_testing) | 475 MB | **340 human sessions over all 25 public games** (lp85 54, others 10–15). Full frame trajectories (FrameDataRaw JSON), won flag, `actions_by_level`, resets. Human win rate per game ranges from 0.14 (bp35) to 1.0 (r11l). | **5** | `SCRATCH/hf/public/magic-sword__arc_agi_3_public_demo_human_testing/` |
| Kepler 1.0 trace corpus | [cveinnt/kepler-arc-agi-3-traces](https://huggingface.co/datasets/cveinnt/kepler-arc-agi-3-traces) | 170 MB | The executable-world-model harness. **Claude Opus 5 = 100.00** (server-verified scorecard 91aa2f10) and GPT-5.6 Sol = 95.97. `runs.jsonl` has per-game `level_actions` and **`notes_md`, a lab notebook giving each game's mechanics level by level**. Also includes the human baseline CSV, the scorer and the agent logs. | **5** | `SCRATCH/hf/public/cveinnt__kepler-arc-agi-3-traces/` |
| Schema harness traces | [schema-harness/arc-agi-3-schema-traces](https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces) (dup: JBrightmanAI/…) | 768 MB | 50 trajectories: 25 GPT-5.6 Sol (mean RHAE 95.35, 24/25 wins) and 25 Claude Opus 4.8 or Fable 5 (mean 98.98, 25/25 wins). Each has `events.jsonl`, snapshots, `baseline_actions.csv` (human baselines per level) and a stdlib RHAE scorer. | **5** | `SCRATCH/hf/public/schema-harness__arc-agi-3-schema-traces/` |
| guanning schema traces | [guanning/arc-agi-3-schema-traces-opus48](https://huggingface.co/datasets/guanning/arc-agi-3-schema-traces-opus48) (0.5 GB), [-gpt56-xhigh](https://huggingface.co/datasets/guanning/arc-agi-3-schema-traces-gpt56-xhigh) (0.41 GB), [-gpt56](https://huggingface.co/datasets/guanning/arc-agi-3-schema-traces-gpt56) (2.6 GB, 100 runs) | 0.4–2.6 GB, **gated (manual approval), 403** | Same `world_model_v5` harness. The gpt56 set is the full sweep split by reasoning effort; its card says effort matters most. | 4 | **not accessible**; access would have to be requested on HF with the Nabidnur account (not done automatically). Card in `SCRATCH/hf/public/_cards/`. |
| guanning/arc3-runs | [link](https://huggingface.co/datasets/guanning/arc3-runs) | 2.28 GB, **gated (manual)** | Raw run directories: sol_xhigh, sol_max, opus48, fable5, *_cc | 3 | not accessible (403) |
| guanning/arc3-ablation-runs | [link](https://huggingface.co/datasets/guanning/arc3-ablation-runs) | 22.3 GB, **gated (manual), 403** | 50 runs: 5 harness ablations (no backtest, BFS, MPC, text world model) × 10 games | 3 | not accessible |
| ARA Labs agent trajectories (about 80 repos) | [AgentNativeResearchLab](https://huggingface.co/AgentNativeResearchLab) `arc-agi3-<harness>-<model>-<game>` | 4 MB–2.9 GB each, about 21 GB in total (16.6 GB of it frame recordings) | One repo per harness × model × game. Harness/model pairs: Claude Code with Opus 4.8, Fable 5 or GLM-5.2; Codex with GPT-5.5 or GPT-5.6 Sol; Antigravity with Gemini 3.1 Pro or Kimi K2.7; Kimi K2.7; Grok 4.5. Games: ar25, ft09, g50t, ls20, r11l, s5i5, su15, tr87, plus ka59, lf52 and wa30. Each has **`solutions/replay/L<n>.txt` (winning action script per level)**, `solutions/GAME.md` (mechanics and gotchas), a `reasoning/` world model, `predictions.jsonl` and accounting. The `phase2-*` repos slice the world model at cost budgets of $13/$27/$51/$76. The `*-agent-trajectories` repos include the ls20 7/7 WIN (10.5k frames). CC-BY-4.0. | 4 | all `arc-agi3-*` repos **without** `recordings/`, `episodes/` and `raw_sessions/` → `SCRATCH/hf/public/AgentNativeResearchLab__*/` |
| ARC-AGI-3 world-model transitions | [fredericowieser/arc-agi-3-wm-traces](https://huggingface.co/datasets/fredericowieser/arc-agi-3-wm-traces) (dup: Ahsna/…) | 904 MB | **14.6M (state, action, next_state) rows**, 64×64 grids. Splits: hhazard (human traces) 14.8k, rollouts 227k, rollouts_positive 6.1M, novelty_gen 8.2M, prior_games, arc_static. | 4 (for learned dynamics / policy pre-training) | `SCRATCH/hf/public/fredericowieser__arc-agi-3-wm-traces/` |
| HHazard transitions | [HHazard/arc-agi-3](https://huggingface.co/datasets/HHazard/arc-agi-3) 1.5 GB, [HHazard/big-arc-agi-3](https://huggingface.co/datasets/HHazard/big-arc-agi-3) 1.9 GB | 1.5 / 1.9 GB | Original transition parquet; its human split is already inside the wm-traces set above | 2 | recorded only |
| Public trace curriculum v3 | [ritwika96/arc-agi-3-public-trace-curriculum-v3](https://huggingface.co/datasets/ritwika96/arc-agi-3-public-trace-curriculum-v3) | 280 MB (27k PNGs) | 5,460 model turns and 22,184 actions derived from the Schema traces. Includes 3,627 misprediction and repair events, and **183 per-level hints generated by Qwen3.6-27B** (157 passed gates). Evaluation uses leave-one-game-out folds. Marked `training_eligible=false`. | 3 | jsonl, hints and manifests (no images) → `SCRATCH/hf/public/ritwika96__…` |
| Procedural games v2, full curriculum, transformation induction | [ritwika96/…](https://huggingface.co/ritwika96) | 0.18–0.52 GB | 330 procedural games and 2,640 levels with verified plans and counterfactuals. The author deprecated them as "too easy". There is also a 130k-episode transformation-induction set. | 2 | recorded only |
| GLM-5.3-Flash expert usage on ARC-AGI-3 | [unity4ar/glm-5.3-flash-arc-agi-3-expert-usage](https://huggingface.co/datasets/unity4ar/glm-5.3-flash-arc-agi-3-expert-usage) | 10 MB | MoE routing counts per layer (43×288) while GLM-5.3-Flash played the 25 games through the **Duck/TAAF harness** on 4×MI355X | 2 (MoE expert-offload sizing idea) | `SCRATCH/hf/public/unity4ar__…` |
| iamseungpil bestiary / skill-discovery / symbolica-bestiary | [iamseungpil](https://huggingface.co/iamseungpil) | 1–85 MB | ls20 world-model "cheat" bestiary (reward hacking of transition-accuracy metrics) and skill-discovery run logs | 2 | `SCRATCH/hf/public/iamseungpil__*` |
| Public game sources | [zarczynski/arc-agi-3-public](https://huggingface.co/datasets/zarczynski/arc-agi-3-public) (dup: AnsteyGeorge) | 3 MB | `environment_files/` for the public games (older snapshot; the Kaggle copy in `SCRATCH/kaggle/comp` is canonical) | 2 | `SCRATCH/hf/public/zarczynski__arc-agi-3-public/` |
| Teaching suite | [dnhkng/arc-agi-3-teaching-suite](https://huggingface.co/datasets/dnhkng/arc-agi-3-teaching-suite) | 1 MB | 100 static grid tasks across 16 families (4 unsolvable). Not interactive. | 1 | downloaded |
| Codex GPT-5.6 Sol g50t | Manusagents/…, Nobody05/… | 1.37 GB | Duplicates of the ARA Labs `codex-gpt5.6sol-g50t` repo | 1 | skipped (duplicate) |
| leonidas123/arc3data | [link](https://huggingface.co/datasets/leonidas123/arc3data) | 2.3 GB | `recordings_converted/` and `tasks/`; no card, unknown provenance | 2 | recorded only |
| ARC-AGI-1/2 only | arcprize/arc_agi_v1/v2_public_eval, arc_agi_2_human_testing; nvidia/Nemotron-SFT/RL-ARC-AGI-v1 (18.4 GB / 0.44 GB); Asap7772/arc-agi-all-Qwen3-* (about 200 repos); Trelis/Qwen3-4B_ds-arc-agi-* | — | The `arcprize` org on HF has **no ARC-AGI-3 data**. The official ARC-AGI-3 human dataset is mirrored by magic-sword. | 1 | recorded only |

Replay scripts downloaded from ARA Labs (count of `L*.txt` files):
- ka59 opus4.8: 7
- su15: 8
- tr87 (agent-trajectories and codex-gpt5.5): 6 each
- ft09 (opus4.8, gpt5.5, gemini): 6 each
- ls20-demo: 6
- wa30: 4–5
- several other runs: 1–2

---

## 3. Public HF: models

| Name | Link | Size / quant | License | What it is / claims | Rel. |
|---|---|---|---|---|---|
| **RadixArk/Qwen3.8-Flash-Next-NVFP4** (our current model) | [link](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | 135 GB repo; NVFP4 W4A4 on **routed experts only**, everything else BF16 | other (Qwen Community 1.0) | Base model: 125B total / 6B active + 51B n-gram embedding + 4B MTP. 48 layers (Gated DeltaNet + QSA sparse attention), 512 experts with top-10 routing, 262k context. The card says "private candidate release", validated on GB300/B300 with SGLang `qwen4_exp` and TP=2. Claims GSM8K 97.27 and AIME26 98.75 pass@1 (BF16 reference 100). Kaggle datasets `keithtyser/qwen3-8-flash-next-nvfp4` and `…-vllm-nvfp4-runtime-v1` repackage it for vLLM on one RTX PRO 6000. 329k downloads. | 5 (context) |
| nvidia/Qwen3.8-Flash-Next-NVFP4 | [link](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 132.7 GB | NVIDIA Open Model License | Official NVIDIA quant, released 2026-08-31. FP8 → NVFP4: GPQA 92.0→91.5, HLE 34.7→35.4, τ²-Telecom 90.8→90.1, Terminal-Bench 2.1 83.3→82.9, IFBench 80.5→81.0. **Alternative with a stronger quantization pedigree; worth an A/B test.** | 4 |
| Qwen/Qwen3.8-Flash-Next (-FP8) | [link](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 360 GB BF16 / 186 GB FP8 | qwen-community-1.0 | "Qwen4 architecture preview". Its "Qwen3.8-Flash" production version has 1M context. | ref |
| **Qwen/Qwen3.6-27B-FP8** | [link](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | 30.9 GB, FP8 | Apache-2.0 | Dense 27B VL model, 262k context, MTP spec-decode (`qwen3_next_mtp`). This is the **Tufa Duck default** (example-run used the `vrfai/Qwen3.6-27B-FP8` mirror). 4.6M downloads. | 4 |
| Qwen/Qwen3.8-27B (+FP8 30.9 GB) | [link](https://huggingface.co/Qwen/Qwen3.8-27B) | 55.6 / 30.9 GB | Apache-2.0 | Newer dense 27B, used by the Shih-Yu-Yeh journal (FP8, LB 2.56) and by U4AR RL | 3 |
| RadixArk/Qwen3.8-27B-NVFP4 (and nvidia/…, `-BF16-LMHead`) | [link](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | 21.9–23.8 GB | Apache-2.0 | Claims GSM8K 97.27 and Terminal-Bench 2.1 (84-task subset, Claude Code) 73.8%. NEXTN MTP. 1.7M downloads. | 3 |
| **Shockem/Qwen3.8-27b-Terse-Coder-NVFP4** / -LoRA | [NVFP4](https://huggingface.co/Shockem/Qwen3.8-27b-Terse-Coder-NVFP4), [LoRA](https://huggingface.co/Shockem/Qwen3.8-27b-Terse-Coder-LoRA) | 21.9 GB (modelopt W4A16) / 1.6 GB (r16) | Apache-2.0 | DPO LoRA with **about 1/10 the chain-of-thought tokens on coding tasks** and correctness preserved. It is the starting policy for U4AR's Duck RL. Useful if our bottleneck is tokens per action within the 9-hour budget. | 3 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | [link](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | 23.5 GB | Apache-2.0 | Small MoE (3B active), MTP. 7.3M downloads. A fast option if we need many cheap calls. | 2 |
| ericmao/arc-agi3-qwen36-{sft,sft-v2,sft-v3,grpo,raw-control,spatial-continuation-100,human100} | [ericmao](https://huggingface.co/ericmao) | LoRA r16 α32, 0.5–2 GB each | none stated | LoRAs on **Qwen3.6-27B** for ARC-AGI-3: SFT, GRPO, human-trace (human100) and spatial-continuation. The cards are empty templates. human100 trained for 0.077 epoch with loss 0.015. **No benchmark claims.** | 2 (idea reference) — configs only in `SCRATCH/hf/public/ericmao__*` |
| star-ga/naestro-agi3-27b | [link](https://huggingface.co/star-ga/naestro-agi3-27b) | 41 MB QLoRA r16 | Apache-2.0 | Goal-inference and plan adapter on Qwen3.6-27B. Trained on the procedural "STARGA v3" corpus. Used as the fallback behind a symbolic mechanic-class cascade. No scores given. | 2 — `SCRATCH/hf/public/star-ga__naestro-agi3-27b/` |
| keithtyser/model-forge-qwen36-27b-ft-v4-nvfp4-dgx-spark | [link](https://huggingface.co/keithtyser/model-forge-qwen36-27b-ft-v4-nvfp4-dgx-spark) | 21.7 GB | Apache-2.0 | By the same "keithtyser" who packages the Kaggle Qwen3.8 runtime our notebook uses. A local fine-tune (v4) of Qwen3.6-27B in NVFP4; purpose unclear. | 2 |
| andyjennings / guychuk `arc-agi-3-grid-jepa` | [link](https://huggingface.co/andyjennings/arc-agi-3-grid-jepa) | code only | — | Grid-JEPA + RSSM world model with planning; no weights or results | 1 (downloaded) |
| 7dsolv/Konaet-UAGI-ARC3-Research-Source | [link](https://huggingface.co/7dsolv/Konaet-UAGI-ARC3-Research-Source) | code | Konaet RCL | **Honest negative result:** 25/25 wins with memorised teacher routes, but **0/25 fresh and 0/25 in a Kaggle neural notebook** | 2 (downloaded) |
| EduDevCommons/ARC_AGI_Wayfinder_Agent | [link](https://huggingface.co/EduDevCommons/ARC_AGI_Wayfinder_Agent) | code | — | Python heuristic agent | 1 (downloaded) |
| arcprize/trm_arc_prize_verification | [link](https://huggingface.co/arcprize/trm_arc_prize_verification) | 4.3 GB | MIT | TRM checkpoints for ARC-AGI-1/2 verification | 1 |
| Divay32/arc-agi-3, samobcoder/arc3-models (29.7 GB ONNX), martineux/arc3 (2025), ARC4NUM/Qwen3.8-Flash-Next-Uncensored-MLX-4bit (107 GB, Apple MLX) | — | — | — | No cards or irrelevant; MLX does not run on Kaggle | 1 |
| XenderYang/harness-duck-models | — | 9 MB | Apache-2.0 | **False positive:** "MicroDuck" robot RL ONNX policies, unrelated to the Tufa Duck | 0 |

HF spaces:
- `federicobravetti/arc-agi-3` (+ `-feat-ls20Agent`, `-feat-m0r0`): ARC-AGI-3-Agents forks with game-specific agents.
- `vandazia/arc-agi-3-lab`
- `iamseungpil/agentness-arena`

All of these were downloaded to `SCRATCH/hf/public/`. Relevance 1–2.

---

## 4. GitHub (cloned with `--depth 1` into `SCRATCH/github/<owner>__<repo>`)

GitHub REST search through `curl` is blocked by the session proxy, which allows repo-scoped endpoints only. Searches were done with the GitHub MCP search tool, and public repos were cloned with git. 43 repos were cloned; 42 succeeded.

### 4a. Official (arcprize)

| Repo | Size | Contents | Rel. |
|---|---|---|---|
| [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) (329★) | 1.2 MB | Official agent template, recordings format and swarm runner | 4 |
| [arcprize/ARC-AGI](https://github.com/arcprize/ARC-AGI) (80★) | 0.9 MB | The `arc_agi` toolkit, source of the `arc-agi` wheel (local `Arcade`, scorecards) | 4 |
| [arcprize/ARCEngine](https://github.com/arcprize/ARCEngine) | 0.7 MB | Game engine (`ARCBaseGame`), needed to write new games | 4 |
| [arcprize/arc-agi-3-benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking) | 1.7 MB | Official frontier-LLM benchmarking harness | 3 |
| [arcprize/ARC-AGI-3-Kaggle-Starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter) | 0.3 MB | Local dev → Kaggle push starter (one command) | 3 |
| [arcprize/docs](https://github.com/arcprize/docs) | 1.4 MB | Mintlify docs source (ARC-AGI-3 API, scoring, recordings) | 3 |
| [arcprize/ARC-AGI-Community-Leaderboard](https://github.com/arcprize/ARC-AGI-Community-Leaderboard) | 0.5 MB | Community submissions with claimed scores and links | 3 |

No official replay or recording *dataset* repo exists on GitHub. The replays are on three.arcprize.org, and `SmartManoj/ARC-AGI-3-Replay` is a viewer for them.

### 4b. Top Kaggle teams and Kaggle-constraint work

| Repo | Size | What it contains | Rel. |
|---|---|---|---|
| **[Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness)** (88★, #1 team, June 30 milestone winner) | **3.0 GB** | `ARC3-Inference/` (the Duck: tool-using solver, prompts, viewer), `tufa-arc-agi-framework/` (TAAF), the Kaggle notebook, and **`example-run/`: a complete run of 25 games × 20 passes with Qwen3.6-27B-FP8**, containing per-pass viewer data, prompts, movies and `score.json`. Mean 1.60 per pass. Best single passes: ft09 25.3, vc33 16.4, ar25/re86 8.3. tr87 and wa30 scored 0 in every pass. This is the upstream of our notebook. | **5** |
| **[tonghuikang/arc3](https://github.com/tonghuikang/arc3)** (#3, 20.53) | 39 MB | "Public version". `autoresearch/`: **imitation-learning per-coordinate policy**. Features are 384 binary values per cell: 4×16 colour groups, 6 previous frames with action and change masks, first and last frame of the previous level, and x/y one-hot. Heads cover action (6), colour (16) and click map (16×64×64). The baseline has hidden size 64 and about 0.4M params. There are 109 architecture variants (ConvNeXt, SE-ResNet, UNet, ViT…). Training uses `only_won=True` on the **"variation" split, i.e. generated variants of games**, about 47k samples, with validation on the base games. Metrics for 106 runs are included. The model class and the training repo are not published. Also contains `leaderboard/` (a Modal Kaggle-LB tracker). | **5** |
| [tonghuikang/inference](https://github.com/tonghuikang/inference) | 1.4 MB | "Stripped down vLLM implementation" (2026-04/07) | 3 |
| [da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution) (Daniel Franzen, #5) | empty | Created 2026-09-25, no commits yet. **Watch it.** | 4 (future) |
| [NVIDIA/dream-team](https://github.com/NVIDIA/dream-team) (probably NVARC) | 29 MB | DreamTeam: 6 agents (observer, simulator, 2 explorers, critic, leader) around an executable world model (paper arXiv 2605.09650). `beam/` agent framework. `game_creator/` plus **25 generated games in `environment_files_generated/`, each with a verified solution**. | 4 |
| [DriesSmit/ARC3-solution](https://github.com/DriesSmit/ARC3-solution) (StochasticGoose, **1st** in the preview, Tufa) | 0.3 MB | CNN predicting which ACTION1–6 or click changes the frame, with hierarchical action→coordinate sampling, a hash-deduplicated buffer and a reset on each new level | 4 |
| [dolphin-in-a-coma/arc-agi-3-just-explore](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore) (**3rd** in the preview) | 2.4 MB | Exploration-only graph search (arXiv 2512.24156). Solved 12/25 private levels, or a median of 17 after a bug fix. | 4 |
| [U4AR/qwen38-arc3-rl](https://github.com/U4AR/qwen38-arc3-rl) | 2.6 MB | LoRA RL of Qwen3.8-27B inside the Duck on 2×H100, starting from Terse-Coder. Reward = levels solved + a bonus for fewer tokens. Duck patches: `watch_video` tool, verbatim history for prefix-cache hits, an episode token budget, and an image-aware context estimate. | 4 |
| [sonpham-org/arc-3](https://github.com/sonpham-org/arc-3) | 55 MB | Instrumented Tufa fork with harness variants, run logs, a reproduction matrix and a GCP spot kit. Site arc3.sonpham.net has a catalog of **927 playable games** (25 official, 571 AI-generated, 252 redbluepill, 50 reviewed "arena"). | 4 |
| [samrishtt/arc-agi-3-kaggle-competition](https://github.com/samrishtt/arc-agi-3-kaggle-competition) | 8.8 MB | Duck v12 notebook plus Forge v20 Go-Explore state-graph agent (LB 1.33), with volatile status-bar masking | 2 |
| [leejianrong/solve-arc-agi-3](https://github.com/leejianrong/solve-arc-agi-3) | 1.9 MB | Clean harness for Qwen3.6-27B-FP8 on vLLM under Kaggle constraints, with ADRs | 2 |
| [Shih-Yu-Yeh/arc3-dev-agi-journal](https://github.com/Shih-Yu-Yeh/arc3-dev-agi-journal) | 1.8 MB | 21 Duck + Qwen3.8-27B-FP8 submissions with local-vs-LB notes. **LB 2.56 came from `MULTIMODAL_UPSCALE` 4→8 (512 px render)**; later module grafts regressed. | 3 |
| [54xkeee/duck-harness-glm53-flash-experiments](https://github.com/54xkeee/duck-harness-glm53-flash-experiments) | 52 MB | GLM-5.3-Flash + Duck with a predict-before-act loop on ls20: 0/16 full solves (in Chinese) | 2 |
| [Mininglamp-AI/polyphony-arc-3](https://github.com/Mininglamp-AI/polyphony-arc-3) | 7.2 MB | Training-free agent for **small open models**. The agent grows a Python "heuristic system" (policy, planner, verifier) in a per-game workspace. | 3 |

### 4c. Frontier-LLM harnesses on the public set (not Kaggle-deployable, but a source of ideas and traces)

| Repo | Size | Claim / contents | Rel. |
|---|---|---|---|
| [ryanbbrown/Retrodict](https://github.com/ryanbbrown/Retrodict) | 4.1 MB | 99.86% RHAE on all 183 levels for $654. Frame log file + code analysis, expectation-checked plans, context resets at 150k tokens, stuck-level escalation after 300 actions. | 3 |
| [Cveinnt/kepler](https://github.com/Cveinnt/kepler) | 4.4 MB | Code for the Kepler traces above: executable `world_model.py`, certified against history, guarded execution. Integrity and incident notes. | 3 |
| [NIMI-research/Tycho](https://github.com/NIMI-research/Tycho) | 2.7 MB | Rendered deterministic Moore machines | 3 |
| [alexisfox7/PRO-LONG](https://github.com/alexisfox7/PRO-LONG) (457★) | 354 MB | 97.4% (arXiv 2607.20064). Programmatic memory: one append-only log searched with code. `research/arc-agi-3/` holds 344 MB of run data. | 3 |
| [pbshgthm/arc-skill](https://github.com/pbshgthm/arc-skill) | 0.5 MB | Claude Code skill: "say what an action will do before you spend it". Opus 5: 100.00 in 7,645 actions. | 3 |
| [feng-rrRay/Continual-Harness-ARC-AGI-3](https://github.com/feng-rrRay/Continual-Harness-ARC-AGI-3), [studio-dots-ai/TELL](https://github.com/studio-dots-ai/TELL) (43.9%, no training), [synchopate/arc-agi-crystalline](https://github.com/synchopate/arc-agi-crystalline) (97.69%), [agno-agi/arc-agi-arcade](https://github.com/agno-agi/arc-agi-arcade), [PrimeIntellect-ai/arc-agi-3-prime-agent](https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent), [Alexyskoutnev/TWIN-ARC-AGI-3](https://github.com/Alexyskoutnev/TWIN-ARC-AGI-3), [ThariqS/ARC-AGI-3-ClaudeCode-SDK](https://github.com/ThariqS/ARC-AGI-3-ClaudeCode-SDK) | 0.3–21 MB | Various memory, world-model and self-improving harness designs | 2 |
| [g-baskin/occam](https://github.com/g-baskin/occam) | 1.5 MB | "Pure algorithmic, no LLM, **57.6% RHAE**, 17/25 games, 27 min CPU", self-reported. **Caveats:**

- Its last fallback is "deepcopy BFS" (`copy.deepcopy` of the env). That is not available through the Kaggle gateway.
- Its main search is reset-replay BFS (from Rudakov et al., the 3rd-place just-explore author) with a per-level budget of up to about 1M actions. Whether that can score depends on how the Kaggle scorer counts actions across RESETs and on the 9-hour limit. Verify this before trusting 57.6%.

A Team AIRIS notebook `arry-occam-solver` exists. Its probe, counter-masking, combo-search and navigation modules are reusable. | 3 |
| [jinbowang1/arc-prize-2026](https://github.com/jinbowang1/arc-prize-2026) | 670 MB | Perfect 100 on ls20, tr87 and ft09, with **documented failure boundaries**. The rules were induced by a human. | 2 |
| [Ephemeral6/harness-v04-site](https://github.com/Ephemeral6/harness-v04-site) | 196 MB | Static site: rules, Lean proofs and replays for all 25 games | 3 (mechanics reference) |
| [TrainLoop/arc-3-harnesses](https://github.com/TrainLoop/arc-3-harnesses) | 4 MB | Claude-written per-level solvers produced by reading game source (`dataset/games`) | 2 |
| [Erikiss/Rebuild-schema-harness…](https://github.com/Erikiss/Rebuild-schema-harness-by-impossible-research) | 0.7 MB | Re-implementation of the Schema harness | 2 |
| [ARA-Labs/ara-ls20](https://github.com/ARA-Labs/ara-ls20) | 1.3 MB | ls20 world-model artifact (same source as the ARA Labs HF data) | 2 |
| [SmartManoj/ARC-AGI-3-Replay](https://github.com/SmartManoj/ARC-AGI-3-Replay), [82deutschmark/arc-explainer](https://github.com/82deutschmark/arc-explainer) (533 MB, bundles the "arena" games) | — | Replay viewer and explainer site | 2 |

### 4d. Extra game environments (important for learned policies and honest held-out validation)

| Source | Count | Path |
|---|---|---|
| [theredbluepill/arc-interactive](https://github.com/theredbluepill/arc-interactive) | **253** community ARCEngine games (Sokoban, flood-fill, memory match, rule switcher…), plus skills to create and check games | `SCRATCH/github/theredbluepill__arc-interactive/` |
| NVIDIA dream-team `environment_files_generated/` | **25** generated games, each with a solution verified to replay to WIN | `SCRATCH/github/NVIDIA__dream-team/environment_files_generated/` |
| sonpham catalog (arc3.sonpham.net) | 927 in total (571 AI-generated, 50 reviewed "arena") | site only; the repo holds harness and run data |
| ritwika96 procedural-games-v2 | 330 games, 2,640 levels (the author says they are too easy) | HF, not downloaded |
| TrainLoop dataset | per-level solvers for the public games | `SCRATCH/github/TrainLoop__arc-3-harnesses/dataset/` |

### Not found

- No repo for "Blind Squirrel" (2nd in the preview). Searches for "StochasticGoose" return nothing; the code is `DriesSmit/ARC3-solution`.
- No public GitHub account under `jeroencottaar`.
- No public code from Lord Han Solo, Yi-Chia Chen, "the last dance" or Third Intelligence.
- The HF author `third-intelligence` only hosts an llm-jp model.

---

## 5. Most useful for us (ranked)

1. **Team AIRIS 4.68 bundle** (`SCRATCH/hf/user/checkmpobbyacckraggle/arc-prize-2026-arc-agi-3/submissions/best/sub_56568844__pub-4.68/`). It is already a better Duck + Qwen3.8-Flash-Next-NVFP4 variant than our 3.85: the r9 gate/budget/driver and r10 "rescue" patches. Diff it against the AGENTFIX notebook first. This is the cheapest gain available.
2. **Tufa `duck-harness` `example-run/`** (500 game runs with Qwen3.6-27B-FP8). It gives a per-game distribution of which games the Duck can score at all. ft09, vc33, ar25, re86, r11l and sb26 have high per-pass ceilings; tr87 and wa30 are always 0. Use it to find variance to exploit: best-of-N passes, time reallocation, early stopping on hopeless games. It is also imitation or critique data in exactly our prompt format.
3. **Tong Hui Kang's imitation-policy recipe** (`tonghuikang/arc3/autoresearch`). The #3 team shows that a **tiny CNN/MLP policy trained by imitation on won trajectories from game variants** is competitive. That model fits easily in our budget and could run alongside the LLM as an action prior or a cheap first pass. Training data for our own version:
   - human sessions (magic-sword, 340 sessions);
   - winning frontier traces (Kepler, Schema, ARA replay scripts);
   - 14.6M transitions (wm-traces);
   - extra games (253 redbluepill + 25 NVIDIA) to generate variants and keep a held-out validation set;
   - the only-won filter and the 384-feature layout are documented in `model.py`.
4. **StochasticGoose (DriesSmit) and just-explore (3rd).** These are proven LLM-free exploration baselines: a frame-change CNN and graph exploration. They are a strong fallback or a hybrid "explore first, LLM second" mode for games where the Duck scores 0.
5. **Mechanics knowledge base for the 25 public games.** Sources are Kepler `runs.jsonl` `notes_md`, ARA `solutions/GAME.md` with replays, Ephemeral6 rules and guanning/Schema traces. Private games are different, but these show which *mechanic classes* recur: navigation, push, click/flood-fill, toggle, target-match. Use them to write better system-prompt priors and tool heuristics, and to compute human baselines (`baseline_actions.csv`).
6. **Serving alternatives worth a single A/B:**
   - `nvidia/Qwen3.8-Flash-Next-NVFP4`, the official quant with published parity numbers;
   - `Shockem/Qwen3.8-27b-Terse-Coder-NVFP4` (22 GB dense, about 10× shorter chain-of-thought), which would let us run many more actions or passes in 9 hours;
   - U4AR's Duck patches (verbatim history for prefix-cache hits, image-aware context estimate, episode token budget) are low-risk throughput wins for our fork.
7. **Watch list:**
   - `da-fr/arc-agi-3-solution` (Daniel Franzen) is empty now, so poll it;
   - `Hisernberg/arc-agi-3-campaign` (the user's own private repo) needs `add_repo` access to be read from this session.

Skip or low value: the ericmao and naestro LoRAs (no evidence); Occam's claimed 57.6% (depends on environment cloning); HHazard, Manusagents and JBrightman duplicates; ARC-AGI-1/2 datasets; and the unrelated projects in the user's account.
