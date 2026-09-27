# Shared context for all research agents (read first)

- Competition: Kaggle `arc-prize-2026-arc-agi-3` (ARC Prize 2026, ARC-AGI-3 track). Deadline 2026-11-02 23:59 UTC. Today 2026-09-27.
- User: Kaggle team rank ~242, best public LB 3.85. Target: 18-20+ (top-3 zone).
- Public LB top (2026-09-27): Tufa Labs 27.29, Lord Han Solo 20.80, Tong Hui Kang 20.53, Yi-Chia Chen 18.80,
  Daniel Franzen 16.68, NVARC3 16.07, "the last dance" 13.70, Third Intelligence 12.31, Matija L & Zhongwei W & Fususu 11.64, rellik13 9.96.
- Hardware on Kaggle: 1x RTX PRO 6000 (96 GB), no internet during submission, 9h notebook budget (32400 s), 1 submission/day.
- User's current notebook = fork of Tufa Labs "Duck harness" (June 30 milestone winner, jeroencottaar), Qwen3.8-Flash-Next NVFP4 on vLLM,
  with "AGENTFIX" runtime patches. Tufa writeup: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133
- Competition data (25 public games, python sources + arc_agi 0.9.8 / arcengine 0.9.3 wheels, ARC-AGI-3-Agents repo) is at:
  SCRATCH/kaggle/comp/  where SCRATCH=/tmp/claude-0/-home-user-arc-agi-3/839fb9f9-4d34-537a-a4fe-103d3cf7eb7f/scratchpad
- User's latest run results (25 public games): SCRATCH/results/ (score.json, benchmark.json, transcripts/, prompts/, artifacts/)
- Credentials: `source ~/.secrets/arc.env` gives GITHUB_TOKEN, HF_TOKEN (user Nabidnur), KAGGLE_API_TOKEN, ARC_API_KEY (three.arcprize.org).
  NEVER write tokens into any file in /home/user/arc-agi-3 or print them in docs.
- Repo (committed deliverables): /home/user/arc-agi-3 . Write markdown findings into /home/user/arc-agi-3/research/ ; large raw downloads into SCRATCH/.
- Do NOT run git commands; the orchestrator commits.
