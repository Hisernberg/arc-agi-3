# 21 - ARC-AGI-3 toolkit internals, offline API, state cloning and the exact scoring formula

Sources read: `arc_agi` 0.9.8 (`base.py`, `local_wrapper.py`, `wrapper.py`, `scorecard.py`, `api.py`,
`server.py`, `remote_wrapper.py`) and `arcengine` 0.9.3 (`base_game.py`, `enums.py`, `camera.py`,
`level.py`, `sprites.py`), installed offline from `SCRATCH/kaggle/comp/arc_agi_3_wheels` into
`SCRATCH/venv` (python 3.12). Every claim below was checked by running code
(harness: `agent/arcenv.py`; scratch scripts in `SCRATCH/`).

## TL;DR

- **Score** = mean over games of a per-game score. Per game: every level `i` (1-indexed) gets
  `s_i = min(115, 100 * (baseline_i / actions_i)^2)` if completed, else 0; game score =
  `sum(i * s_i) / sum(i)` over all levels, then capped at `100 * sum(i for completed levels) / sum(i)`.
  Level `i` counts every action from the moment level `i` started until it was completed, including
  failed attempts and level RESETs. Late levels weigh most: the last of 7 levels is worth 25% of the game.
- **Offline play works fully on CPU** with just the wheels + `environment_files`. One step takes about 0.3-2 ms.
- **State cloning works.** A plain `copy.deepcopy(game)` is exact for 23/25 games. bp35 and lf52
  key their undo tables by the builtin `id()` of live objects. For those two, deepcopy diverges
  (bp35) or blows up memory until the process is OOM-killed (lf52: 11.5 GB). `arcenv.deepcopy_remap_ids`
  rewrites stale ids and makes all 25 games clone-exact: tested over 200 random actions with
  snapshots every 25 steps, comparing state, levels, the settled frame and the animation length.
  A clone takes about 1-11 ms. All
  games are deterministic under replay. The only randomness is lf52's `np.random.shuffle` dissolve
  animation and tr87's fixed-seed `random.Random`.
- **The online API works** with our key. `GET /api/games` lists **exactly the same 25 games with the same
  versions and baselines** as the local metadata. No extra games are exposed. The probe used 1 RESET
  and 1 ACTION1 on ls20, then closed the scorecard.

## 1. Running offline

```python
from arc_agi import Arcade, OperationMode
arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=".../environment_files")
env = arc.make("ls20")            # LocalEnvironmentWrapper; performs the initial RESET
fd = env.step(GameAction.ACTION1)  # FrameDataRaw
sc = arc.get_scorecard()           # EnvironmentScorecard (computed with the local ScorecardManager)
```

`LocalEnvironmentWrapper` finds `<local_dir>/<class_name.lower()>.py`, `exec`s it into a fresh module
and instantiates class `game_id[0].upper()+game_id[1:4]` (e.g. `Ls20`). The class is cached per process.
Every step calls `game.perform_action(ActionInput(id, data), raw=True)`, stamps `guid`/`game_id`
on the result and feeds it to the scorecard (`_set_last_response`). `OPERATION_MODE` from the environment overrides
the constructor, and `COMPETITION` always wins. `NORMAL` mode downloads games from the API. `ONLINE` and
`COMPETITION` use `RemoteEnvironmentWrapper`, which does `POST {base}/api/cmd/{RESET|ACTIONn}` with
`{game_id, guid, x, y}`.

Our harness `agent/arcenv.py` (`ArcEnv(game_id)`) does the same without Arcade/Flask. It adds
`clone()`, the Kaggle competition-mode RESET semantics, exact scoring through the official
`Scorecard`/`EnvironmentScorecard` classes, `valid_actions()` (engine-internal meaningful clicks) and
ASCII rendering. CLI: `python agent/arcenv.py list|show <g> [acts]|test|online`.

## 2. Actions

| id | name | payload | convention in the 25 games |
|---|---|---|---|
| 0 | RESET | - | level reset, or full reset (see section 4) |
| 1-4 | ACTION1-4 | - | up / down / left / right (every keyboard game; sp80 remaps them on rotated levels) |
| 5 | ACTION5 | - | "interact / confirm / cycle selection / run" (game specific) |
| 6 | ACTION6 | `{"x":0..63,"y":0..63}` | click at **display** pixel (x = column, y = row) |
| 7 | ACTION7 | - | undo (ar25, bp35, lf52, sb26, sk48, su15) |

- `available_actions` is a static per-game list, returned with every frame. It is the only affordance
  hint. `tags` in metadata (`keyboard`/`click`/`keyboard_click`) are coarse, and tu93 is tagged
  keyboard_click while it only exposes 1-4.
- ACTION6 coordinates are display coordinates. Games map them with `camera.display_to_grid(x, y)`:
  `scale = min(64//w, 64//h)`, letterbox padding `(64 - w*scale)//2`, and `grid = (x - pad)//scale + camera.x`.
  For a 16x16 camera, one grid cell is a 4x4 pixel block. A click on the letterbox returns `None`,
  a no-op that is still counted.
- Pydantic validates `x`,`y` to 0..63 (`ComplexAction`). Actions that the game ignores (bumping a wall,
  ACTION7 with an empty stack, clicking empty space) **still cost one action**.
- `ActionInput.reasoning` is an opaque blob of at most 16 KB, echoed back and ignored by games.

## 3. Frames and state

- `FrameDataRaw.frame` is a **list** of `np.ndarray (64, 64) int8`, one per engine sub-step. The engine loops
  `step(); render()` until the game calls `complete_action()`, with a hard limit of 1000 frames. Most actions
  return 1 frame. Animations return more: cd82 ACTION5 15, g50t moves 7, sc25 first action 22 (a level-1
  tutorial demo), sp80 ACTION5 22, sb26 ACTION5 42. **The last frame is the settled state.**
  Over REST, `FrameData.frame` is `list[list[list[int]]]`.
- Colors 0-15 (`arc_agi/rendering.py`): 0 white, 1 off-white, 2 light grey, 3 grey, 4 off-black,
  5 black, 6 magenta, 7 pink, 8 red, 9 blue, 10 light blue, 11 yellow, 12 orange, 13 maroon,
  14 green, 15 purple. Each game picks its own background/letterbox.
- Rendering: sprites are drawn by layer, and pixel value -1 is transparent. The camera viewport (often 8x8 to 64x64)
  is **upscaled by an integer factor** and letterboxed into 64x64. Then `RenderableUserDisplay`
  interfaces draw HUDs directly into the 64x64 frame: step-budget bars, lives, and so on.
- `state`: `NOT_PLAYED` before the first reset, then `NOT_FINISHED`, `WIN` (last level done) or `GAME_OVER`.
  After `GAME_OVER` or `WIN` any non-RESET action returns an **empty frame list**. Over REST this is a
  400 `GAME_NOT_STARTED_ERROR`. It is not counted, and the only way on is RESET.
- `levels_completed` = `game._score` (incremented by `next_level()`). `win_levels` = number of levels.
  The level index is not exposed, but it equals `levels_completed` while playing.
- `full_reset` is set on the raw frame. The REST server drops it (always `False` over HTTP).

## 4. RESET semantics (important)

`ARCBaseGame.handle_reset()`:
```python
if os.getenv("ONLY_RESET_LEVELS") == "true" and self._state != GameState.WIN:
    self.level_reset()
elif self._action_count == 0 or self._state == GameState.WIN:
    self.full_reset()          # back to level 1, levels_completed = 0
else:
    self.level_reset()         # restart current level
```
`_action_count` is zeroed by `set_level()`, so it is 0 right after a level transition **and right
after a level reset**. Consequences:

| situation | local / online NORMAL | Kaggle gateway (`RestAPI`, competition_mode) |
|---|---|---|
| RESET after >= 1 action in level | level reset, **+1 action** | same |
| RESET with 0 actions in the level (just entered level k, or RESET twice) | **FULL reset to level 1**, new "play" on the scorecard | swallowed (no-op) but still **+1 action** (the server re-feeds the last frame to the scorecard) |
| RESET after WIN | full reset | full reset |

In competition mode the server also refuses a second scorecard, refuses to create a second
environment per game (you must keep the same `guid`), hides the scorecard until close, and on close
instantiates every unplayed game so that it scores 0. Verified with `ArcEnv(..., competition=True/False)`.

## 5. Scoring: exact code

`arc_agi/scorecard.py`, `Card.set_levels_completed` records `(levels_completed, cumulative_actions)`
each time `levels_completed` changes. `Scorecard.update_scorecard` counts: ACTION1-7 gives `+1 action`.
RESET with `full_reset` gives a new play (counters restart). RESET without it gives `+1 action, +1 reset`.

Per level (`EnvironmentScoreCalculator.add_level`):
```python
if completed:
    score = ((baseline_actions / actions_taken) ** 2) * 100
    score = min(score, 115.0)          # per-level cap 115
else:
    score = 0.0                        # but the level still enters the weighted mean
```
Per game (`to_score`), with `weight = level_index` (1-based):
```python
total_score  = sum(level_scores[i] * weight_i)
total_weight = sum(weight_i)                              # ALL levels of the game
max_weights  = sum(weight_i for levels with score > 0)
score = min(total_score / total_weight, max_weights / total_weight * 100)
```
Level actions (`_calculate_score`): `level_actions[i] = actions_by_level[i][1] - actions_by_level[i-1][1]`,
i.e. cumulative actions at completion of level i minus at completion of level i-1. Actions after the
last completion go to the uncompleted level (score 0).
Across runs (plays) of one game: `EnvironmentScoreList.score = max(run.score)`. The total is
`EnvironmentScorecard.score = mean(env.score for env in environments)`. In competition mode that covers
all games (unplayed = 0). This matches the published RHAE methodology (docs_methodology: "upper median
human" baseline, square, cap 1.15, level-index weights, completion cap, average over games).
`arcenv.score_from_level_actions()` is a closed-form re-implementation. It equals
`EnvironmentScoreCalculator` on 200 random synthetic plays, and `ArcEnv.score()` calls the official classes.

Worked check (vc33, 7 levels, baselines 7 18 44 61 131 34 152): BFS solves level 1 in 3 clicks, so
`s_1 = min(115, 100*(7/3)^2) = 115`. The game score is `min(115*1/28, 100*1/28) = 3.571`, which the
official scorecard reproduces.

### What this implies

- Efficiency is squared: 1.1x baseline actions gives 82.6, 1.5x gives 44.4, 2x gives 25, 3x gives 11.1, 5x gives 4, 10x gives 1.
  **Completing a level at any cost is worth far more than not completing it** (the weight is paid
  anyway), but wasted exploration is punished quadratically.
- Beating the baseline pays up to 115 on that level, and it only helps to offset inefficient levels. When all
  levels are done the game is capped at 100. Baselines are "upper-median first-time human" counts
  and contain human exploration. Offline BFS optima for level 1 are usually far below them:
  cd82 5 vs 55, ft09 4 vs 43, sp80 4 vs 39, r11l 3 vs 22, sk48 14 vs 61, dc22 20 vs 59, lf52 8 vs 32,
  ka59 11 vs 28, lp85 5 vs 17. Over the 15 games solved, the optimum is 151 vs 486 human actions (31%).
  Full list in `20_game_mechanics.md`. Knowing the rules is worth the 115 cap.
- Weights: level k of an n-level game weighs `2k/(n(n+1))`. The first level alone caps the game at
  1.8-4.8%. Completing the first 4 levels at baseline efficiency caps it at 18-48% (table below).
  Getting deep into games matters more than being perfect on early levels.
- Every retry and RESET is charged to the current level. A GAME_OVER costs everything spent in the
  level plus the RESET, and the level restarts from scratch with the counter still running.

### Per-game weight table (cap if the first k levels are completed at >= baseline efficiency)

| game | tags | levels | sum of weights | sum of baselines | baselines per level | cap k=1 / 2 / 3 / 4 (%) |
|---|---|---|---|---|---|---|
| ar25 | keyboard_click | 8 | 36 | 748 | 32 50 75 37 89 159 233 73 | 2.8 / 8.3 / 16.7 / 27.8 |
| bp35 | keyboard_click | 9 | 45 | 651 | 21 48 44 38 33 87 86 131 163 | 2.2 / 6.7 / 13.3 / 22.2 |
| cd82 | keyboard_click | 6 | 21 | 171 | 55 8 41 21 23 23 | 4.8 / 14.3 / 28.6 / 47.6 |
| cn04 | keyboard_click | 6 | 21 | 789 | 29 54 85 300 208 113 | 4.8 / 14.3 / 28.6 / 47.6 |
| dc22 | keyboard_click | 6 | 21 | 1228 | 59 102 67 98 324 578 | 4.8 / 14.3 / 28.6 / 47.6 |
| ft09 | - | 6 | 21 | 208 | 43 12 23 28 65 37 | 4.8 / 14.3 / 28.6 / 47.6 |
| g50t | keyboard | 7 | 28 | 879 | 78 175 179 230 96 54 67 | 3.6 / 10.7 / 21.4 / 35.7 |
| ka59 | keyboard_click | 7 | 28 | 730 | 28 109 51 51 33 132 326 | 3.6 / 10.7 / 21.4 / 35.7 |
| lf52 | click | 10 | 55 | 1339 | 32 81 60 71 205 148 244 109 164 225 | 1.8 / 5.5 / 10.9 / 18.2 |
| lp85 | click | 8 | 36 | 388 | 17 38 31 16 41 60 26 159 | 2.8 / 8.3 / 16.7 / 27.8 |
| ls20 | keyboard | 7 | 28 | 776 | 22 123 73 84 96 192 186 | 3.6 / 10.7 / 21.4 / 35.7 |
| m0r0 | keyboard_click | 6 | 21 | 1107 | 30 111 203 26 500 237 | 4.8 / 14.3 / 28.6 / 47.6 |
| r11l | click | 6 | 21 | 233 | 22 33 51 26 52 49 | 4.8 / 14.3 / 28.6 / 47.6 |
| re86 | keyboard_click | 8 | 36 | 1255 | 26 42 86 108 189 139 424 241 | 2.8 / 8.3 / 16.7 / 27.8 |
| s5i5 | click | 8 | 36 | 638 | 20 89 106 54 162 38 86 83 | 2.8 / 8.3 / 16.7 / 27.8 |
| sb26 | keyboard_click | 8 | 36 | 213 | 18 28 18 19 31 23 58 18 | 2.8 / 8.3 / 16.7 / 27.8 |
| sc25 | keyboard_click | 6 | 21 | 350 | 36 6 32 83 143 50 | 4.8 / 14.3 / 28.6 / 47.6 |
| sk48 | keyboard_click | 8 | 36 | 1070 | 61 177 101 103 230 181 125 92 | 2.8 / 8.3 / 16.7 / 27.8 |
| sp80 | keyboard_click | 6 | 21 | 518 | 39 58 25 148 96 152 | 4.8 / 14.3 / 28.6 / 47.6 |
| su15 | click | 9 | 45 | 361 | 22 42 26 115 36 31 8 40 41 | 2.2 / 6.7 / 13.3 / 22.2 |
| tn36 | click | 7 | 28 | 317 | 32 72 26 40 30 55 62 | 3.6 / 10.7 / 21.4 / 35.7 |
| tr87 | keyboard | 6 | 21 | 414 | 54 58 40 45 71 146 | 4.8 / 14.3 / 28.6 / 47.6 |
| tu93 | keyboard_click | 9 | 45 | 462 | 19 16 34 42 123 80 14 23 111 | 2.2 / 6.7 / 13.3 / 22.2 |
| vc33 | click | 7 | 28 | 447 | 7 18 44 61 131 34 152 | 3.6 / 10.7 / 21.4 / 35.7 |
| wa30 | keyboard | 9 | 45 | 1843 | 71 119 183 98 368 68 79 442 415 | 2.2 / 6.7 / 13.3 / 22.2 |

183 levels in total. The 25 games have 6-10 levels (mean 7.3).

## 6. State cloning / lookahead

- `copy.deepcopy(game)` gives an independent game. `ArcEnv.clone()` also copies the scorecard and
  counters, and shares the read-only `_clean_levels` templates (about 2x faster: 0.5-11 ms per clone,
  60 KB-1.2 MB per clone).
- Fidelity test (`python agent/arcenv.py test`): for every game, 200 random actions (auto-RESET on
  GAME_OVER) with snapshots every 25 actions. Each snapshot replays the suffix and must reproduce
  state, levels and the last frame, plus the animation length. A fresh instance must reproduce the whole run
  (determinism). **25/25 OK.**
- Pitfall: bp35 (`keunykhwkoi[id(obj)]`) and lf52 (`ainlnxnlazs[id(obj)]`) store `id()`-keyed
  undo/snapshot tables. With plain deepcopy the int keys point at the original objects: bp35's clone
  diverged at the 13th action, and lf52's clone replayed 16-frame animations, slowed down and ate
  memory until OOM. Fix: `deepcopy_remap_ids()` takes the deepcopy memo `old_id -> new_obj` and rewrites every int
  equal to an old id (dict keys and values, lists, tuples, namedtuples, sets, attributes, slots).
  `ArcEnv(clone_mode="auto")` uses it only for games whose source calls `id(` (bp35, lf52).
- Pickle does not work out of the box (exec'd module). `arcenv` registers the module as
  `sys.modules["arcgame_<id>"]`, so fork-based multiprocessing can share classes. Pickled snapshots
  would still need the id-remap for bp35/lf52.
- Randomness: lf52 uses the global `np.random.shuffle` only for a dissolve transition (visual),
  and tr87 seeds `random.Random` with fixed per-level seeds. Everything else is deterministic, so replaying an
  action list from RESET is also an exact (slower) way to restore a state.
- `game._get_valid_actions()` (ARC "graph builder" API) lists meaningful clicks per state and is
  handy for offline search (`ArcEnv.valid_actions()`). Caveat: bp35 and lf52 set a module global
  `GRAPH_BUILDER=True` inside it, which disables their undo recording. `arcenv` restores it.
  This API is **not** available through the Kaggle gateway.
- Speed: 0.3-2.2 ms per step on this CPU. A BFS with clones expands about 40-400 nodes/s per core (clone + step + hashing).

## 7. Online API (three.arcprize.org)

- `GET /api/games` (header `X-API-Key`) returns 200 with 25 entries `{game_id, title, tags, baseline_actions}`:
  `ar25-0c556536 ... wa30-ee6fef47`. These are identical to the local `metadata.json` (ids, versions,
  baselines, tags). **No hidden or extra games are exposed.** `GET /api/games/<id>` adds `default_fps`.
- Probe: `POST /api/scorecard/open` gives a `card_id`. `POST /api/cmd/RESET {game_id, card_id}` returns 1 frame of
  64x64, `guid`, `available_actions`, `win_levels`. `POST /api/cmd/ACTION1 {game_id, guid}` works.
  `GET /api/scorecard/<id>` and `POST /api/scorecard/close` return the same `EnvironmentScorecard` JSON as the local
  scorer (`level_actions`, `level_baseline_actions`, `level_scores`, `tags_scores`). The server runs the
  same scorecard code. Total cost: 2 actions.
- `/api/games/anonkey` hands out anonymous keys, which `Arcade` uses when `ARC_API_KEY` is empty.

## 8. Kaggle notes

- The notebook plays through a gateway with `OperationMode.COMPETITION` (`ARC_BASE_URL`). The gateway is
  presumably the same `RestAPI(competition_mode=True)` server code shipped in the wheel. If so, section 4 semantics apply: one scorecard, one guid per game,
  RESET at a level start is swallowed but charged, and unplayed games score 0.
- `ONLY_RESET_LEVELS=true` changes `handle_reset` in the process that runs the games. Setting it in the
  agent (client) process has no effect on a remote gateway.
- All frames of an action are returned. An agent that only looks at `frame[-1]` sees the settled
  state, and intermediate frames show the motion (useful for learning dynamics).
