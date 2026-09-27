# 20 - Mechanics catalog of the 25 public ARC-AGI-3 games (+ cross-game taxonomy)

Method: I read each game's source (`SCRATCH/kaggle/comp/environment_files/<g>/<ver>/<g>.py`,
obfuscated names but readable logic) and dumped every level (sprites, tags, level data, start frame):
`SCRATCH/catalog/<g>_levels.txt`. I probed every action from the level-1 start state on exact clones:
`SCRATCH/catalog/<g>_probe.txt`, summary `SCRATCH/probe_summary.txt`. I also ran a clone-based BFS for
the shortest level-1 solution (`SCRATCH/bfs_level1.py`, results `SCRATCH/bfs_l1*.jsonl`); every
solution found was re-verified by replay. The harness is `agent/arcenv.py` (see `21_env_and_scoring.md`).

Conventions: ACTION1/2/3/4 = up/down/left/right, ACTION5 = interact, ACTION6 = click(x, y), ACTION7 = undo.
"Budget" means a per-level action budget drawn as a HUD bar. Running out gives GAME_OVER, which needs a
RESET that is charged to the level. Colors: 0 white, 1 off-white, 2 lt-grey, 3 grey, 4 off-black,
5 black, 6 magenta, 7 pink, 8 red, 9 blue, 10 lt-blue, 11 yellow, 12 orange, 13 maroon, 14 green, 15 purple.
Items marked *(inferred)* come from code I skimmed rather than traced exactly.

## Summary table

"L1 opt" is the shortest level-1 solution found by BFS (verified), next to the human baseline.
"t/o" means BFS timed out: 240 s, frontier capped at 4000, so it is only a depth bound.

| game | tag / actions | lv | sum of baselines | genre (one line) | L1 opt / human |
|---|---|---|---|---|---|
| ar25 | kb+click 1-7 | 8 | 748 | move pieces and mirror axes so pieces + reflections cover targets | **15** / 32 |
| bp35 | kb+click 3,4,6,7 | 9 | 651 | side-view gravity platformer; click to break/toggle/flip gravity; reach gem | see note |
| cd82 | kb+click 1-6 | 6 | 171 | move paint bucket around canvas, pick color, pour halves/triangles to copy target | **5** / 55 |
| cn04 | kb+click 1-6 | 6 | 789 | jigsaw: select/move/rotate pieces until all connector pixels pair up | t/o d7 / 29 |
| dc22 | kb+click 1-4,6 | 6 | 1228 | walk on tiles to goal; buttons toggle bridges; crane carries bridge pieces | **20** / 59 |
| ft09 | click 6 | 6 | 208 | lights-out-like color cycling to satisfy neighbor-equality clues | **4** / 43 |
| g50t | kb 1-5 | 7 | 879 | Braid-like: rewind to spawn a ghost replaying your path; ghosts hold plates | see note |
| ka59 | kb+click 1-4,6 | 7 | 730 | multi-block Sokoban: pushed blocks slide; bombs; fill target frames | **11** / 28 |
| lf52 | click(+kb) 1-4,6,7 | 10 | 1339 | peg solitaire (jump & capture same color) on sliding platforms | see note |
| lp85 | click 6 | 8 | 388 | buttons rotate tiles around interlocking loops; put marked tiles on goals | **5** / 17 |
| ls20 | kb 1-4 | 7 | 776 | maze; walk over changers to set key shape/color/rotation; open doors | **13** / 22 |
| m0r0 | kb+click 1-6 | 6 | 1107 | mirrored twins move together (x-mirrored); make pairs meet | **15** / 30 |
| r11l | click 6 | 6 | 233 | bodies sit at centroid of their legs; move legs so bodies reach targets | **3** / 22 |
| re86 | kb+click 1-5 | 8 | 1255 | move outline shapes (cycle selection) to compose target picture; recolor wells | t/o d9 / 26 |
| s5i5 | click 6 | 8 | 638 | robot arms: sliders extend/shrink colored segments, buttons rotate; reach targets | **13** / 20 |
| sb26 | kb+click 5,6,7 | 8 | 213 | drag color tokens into program slots (with subroutine calls); run to match sequence | t/o d5 / 18 |
| sc25 | kb+click 1-4,6 | 6 | 350 | maze + 3x3 rune grid: draw spell patterns (fire / enlarge / teleport) to reach exit | see note |
| sk48 | kb+click 1-4,6,7 | 8 | 1070 | telescoping arms push colored blocks; reproduce reference color order | **14** / 61 |
| sp80 | kb+click 1-6 | 6 | 518 | place deflector bars, pour water, fill all cups, avoid forbidden zones | **4** / 39 |
| su15 | click 6,7 | 9 | 361 | click = vacuum pulse pulling fruits; same-level fruits merge; deliver to goal | t/o d4 / 22 |
| tn36 | click 6 | 7 | 317 | toggle bits of instruction rows (opcodes), run program so piece reaches target | see note |
| tr87 | kb 1-4 | 6 | 414 | glyph translation by rewrite rules; cursor + cycle glyph variants | t/o d8 / 54 |
| tu93 | kb 1-4 (tag kb+click) | 9 | 462 | arrow through corridor maze to exit; sentries, patrols, mimics | **18** / 19 |
| vc33 | click 6 | 7 | 447 | communicating vessels: pumps move liquid between paired columns; floats to marks | **3** / 7 |
| wa30 | kb 1-5 | 9 | 1843 | warehouse: grab/drag boxes into zone; helper NPCs carry, thief NPCs steal | see note |

Notes: rows marked "see note" were still running or unsolved when this table was frozen; section 4 has
the final BFS results. In 15 of the 16 games solved so far, level 1 was solved in ≤ 50% of the human
baseline, the exception being tu93 (18 vs 19). Humans' baselines contain exploration, so an agent that
already understands a game can hit the 115 cap per level. The whole difficulty is *learning the game cheaply*.

---

## Per-game entries

### ar25 - Mirror / kaleidoscope cover (keyboard_click, actions 1-7)
- **Levels 8**, baselines 32 50 75 37 89 159 233 73. Budget (StepCounter) 64 64 128 128 128 320 320 320,
  drawn as an energy bar in column x=63 whose color changes every 64 steps.
- **Objects**: movable pieces (tag `0006`, black shapes), mirror axes (tag `0003`, light-blue lines:
  `0054` vertical axis reflects x, `0002` horizontal axis reflects y), fixed pieces (`0056`, not
  selectable), target cells (`0001`, dots). All reflections of every piece in all mirrors (compositions up to
  depth 12) are drawn in off-black.
- **Actions**: ACTION1-4 move the selected object 1 cell (a vertical mirror only moves left/right, a horizontal
  one only up/down). ACTION5 cycles the selection. ACTION6 click selects the object under the cursor
  (pieces preferred over mirrors). ACTION7 undo: restores positions and costs an action, but not budget.
  Moves and ACTION5 cost budget. Some pieces rotate 90° whenever their distance to their mirror
  changes (tags `0040`/`0044`). Some reflect only horizontally or only vertically.
- **Win**: every target cell is covered by a piece or by a reflection. **Lose**: budget exhausted.
- **Skill**: symmetry/reflection geometry, choosing which object to move.
- **Optimal policy**: solve for the piece offset and mirror offset analytically (targets = union of the
  piece's images under the mirror group), select by click (1 action), then move straight there. BFS L1 = 15 (human 32).

### bp35 - Gravity platformer with editable terrain (keyboard_click, 3,4,6,7)
- **Levels 9**, baselines 21 48 44 38 33 87 86 131 163. Custom engine (6-px cells, camera scrolls
  vertically). Budget bar on row 63: 64 actions (levels 1-6) or 128 (levels 7-9).
- **Actions**: ACTION3/4 walk left/right one cell. With no support the player falls under gravity until
  it lands, with a multi-frame animation. Walking into a wall plays a bump. ACTION6 clicks blocks:
  breakable (`qclfkhjnaac`) is destroyed; spawner (`etlsaqqtjvn`) grows blocks into its 4 empty neighbours;
  toggle (`yuuqpmlxorv`<->`oonshderxef`) switches solid/passable; gravity switch (`lrpkmzabbfa`)
  **flips gravity** (up/down). ACTION7 undo (engine snapshot stack). There is no ACTION1/2: vertical motion comes only from gravity.
- **Win**: land on / walk into the gem (`fjlzdjxhant`). **Lose**: land on spikes; in levels 1-3 a
  rising hazard climbs 1 cell every 2 actions and kills when level with the player; budget exhausted.
- **Skill**: physics (gravity/falling) prediction, terrain editing as tool use, scrolling world.
- **Optimal**: plan the fall path to the gem, then click only the blocks that must change (clicks are
  as expensive as steps).

### cd82 - Paint the canvas (keyboard_click, 1-6)
- **Levels 6**, baselines 55 8 41 21 23 23. Budget 100 actions per level (HUD).
- **Objects**: a 10x10 canvas, a target picture (the two diagonals are ignored in the comparison), a
  paint bucket sitting on one of 8 ring positions around the canvas (3x3 grid minus centre), and a color palette.
- **Actions**: ACTION1-4 move the bucket around the ring. ACTION6 on a palette swatch picks the color. ACTION5
  pours (15-frame animation): from a side position it fills the adjacent half (5 rows/cols), from a corner
  the triangular half. In some levels, clicking the small nozzle at a side position paints a 3x4 strip in the middle
  of that side. Later paint overwrites earlier paint.
- **Win**: canvas equals target (off-diagonal cells). **Lose**: 100 actions.
- **Skill**: layered painting order (reverse-engineer occlusion), color selection.
- **Optimal**: decompose the target into an ordered stack of half/triangle fills (bottom layer first),
  and minimize ring moves and color switches. BFS L1 = **5** (human 55!).

### cn04 - Jigsaw connector matching (keyboard_click, 1-6)
- **Levels 6**, baselines 29 54 85 300 208 113. Budget MaxSteps 75 100 125 125 150 200 (bar in row 0).
- **Objects**: pieces on a 20x20 grid with red (8) or maroon (13) connector pixels on their edges.
  The selected piece is drawn highlighted/white.
- **Actions**: ACTION6 selects a piece (clicking the selected piece again cycles stacked variants or
  deselects). ACTION1-4 move the selected piece 1 cell, clamped to the grid. ACTION5 rotates it 90°, or cycles
  through alternative shapes stacked at the same spot.
- **Rule**: a connector pixel is satisfied (turns grey 3) when exactly two pieces' connector pixels
  overlap in one cell. **Win**: every connector of every visible piece is satisfied. **Lose**: budget.
- **GreyMasking** (levels 3-6): unselected pieces are drawn grey, so their connectors are hidden until
  selected. This is partial observability plus memory.
- **Skill**: spatial assembly, rotation, working memory. **Optimal**: infer a global placement where all
  connectors pair up, then move each piece along its shortest path. BFS timed out at depth 7.

### dc22 - Tile walker with switches, bridges and a crane (keyboard_click, 1-4,6)
- **Levels 6**, baselines 59 102 67 98 324 578. Budget 128 192 192 192 512 1024, plus lives.
- **Actions**: ACTION1-4 move the player (`jfva`) one step, and only onto walkable tiles. ACTION6 clicks: letter
  switches (`buezna` + one-letter tag) swap groups of tiles between variants (bridges appear or
  vanish); a pressure plate (`piyqze`) enables a switch letter; an on-screen crane (`crzsjq`) has
  arrow buttons (`up/dowlja/lersnf/riidpd`) and a grab button (`grawwq`), moves on a 4x4 rail grid and carries an object or bridge piece.
- If a click removes the tile under the player, it falls: a 14-frame animation, state rolled back to
  before the click, and one life lost. **Win**: player on the goal tile (`goknoi`). **Lose**: lives or budget exhausted.
- **Skill**: causal discovery (which switch changes which tiles), tool use (crane), path planning.
  BFS L1 = **20** (human 59).

### ft09 - Color-constraint tiles, lights-out family (click only)
- **Levels 6**, baselines 43 12 23 28 65 37. Budget (clicks) 32 32 96 96 128 128.
- **Objects**: 3x3-px tiles at a 4-px pitch. Palette per level (`cwU`, 2-3 colors). Clue tiles (`bsT`): the centre
  color c plus 8 surrounding markers. A white marker means that neighbour must be color c, otherwise it must differ.
- **Actions**: ACTION6 on a normal tile (`Hkx`) cycles its color. On an `NTi` tile it cycles the neighbours
  given by the tile's own magenta pattern (lights-out neighbourhood). In level 1, clicking empty space flashes a hint.
- **Win**: all clues satisfied. **Lose**: budget.
- **Skill**: constraint satisfaction, linear algebra mod k (lights-out). **Optimal**: solve the
  system, click each tile the minimal number of times. BFS L1 = **4** (human 43).

### g50t - Time-loop ghosts, Braid-like (keyboard, 1-5)
- **Levels 7**, baselines 78 175 179 230 96 54 67. Timer: a bar sprite slides left 1 px every 2 actions,
  and the level is lost when it is gone.
- **Actions**: ACTION1-4 move the avatar one grid cell along corridors (7-frame animation; blocked moves
  do nothing but still cost). ACTION5 = **rewind**: the recorded path plays back in reverse (17 frames),
  the avatar returns to the spawn, and the recording becomes a **ghost** that repeats those moves in lockstep with
  your next moves. The number of ghost slots is shown by the indicators at the top-left.
- **Objects**: pressure plates / buttons that open doors and bridges while an avatar or ghost stands on them;
  auto-moving objects that step every turn *(inferred)*; the goal marker.
- **Win**: avatar reaches the goal. **Lose**: timer runs out, or the avatar is killed.
- **Skill**: temporal planning, cooperating with one's own past (multi-agent from single control), causal
  plate-to-door mapping. **Optimal**: plan ghost paths that hold the plates at the right moments, and keep the
  live run short.

### ka59 - Multi-block Sokoban with sliding and bombs (keyboard_click, 1-4,6)
- **Levels 7**, baselines 28 109 51 51 33 132 326. Budget 100 127 100 127 100 150 200 (bar on row 63).
- **Actions**: ACTION6 selects one of several player blocks (`0022`; the selected one has a white centre).
  ACTION1-4 move it one 3-px cell. Walls (`0029`) and gulches (`0015`) block the player. Walking into
  another movable object (player blocks, crates `0001`, bombs `0003`) pushes it, and pushed objects
  **keep sliding** over the next frames (about 5 cells, longer across gulches) until they hit a wall.
- Bombs: each player move fills one row of every bomb's fuse; when full it explodes and blasts
  neighbours 3 px per frame. Enemies (tag `Enemy`) chase the crate and you lose if they reach it.
- **Win**: every target frame `0010` contains a player block and every `0027` frame contains a crate.
- **Skill**: Sokoban planning with momentum, multi-agent selection. BFS L1 = **11** (human 28).

### lf52 - Peg solitaire on moving platforms (click + arrows, 1-4,6,7)
- **Levels 10**, baselines 32 81 60 71 205 148 244 109 164 225. Custom engine (6-px cells, scrolling).
  Hidden move counter: lose at 64 (level 1), 320 (levels 2-5) or 640 (levels 6-10). Undo adds to it.
- **Actions**: ACTION6 on a peg shows up to 4 jump targets 2 cells away. ACTION6 on a target makes the peg jump
  over the neighbour. If the jumped peg has the **same color** it is captured (removed). Landing needs a
  platform tile. ACTION1-4 slide all movable platform blocks one cell, carrying the pegs on them (the camera
  pans in some levels). ACTION7 undo. When the position is dead, a restart button appears bottom-left,
  and clicking it resets the level.
- **Win**: one peg of the color remains (two in levels 6-7; blue pegs are excluded in levels 8+).
- **Skill**: combinatorial search with irreversible moves, undo as a resource.

### lp85 - Interlocking loop rotations (click only)
- **Levels 8**, baselines 17 38 31 16 41 60 26 159. Budget 13 60 80 150 80 80 80 80. Note that level 1's
  budget of 13 is *below* the human 17: humans failed and reset.
- **Actions**: ACTION6 on a button `button_<group>_<L|R>` rotates every tile on that group's closed
  track one position left or right. Tracks intersect, so shared tiles move between loops, as in Hungarian rings / Rubik-style puzzles.
- **Win**: every marked tile `bghvgbtwcb` is on a `goal` cell and every `fdgmtkfrxl` is on a `goal-o` cell.
- **Skill**: permutation-group reasoning, commutators. BFS L1 = **5** (human 17).

### ls20 - Key-lock maze (keyboard, 1-4)
- **Levels 7**, baselines 22 123 73 84 96 192 186. Per-life budget 42 steps (yellow bar). Level 1 costs 1 per
  move, later levels 2 per move. **3 lives** (red squares); losing a life resets the position and key.
  Refill pickups restore the bar. Fog of war in later levels.
- **Objects**: 5x5 player block moving 5 px; walls; a key display (bottom-left) showing the current key
  shape, color and rotation; changer tiles: shape changer cycles 6 shapes, color changer cycles 4 colors,
  rotation changer rotates 90°. Changers can ride on tracks. Doors (`rjlbuycveu`) accept the key only if it
  matches their shape, color and rotation (a wrong key blocks). Bounce pads launch the player until a wall.
- **Win**: pass through every door. **Lose**: all 3 lives spent.
- **Skill**: attribute matching (key-lock), route planning through changers with a resource budget.
  BFS L1 = **13** (human 22).

### m0r0 - Mirrored twins (keyboard_click, 1-6)
- **Levels 6**, baselines 30 111 203 26 500 237. Budget 150 actions.
- **Actions**: ACTION1-4 move all twins at once: `...-leklkn` moves as pressed, `...-rivmdg` is mirrored
  horizontally, and in 4-twin levels the `boweok` pair is mirrored vertically. Each twin is blocked individually by walls.
  ACTION6 selects a movable block (`mosdlc`); arrows then move the block instead of the twins, and clicking
  elsewhere deselects. ACTION5 has no effect.
- A pair that lands on the same cell, or swaps through each other, **merges** and is done. Colored plates open the matching
  colored gates while a twin stands on them. Hazard tiles (checkerboard) flash and reset the twins and
  blocks to their start positions.
- **Win**: all twins merged. **Skill**: coordinated mirrored control, using walls to desynchronize.
  BFS L1 = **15** (human 30).

### r11l - Centroid bodies (click only)
- **Levels 6**, baselines 22 33 51 26 52 49. Budget 60 actions. 5 hazard strikes lose the level.
- **Mechanic**: each body (`roefwu-X`) is drawn at the **centroid of its legs** (`roefwulewcui-X`).
  ACTION6 on a leg selects it (it turns white). ACTION6 on empty space moves the selected leg there
  (animated; blocked by walls `wakneh`), and the body follows the new centroid. Paint blobs recolor some
  bodies, which must match their target colors.
- **Win**: every body overlaps its target (`flkdtg-X`), with matching colors where required. **Lose**: 5 strikes
  from bodies touching hazards (`defgjl`), or the budget.
- **Skill**: geometric reasoning (averages / inverse problems). BFS L1 = **3** (human 22).

### re86 - Stencil composition (keyboard_click, 1-5)
- **Levels 8**, baselines 26 42 86 108 189 139 424 241. Budget 100 100 200 200 250 200 300 400.
- **Actions**: ACTION5 selects the next shape. ACTION1-4 move the selected outline shape (cross, box) by 3 px.
  Paint wells (`0007`) recolor a shape that moves over them. Obstacles (`0003`) deform shapes pushed against
  them: boxes change aspect ratio, crosses shift their bars.
- **Win**: every colored pixel of the target picture (`0054`) is reproduced by the composite of the shapes.
- **Skill**: spatial composition, recoloring and deformation by tool contact. BFS timed out at depth 9.

### s5i5 - Articulated robot arms (click only)
- **Levels 8**, baselines 20 89 106 54 162 38 86 83. Budget 50 150 200 100 150 150 200 200.
- **Mechanic**: colored segments form kinematic chains (children follow parents). A slider bar per color:
  clicking its far half extends all segments of that color by one unit, the near half shrinks them.
  Colored rotate buttons turn that color's segments 90° about their base. Moves causing overlaps are reverted.
- **Win**: every target (`0087`) has an end effector (`0064`) on it. **Skill**: forward/inverse kinematics.
  BFS L1 = **13** (human 20).

### sb26 - Color-sequence programming with subroutines (keyboard_click, 5,6,7)
- **Levels 8**, baselines 18 28 18 19 31 23 58 18. Energy 64: each placement/swap and each run costs 1.
- **Objects**: target color sequence (top row); program frames with N slots; color tokens in the inventory
  and in slots; "call" tokens (`vgszefyyyp`) that jump into the frame of their color and return
  afterwards (subroutines).
- **Actions**: ACTION6 on a token then ACTION6 on a slot or token places or swaps it. ACTION5 **runs**: a cursor
  steps through the slots (long animation, about 42 frames) and each token must equal the next target color. A mismatch
  flashes and aborts. ACTION7 undoes the last placement.
- **Win**: whole sequence produced. **Skill**: program synthesis, recursion, sequence alignment.
  BFS timed out at depth 5 (branching is large).

### sc25 - Spell-casting maze (keyboard_click, 1-4,6)
- **Levels 6**, baselines 36 6 32 83 143 50. Mana budget 50 25 50 35 65 60 (moves and rune toggles both cost 1).
- **Actions**: ACTION1-4 walk 2 px (4 when enlarged). ACTION6 on the 3x3 rune grid toggles a cell. When the
  pattern equals a spell allowed in this level, it casts: vertical line = fire (burns obstacles ahead),
  diamond ring = enlarge x2, corner = swap/teleport *(inferred)*. Spell-book icons play a demo when
  clicked. Levels 1-3 auto-play the demo on the **first action** (22 frames, no effect otherwise).
  Pickups refund mana.
- **Win**: walk into the exit (it slides in). **Lose**: mana exceeded.
- **Skill**: learn symbolic commands from a demonstration, then use them to remove obstacles.

### sk48 - Telescoping arms, color order (keyboard_click, 1-4,6,7)
- **Levels 8**, baselines 61 177 101 103 230 181 125 92. Budget 196 moves.
- **Mechanic**: each base has an arm of segments. An arrow along the arm direction extends it one segment
  (pushing blocks ahead). The opposite arrow retracts it. A perpendicular arrow slides the base along its
  rail and drags or pushes blocks. ACTION6 selects which base you control. ACTION7 undo.
- **Win**: for every base pair, the colored blocks along the controlled arm equal, in order, the colors
  along the reference arm. **Skill**: ordering objects by pushing, extendable tools.
  BFS L1 = **14** (human 61).

### sp80 - Water routing (keyboard_click, 1-6)
- **Levels 6**, baselines 39 58 25 148 96 152. Step budget 30 45 100 120 100 120. The 4th failed pour loses.
- **Actions**: ACTION6 selects a deflector bar. ACTION1-4 move it 1 cell (it keeps a margin around cups
  and the top rows). ACTION5 **pours**: water falls from the sources, splits left/right around bars, and
  diagonal deflectors send it sideways (22-frame simulation).
- **Win**: every cup fills from above and no forbidden block gets wet. After a failure the water is cleared.
  **Rotated levels** (180°): the board is upside down and the arrows and click coordinates are remapped.
- **Skill**: physics prediction (flow), counterfactual planning, adapting to remapped controls.
  BFS L1 = **4** (human 39).

### su15 - Vacuum and merge (click 6, undo 7)
- **Levels 9**, baselines 22 42 26 115 36 31 8 40 41. Budget 32 or 48.
- **Mechanic**: a click in the play area emits a shrinking attraction circle. Everything inside is
  pulled toward the click point over about 5 frames. Two fruits of the same level that touch **merge**
  into the next level (Suika/2048). Enemies are also displaced; an enemy hitting a fruit downgrades it,
  and a level-0 fruit is destroyed.
- **Win**: goal zones hold exactly the required counts, e.g. `[2,1]` = one level-2 fruit.
  **Lose**: budget, or no fruit left. **Skill**: continuous aiming, merge arithmetic. BFS timed out at depth 4
  (256 click positions).

### tn36 - Program the piece with bit-encoded instructions (click only)
- **Levels 7**, baselines 32 72 26 40 30 55 62. Timer bar: 1 px per click (1 px per 2 clicks from level 6 on);
  lose when it runs out.
- **Mechanic**: a program panel of instruction rows. Clicking a cell toggles a bit, and a row's bits form an
  opcode: 1 left, 2 right, 3 down, 33 up, 5/6/7/16 rotate, 8/9 scale +/-, 10-13 double moves,
  14/15/63 recolor. Clicking the run button executes the rows in order with animation. Walls block. Blinking
  hazards (toggle every 3 steps) kill. Some levels show a read-only reference program.
- **Win**: the piece ends on the target outline with the same scale. **Skill**: program synthesis,
  decoding opcode semantics by experiment.

### tr87 - Glyph translation by rewrite rules (keyboard, 1-4)
- **Levels 6**, baselines 54 58 40 45 71 146. Budget 128 (256 on level 6).
- **Mechanic**: the top area lists rules `glyph-sequence -> glyph-sequence`. The bottom shows a source row and an answer
  row. ACTION3/4 move the cursor over answer glyphs. ACTION1/2 cycle the selected glyph through its
  variants. Initial answer glyphs are scrambled with a fixed seed.
- **Win**: answer = source rewritten by segmenting it into rule left-hand sides and concatenating the right-hand sides.
  Twists: `double_translation` (chained rules), `alter_rules` (you edit the rules instead),
  `tree_translation`. **Skill**: analogy, rule induction, symbol substitution. BFS timed out at depth 8.
- **Optimal**: compute the required glyph per slot, and per slot press min(up, down) plus the cursor moves.

### tu93 - Corridor maze with enemies (keyboard 1-4; metadata says keyboard_click)
- **Levels 9**, baselines 19 16 34 42 123 80 14 23 111. Budget 50 50 35 20 50 60 30 50 50.
- **Mechanic**: the arrow avatar jumps one corridor segment per action (8-frame animation). Blocked moves cost a step
  but do nothing. Enemies: sentries wake and charge when you are in line with them; patrols bounce along
  corridors; mimics copy your moves. Contact kills.
- **Win**: reach the exit. **Skill**: navigation with adversary prediction and timing.
  BFS L1 = **18** (human 19).

### vc33 - Communicating vessels (click only)
- **Levels 7**, baselines 7 18 44 61 131 34 152. Budget 50 50 75 50 200 50 200.
- **Mechanic**: liquid columns come in pairs. Clicking a pump (`0022`) moves one unit of liquid from one column
  to its partner (the surface moves by the level's gravity step of 2-3 px), and floats ride the surfaces.
  Clicking a gate (`0004`) when the two sides line up transfers the floats across. The gravity direction differs per
  level (down, up, left, right).
- **Win**: every float sits level with the marker of its color. **Skill**: quantity/level balancing,
  conservation. BFS L1 = **3** (human 7).

### wa30 - Warehouse with helper and thief NPCs (keyboard, 1-5)
- **Levels 9**, baselines 71 119 183 98 368 68 79 442 415 (largest sum, 1843). Budget 200 70 100 100 125
  75 125 150 70.
- **Actions**: ACTION1-4 move the avatar 1 cell and turn it to face that way. ACTION5 grabs the box it faces,
  after which the box moves rigidly with the avatar; pressing again releases it. ACTION5 facing a thief removes the thief.
- **NPCs**: helpers (`kdweefinfi`) BFS to free boxes and carry them into the zone. Thieves (`ysysltqlke`) carry
  boxes to their own zone. Some cells are no-walk.
- **Win**: every box is inside a target zone and released. **Skill**: transport planning, exploiting and
  countering NPC policies.

---

## Cross-game taxonomy

### A. Control schemes (how the agent acts)
| scheme | games |
|---|---|
| Direct avatar movement with arrows | ls20, g50t, tu93, wa30, dc22, sc25, bp35 (left/right only), m0r0 (mirrored multi-avatar) |
| Select-then-move (click or ACTION5 picks the object, arrows move it) | ar25, cn04, ka59, re86, sp80, sk48, m0r0 (blocks) |
| Cursor + value cycling (arrows select a slot, up/down change its value) | tr87, cd82 (ring position) |
| Click-select-then-click-target | r11l (leg -> location), lf52 (peg -> landing), sb26 (token -> slot) |
| Pure click buttons / toggles | ft09, lp85, vc33, s5i5, tn36, su15, sc25 (runes), dc22 (switches, crane), bp35 (blocks) |
| Commit / "run" action (plan first, then execute a simulation) | sp80 (pour), cd82 (pour paint), sb26 (run program), tn36 (run button), g50t (rewind -> ghost) |
| Undo (ACTION7) available | ar25, bp35, lf52, sb26, sk48, su15 |

### B. Mechanic primitives and where they appear
| primitive | games |
|---|---|
| Grid navigation / pathfinding with walls | ls20, tu93, g50t, dc22, sc25, wa30, bp35, m0r0, ka59 |
| Pushing, Sokoban, sliding until blocked | ka59, sk48, wa30 (carry), re86 (push against obstacles) |
| Gravity / falling / fluid | bp35 (flip gravity), sp80 (water), vc33 (liquid levels, per-level gravity direction), su15 (attraction) |
| Key-lock / attribute matching (shape, color, rotation) | ls20, r11l (colors), cn04 (connectors), sk48 (color order) |
| Pattern or picture reproduction | cd82, re86, ar25 (cover set), ft09 (clues), sb26 (sequence), tr87 (translation) |
| Toggles / cycles over discrete states (mod-k) | ft09, tn36 (bits), sc25 (runes), tr87 (glyph variants), bp35 (toggle blocks), dc22 (switch groups) |
| Rotation / reflection | ar25 (mirrors), cn04, ls20, s5i5, tn36, sp80 (rotated boards) |
| Permutation / cyclic shift | lp85, tr87 cursor, cd82 ring |
| Coupled or mirrored control, self-copies | m0r0 (mirror twins), g50t (ghosts), ar25 (reflections), s5i5 (kinematic chains) |
| Merge / combine / capture | m0r0 (twins merge), su15 (fruit merge), lf52 (peg capture) |
| NPCs / enemies with simple policies | tu93, wa30, ka59 (chasers), g50t (movers), bp35 (rising hazard) |
| Programs / sequences / subroutines | sb26, tn36, g50t (recorded path), tr87 (rewrite rules) |
| Demonstration / hint to imitate | sc25 (spell demo), ft09 L1 (hint flash), ar25 L2 (blink) |
| Partial observability | cn04 (GreyMasking), ls20 (fog), bp35 and lf52 (scrolling camera) |
| Resource budget as HUD bar (all games), lives or strikes | ls20 (3 lives + refills), dc22 (lives), r11l (5 strikes), sp80 (4 pours), sb26 (energy) |

### C. Level-design regularities (useful priors)
- Level 1 is a tutorial with 1-2 mechanics and a tiny state space: BFS optimum is 3-20 actions, and
  vc33, lp85, ft09, cd82 and sp80 each take 5 or fewer. Every later level adds one new element: tr87 adds
  `double_translation`, then `alter_rules`, then `tree_translation`; ls20 adds refills, fog and moving changers; sp80 adds
  rotation and diagonal deflectors; vc33 adds gates; ka59 adds crates, bombs and enemies. **The rules persist across levels within a game.**
  Knowledge carried from level k to level k+1 is the main lever, and later levels carry the most weight.
- The HUD always shows the remaining budget (bar on row 0 or 63, column 63, or a sprite bar). An agent
  must segment it out of the state and use it to know how much it can explore.
- Selection feedback is visual (white or highlighted object). Blocked moves and no-op clicks produce
  zero pixel change except the HUD tick. This is the cheapest signal for learning affordances.
- Win is always a static predicate over object configuration (positions, colors, adjacency or
  counts) checked after each action. There are no score counters to maximize.

### D. Primitives an agent must master to generalize to hidden games
1. **Affordance probing with minimal waste**: try each available action once, click each distinct object
   once, and diff the frames (use the last frame, mask the HUD). Probes of this kind cracked most level-1 semantics here.
2. **Object-centric perception**: connected components by color, identifying the avatar (whatever moves with
   the arrows) and the selection highlight, tracking objects across frames, reading HUD bars.
3. **Goal inference from layout**: outlines, silhouettes and markers of the same color or shape
   (targets, cups, goals); reference panels to copy (cd82, re86, sk48, tr87, sb26).
4. **World-model building and planning**: after a few probes, model the transition (move by k px,
   push, slide, gravity, cycles mod k) and plan with BFS/A* *inside the model*. The Kaggle API cannot
   clone states, so lookahead must come from the agent's own model. Offline, `ArcEnv.clone()` gives exact
   lookahead for training and evaluating such models.
5. **Budget-aware exploration**: efficiency is squared per level and later levels weigh more, so explore
   while cheap (level 1 weighs 1), and avoid GAME_OVER (it costs the RESET plus the whole wasted attempt).
6. **Commit-style actions and delayed effects**: pour, run, rewind. Predict before committing, and read
   the multi-frame animation for information.
7. **Transfer within a game**: carry the learned rules and object roles to the next level, and expect one new twist.
8. **Symbolic and program reasoning**: sequences, rewrite rules, bit-encoded opcodes, and symmetry / group
   actions (mirrors, permutations, mod-k toggles).
