# Open Issues Log

For a full inventory of what's built, see `README.md`'s "Course map"
section. This file only tracks things that are genuinely unresolved.

## 1. Waypoint-following has no local obstacle avoidance (diagnosed, not fixed)

`demo_qlearning_live.py`/`demo_ga_live.py`'s `action_toward()` steers
purely by bearing to the target node — no local reactive obstacle
avoidance. **Reproduced deliberately and diagnosed exactly:**

```bash
python3 demo_ga_live.py --map u_shape --generations 100 --spawn_seed 8
```

Robot spawns near `n17` (right side of the map); the learned policy's
next target is `n12` (left side) — a straight line the GRID considers
clear (node-center-to-node-center line of sight with the default 20px
clearance), but which runs directly along the OUTER FACE of the U-shape's
right wall for a long stretch, not just a single close point near a
corner. The robot drifts into contact with that wall while course-
correcting, and since `action_toward` has no way to notice or route
around an obstacle, it just keeps commanding "forward," pinned against
the wall for the rest of the episode. See
`screenshots/ga_stuck_bug_frame.png` (the robot visibly pressed against
the wall, heading pointed straight into it) and
`screenshots/ga_stuck_bug_example.gif`.

This changes which fix candidate looks more promising: since the failure
mode is a long parallel graze along a wall face (not just a tight corner),
(1) increasing `grid_world.build_grid`'s clearance parameter is a
plausible fix but may need to be substantial, not a small bump, to clear
an entire parallel run alongside a wall — worth testing empirically, not
assuming a small increase suffices. (2) Blending in local reactive
avoidance (Braitenberg-style repulsion from high sensor readings) remains
the more robust fix in principle, since it would catch this regardless of
how the grid's edges were built. Neither has been implemented or compared
yet.

**Correction, found by a real cross-machine test:** this was originally
documented as "reliably reproduces the bug" — that overclaimed it.
Reran 3x in this sandbox (Python 3.12) and got the identical stuck result
every time (fully deterministic *within one environment*), but a
different machine (Python 3.9, different numpy/numba versions) ran the
exact same command and reached the food successfully. Since this is a
borderline case — the robot's path GRAZES a wall rather than cleanly
crossing or clearing it — tiny floating-point differences between Python/
numpy/numba versions are enough to tip it either way. **`--spawn_seed 8`
reliably reproduces the bug on some machines and not on others; it is not
a universal repro.** Anyone using this for a classroom demo should verify
on their own machine first, not assume it will trigger live. This doesn't
change the diagnosis (the steering still has no reactive obstacle
avoidance) — it just means the specific seed used to demonstrate it isn't
portable across environments the way a normal fixed-seed repro would be.

**Note for reproducing other cases:** both live-demo scripts now take
`--spawn_seed` to fix the robot's starting position independently of
training's `--seed`, specifically to make bugs like this reproducible on
demand instead of stumbled into by chance.

## 2. GIF frame-count discrepancy (not investigated)

While reproducing issue #1, `demo_ga_live.py` printed "Saved GIF: ...
(300 frames)" but both `imageio.mimread` and PIL's own frame iterator
only read back 7 frames from the saved file — a real discrepancy between
what the script believed it wrote and what the GIF file actually
contains. Possibly related to `imageio.mimsave` handling many
near-identical consecutive frames (the robot barely moves while stuck) in
some lossy/deduplicating way — a *successful* run afterward showed
matching frame counts (32 written, 32 read back), so this may be specific
to long runs of near-duplicate frames. Not investigated further. Worth
checking before relying on GIF length as a proxy for episode length in
any future analysis.

## 3. GA vs. Q-learning: interaction-count comparison is apples-to-oranges

`qlearning.py` now prints the ACTUAL total environment interactions used
during training (measured: 23,787 across 2000 episodes on `u_shape`).
`ga_learning.py` only prints a WORST-CASE upper bound
(`population_size x generations x states x max_steps`, e.g. 3,168,000),
not the actual count, because `rollout_fitness()` isn't instrumented to
count real steps (most rollouts terminate long before `max_steps` once a
chromosome is any good). Activity 6 in the exercises docx deliberately
asks students to notice this asymmetry rather than presenting a clean
number — but if a real apples-to-apples comparison is wanted later,
`rollout_fitness()` needs a step counter added and threaded back up
through `train_ga()`.

## 4. Not yet measured: does GA's policy tie between actions the way Q-learning's does?

Section 13.4 of `TESTING.md` found that Q-learning's flat `-1`-per-step
reward makes many states converge to TIED Q-values across several
actions (since the reward only encodes hop-count to the goal, not real
distance — same effect BFS has vs. A*). Whether GA's evolved policies
show the same tie pattern, or something different (since GA never
computes a value function at all, only picks a single action per gene),
has not been checked. Flagged as a good "don't assume, run it" question
in Activity 6, but not answered anywhere in this repo yet.

## If resuming with a fresh context window

Point the new conversation at this file plus `TESTING.md` and `README.md`
in the delivered `training_wheels_level0.zip` — `README.md`'s course map
gives the full built inventory, `TESTING.md` has every module's detailed
test log, and this file has everything still genuinely open.
