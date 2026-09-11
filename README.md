# Training Wheels: Russell & Norvig Agent Types on the Foraging Simulator

This is **Level 0** of a teaching arc built on top of the existing
foraging simulator:

- **Level 0 (this package)**: single-robot foraging, a thin documented
  wrapper, no social channel. Used to walk through the five classic agent
  architectures (table-driven, simple reflex, model-based reflex,
  goal-based, utility-based) plus a learning agent, one at a time.
- **Level 1**: the full `foraging_env.py` as-is, social channel included,
  no wrapper -- "take off the training wheels".
- **Level 2**: `essim3d.py`, real physics (PyBullet), two robots.

This README only covers Level 0.

## Course map

Everything below is built and tested (see `TESTING.md` for the full test
log of each). All commands run from inside `training_wheels/`, on the
`u_shape` map unless noted. Static-map modules (goal-based search
onward) need `u_shape` or `n_shape`, never `default` — see each module's
own docstring for why.

| # | Agent type | Files | Try it |
|---|---|---|---|
| 1 | PEAS analysis | `peas_foraging.json` | read it; see the "wired" section below, and Activity 0 |
| 2 | Table-driven | `agents/table_driven_agent.py` | `python3 agents/table_driven_agent.py` |
| 3 | Simple reflex | `agents/simple_reflex_agent.py` | `python3 demo_level0_reflex.py --agent simple` |
| 4 | Model-based reflex | `agents/model_based_reflex_agent.py` | `python3 demo_level0_reflex.py --agent model_based` |
| 5 | Goal-based (search) | `grid_world.py`, `search_policies.py`, `visualize_search.py` | `python3 visualize_search.py --algorithm astar` |
| 6 | Utility-based | `visualize_utility.py` | `python3 visualize_utility.py` |
| 7 | CSP | `csp_spawn.py` | `python3 csp_spawn.py --map csp_dense` |
| 8 | Minimax | `minimax_search.py`, `visualize_minimax.py` | `python3 visualize_minimax.py --predator n0 --prey n10` |
| 9 | Learning: Q-learning | `qlearning.py`, `visualize_qlearning.py`, `demo_qlearning_live.py` | `python3 demo_qlearning_live.py --episodes 2000` |
| 10 | Learning: GA | `ga_learning.py`, `visualize_ga.py`, `visualize_ga_population.py`, `demo_ga_live.py` | `python3 demo_ga_live.py --generations 100` |

Course exercises with student + teacher sections for all **7** activities
(PEAS wired-vs-narrative, table-driven, reflex comparison, CSP, minimax,
Q-learning, GA vs. Q-learning) are in `exercises/AI_Agents_Exercises.docx`.
`EXERCISES_COMMANDS.md` has just the commands for each activity, without
the pedagogical framing, for quick copy-paste during class. Open issues
and things flagged but not yet resolved are tracked in `STATUS.md`, not
here. **`CHANGELOG.md` lists exactly which files changed and why, entry
by entry — check it after this package is updated instead of re-diffing
everything by hand.**

## Installation

Copy this whole `training_wheels/` folder into your simulator repo — as a
top-level sibling of `foraging_env.py`, one level deeper, wherever is
convenient. **No flattening needed**: `edu_env.py` and every module in
`agents/` that needs the simulator's code walk upward from their own file
location until they find `foraging_env.py`, and add that folder to
`sys.path` automatically. This works the same whether you run
`demo_level0_reflex.py` from the simulator root, from inside
`training_wheels/`, or import these modules from anywhere else — it's
based on each file's own location on disk, never on the current working
directory.

(An earlier version of this package required manually flattening the
folder structure, which turned out to be a real, repeated source of
`ModuleNotFoundError: No module named 'foraging_env'` — see
`TESTING.md` and `REPO_VERSIONING.md` for the story. That requirement is
gone now; you should not need to move any files by hand.)

You still need a `worlds/` subfolder with `random_obstacles.json`,
`u_shape.json`, `n_shape.json` at your simulator's root, for
`config_loader.load_world_file` to find them — that's a requirement of the
original project, unrelated to `training_wheels/`. **New:** also copy
`training_wheels/worlds/csp_dense.json` into that same `worlds/` folder —
it's the dense obstacle map used by the CSP module.

**After installing, run the one-command check:**

```bash
python3 verify_install.py
```

Runs a lightweight version of every module's smoke test and prints a
pass/fail table. If everything shows `OK`, you're set up correctly — no
need to work through `TESTING.md` by hand first. If something fails, it
tells you which check failed; go to the matching numbered section in
`TESTING.md` for the full diagnostic and troubleshooting table (this
script only tells you WHAT failed, not why).

## Structure

```
training_wheels/
├── peas_foraging.json          PEAS analysis, wired vs. narrative fields
├── verify_install.py            one-command check: is everything installed correctly?
├── edu_env.py                  make_edu_env(): the only wired surface
├── demo_level0_reflex.py       runner for the two reflex-family agents
├── grid_world.py                coarse navigation graph (any map, snapshot per episode)
├── search_policies.py           DFS / BFS / A*, one swappable signature
├── worlds/
│   └── csp_dense.json           dense obstacle map, copy into your simulator's worlds/
├── visualize_search.py          static image: frozen robot + grid + highlighted plan
├── visualize_utility.py         static image: shortest route vs. utility-based route, side by side
├── csp_spawn.py                  backtracking vs. generate-and-test for robot+food placement
├── minimax_search.py             minimax + alpha-beta, predator-prey over the grid
├── visualize_minimax.py          static image: predator/prey chase trajectory
├── qlearning.py                   tabular Q-learning, blind to the graph's edges
├── visualize_qlearning.py         static image: learned policy as arrows on the grid
├── demo_qlearning_live.py         Phase 2: drives the real robot with the trained table, saves a GIF
├── ga_learning.py                  genetic algorithm, evolves full policies (evolution.json defaults)
├── visualize_ga.py                 static image: GA's best policy as arrows on the grid
├── demo_ga_live.py                 Phase 2 for GA: drives the real robot, saves a GIF
├── visualize_ga_population.py      shows multiple individuals side by side -- makes it visibly a GA
└── agents/
    ├── base_agent.py           common Agent interface (act / reset)
    ├── table_driven_agent.py   discussion-only: combinatorics demo
    ├── simple_reflex_agent.py       wraps controllers.BraitenbergController
    └── model_based_reflex_agent.py  wraps controllers.HeuristicCalibrationController
```

**Open design question, resolved for Q-learning/GA but worth knowing
about:** the common `Agent` interface only gives `act()` the percept
vector (proximity + food bearing + odor) — it deliberately does not
include the robot's absolute (x, y, θ) pose, because the reflex-family
agents don't need it and shouldn't have access to it. `demo_qlearning_live.py`
and `demo_ga_live.py` are the two places in this course that read
`env.agent_pos`/`env.agent_theta` directly, as a documented privileged-
information exception, to follow a plan defined over `grid_world.py`'s
graph — see `TESTING.md` Sections 13.5 and 14 for the full reasoning.

## The "wired" idea in `peas_foraging.json`

Every PEAS field carries a `"wired"` flag:

- `"wired": true` means the field has a real, direct config parameter in
  `ForagingEnvConfig` that `edu_env.make_edu_env()` exposes -- editing the
  JSON's `"value"` and passing the matching argument to `make_edu_env()`
  genuinely changes the simulator's behavior. The `"config_key"` field
  names exactly which `ForagingEnvConfig` attribute it maps to.
- `"wired": false` means the field is an accurate *description* of the
  task (grounded in reading the environment source), but there is no dial
  in the code that controls it -- it is an emergent property of how
  `foraging_env.py` was written, not a parameter.

**This is Activity 0** in `exercises/AI_Agents_Exercises.docx`: before
looking at the code, students guess `"wired": true/false` for each field
on their own, then verify by reading `edu_env.py` (short) and, only if
they want to confirm the "false" ones, `foraging_env.py` (long). The
point isn't memorizing the answer key -- it's practicing "verify against
the source" instead of trusting a description (from a professor, a
teammate, or an AI) at face value.

This distinction also opens a real discussion: Russell & Norvig describe
properties like observability as categorical properties of an environment.
In practice, "partial observability" here is really the end result of a
dozen small engineering decisions (sensor range, sensor count, noise
model, sampling rate) -- there was never a single "observability" variable
to set. That gap between the textbook abstraction and an actual codebase
is worth naming explicitly, not glossing over.

## Agent interface

Every agent type implements the same two methods
(`agents/base_agent.py::Agent`):

```python
agent.act(percept, info=None) -> action_id   # 0..4
agent.reset()                                # clear episode state, if any
```

This means `demo_level0_reflex.py` never needs to change when a new agent
type is added -- only which class gets instantiated. That uniformity is
itself the lesson: from the outside, a table-driven agent, a simple reflex
agent, and a full learning agent are indistinguishable. The interesting
differences are all inside `act()`.

**One documented exception:** `ModelBasedReflexAgent` augments its percept
by one value before handing it to the wrapped controller (which was
originally built for a two-robot scenario and needs a "which wall to
follow" signal this course level doesn't otherwise provide). This is the
only place in Level 0 where an agent's input is anything other than
exactly what `ForagingEnv` returns -- done in the open, in
`agents/model_based_reflex_agent.py`, specifically so it's easy to find
and question. Full story, including the before/after numbers this fixed,
is in `TESTING.md`.
