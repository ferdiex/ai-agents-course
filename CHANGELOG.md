# Changelog

Newest entries at the top. Each entry lists exactly which files changed
and why — use this instead of downloading the full zip every time.

---

## Entry 3 — Correct spawn_seed=8 bug repro claim

**Files:** `exercises/AI_Agents_Exercises.docx`, `STATUS.md`

Both files previously said `--spawn_seed 8` "reliably reproduces" the
stuck-robot bug. Tested 3x in one environment (Python 3.12): identical
stuck result every time. But the same command on a different machine
(Python 3.9) reached the food successfully.

Real finding, not a contradiction to hide: the failure is a **borderline
graze**, not a clean crash — the robot's straight-line path skims a wall
without technically crossing it. Tiny floating-point differences between
Python/numpy/numba versions are apparently enough to tip that borderline
case either way. `--spawn_seed 8` is deterministic *within one
environment* but not portable *across* environments. Both files now say
to verify on your own machine before relying on it for a live demo.

## Entry 2 — Document the stuck-robot bug as a reproducible bonus

**Files:** `exercises/AI_Agents_Exercises.docx`

Added a bonus note to Activity 6's "Variante si sobra tiempo": running
`demo_ga_live.py --map u_shape --generations 100 --spawn_seed 8`
reproduces the known stuck-robot bug on demand (later corrected in
Entry 3 above — turned out not to be universal across environments).

## Entry 1 — Add Activity 0 (PEAS wired vs. narrative), bin definition, README fixes

**Files:** `exercises/AI_Agents_Exercises.docx`, `EXERCISES_COMMANDS.md`, `README.md`

- Added Activity 0 (PEAS: Wired vs. Narrative) — new, placed before
  Activity 1, with student task + teacher's manual (10 PEAS fields, 3 of
  which are actually wired: `dynamics`, `agents`, `map`).
- Added a "bin/bucket" definition at the start of Activity 1.
- `EXERCISES_COMMANDS.md`: added the Activity 0 section.
- `README.md`: fixed "6 activities" → "7 activities" in the course map
  (stale after adding Activity 0); pointed the "wired" section at
  Activity 0 instead of describing it as a loose suggestion; added
  `EXERCISES_COMMANDS.md` to the course-map description.

---

*Everything before Entry 1 (all 14 code modules, `TESTING.md`,
`STATUS.md`, `REPO_VERSIONING.md`, `verify_install.py`, Activities 1-6)
predates this changelog — that's the state of the initial
`ai-agents-course` upload. See `README.md`'s course map for the full
built inventory.*
