# Repo Versioning Runbook — Finding (or Creating) the Paper-Frozen State

This is a one-time investigative task, to be run against your actual
simulator repository (not this sandbox — I only ever had loose copies of
individual files here, never your real git history).

Goal: before the course repo references any tag of the simulator, know
with reasonable confidence which commit corresponds to the state used for
the paper's results, so the two never get confused later.

---

## Step 1 — See what you already have

```bash
git fetch --tags                       # tags created on GitHub's web UI don't
                                        # show up locally until you fetch them
git tag -l -n99                        # list all tags WITH their messages
git log --oneline --all --graph --decorate
git status                             # anything uncommitted right now?
git stash list                         # anything stashed and forgotten?
```

If `git status` or `git stash list` show anything, stop here and think
before proceeding — it's common for the "real" state used for a paper's
final runs to be sitting uncommitted or stashed, especially quick
parameter tweaks made right before generating final figures.

## Step 2 — Narrow down by time window

If you remember roughly when you generated the paper's results (submission
date, or when you ran the batch of `gru_diagnostic_*.csv` files that fed
`consolidate_runs.py`):

```bash
git log --oneline --since="2026-XX-XX" --until="2026-YY-YY"
```

Cross-reference against file timestamps if you still have the original
result CSVs or figures anywhere (even outside git) — the modification date
on a `gru_diagnostic_003.csv` or a saved plot can point you to the right
week even if the commit message doesn't say anything useful.

## Step 3 — Narrow down by content

If the paper describes specific ablations (e.g. `social_mode=shuffled`,
`social_variant=angle_only`, a particular `hunger_decay` tuning, a specific
version of the social attention filter in `GRUController.act`), search for
when that exact logic appeared or changed:

```bash
git log --oneline --grep="paper" --grep="final" --grep="results" -i --all
git log -p --follow controllers.py | less     # read through the actual diffs
git blame controllers.py                       # who/when touched each line
```

Comments already in the code are useful anchors — e.g. `controllers.py`
has inline notes like "SINTONÍA FINA" marking specific behavioral tuning
passes. If the paper's described behavior matches a tuning pass that has
such a comment, that commit (or the one right after it) is a strong
candidate.

## Step 4 — Decide, and tag immediately

**If you found a confident candidate commit:**

```bash
git tag -a v1.0-paper <commit-hash> -m "State used for [paper name/venue]"
git push origin v1.0-paper
```

**If you cannot confidently recover it:** tag your best estimate anyway,
but say so honestly in the tag message — a documented "best guess, not
verified" is more useful later than silence, and much more useful than a
tag that quietly claims more certainty than you actually have:

```bash
git tag -a v1.0-paper-approx <commit-hash> -m "Best estimate of the state used for [paper]; not independently verified"
git push origin v1.0-paper-approx
```

**Either way, also create a forward-looking course tag**, independent of
whatever you conclude about the paper, on whatever commit you want the
course to reference going forward (likely current `HEAD`, or `main`):

```bash
git tag -a v1.0-course -m "Simulator baseline referenced by the AI course materials (training_wheels/)"
git push origin v1.0-course
```

From this point on, if you need to change `foraging_env.py`/`controllers.py`
for a course-related reason, bump to `v1.1-course`, `v1.2-course`, etc. —
never move `v1.0-course` after students have started depending on it, and
never touch `v1.0-paper`(-approx) again at all.

---

## Course repo README template (fill in the tag once you have it)

This is the section to drop into the course repo's `README.md`, replacing
the current ad-hoc "copy these files in" instructions with a pinned,
verifiable install step:

```markdown
## Installation

1. Clone the simulator at the exact version this course was built and
   tested against:

   ```bash
   git clone --branch v1.0-course https://github.com/<your-org>/<your-simulator-repo>.git
   cd <your-simulator-repo>
   ```

2. Copy this repository's `training_wheels/` folder in anywhere inside
   that cloned folder (as a sibling of `foraging_env.py`, one level
   deeper, wherever's convenient — no manual flattening required, see
   `training_wheels/README.md`).

3. Install dependencies: `pip install gymnasium numpy pygame numba`

4. Verify your setup by following `training_wheels/TESTING.md`,
   Sections 1-4, before attempting anything with rendering.
```

Do not publish the course repo with this section pointing at `main` or at
no tag at all — that reopens exactly the ambiguity this whole exercise is
meant to close. If `v1.0-course` doesn't exist yet when you're ready to
publish, that's a signal to go finish Steps 1-4 above first, not to ship
with a moving target.
