## Installation

1. Clone the simulator at the exact version this course was built and
   tested against:

```bash
   git clone --branch v1.0-course https://github.com/ferdiex/essim2d3d.git
   cd essim2d3d
```

2. Copy this whole `training_wheels/` folder into the cloned repo — as a
   top-level sibling of `foraging_env.py`, or one level deeper, wherever
   is convenient. No flattening needed: `edu_env.py` and every module
   that needs the simulator's code walk upward from their own file
   location until they find `foraging_env.py`.

3. Make sure a `worlds/` subfolder exists at the simulator root
   containing `random_obstacles.json`, `u_shape.json`, `n_shape.json`
   (from the original project) plus `training_wheels/worlds/csp_dense.json`
   (copy it in — it's the dense obstacle map used by the CSP module).

4. Install dependencies:

```bash
   pip install gymnasium numpy pygame numba
```

5. Run the one-command diagnostic check:

```bash
   cd training_wheels
   python3 verify_install.py
```

   Expected: a pass/fail table, 12/12 checks passing. If anything fails,
   the report tells you which check — go to the matching numbered section
   in `TESTING.md` for the full diagnostic and troubleshooting table.
