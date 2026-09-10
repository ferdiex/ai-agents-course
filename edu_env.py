# -*- coding: utf-8 -*-
"""
Thin, documented wrapper around ForagingEnv for the "training wheels"
course level.

This module intentionally exposes only the configuration knobs that are
marked "wired": true in peas_foraging.json. Everything else (the social
channel, sensor noise parameters, rendering internals, etc.) is fixed to a
sane default and treated as out of scope for this course level.

Why this file exists (read this before skipping to the code)
--------------------------------------------------------------
foraging_env.py is real project code: raycasting via numba, a social
communication channel with several ablation modes, per-agent path
tracing for rendering, and more. That is a lot to read just to answer a
simple question like "does changing num_agents in the PEAS json actually
do anything?".

This wrapper is the ONLY file students need to open to verify, in a couple
of minutes, that every field marked "wired": true in peas_foraging.json
really does flow into ForagingEnvConfig. Everything below this function is
optional reading for the curious.
"""

import os
import sys


def _add_simulator_root_to_path():
    """
    Walk upward from this file's own location until we find a folder that
    contains foraging_env.py (the simulator's root), and add it to
    sys.path if it isn't already there.

    Why this exists: training_wheels/ can be dropped anywhere inside (or
    right next to) the simulator repo -- as a top-level sibling folder, as
    a subfolder one level in, nested deeper, whatever -- without anyone
    having to manually move files around first. This function finds the
    simulator no matter where training_wheels/ ended up, using the file's
    own path (__file__), never the current working directory, so it works
    the same whether this module is imported or run as a script, and from
    whatever directory the terminal happens to be in.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(here, "foraging_env.py")):
            if here not in sys.path:
                sys.path.insert(0, here)
            return
        parent = os.path.dirname(here)
        if parent == here:  # reached the filesystem root, never found it
            raise ImportError(
                "Could not locate foraging_env.py by walking up from "
                f"{os.path.abspath(__file__)}. Make sure training_wheels/ "
                "lives somewhere inside (or next to) your simulator repo."
            )
        here = parent


_add_simulator_root_to_path()

from foraging_env import ForagingEnv, ForagingEnvConfig


def make_edu_env(map_name="default", num_agents=1, randomize_food=False,
                  randomize_spawns=False, render_mode=None):
    """
    Build a ForagingEnv configured for the course's Level 0 (training wheels).

    Parameters
    ----------
    map_name : str
        One of "default", "u_shape", "n_shape", "csp_dense". Wired to
        ForagingEnvConfig.map_name (peas_foraging.json field: "map").
        "csp_dense" requires worlds/csp_dense.json to exist -- see
        training_wheels/worlds/csp_dense.json and copy it into your
        simulator's own worlds/ folder.
    num_agents : int
        1 for every single-robot exercise in this course level (simple
        reflex, model-based reflex, goal-based, CSP, utility-based,
        learning). 2 is reserved for the predator-prey minimax module --
        do not use it anywhere else at this level.
        Wired to ForagingEnvConfig.num_agents (PEAS field: "agents").
    randomize_food : bool
        Wired to ForagingEnvConfig.randomize_food (PEAS field: "dynamics").
    randomize_spawns : bool
        Wired to ForagingEnvConfig.randomize_spawns (PEAS field: "dynamics").
    render_mode : str or None
        Passed straight to ForagingEnv. Use "human" to open a window.

    Returns
    -------
    ForagingEnv
        A gymnasium-style environment:
            obs_list, info = env.reset()
            obs_list, reward, terminated, truncated, info = env.step(actions)
        obs_list and actions are lists indexed by agent id, even when
        num_agents == 1 (i.e. obs_list[0], actions = [action]).
    """
    config = ForagingEnvConfig(
        map_name=map_name,
        num_agents=num_agents,
        # u_env=True is the only lever this wrapper has to stop
        # ForagingEnv.reset() from overwriting a map's obstacles with its
        # own hardcoded 5-block layout (see foraging_env.py's reset() --
        # the check is a hardcoded map_name list we can't edit without
        # touching the simulator itself). It also happens to change how
        # food is placed and excludes a central box from robot spawns;
        # csp_dense is deliberately designed around those side effects
        # rather than fighting them -- see TESTING.md for the full story.
        u_env=(map_name in ("u_shape", "csp_dense")),
        max_steps=1000,  # matches essim2d.py's convention (default is 600)
        randomize_food=randomize_food,
        randomize_spawns=randomize_spawns,
        social_mode="off",  # out of scope at this course level, fixed on purpose
    )
    return ForagingEnv(config=config, render_mode=render_mode)
