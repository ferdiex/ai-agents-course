# -*- coding: utf-8 -*-
"""
Model-based reflex agent: like a simple reflex agent, but it keeps a small
internal state -- a belief about what is currently happening -- so it can
keep acting sensibly even when the current percept alone is ambiguous or
briefly misleading.

This wraps the existing HeuristicCalibrationController. Its phase / timer /
turn_dir / subphase attributes ARE that internal model: once it decides it
is "backing off" from a wall, it keeps backing off for a few steps even if
the sensor that triggered the maneuver stops firing in the meantime. A
simple reflex agent has no way to do that -- it would re-decide from
scratch on every single step, using only what the sensors say right now.

This is the qualitative line between simple reflex and model-based reflex:
not "does it have any state at all" (the simple reflex agent has a stuck
counter too), but "does that state represent a belief about the world that
outlives a single percept".

A deliberate augmentation, and why it exists
----------------------------------------------
HeuristicCalibrationController was originally built for the two-robot
social scenario: index 10 of its input carries a signal from the OTHER
robot, and the controller uses its sign to pick which wall to follow
(`preferred = +1 if soc > 0 else -1`). In this single-robot course level,
that percept only has 10 entries (no social channel -- see
peas_foraging.json's "sensors" field), so `len(obs) > 10` is always False
and `soc` is always 0.0. Left alone, this doesn't corrupt the controller
with anything foreign -- it simply starves it of the one input its
wall-side decision depends on, so it silently defaults to `preferred = -1`
FOREVER, on every map, regardless of where the food actually is. That is
exactly what an early version of this file did, and it measurably hurt
this agent's performance versus the simpler Braitenberg-based reflex agent
on the u_shape map -- see TESTING.md for the actual numbers.

Rather than edit controllers.py (the original simulator's file, which this
course package deliberately never touches -- see REPO_VERSIONING.md), this
class synthesizes the missing index-10 input from something the agent
already legitimately has: the food bearing at percept[8] (`rel_angle`).
This is the ONE place in this entire course level where an agent's input
is augmented beyond what ForagingEnv itself provides, and it is done here,
in the open, specifically so it's easy to point at and question -- not
buried inside a copy of someone else's controller.
"""

import os
import sys


def _add_simulator_root_to_path():
    """See edu_env.py for the full explanation of what/why this does."""
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(here, "foraging_env.py")):
            if here not in sys.path:
                sys.path.insert(0, here)
            return
        parent = os.path.dirname(here)
        if parent == here:
            raise ImportError(
                "Could not locate foraging_env.py by walking up from "
                f"{os.path.abspath(__file__)}. Make sure training_wheels/ "
                "lives somewhere inside (or next to) your simulator repo."
            )
        here = parent


_add_simulator_root_to_path()

from controllers import HeuristicCalibrationController
from .base_agent import Agent


class ModelBasedReflexAgent(Agent):
    """Wall-following agent driven by an explicit internal state machine."""

    def __init__(self, agent_idx=0):
        self._controller = HeuristicCalibrationController(agent_idx=agent_idx)

    def act(self, percept, info=None):
        # See the module docstring: index 10 is where the wrapped
        # controller expects a social signal that doesn't exist for a
        # single robot. We reuse the food bearing already at percept[8] in
        # its place, so the preferred wall-following side depends on where
        # the food actually is instead of being stuck on one fixed side.
        food_bearing = percept[8]
        augmented_percept = list(percept) + [food_bearing]
        return self._controller.act(augmented_percept)

    def reset(self):
        self._controller.reset()
