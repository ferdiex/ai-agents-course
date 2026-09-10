# -*- coding: utf-8 -*-
"""
Simple reflex agent: chooses an action from the CURRENT percept only, using
fixed condition-action rules. It keeps no memory of the past and does not
reason about any goal -- it just reacts.

This wraps the Braitenberg-style controller that already exists in the main
project (controllers.BraitenbergController) behind the course's common
Agent interface, so it can be swapped in and out of the runner exactly like
every other agent type in this package.

Note on "no memory": BraitenbergController does keep a small stuck-escape
counter internally. That is a pragmatic patch, not a model of the world --
it does not represent any belief about the environment, just "how many
steps have I been stuck". We still classify this as a simple reflex agent;
the model-based reflex agent (next module) is qualitatively different
because its internal state IS an explicit model of what phase of a
maneuver it believes itself to be in.
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

from controllers import BraitenbergController
from .base_agent import Agent


class SimpleReflexAgent(Agent):
    """Braitenberg-style vehicle: obstacle avoidance from raw sensors only."""

    def __init__(self, model_path=None, num_agents=1):
        """
        Parameters
        ----------
        model_path : str or None
            Path to a JSON with 'left_weights'/'right_weights' (e.g.
            braitenberg_avoidance.json). Without it, falls back to a fixed
            heuristic threshold rule -- still a simple reflex agent either
            way.
        num_agents : int
            Kept for interface consistency; unused by this controller.
        """
        self._controller = BraitenbergController(model_path=model_path,
                                                   num_agents=num_agents)

    def act(self, percept, info=None):
        return self._controller.act(percept, info)

    def reset(self):
        self._controller.reset()
