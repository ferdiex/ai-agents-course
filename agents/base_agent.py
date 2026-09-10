# -*- coding: utf-8 -*-
"""
Common interface shared by every agent type in this course.

Following Russell & Norvig's terminology: an agent is characterized by its
*agent program* -- the function mapping a percept (or percept sequence, for
agents that keep memory) to an action. Every concrete agent class in this
package implements the same external contract (act / reset), so that
swapping one agent type for another never requires touching the runner
script -- only what happens INSIDE act() changes.

This is the whole point of the course exercise: table-driven, simple
reflex, model-based reflex, goal-based, utility-based, and learning agents
all look identical from the outside. The difference between them is
entirely a design decision about what act() does internally (and, for
model-based/learning agents, what state reset() needs to clear).
"""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Abstract base class for all agent types used in this course."""

    @abstractmethod
    def act(self, percept, info=None):
        """
        Map the current percept (and optionally extra environment info) to
        a single discrete action.

        Parameters
        ----------
        percept : array-like
            The observation vector returned by the environment for this
            agent. See ForagingEnv._get_obs_single for its exact layout:
            8 proximity readings, food bearing, odor (social signal is
            excluded at this course level -- see edu_env.py).
        info : dict, optional
            Extra environment info (e.g. individual_success), not part of
            the percept itself but occasionally useful.

        Returns
        -------
        int
            A discrete action id:
                0 = forward
                1 = turn left
                2 = turn right
                3 = reverse
                4 = signal / wander
        """
        raise NotImplementedError

    def reset(self):
        """
        Clear any internal state between episodes.

        Stateless agents (e.g. a simple reflex agent) can leave this as a
        no-op. Agents with memory (model-based reflex, goal-based agents
        with a cached plan, learning agents with an exploration schedule,
        etc.) should override this.
        """
        pass
