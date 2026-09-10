# -*- coding: utf-8 -*-
"""
Level 0 demo: run single-robot foraging episodes with either a simple
reflex agent or a model-based reflex agent, and compare what happens.

Usage
-----
    python demo_level0_reflex.py --agent simple
    python demo_level0_reflex.py --agent model_based --map u_shape
    python demo_level0_reflex.py --agent model_based --map u_shape --render

Try this in class: run the SAME agent on the SAME map, but toggle
"randomize_spawns"/"randomize_food" in edu_env.make_edu_env() and watch
whether each agent type keeps working or starts failing. That failure mode
is usually the best way to make the difference between reflex types
concrete instead of definitional.
"""

import argparse
import time

from edu_env import make_edu_env
from agents.simple_reflex_agent import SimpleReflexAgent
from agents.model_based_reflex_agent import ModelBasedReflexAgent


def build_agent(kind):
    if kind == "simple":
        return SimpleReflexAgent(model_path="braitenberg_avoidance.json")
    if kind == "model_based":
        return ModelBasedReflexAgent(agent_idx=0)
    raise ValueError(f"Unknown agent kind: {kind}")


def main():
    parser = argparse.ArgumentParser(description="Level 0 reflex agent demo")
    parser.add_argument("--agent", choices=["simple", "model_based"], default="simple")
    parser.add_argument("--map", default="default", choices=["default", "u_shape", "n_shape"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true", help="Open a window and draw every step")
    parser.add_argument("--sleep", type=float, default=0.02,
                         help="Seconds to pause between rendered frames (only with --render)")
    args = parser.parse_args()

    env = make_edu_env(
        map_name=args.map,
        num_agents=1,
        render_mode="human" if args.render else None,
    )
    agent = build_agent(args.agent)

    for ep in range(1, args.episodes + 1):
        obs_list, info = env.reset()
        agent.reset()
        done = False

        while not done:
            action = agent.act(obs_list[0], info)
            obs_list, _, terminated, truncated, info = env.step([action])

            # NOTE: ForagingEnv only ever draws to the window when render()
            # is called explicitly -- passing render_mode="human" alone
            # does not open or update a window by itself. Skipping this
            # call is a real bug that shipped in an earlier version of this
            # file: the episode would run to completion silently, with no
            # window ever appearing, and no error either.
            if args.render:
                env.render()
                time.sleep(args.sleep)

            done = terminated or truncated

        status = "SUCCESS" if info["success"] else "FAIL"
        print(f"Episode {ep}: {status} in {info['step_count']} steps")

    env.close()


if __name__ == "__main__":
    main()

