# -*- coding: utf-8 -*-
"""
Phase 2: use the Q-table trained in qlearning.py to actually drive the
real robot in ForagingEnv, not just inspect the policy as arrows on a
static grid.

The same design exception flagged (and deferred) back when goal-based
search was discussed applies here: the common Agent interface only gives
act() the percept vector, which does not include the robot's absolute
(x, y, theta) pose. Following a policy defined over grid NODES requires
knowing which node the robot is actually near, which the percept alone
cannot answer. This script is the one place that reads env.agent_pos /
env.agent_theta directly -- a deliberate, documented exception, not
something smuggled into the general Agent interface.

How it works
---------------
1. Train the Q-table exactly as in qlearning.py (instant, over the
   abstract graph).
2. Pick an actual starting node from the robot's real spawn position.
3. Follow the greedy policy node by node: steer toward the current
   target node's (x, y) position; once within arrival_radius of it,
   advance to whatever the policy says is next from there.
4. Record frames and save as an animated GIF -- the actual payoff of
   this module: seeing the abstract table translate into real movement.

Usage
-----
    python demo_qlearning_live.py --map u_shape --episodes 2000
"""

import argparse
import math

import imageio
import numpy as np
import pygame

from edu_env import make_edu_env
from grid_world import build_grid, nearest_visible_node
from qlearning import train_qlearning, greedy_policy, print_q_table
from visualize_qlearning import draw_policy


def action_toward(robot_pos, robot_theta, target_pos, turn_threshold=0.12):
    """Same bearing convention ForagingEnv itself uses for the food
    bearing in _get_obs_single -- so this steers exactly the way the
    environment's own geometry expects."""
    dx = target_pos[0] - robot_pos[0]
    dy = target_pos[1] - robot_pos[1]
    rel_angle = ((math.atan2(dy, dx) - robot_theta + math.pi) % (2 * math.pi) - math.pi) / math.pi
    if rel_angle > turn_threshold:
        return 2
    elif rel_angle < -turn_threshold:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Drive the real robot using a trained Q-table")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape"])
    parser.add_argument("--episodes", type=int, default=2000, help="Q-learning training episodes")
    parser.add_argument("--arrival_radius", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn_seed", type=int, default=None,
                         help="Seeds the robot's spawn position too (separate from Q-learning training's --seed)")
    args = parser.parse_args()

    if args.spawn_seed is not None:
        np.random.seed(args.spawn_seed)

    env = make_edu_env(map_name=args.map, num_agents=1)
    obs_list, info = env.reset()

    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))

    print(f"Map: {args.map}  |  {len(nodes)} nodes  |  training Q-table for {args.episodes} episodes...")
    Q, steps_per_episode = train_qlearning(nodes, edges, episodes=args.episodes, seed=args.seed)
    policy = greedy_policy(Q, nodes)
    correct = sum(1 for s, a in policy.items() if a in edges.get(s, ()))
    print(f"Trained. {correct}/{len(policy)} states point at a real edge.\n")

    # Same policy-arrows image visualize_qlearning.py produces on its own --
    # generated here too so one command gives both the static "what did it
    # learn" picture and the live "watch it actually drive" GIF below.
    env.render()
    draw_policy(env, nodes, edges, policy)
    policy_out_path = (args.out or f"qlearning_live_{args.map}").rsplit(".", 1)[0] + "_policy.png"
    pygame.image.save(env.window, policy_out_path)
    print(f"Saved policy image: {policy_out_path}\n")

    robot_pos = (float(env.agent_pos[0][0]), float(env.agent_pos[0][1]))
    current_node = nearest_visible_node(robot_pos, nodes, env.obstacles)
    target_node = policy.get(current_node, "food")
    print(f"Robot spawned near {current_node} {nodes[current_node]}")
    print(f"Following policy: {current_node} -> {target_node} -> ... -> food\n")

    frames = []
    node_log = [current_node]
    success = False

    for step in range(args.max_steps):
        robot_pos = (float(env.agent_pos[0][0]), float(env.agent_pos[0][1]))
        robot_theta = float(env.agent_theta[0])

        action = action_toward(robot_pos, robot_theta, nodes[target_node])
        obs_list, _, terminated, truncated, info = env.step([action])

        frame = env.render()
        if step % 2 == 0:
            frames.append(frame)

        if info["success"]:
            success = True
            print(f"Reached the food at step {step}.")
            break

        dist_to_target = math.dist(robot_pos, nodes[target_node])
        if dist_to_target < args.arrival_radius:
            current_node = target_node
            node_log.append(current_node)
            if current_node == "food":
                continue  # env's own success check above will catch it shortly
            target_node = policy.get(current_node, "food")
            print(f"  step {step}: arrived at {current_node}, next target -> {target_node}")

        if terminated or truncated:
            break

    print(f"\nNode sequence followed: {' -> '.join(node_log)}")
    print(f"Result: {'SUCCESS' if success else 'did not reach the food within max_steps'}")

    out_path = args.out or f"qlearning_live_{args.map}.gif"
    imageio.mimsave(out_path, frames, fps=20, loop=0)
    print(f"Saved GIF: {out_path}  ({len(frames)} frames)")
    print(f"Saved policy image: {policy_out_path}")

    env.close()


if __name__ == "__main__":
    main()
