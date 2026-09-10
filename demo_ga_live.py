# -*- coding: utf-8 -*-
"""
Phase 2 for the GA module: drive the real robot in ForagingEnv using the
GA's best evolved chromosome, exactly the way demo_qlearning_live.py does
for Q-learning -- same action_toward() steering, same node-by-node
waypoint following, same privileged (x, y, theta) access exception
(see demo_qlearning_live.py's docstring for the full explanation, not
repeated here).

Usage
-----
    python demo_ga_live.py --map u_shape --generations 100
"""

import argparse
import math

import imageio
import numpy as np
import pygame

from edu_env import make_edu_env
from grid_world import build_grid, nearest_visible_node
from ga_learning import train_ga
from visualize_qlearning import draw_policy
from demo_qlearning_live import action_toward


def main():
    parser = argparse.ArgumentParser(description="Drive the real robot using an evolved GA policy")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape"])
    parser.add_argument("--population_size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--arrival_radius", type=float, default=25.0)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn_seed", type=int, default=None,
                         help="Seeds the robot's spawn position too (separate from GA training's --seed)")
    args = parser.parse_args()

    if args.spawn_seed is not None:
        np.random.seed(args.spawn_seed)

    env = make_edu_env(map_name=args.map, num_agents=1)
    obs_list, info = env.reset()

    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))

    print(f"Map: {args.map}  |  {len(nodes)} nodes  |  training GA for {args.generations} generations...")
    policy, fitness_curve, stats_per_gen, _ = train_ga(nodes, edges, population_size=args.population_size,
                                       generations=args.generations, seed=args.seed)
    correct = sum(1 for s, a in policy.items() if a in edges.get(s, ()))
    print(f"Trained. {correct}/{len(policy)} states point at a real edge.\n")

    env.render()
    draw_policy(env, nodes, edges, policy)
    policy_out_path = (args.out or f"ga_live_{args.map}").rsplit(".", 1)[0] + "_policy.png"
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

        if math.dist(robot_pos, nodes[target_node]) < args.arrival_radius:
            current_node = target_node
            node_log.append(current_node)
            if current_node != "food":
                target_node = policy.get(current_node, "food")
                print(f"  step {step}: arrived at {current_node}, next target -> {target_node}")

        if terminated or truncated:
            break

    print(f"\nNode sequence followed: {' -> '.join(node_log)}")
    print(f"Result: {'SUCCESS' if success else 'did not reach the food within max_steps'}")

    out_path = args.out or f"ga_live_{args.map}.gif"
    imageio.mimsave(out_path, frames, fps=20, loop=0)
    print(f"Saved GIF: {out_path}  ({len(frames)} frames)")
    print(f"Saved policy image: {policy_out_path}")

    env.close()


if __name__ == "__main__":
    main()
