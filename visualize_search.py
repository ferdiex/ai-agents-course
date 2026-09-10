# -*- coding: utf-8 -*-
"""
Render a single static image: the robot's starting position, the full
navigation grid overlaid on top of it, and the path a chosen search policy
(dfs/bfs/astar) finds from the robot to the food -- highlighted in a
different color.

This deliberately does NOT move the robot or run the simulation loop. It
is a snapshot of classical planning's core assumption: the world is fully
known and static at planning time. If the robot actually had to walk this
plan step by step, it would eventually need to re-sense and re-plan
periodically (the classic "sense-plan-act" cycle, going back to Shakey the
robot at SRI in the 1970s) rather than trust one photo forever -- that is
a deliberate next step, not something this script does.

The grid is built from THIS run's own environment, right after its own
reset() -- not from a separate throwaway environment -- so it is valid
for any map, including "default" (whose obstacles only ever change
between episodes, never during one).

Usage
-----
    python visualize_search.py --map u_shape --algorithm astar
    python visualize_search.py --map u_shape --algorithm bfs
    python visualize_search.py --map u_shape --algorithm dfs

Use the SAME --seed across algorithm runs (default: 42) to compare
dfs/bfs/astar from the exact same starting position -- otherwise each run
gets an independently randomized spawn and the comparison isn't fair.
"""

import argparse

import numpy as np
import pygame

from edu_env import make_edu_env
from grid_world import build_grid, nearest_visible_node
from search_policies import POLICIES, path_length


NODE_COLOR = (30, 90, 200)
FOOD_COLOR = (255, 140, 0)
EDGE_COLOR = (185, 185, 185)
PATH_COLOR = (40, 170, 40)
ROBOT_COLOR = (220, 20, 60)
LABEL_COLOR = (0, 0, 0)


def draw_grid_overlay(env, nodes, edges, path=None, robot_to_start=None):
    """Draw the full grid, an optional highlighted path, and an optional
    robot-to-nearest-node connector directly onto env.window."""
    pygame.font.init()
    font = pygame.font.SysFont(None, 18)

    # All edges first (thin gray), so the path and nodes sit visibly on top.
    for a, neighbors in edges.items():
        for b in neighbors:
            pygame.draw.line(env.window, EDGE_COLOR, nodes[a], nodes[b], 1)

    if path:
        for i in range(len(path) - 1):
            pygame.draw.line(env.window, PATH_COLOR, nodes[path[i]], nodes[path[i + 1]], 4)

    path_set = set(path) if path else set()
    for node_id, pos in nodes.items():
        if node_id == "food":
            color = FOOD_COLOR
        elif node_id in path_set:
            color = PATH_COLOR
        else:
            color = NODE_COLOR
        pos_i = (int(pos[0]), int(pos[1]))
        pygame.draw.circle(env.window, color, pos_i, 6)
        label = font.render(node_id, True, LABEL_COLOR)
        env.window.blit(label, (pos_i[0] + 8, pos_i[1] - 8))

    if robot_to_start:
        robot_pos, start_node = robot_to_start
        pygame.draw.line(env.window, ROBOT_COLOR, robot_pos, nodes[start_node], 2)
        pygame.draw.circle(env.window, ROBOT_COLOR, (int(robot_pos[0]), int(robot_pos[1])), 8, 2)


def main():
    parser = argparse.ArgumentParser(description="Static snapshot: grid + search path overlay")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape", "default"])
    parser.add_argument("--algorithm", default="astar", choices=list(POLICIES.keys()))
    parser.add_argument("--seed", type=int, default=42,
                         help="Fixes the robot's spawn so dfs/bfs/astar can be compared from the same start")
    parser.add_argument("--out", default=None, help="Output PNG path (default: search_<algorithm>_<map>.png)")
    args = parser.parse_args()

    # ForagingEnv's spawn sampling (_sample_pose) draws from the GLOBAL
    # numpy RNG, not the per-instance one gymnasium's reset(seed=...) sets
    # up -- so this is the seed call that actually controls reproducibility
    # here, not env.reset(seed=...) by itself.
    np.random.seed(args.seed)

    env = make_edu_env(map_name=args.map, num_agents=1)
    obs_list, info = env.reset()

    # Built from THIS episode's own obstacles/food -- valid for any map.
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))

    robot_pos = (float(env.agent_pos[0][0]), float(env.agent_pos[0][1]))
    start_node = nearest_visible_node(robot_pos, nodes, env.obstacles)

    policy = POLICIES[args.algorithm]
    path = policy(edges, nodes, start_node, "food")

    env.render()
    draw_grid_overlay(env, nodes, edges, path=path, robot_to_start=(robot_pos, start_node))

    out_path = args.out or f"search_{args.algorithm}_{args.map}.png"
    pygame.image.save(env.window, out_path)

    if path is None:
        print(f"{args.algorithm.upper()}: no path found from {start_node} to food.")
    else:
        print(f"{args.algorithm.upper()}: {len(path)} nodes, length={path_length(nodes, path):.1f}px")
        print(f"Path: {path}")
    print(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
