# -*- coding: utf-8 -*-
"""
Render a single static image comparing two routes from the same frozen
starting position to the food:

  - the plain shortest path (astar_path: minimizes distance only)
  - the utility-based path (utility_astar_path: trades distance off
    against how close the route passes to a wall)

Both drawn on the SAME image so the difference is visible at a glance,
not just in two numbers printed to a terminal.

Usage
-----
    python visualize_utility.py --map u_shape --seed 5
    python visualize_utility.py --map u_shape --seed 5 --risk_weight 6 --comfort_distance 60
"""

import argparse

import numpy as np
import pygame

from edu_env import make_edu_env
from grid_world import build_grid, nearest_visible_node
from search_policies import astar_path, utility_astar_path, path_length


NODE_COLOR = (150, 150, 150)
FOOD_COLOR = (255, 140, 0)
EDGE_COLOR = (200, 200, 200)
SHORTEST_COLOR = (220, 30, 30)     # red: plain shortest path
UTILITY_COLOR = (40, 110, 220)     # blue: utility-based path
SHARED_COLOR = (140, 60, 180)      # purple: edges both routes agree on
ROBOT_COLOR = (30, 160, 30)
LABEL_COLOR = (0, 0, 0)


def _edge_set(path):
    if not path:
        return set()
    return {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}


def draw_comparison(env, nodes, edges, shortest_path, utility_path, robot_to_start):
    pygame.font.init()
    font = pygame.font.SysFont(None, 18)

    for a, neighbors in edges.items():
        for b in neighbors:
            pygame.draw.line(env.window, EDGE_COLOR, nodes[a], nodes[b], 1)

    shortest_edges = _edge_set(shortest_path)
    utility_edges = _edge_set(utility_path)
    shared_edges = shortest_edges & utility_edges

    def draw_path_edges(edge_ids, color, width):
        for pair in edge_ids:
            a, b = tuple(pair)
            pygame.draw.line(env.window, color, nodes[a], nodes[b], width)

    draw_path_edges(shortest_edges - shared_edges, SHORTEST_COLOR, 4)
    draw_path_edges(utility_edges - shared_edges, UTILITY_COLOR, 4)
    draw_path_edges(shared_edges, SHARED_COLOR, 4)

    on_any_path = shortest_edges | utility_edges
    for node_id, pos in nodes.items():
        pos_i = (int(pos[0]), int(pos[1]))
        color = FOOD_COLOR if node_id == "food" else NODE_COLOR
        pygame.draw.circle(env.window, color, pos_i, 6)
        label = font.render(node_id, True, LABEL_COLOR)
        env.window.blit(label, (pos_i[0] + 8, pos_i[1] - 8))

    robot_pos, start_node = robot_to_start
    pygame.draw.line(env.window, ROBOT_COLOR, robot_pos, nodes[start_node], 2)
    pygame.draw.circle(env.window, ROBOT_COLOR, (int(robot_pos[0]), int(robot_pos[1])), 8, 2)

    # Small legend, top-left corner.
    legend = [
        (SHORTEST_COLOR, "shortest (astar)"),
        (UTILITY_COLOR, "utility-based"),
        (SHARED_COLOR, "both agree"),
    ]
    for i, (color, text) in enumerate(legend):
        y = 10 + i * 18
        pygame.draw.line(env.window, color, (10, y + 6), (30, y + 6), 4)
        env.window.blit(font.render(text, True, LABEL_COLOR), (36, y))


def main():
    parser = argparse.ArgumentParser(description="Compare shortest vs. utility-based routes on one image")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape", "default"])
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--risk_weight", type=float, default=5.0,
                         help="How heavily to penalize hugging a wall (0 = identical to plain astar)")
    parser.add_argument("--comfort_distance", type=float, default=60.0,
                         help="Pixel clearance below which an edge starts costing extra")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()

    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))

    robot_pos = (float(env.agent_pos[0][0]), float(env.agent_pos[0][1]))
    start_node = nearest_visible_node(robot_pos, nodes, env.obstacles)

    shortest = astar_path(edges, nodes, start_node, "food")
    utility = utility_astar_path(edges, nodes, env.obstacles, start_node, "food",
                                   risk_weight=args.risk_weight,
                                   comfort_distance=args.comfort_distance)

    env.render()
    draw_comparison(env, nodes, edges, shortest, utility, (robot_pos, start_node))

    out_path = args.out or f"utility_vs_shortest_{args.map}.png"
    pygame.image.save(env.window, out_path)

    def describe(name, path):
        if path is None:
            print(f"{name}: no path found")
        else:
            print(f"{name}: {path} | length={path_length(nodes, path):.1f}px")

    describe("Shortest (astar)  ", shortest)
    describe("Utility-based     ", utility)
    print(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
