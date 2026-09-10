# -*- coding: utf-8 -*-
"""
Draw the GA's best chromosome (a full policy) as arrows on the grid --
the exact same diagram visualize_qlearning.py produces, reusing its
drawing code directly. Same green/red convention: green if an arrow
points at a real edge, red if it doesn't.

Usage
-----
    python visualize_ga.py --map u_shape --generations 100
"""

import argparse

import pygame

from edu_env import make_edu_env
from grid_world import build_grid
from ga_learning import train_ga
from visualize_qlearning import draw_policy


def main():
    parser = argparse.ArgumentParser(description="Visualize the GA's learned policy on the grid")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape"])
    parser.add_argument("--population_size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))

    print(f"Map: {args.map}  |  {len(nodes)} nodes  |  training GA for {args.generations} generations...")
    best_policy, fitness_curve, stats_per_gen, _ = train_ga(nodes, edges, population_size=args.population_size,
                                            generations=args.generations, seed=args.seed)

    env.render()
    draw_policy(env, nodes, edges, best_policy)

    out_path = args.out or f"ga_policy_{args.map}_gen{args.generations}.png"
    pygame.image.save(env.window, out_path)

    correct = sum(1 for s, a in best_policy.items() if a in edges.get(s, ()))
    print(f"\n{correct}/{len(best_policy)} states learned a correct move")
    print(f"Fitness -- generation 0: {fitness_curve[0]:.1f}  |  generation {len(fitness_curve)-1}: {fitness_curve[-1]:.1f}")
    print(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
