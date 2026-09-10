# -*- coding: utf-8 -*-
"""
The one visualization that makes this unambiguously a GA and not "RL with
extra steps": a montage showing several DIFFERENT individuals from the
population side by side, at generation 0 (random, wildly different from
each other) versus the final generation (converged, mostly similar and
mostly correct).

Q-learning has no equivalent of this picture -- it only ever has ONE
policy at any point in training, refined incrementally. A GA always has a
whole population of competing candidate solutions; this is what that
actually looks like, not just a claim in a docstring.

Usage
-----
    python visualize_ga_population.py --map u_shape --generations 100
"""

import argparse

import pygame

from edu_env import make_edu_env
from grid_world import build_grid
from ga_learning import train_ga, rollout_fitness
from visualize_qlearning import draw_policy

THUMB_W, THUMB_H = 260, 195
PADDING = 20
LABEL_H = 26


def render_thumbnail(env, nodes, edges, chromosome):
    env.render()
    draw_policy(env, nodes, edges, chromosome)
    full = env.window.copy()
    return pygame.transform.smoothscale(full, (THUMB_W, THUMB_H))


def main():
    parser = argparse.ArgumentParser(description="Show GA population diversity: generation 0 vs. final")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape"])
    parser.add_argument("--population_size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--n_individuals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))

    gen0, genN = 0, args.generations - 1
    print(f"Training GA for {args.generations} generations, "
          f"capturing the population at generation {gen0} and {genN}...")
    _, _, _, snapshots = train_ga(nodes, edges, population_size=args.population_size,
                                    generations=args.generations, seed=args.seed,
                                    snapshot_generations=[gen0, genN])

    states = [n for n in nodes if n != "food"]
    n = args.n_individuals

    pygame.font.init()
    font = pygame.font.SysFont(None, 20)
    title_font = pygame.font.SysFont(None, 26)

    canvas_w = n * THUMB_W + (n + 1) * PADDING
    canvas_h = 2 * (THUMB_H + LABEL_H) + 3 * PADDING + 40
    canvas = pygame.Surface((canvas_w, canvas_h))
    canvas.fill((255, 255, 255))
    canvas.blit(title_font.render("GA population: diversity at generation 0 vs. after evolving",
                                    True, (0, 0, 0)), (PADDING, 8))

    rows = [
        (f"Generation {gen0} (random, diverse)", gen0),
        (f"Generation {genN} (evolved, converged)", genN),
    ]
    for row, (gen_label, gen_num) in enumerate(rows):
        y0 = 40 + row * (THUMB_H + LABEL_H + PADDING)
        canvas.blit(font.render(gen_label, True, (0, 0, 0)), (PADDING, y0))
        population = snapshots[gen_num][:n]
        for i, chrom in enumerate(population):
            thumb = render_thumbnail(env, nodes, edges, chrom)
            x = PADDING + i * (THUMB_W + PADDING)
            y = y0 + LABEL_H
            canvas.blit(thumb, (x, y))
            fitness = rollout_fitness(chrom, edges, "food", states, max_steps=60)
            canvas.blit(font.render(f"individual #{i+1}: fitness={fitness:.0f}", True, (0, 0, 0)),
                        (x, y + THUMB_H + 2))

    out_path = args.out or f"ga_population_{args.map}.png"
    pygame.image.save(canvas, out_path)
    print(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
