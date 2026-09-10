# -*- coding: utf-8 -*-
"""
Draw the learned greedy policy as an arrow from every node to whatever
node Q-learning currently believes is the best move from there.

Green arrow: points at a real edge (learned correctly).
Red arrow: points at something that ISN'T a real edge -- that state
was never explored enough during training to learn better. Seeing even
one of these is a legitimate, useful result: it's direct visual evidence
of under-exploration, not a bug in the code.

Usage
-----
    python visualize_qlearning.py --map u_shape --episodes 2000
    python visualize_qlearning.py --map u_shape --episodes 200   # too few -- watch red arrows appear
"""

import argparse
import math

import pygame

from edu_env import make_edu_env
from grid_world import build_grid
from qlearning import train_qlearning, greedy_policy

NODE_COLOR = (170, 170, 170)
FOOD_COLOR = (255, 140, 0)
EDGE_COLOR = (225, 225, 225)
CORRECT_COLOR = (30, 160, 30)
WRONG_COLOR = (220, 20, 60)
LABEL_COLOR = (0, 0, 0)


def _draw_arrow(surface, color, start, end, width=3, head=8):
    pygame.draw.line(surface, color, start, end, width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    base1 = (end[0] - ux * head + px * head * 0.6, end[1] - uy * head + py * head * 0.6)
    base2 = (end[0] - ux * head - px * head * 0.6, end[1] - uy * head - py * head * 0.6)
    pygame.draw.polygon(surface, color, [tip, base1, base2])


def draw_policy(env, nodes, edges, policy, goal="food"):
    pygame.font.init()
    font = pygame.font.SysFont(None, 16)
    env.render()

    for a, neighbors in edges.items():
        for b in neighbors:
            pygame.draw.line(env.window, EDGE_COLOR, nodes[a], nodes[b], 1)

    for node_id, pos in nodes.items():
        color = FOOD_COLOR if node_id == goal else NODE_COLOR
        pos_i = (int(pos[0]), int(pos[1]))
        pygame.draw.circle(env.window, color, pos_i, 6)
        env.window.blit(font.render(node_id, True, LABEL_COLOR), (pos_i[0] + 8, pos_i[1] - 8))

    correct, wrong = 0, 0
    for state, action in policy.items():
        start = nodes[state]
        end = nodes[action]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        end_pulled = (start[0] + dx * 0.85, start[1] + dy * 0.85) if length > 1e-6 else end
        is_real = action in edges.get(state, ())
        color = CORRECT_COLOR if is_real else WRONG_COLOR
        correct += is_real
        wrong += not is_real
        _draw_arrow(env.window, color, start, end_pulled)

    big_font = pygame.font.SysFont(None, 22)
    env.window.blit(big_font.render(f"Learned policy: {correct}/{correct+wrong} arrows point at a real edge",
                                     True, (0, 0, 0)), (10, 10))
    legend_y = 34
    pygame.draw.line(env.window, CORRECT_COLOR, (10, legend_y + 6), (30, legend_y + 6), 4)
    env.window.blit(font.render("correct (real edge)", True, LABEL_COLOR), (36, legend_y))
    pygame.draw.line(env.window, WRONG_COLOR, (10, legend_y + 22), (30, legend_y + 22), 4)
    env.window.blit(font.render("wrong (under-explored)", True, LABEL_COLOR), (36, legend_y + 16))


def main():
    parser = argparse.ArgumentParser(description="Visualize a learned Q-learning policy on the grid")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))

    print(f"Map: {args.map}  |  {len(nodes)} nodes  |  training for {args.episodes} episodes...")
    Q, steps_per_episode = train_qlearning(nodes, edges, episodes=args.episodes, seed=args.seed)
    policy = greedy_policy(Q, nodes)

    env.render()
    draw_policy(env, nodes, edges, policy)

    out_path = args.out or f"qlearning_policy_{args.map}_ep{args.episodes}.png"
    pygame.image.save(env.window, out_path)

    correct_edges = sum(1 for s, a in policy.items() if a in edges.get(s, ()))
    first10 = sum(steps_per_episode[:10]) / 10
    last10 = sum(steps_per_episode[-10:]) / 10
    print(f"\n{correct_edges}/{len(policy)} states learned a correct move")
    print(f"Avg steps/episode -- first 10: {first10:.1f}  |  last 10: {last10:.1f}")
    print(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
