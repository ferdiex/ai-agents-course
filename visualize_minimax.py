# -*- coding: utf-8 -*-
"""
Render the predator-prey chase as a single static image: both
trajectories traced on top of the navigation grid, from a frozen starting
snapshot -- same "one photo, not a live simulation" spirit as
visualize_search.py and visualize_utility.py.

Uses the +1 / 0 / -1 zero-sum evaluation (minimax_search.py's
simulate_chase_zerosum), not the distance-based one used for the
alpha-beta pruning table in TESTING.md Section 12.1 -- that one is about
counting nodes explored, this one is about the actual game result.

A note on colors: ForagingEnv.render() always draws its own single robot
(the leftover from the un-wrapped simulator) in a blue close to
(40,120,220). An earlier version of this script colored the prey almost
exactly that same blue, making the two impossible to tell apart. The prey
now uses a clearly different color (magenta) specifically to avoid that
collision -- if you still see only one moving marker, that lone blue dot
is the simulator's own idle robot, not part of this game at all.

Usage
-----
    python visualize_minimax.py --map u_shape --depth 3 --predator n0 --prey n10   # capture
    python visualize_minimax.py --map u_shape --depth 3 --predator n0 --prey n11   # draw (real cycle)
"""

import argparse
import math

import pygame

from edu_env import make_edu_env
from grid_world import build_grid
from minimax_search import simulate_chase_zerosum

PREDATOR_COLOR = (220, 20, 60)     # red
PREY_COLOR = (200, 0, 200)         # magenta -- deliberately far from ForagingEnv's robot blue
CAPTURE_COLOR = (30, 160, 30)      # green: +1
DRAW_COLOR = (230, 150, 0)         # orange: 0
ESCAPE_COLOR = (120, 120, 120)     # gray: -1, prey reached safe distance
NODE_COLOR = (170, 170, 170)
EDGE_COLOR = (215, 215, 215)
LABEL_COLOR = (0, 0, 0)

OUTCOME_STYLE = {
    "capture": (CAPTURE_COLOR, "CAPTURE (+1): predator wins"),
    "draw": (DRAW_COLOR, "DRAW (0): still in the danger zone, no capture yet"),
    "prey_escaped": (ESCAPE_COLOR, "PREY ESCAPED (-1): reached safe distance"),
}


def _offset_path(path_points, spread=6):
    """
    Returns a list of points to draw as connected segments, where each
    segment is nudged perpendicular to its own direction, alternating
    sides by ply index. Without this, a "there and back" trip along the
    same edge draws as one overlapping line and looks like a single
    one-way move -- exactly the confusion this function exists to avoid.
    """
    if len(path_points) < 2:
        return list(path_points)
    offset_points = [path_points[0]]
    for i in range(len(path_points) - 1):
        x0, y0 = path_points[i]
        x1, y1 = path_points[i + 1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            offset_points.append(path_points[i + 1])
            continue
        # perpendicular unit vector, alternating side by ply index
        px, py = -dy / length, dx / length
        side = 1 if i % 2 == 0 else -1
        ox, oy = px * spread * side, py * spread * side
        offset_points.append((x1 + ox, y1 + oy))
    return offset_points


def draw_chase(env, nodes, edges, trajectory, outcome):
    pygame.font.init()
    font = pygame.font.SysFont(None, 16)
    ply_font = pygame.font.SysFont(None, 15)
    big_font = pygame.font.SysFont(None, 22)
    env.render()

    for a, neighbors in edges.items():
        for b in neighbors:
            pygame.draw.line(env.window, EDGE_COLOR, nodes[a], nodes[b], 1)
    for node_id, pos in nodes.items():
        pos_i = (int(pos[0]), int(pos[1]))
        pygame.draw.circle(env.window, NODE_COLOR, pos_i, 5)
        env.window.blit(font.render(node_id, True, LABEL_COLOR), (pos_i[0] + 7, pos_i[1] - 7))

    predator_path = [nodes[p] for p, _ in trajectory]
    prey_path = [nodes[q] for _, q in trajectory]

    predator_offset = _offset_path(predator_path, spread=7)
    prey_offset = _offset_path(prey_path, spread=7)

    if len(predator_offset) > 1:
        pygame.draw.lines(env.window, PREDATOR_COLOR, False, predator_offset, 3)
    if len(prey_offset) > 1:
        pygame.draw.lines(env.window, PREY_COLOR, False, prey_offset, 3)

    # Ply-number labels at each offset point, so the order (and any
    # back-and-forth) is readable directly off the image, not just
    # inferred from a single overlapping line.
    for i, pt in enumerate(predator_offset):
        pygame.draw.circle(env.window, PREDATOR_COLOR, [int(c) for c in pt], 3)
        env.window.blit(ply_font.render(str(i), True, PREDATOR_COLOR), (pt[0] + 4, pt[1] + 4))
    for i, pt in enumerate(prey_offset):
        pygame.draw.circle(env.window, PREY_COLOR, [int(c) for c in pt], 3)
        env.window.blit(ply_font.render(str(i), True, PREY_COLOR), (pt[0] - 14, pt[1] - 14))

    # Hollow rings mark the starting positions (ply 0, un-offset -- the
    # true starting node).
    pygame.draw.circle(env.window, PREDATOR_COLOR, [int(c) for c in predator_path[0]], 10, 2)
    pygame.draw.circle(env.window, PREY_COLOR, [int(c) for c in prey_path[0]], 10, 2)

    outcome_color, outcome_text = OUTCOME_STYLE[outcome]

    if outcome == "capture":
        pygame.draw.circle(env.window, outcome_color, [int(c) for c in predator_path[-1]], 13, 4)
    elif outcome == "draw":
        pygame.draw.circle(env.window, outcome_color, [int(c) for c in predator_path[-1]], 13, 4)
        pygame.draw.circle(env.window, outcome_color, [int(c) for c in prey_path[-1]], 13, 4)
    else:  # prey_escaped
        pygame.draw.circle(env.window, PREDATOR_COLOR, [int(c) for c in predator_path[-1]], 10)
        pygame.draw.circle(env.window, outcome_color, [int(c) for c in prey_path[-1]], 13, 4)

    # Legend: the three possible outcomes, with the one that happened
    # highlighted in bold at the top.
    y = 10
    env.window.blit(big_font.render(outcome_text, True, outcome_color), (10, y))
    y += 26
    for key, (color, text) in OUTCOME_STYLE.items():
        marker = ">> " if key == outcome else "    "
        pygame.draw.line(env.window, color, (10, y + 6), (30, y + 6), 4)
        env.window.blit(font.render(marker + text, True, LABEL_COLOR), (36, y))
        y += 17

    y += 6
    pygame.draw.line(env.window, PREDATOR_COLOR, (10, y + 6), (30, y + 6), 4)
    env.window.blit(font.render("predator path", True, LABEL_COLOR), (36, y))
    y += 17
    pygame.draw.line(env.window, PREY_COLOR, (10, y + 6), (30, y + 6), 4)
    env.window.blit(font.render("prey path", True, LABEL_COLOR), (36, y))


def main():
    parser = argparse.ArgumentParser(description="Visualize a zero-sum minimax predator-prey chase")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape", "default", "csp_dense"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--safe_distance", type=float, default=250.0,
                         help="Distance (px) at which the prey is considered to have escaped for good")
    parser.add_argument("--predator", default=None, help="Starting node id, e.g. n0 (default: auto-picks farthest pair)")
    parser.add_argument("--prey", default=None, help="Starting node id, e.g. n10")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height, food_pos=None)

    if args.predator and args.prey:
        predator_start, prey_start = args.predator, args.prey
    else:
        node_ids = sorted(nodes.keys())
        predator_start, prey_start = max(
            ((a, b) for a in node_ids for b in node_ids if a != b),
            key=lambda pair: math.dist(nodes[pair[0]], nodes[pair[1]]))

    state = (predator_start, prey_start, "predator")
    print(f"Map: {args.map}  |  depth={args.depth}  |  {len(nodes)} nodes")
    print(f"Predator starts {predator_start}, prey starts {prey_start}\n")
    trajectory, outcome = simulate_chase_zerosum(state, edges, nodes, args.depth,
                                                    safe_distance=args.safe_distance,
                                                    max_rounds=args.rounds, verbose=True)

    env.render()
    draw_chase(env, nodes, edges, trajectory, outcome)

    out_path = args.out or f"minimax_{outcome}_{args.map}_depth{args.depth}.png"
    pygame.image.save(env.window, out_path)

    print(f"\nResult: {outcome.upper()} after {len(trajectory)-1} plies")
    print(f"Saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
