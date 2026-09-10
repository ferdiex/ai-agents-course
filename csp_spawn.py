# -*- coding: utf-8 -*-
"""
CSP formulation of "where do the robot and the food go?" -- the same
problem ForagingEnv._sample_pose()/reset() already solves today via
generate-and-test (draw a random point, check it, throw it away if it
fails, repeat up to 1000 times).

Like the classic N-Queens CSP, there is no path, no movement, no
trajectory here -- the SOLUTION is a single valid assignment:
(robot_position, food_position) that satisfies every constraint at once.
What we compare is how much WORK each method needs to find that
assignment, not how "good" it is.

Variables : robot_position, food_position
Domain    : nodes of grid_world's own coarse navigation grid (the SAME
            graph goal-based search uses), split into two subsets by
            which region each node falls in (see ROBOT_REGION/FOOD_REGION)
Constraints:
  - unary:  already enforced by construction -- grid_world.build_grid()
            only keeps nodes with a clearance margin from every obstacle,
            so every domain member already satisfies "not on an obstacle"
  - binary: 100 <= distance(robot_position, food_position) <= 300

A note on an earlier version of this file
--------------------------------------------
An earlier attempt used a much finer domain (a dense 434+96-point grid)
covering the same regions. That version made backtracking dramatically
WORSE than generate-and-test (538 checks vs. ~14), not better -- and that
result was correct, not a bug. Filtering (or even just enumerating) a
domain that large has a fixed cost proportional to its size, while
generate-and-test's cost is proportional to 1/(success probability). With
domain size 530 and a success probability of ~6.8% (~15 draws expected),
530 >> 15, so blind sampling wins outright. Systematic search only beats
random sampling when the domain is SMALL relative to how rare valid
combinations are -- not simply "when the problem is hard". Reusing
grid_world's coarse graph (typically 15-30 nodes total) instead of a fine
grid is what finally makes the domain small enough for that trade to favor
backtracking. See TESTING.md for the numbers from both attempts.

Usage
-----
    python csp_spawn.py --map csp_dense --seed 5
    python csp_spawn.py --map csp_dense --trials 200   # averaged comparison
"""

import argparse
import math

import numpy as np
import pygame

from edu_env import make_edu_env
from grid_world import build_grid

ROBOT_REGION = ((100, 700), (40, 300))   # matches _sample_pose's agent_idx=0 range
FOOD_REGION = ((280, 500), (280, 430))   # matches the u_env randomize_food range
DISTANCE_RANGE = (100, 300)


def _in_region(pt, region):
    (x0, x1), (y0, y1) = region
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def build_csp_domains(env):
    """
    Build grid_world's coarse graph from THIS run's own obstacles (see
    grid_world.py's note on why that matters), then split its nodes into
    the robot-candidate and food-candidate subsets. No obstacle-clearance
    check needed here -- build_grid() already only keeps nodes with a
    clearance margin from every obstacle, so unary node-consistency is
    already done by construction.
    """
    nodes, _ = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                            columns=6, rows=5)
    robot_candidates = [pos for pos in nodes.values() if _in_region(pos, ROBOT_REGION)]
    food_candidates = [pos for pos in nodes.values() if _in_region(pos, FOOD_REGION)]
    return robot_candidates, food_candidates


def solve_backtracking(robot_candidates, food_candidates, distance_range=DISTANCE_RANGE):
    """
    Systematic search over the two (already node-consistent) domains: try
    every pair in a fixed order, stop at the first one satisfying the
    binary distance constraint.

    Returns (robot_pos, food_pos, nodes_visited, proved_no_solution).
    If the loop runs out without finding anything, proved_no_solution is
    True: every single pair in the domain was checked, so there being no
    valid one is a CERTAIN fact, not a guess -- this is completeness, the
    one advantage generate-and-test can never offer, regardless of how
    many attempts it's given.
    """
    nodes_visited = 0
    for robot in robot_candidates:
        for food in food_candidates:
            nodes_visited += 1
            if distance_range[0] <= math.dist(robot, food) <= distance_range[1]:
                return robot, food, nodes_visited, False
    return None, None, nodes_visited, True


def solve_generate_and_test(robot_candidates, food_candidates, distance_range=DISTANCE_RANGE,
                              max_attempts=200000, seed=0):
    """
    Draw i.i.d. random picks (with replacement) from the SAME two domains
    backtracking uses, until a compatible pair turns up.

    Returns (robot_pos, food_pos, attempts, proved_no_solution).
    proved_no_solution is ALWAYS False here, even when max_attempts is
    exhausted -- that's the point being made: running out of attempts
    means "didn't find one this time", never "there isn't one". Compare
    against solve_backtracking's honest True when it genuinely proves
    impossibility.
    """
    rng = np.random.RandomState(seed)
    for attempt in range(1, max_attempts + 1):
        robot = robot_candidates[rng.randint(len(robot_candidates))]
        food = food_candidates[rng.randint(len(food_candidates))]
        if distance_range[0] <= math.dist(robot, food) <= distance_range[1]:
            return robot, food, attempt, False
    return None, None, max_attempts, False


def draw_solution(env, robot_pos, food_pos, label_prefix, out_path):
    pygame.font.init()
    font = pygame.font.SysFont(None, 22)
    env.render()
    pygame.draw.circle(env.window, (220, 20, 60), (int(robot_pos[0]), int(robot_pos[1])), 10, 3)
    pygame.draw.circle(env.window, (255, 140, 0), (int(food_pos[0]), int(food_pos[1])), 10, 3)
    label = font.render(f"{label_prefix}: robot + food placed", True, (0, 0, 0))
    env.window.blit(label, (10, 10))
    pygame.image.save(env.window, out_path)


def main():
    parser = argparse.ArgumentParser(description="CSP: robot+food placement, backtracking vs generate-and-test")
    parser.add_argument("--map", default="csp_dense", choices=["default", "csp_dense", "u_shape", "n_shape"])
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--trials", type=int, default=1,
                         help="Run more than once (different seeds) to compare AVERAGE cost, not just one instance")
    parser.add_argument("--dist_min", type=float, default=DISTANCE_RANGE[0])
    parser.add_argument("--dist_max", type=float, default=DISTANCE_RANGE[1])
    parser.add_argument("--max_attempts", type=int, default=5000,
                         help="Generate-and-test gives up after this many attempts")
    args = parser.parse_args()
    distance_range = (args.dist_min, args.dist_max)

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()

    robot_candidates, food_candidates = build_csp_domains(env)
    print(f"Domain sizes: {len(robot_candidates)} robot candidates, {len(food_candidates)} food candidates "
          f"({len(robot_candidates) + len(food_candidates)} total, {len(robot_candidates)*len(food_candidates)} possible pairs)")
    print(f"Required distance range: {distance_range}")

    if args.trials == 1:
        robot_bt, food_bt, nodes_bt, proved_bt = solve_backtracking(robot_candidates, food_candidates, distance_range)
        robot_gt, food_gt, attempts_gt, proved_gt = solve_generate_and_test(
            robot_candidates, food_candidates, distance_range, max_attempts=args.max_attempts, seed=args.seed)

        print(f"\nMap: {args.map}")
        if robot_bt is not None:
            print(f"Backtracking      : FOUND a solution after checking {nodes_bt}/{len(robot_candidates)*len(food_candidates)} pairs -> robot={robot_bt}, food={food_bt}")
        else:
            print(f"Backtracking      : checked all {nodes_bt} pairs -> PROVED no solution exists. Certain, not a guess.")

        if robot_gt is not None:
            print(f"Generate-and-test : found one after {attempts_gt} random attempts -> robot={robot_gt}, food={food_gt}")
        else:
            print(f"Generate-and-test : exhausted {args.max_attempts} random attempts, found nothing -- "
                  f"but this does NOT prove no solution exists. It never can, no matter how high --max_attempts goes.")

        if robot_bt is not None:
            draw_solution(env, robot_bt, food_bt, "Backtracking", f"csp_backtracking_{args.map}.png")
            print(f"Saved: csp_backtracking_{args.map}.png")
        if robot_gt is not None:
            draw_solution(env, robot_gt, food_gt, "Generate-and-test", f"csp_generate_and_test_{args.map}.png")
            print(f"Saved: csp_generate_and_test_{args.map}.png")
    else:
        bt_counts, gt_counts, gt_found_count = [], [], 0
        for t in range(args.trials):
            _, _, nodes_bt, _ = solve_backtracking(robot_candidates, food_candidates, distance_range)
            _, _, attempts_gt, _ = solve_generate_and_test(robot_candidates, food_candidates, distance_range,
                                                              max_attempts=args.max_attempts, seed=t)
            bt_counts.append(nodes_bt)
            gt_counts.append(attempts_gt)
            if attempts_gt < args.max_attempts:
                gt_found_count += 1

        print(f"\nMap: {args.map}  |  {args.trials} trials")
        print(f"Backtracking      : avg {np.mean(bt_counts):.1f} node visits (min={min(bt_counts)}, max={max(bt_counts)}) "
              f"-- CONSTANT, and always ends with a certain answer either way")
        print(f"Generate-and-test : avg {np.mean(gt_counts):.1f} attempts    (min={min(gt_counts)}, max={max(gt_counts)}), "
              f"succeeded in {gt_found_count}/{args.trials} trials before hitting --max_attempts")

    env.close()


if __name__ == "__main__":
    main()
