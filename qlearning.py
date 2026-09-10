# -*- coding: utf-8 -*-
"""
Tabular Q-learning over grid_world's coarse graph -- but, unlike every
other module in this course, the agent here is NEVER handed the edge
list. It only discovers which moves work by trying them.

Why this matters (read before assuming this duplicates goal_based search)
----------------------------------------------------------------------------
grid_world.py's graph already excludes every edge that crosses an
obstacle -- goal-based search (A*) and this module both move over that
SAME graph, but they use it completely differently:

  - A* is HANDED the edge list and instantly computes the shortest path.
  - This module's action space is "attempt to move to node X", for EVERY
    node X in the graph, not just the real neighbors. If (current, X)
    isn't a real edge, the attempt just fails -- the agent stays put and
    pays a penalty, exactly like bumping into a wall it didn't know was
    there. Over many episodes, trial and error is what teaches it which
    attempts actually work.

If we handed this module the edge list the way A* gets it, Q-learning
would just rediscover the same shortest path A* already computed
instantly, far more slowly -- an honest but uninteresting demo. Learning
BLIND is what actually demonstrates why reinforcement learning exists as
a different idea from planning: it works even when the agent doesn't
already know the map.

This is also, loosely, the same move as training on a compressed/latent
representation instead of raw pixels -- except this "latent space" was
hand-designed by us (explicit x,y coordinates, human-readable node ids),
not learned by a neural net. Cheap imitation of the real thing, useful
for teaching the RL mechanics without the cost of a real perception
pipeline.

Static-map requirement: same as goal-based search, CSP, and minimax --
this needs a graph that doesn't change between episodes, so u_shape/n_shape
(or any map, as long as it's built from one frozen snapshot) are the
appropriate choices, not "default".

Usage
-----
    python qlearning.py --map u_shape --episodes 2000
"""

import argparse
import random

from grid_world import build_grid


def train_qlearning(nodes, edges, episodes=2000, alpha=0.5, gamma=0.9,
                      epsilon_start=1.0, epsilon_end=0.05,
                      step_penalty=-1.0, invalid_penalty=-5.0, goal_reward=100.0,
                      goal="food", max_steps_per_episode=200, seed=0):
    """
    Q[state, action] where action means "attempt to move to this node id"
    -- the FULL set of nodes, not just state's real neighbors. edges is
    used here only by the ENVIRONMENT, to decide whether an attempted move
    succeeds; the learning update itself never looks at edges directly.

    Returns (Q, steps_per_episode) -- the second is exactly the learning
    curve you'd plot for any RL agent: episodes should need fewer steps
    to reach the goal as training progresses, if learning is working.
    """
    rng = random.Random(seed)
    all_nodes = list(nodes.keys())
    Q = {(s, a): 0.0 for s in all_nodes for a in all_nodes if a != s}
    steps_per_episode = []

    for ep in range(episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / max(1, episodes - 1))
        state = rng.choice([n for n in all_nodes if n != goal])
        steps = 0

        for _ in range(max_steps_per_episode):
            steps += 1
            actions = [a for a in all_nodes if a != state]

            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(actions, key=lambda a: Q[(state, a)])

            if action in edges.get(state, ()):
                next_state = action
                reward = goal_reward if next_state == goal else step_penalty
            else:
                next_state = state  # bumped into a "wall" it didn't know was there
                reward = invalid_penalty

            best_next = max(Q[(next_state, a)] for a in all_nodes if a != next_state)
            Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])

            state = next_state
            if state == goal:
                break

        steps_per_episode.append(steps)

    return Q, steps_per_episode


def greedy_policy(Q, nodes, goal="food"):
    """The action Q-learning currently believes is best from every node --
    NOT guaranteed to match a real edge if that state was never explored
    enough to learn otherwise. See visualize_qlearning.py for how this
    shows up visually (a policy arrow pointing at a real neighbor vs. one
    still pointing at a wall)."""
    all_nodes = list(nodes.keys())
    policy = {}
    for s in all_nodes:
        if s == goal:
            continue
        actions = [a for a in all_nodes if a != s]
        policy[s] = max(actions, key=lambda a: Q[(s, a)])
    return policy


def print_q_table(Q, nodes, edges, goal="food", top_k=3):
    """
    Print the actual table, not just the policy derived from it -- this is
    literally the same kind of object the table-driven agent's lookup
    table was (Activity 1), except this one filled itself in through
    experience instead of being hand-written. For each state, shows the
    top `top_k` actions by Q-value, marking which ones are real edges.
    """
    all_nodes = sorted(nodes.keys())
    print(f"{'state':>6} | {'#1 action (Q-value)':>22} | {'#2 action (Q-value)':>22} | {'#3 action (Q-value)':>22}")
    print("-" * 82)
    for s in all_nodes:
        if s == goal:
            continue
        actions = [a for a in nodes if a != s]
        ranked = sorted(actions, key=lambda a: Q[(s, a)], reverse=True)[:top_k]
        cells = []
        for a in ranked:
            mark = "" if a in edges.get(s, ()) else "*"
            cells.append(f"{a}{mark} ({Q[(s, a)]:+7.1f})")
        while len(cells) < top_k:
            cells.append("")
        print(f"{s:>6} | {cells[0]:>22} | {cells[1]:>22} | {cells[2]:>22}")
    print("\n(* marks an action that is NOT a real edge -- Q-learning still")
    print(" assigns it a number, same as any other action, it just usually")
    print(" learns a low one after enough attempts fail.)")


if __name__ == "__main__":
    from edu_env import make_edu_env

    parser = argparse.ArgumentParser(description="Train tabular Q-learning on the coarse navigation grid")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))
    env.close()

    print(f"Map: {args.map}  |  {len(nodes)} nodes  |  training for {args.episodes} episodes...")
    Q, steps_per_episode = train_qlearning(nodes, edges, episodes=args.episodes, seed=args.seed)

    policy = greedy_policy(Q, nodes)
    correct_edges = sum(1 for s, a in policy.items() if a in edges.get(s, ()))
    print(f"\nGreedy policy learned for {len(policy)} states; "
          f"{correct_edges}/{len(policy)} point at a real edge "
          f"(the rest never got explored enough to learn better).")

    first10 = sum(steps_per_episode[:10]) / 10
    last10 = sum(steps_per_episode[-10:]) / 10
    print(f"Avg steps/episode -- first 10 episodes: {first10:.1f}  |  last 10 episodes: {last10:.1f}")
    print(f"Total real environment interactions across all {args.episodes} episodes: {sum(steps_per_episode)}")

    print("\n--- The actual Q-table (top 3 actions per state) ---\n")
    print_q_table(Q, nodes, edges)
