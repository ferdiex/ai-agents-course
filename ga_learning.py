# -*- coding: utf-8 -*-
"""
Genetic algorithm over the same "learn to navigate the grid, blind to the
edge list" problem qlearning.py solves -- same reward scheme, same
never-given-the-edges rule, but searching the space of POLICIES directly
instead of updating a value table one experience at a time.

Chromosome = a full policy: one gene per non-goal state, each gene an
action (a target node id, not necessarily a real edge -- same action
space as qlearning.py, "attempt to move to node X" for any X).

Default GA parameters are lifted directly from this project's own
evolution.json (population_size=24, generations=100, mutation_rate=0.2,
crossover_rate=0.7, elitism=2) -- a deliberate continuity thread, not a
coincidence: this project already used a GA for something else, and this
module borrows its tuning rather than inventing new numbers from scratch.

Usage
-----
    python ga_learning.py --map u_shape --generations 100
"""

import argparse
import random

from grid_world import build_grid

STEP_PENALTY = -1.0
INVALID_PENALTY = -5.0
GOAL_REWARD = 100.0


def _random_chromosome(all_nodes, goal, rng):
    states = [n for n in all_nodes if n != goal]
    return {s: rng.choice([a for a in all_nodes if a != s]) for s in states}


def rollout_fitness(chromosome, edges, goal, start_states, max_steps=60):
    """Same reward scheme as qlearning.py's train_qlearning, so the two
    modules' results are directly comparable, not just similar in spirit."""
    total = 0.0
    for start in start_states:
        state = start
        for _ in range(max_steps):
            if state == goal:
                break
            action = chromosome[state]
            if action in edges.get(state, ()):
                next_state = action
                reward = GOAL_REWARD if next_state == goal else STEP_PENALTY
            else:
                next_state = state  # invalid attempt: bounced back
                reward = INVALID_PENALTY
            total += reward
            state = next_state
            if state == goal:
                break
    return total / len(start_states)


def _tournament_select(scored, k, rng):
    contenders = rng.sample(scored, k)
    return max(contenders, key=lambda x: x[1])[0]


def _crossover(parent1, parent2, states, rng):
    return {s: (parent1[s] if rng.random() < 0.5 else parent2[s]) for s in states}


def _mutate(chromosome, all_nodes, mutation_rate, rng):
    out = dict(chromosome)
    for s in out:
        if rng.random() < mutation_rate:
            out[s] = rng.choice([a for a in all_nodes if a != s])
    return out


def train_ga(nodes, edges, goal="food", population_size=24, generations=100,
              mutation_rate=0.2, crossover_rate=0.7, elitism=2,
              tournament_k=3, max_steps=60, seed=0, snapshot_generations=None):
    """
    Returns (best_chromosome, best_fitness_per_gen, stats_per_gen, snapshots).

    stats_per_gen: list of (best, mean, worst) fitness for EVERY generation
    -- this is the population-level view Q-learning has no equivalent of,
    since Q-learning only ever has one agent, never a population with a
    spread that narrows over time.

    snapshots: {generation_number: [chromosomes...]} for whichever
    generations were listed in snapshot_generations (e.g. [0, generations-1]
    to compare the starting population against the final one) -- the whole
    population at that point, unsorted, so a caller can show several
    DIFFERENT individuals side by side, not just the single best one.
    """
    rng = random.Random(seed)
    all_nodes = list(nodes.keys())
    states = [n for n in all_nodes if n != goal]
    snapshot_generations = set(snapshot_generations or [])

    population = [_random_chromosome(all_nodes, goal, rng) for _ in range(population_size)]
    best_fitness_per_gen = []
    stats_per_gen = []
    snapshots = {}
    best_chromosome, best_fitness = None, float("-inf")

    for gen in range(generations):
        if gen in snapshot_generations:
            snapshots[gen] = [dict(c) for c in population]

        scored = [(c, rollout_fitness(c, edges, goal, states, max_steps)) for c in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        fitnesses = [f for _, f in scored]
        stats_per_gen.append((fitnesses[0], sum(fitnesses) / len(fitnesses), fitnesses[-1]))

        if scored[0][1] > best_fitness:
            best_fitness, best_chromosome = scored[0][1], scored[0][0]
        best_fitness_per_gen.append(scored[0][1])

        next_population = [c for c, _ in scored[:elitism]]
        while len(next_population) < population_size:
            p1 = _tournament_select(scored, tournament_k, rng)
            p2 = _tournament_select(scored, tournament_k, rng)
            child = _crossover(p1, p2, states, rng) if rng.random() < crossover_rate else dict(p1)
            child = _mutate(child, all_nodes, mutation_rate, rng)
            next_population.append(child)
        population = next_population

    if (generations - 1) in snapshot_generations:
        # capture the population AFTER the last generation's reproduction,
        # i.e. the one nobody has scored yet -- use the last scored one
        # instead, which is what actually produced best_chromosome.
        snapshots[generations - 1] = [c for c, _ in scored]

    return best_chromosome, best_fitness_per_gen, stats_per_gen, snapshots


if __name__ == "__main__":
    from edu_env import make_edu_env

    parser = argparse.ArgumentParser(description="Train a GA policy on the coarse navigation grid")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape"])
    parser.add_argument("--population_size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--mutation_rate", type=float, default=0.2)
    parser.add_argument("--crossover_rate", type=float, default=0.7)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))
    env.close()

    print(f"Map: {args.map}  |  {len(nodes)} nodes  |  "
          f"pop={args.population_size}  gens={args.generations}  "
          f"(defaults lifted from this project's own evolution.json)")

    best, fitness_curve, stats_per_gen, _ = train_ga(nodes, edges, population_size=args.population_size,
                                     generations=args.generations, mutation_rate=args.mutation_rate,
                                     crossover_rate=args.crossover_rate, elitism=args.elitism,
                                     seed=args.seed)

    correct = sum(1 for s, a in best.items() if a in edges.get(s, ()))
    print(f"\nBest chromosome: {correct}/{len(best)} states point at a real edge")
    print(f"Fitness -- generation 0: {fitness_curve[0]:.1f}  |  "
          f"generation {len(fitness_curve)-1}: {fitness_curve[-1]:.1f}")
    b0, m0, w0 = stats_per_gen[0]
    bN, mN, wN = stats_per_gen[-1]
    print(f"Population spread -- generation 0: best={b0:.1f} mean={m0:.1f} worst={w0:.1f}")
    print(f"Population spread -- generation {len(stats_per_gen)-1}: best={bN:.1f} mean={mN:.1f} worst={wN:.1f}")
    worst_case_budget = args.population_size * args.generations * (len(nodes) - 1) * 60
    print(f"\nWorst-case interaction budget (population x generations x states x max_steps): {worst_case_budget:,}")
    print("(An upper bound, not the actual count -- most rollouts end long before max_steps once")
    print(" a chromosome is any good. Actual instrumentation would need to be added to measure exactly.)")
