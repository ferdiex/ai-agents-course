# -*- coding: utf-8 -*-
"""
Single-command installation check: runs a lightweight version of every
module's own smoke test (the same things TESTING.md walks through by
hand, Sections 1-4 and beyond) and prints one pass/fail report at the
end. Does not replace TESTING.md -- if something fails here, go to the
matching TESTING.md section for the full diagnostic detail and
troubleshooting table.

Usage
-----
    python3 verify_install.py
"""

import sys
import traceback

RESULTS = []


def check(name):
    """Decorator: run a check function, catch anything, record pass/fail."""
    def decorator(fn):
        try:
            detail = fn()
            RESULTS.append((name, True, detail or ""))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return decorator


print("Running installation checks (this takes a few seconds)...\n")

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@check("1. peas_foraging.json is valid JSON")
def _():
    import json
    json.load(open("peas_foraging.json"))


@check("2. table_driven_agent.py runs standalone (no simulator needed)")
def _():
    import subprocess
    result = subprocess.run([sys.executable, "agents/table_driven_agent.py"],
                             capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-300:])
    if "rows" not in result.stdout:
        raise RuntimeError("unexpected output")


@check("3. Core imports resolve (edu_env, agents, foraging_env)")
def _():
    global make_edu_env
    from edu_env import make_edu_env
    from agents.simple_reflex_agent import SimpleReflexAgent
    from agents.model_based_reflex_agent import ModelBasedReflexAgent


@check("4. worlds/ has the four required world files")
def _():
    import edu_env
    sim_root = os.path.dirname(os.path.abspath(edu_env.__file__))
    # edu_env.py's own path-walking already found the simulator root once;
    # re-derive it the same way rather than assuming a relative path here.
    here = sim_root
    while not os.path.isfile(os.path.join(here, "foraging_env.py")):
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError("could not locate foraging_env.py")
        here = parent
    missing = [f for f in ("u_shape.json", "n_shape.json", "random_obstacles.json", "csp_dense.json")
               if not os.path.isfile(os.path.join(here, "worlds", f))]
    if missing:
        raise RuntimeError(f"missing in worlds/: {missing}")


@check("5. Environment builds and steps (default map)")
def _():
    env = make_edu_env(map_name="default", num_agents=1)
    obs_list, info = env.reset()
    assert obs_list[0].shape == (10,), f"unexpected obs shape {obs_list[0].shape}"
    env.step([0])
    env.close()


@check("6. Reflex agents act without error (u_shape, 1 episode)")
def _():
    from agents.simple_reflex_agent import SimpleReflexAgent
    env = make_edu_env(map_name="u_shape", num_agents=1)
    agent = SimpleReflexAgent(model_path="braitenberg_avoidance.json")
    obs_list, info = env.reset()
    agent.reset()
    for _ in range(20):
        action = agent.act(obs_list[0], info)
        obs_list, _, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            break
    env.close()


@check("7. grid_world builds a graph on u_shape (10-30 nodes expected)")
def _():
    from grid_world import build_grid
    env = make_edu_env(map_name="u_shape", num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))
    env.close()
    if not (10 <= len(nodes) <= 40):
        raise RuntimeError(f"unexpected node count: {len(nodes)}")
    return f"{len(nodes)} nodes"


@check("8. search_policies finds a path (DFS/BFS/A*)")
def _():
    from grid_world import build_grid
    from search_policies import POLICIES
    env = make_edu_env(map_name="u_shape", num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))
    env.close()
    start = next(n for n in nodes if n != "food")
    for name, policy in POLICIES.items():
        path = policy(edges, nodes, start, "food")
        if path is None:
            raise RuntimeError(f"{name} found no path from {start} to food")


@check("9. csp_spawn solves the dense map (backtracking + generate-and-test)")
def _():
    from csp_spawn import build_csp_domains, solve_backtracking, solve_generate_and_test
    env = make_edu_env(map_name="csp_dense", num_agents=1)
    env.reset()
    robot_c, food_c = build_csp_domains(env)
    env.close()
    r1 = solve_backtracking(robot_c, food_c)
    r2 = solve_generate_and_test(robot_c, food_c, seed=0)
    if r1[0] is None or r2[0] is None:
        raise RuntimeError("no solution found by one or both methods")


@check("10. minimax_search agrees between plain and alpha-beta")
def _():
    from grid_world import build_grid
    from minimax_search import minimax_decision, alphabeta_decision
    env = make_edu_env(map_name="u_shape", num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height, food_pos=None)
    env.close()
    ids = sorted(nodes.keys())
    state = (ids[0], ids[-1], "predator")
    m1 = minimax_decision(state, edges, nodes, 3)
    m2 = alphabeta_decision(state, edges, nodes, 3)
    if m1[0] != m2[0] or abs(m1[1] - m2[1]) > 1e-9:
        raise RuntimeError(f"plain and alpha-beta disagree: {m1} vs {m2}")


@check("11. qlearning trains and improves (short run)")
def _():
    from grid_world import build_grid
    from qlearning import train_qlearning
    env = make_edu_env(map_name="u_shape", num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))
    env.close()
    Q, steps = train_qlearning(nodes, edges, episodes=150, seed=0)
    first10 = sum(steps[:10]) / 10
    last10 = sum(steps[-10:]) / 10
    if last10 >= first10:
        raise RuntimeError(f"no improvement: first10={first10:.1f} last10={last10:.1f}")


@check("12. ga_learning trains and improves (short run)")
def _():
    from grid_world import build_grid
    from ga_learning import train_ga
    env = make_edu_env(map_name="u_shape", num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=tuple(env.food_pos))
    env.close()
    best, fitness_curve, stats, _ = train_ga(nodes, edges, population_size=12, generations=15, seed=0)
    if fitness_curve[-1] <= fitness_curve[0]:
        raise RuntimeError(f"no improvement: gen0={fitness_curve[0]:.1f} genN={fitness_curve[-1]:.1f}")


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

print(f"{'Check':<62} {'Result'}")
print("-" * 78)
n_pass = 0
for name, ok, detail in RESULTS:
    status = "OK" if ok else "FAIL"
    n_pass += ok
    line = f"{name:<62} {status}"
    print(line)
    if not ok:
        print(f"    -> {detail}")
    elif detail:
        print(f"    -> {detail}")

print("-" * 78)
print(f"{n_pass}/{len(RESULTS)} checks passed.")

if n_pass < len(RESULTS):
    print("\nSomething failed above. Find the matching section number in TESTING.md")
    print("for the full diagnostic and troubleshooting table -- this script only")
    print("tells you WHAT failed, not why; TESTING.md has the why.")
    sys.exit(1)
else:
    print("\nEverything checks out. You're set up correctly.")
