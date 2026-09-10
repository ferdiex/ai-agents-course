# -*- coding: utf-8 -*-
"""
Interchangeable search policies over the grid_world graph.

Every function here has the same signature -- (edges, nodes, start, goal) ->
list of node ids from start to goal, or None if unreachable -- so a
goal-based agent can treat "which search algorithm to use" as a single
swappable argument, the same way the rest of this course treats "which
agent type to use" as a swappable class.

DFS is written to mirror the two-clause Prolog rule directly:

    path(X, Y) :- edge(X, Y).
    path(X, Z) :- edge(X, Y), path(Y, Z).

with exactly one addition Prolog's bare rule doesn't have: a visited set.
Try removing it and running dfs_path on this module's own test graph
(which has cycles, since edges come from mutual line-of-sight) -- it will
recurse forever, which is the point: naive backward-chaining resolution
has no notion of "already tried this", and a graph with a cycle needs one.
"""

import heapq
import math
from collections import deque


def dfs_path(edges, nodes, start, goal):
    """Depth-first search. Not guaranteed shortest -- just A path, if any.

    Neighbors are visited in sorted (not arbitrary set-iteration) order.
    This matters more than it looks: Python randomizes string hashing per
    process by default, so iterating a plain `set` of neighbor ids gives a
    DIFFERENT order every time you run the script -- meaning two students
    running the exact same code on the exact same map could get two
    different (both valid) DFS paths, with no way to compare notes in
    class. Sorting removes that source of confusion entirely."""

    def _dfs(node, visited):
        if node == goal:
            return [node]
        visited.add(node)
        for neighbor in sorted(edges.get(node, ())):
            if neighbor in visited:
                continue
            sub_path = _dfs(neighbor, visited)
            if sub_path is not None:
                return [node] + sub_path
        return None

    return _dfs(start, set())


def bfs_path(edges, nodes, start, goal):
    """Breadth-first search. Guaranteed fewest EDGES (hops), not necessarily
    the shortest total distance -- that distinction matters once nodes are
    spaced unevenly, which is exactly why A* exists."""
    frontier = deque([start])
    came_from = {start: None}

    while frontier:
        node = frontier.popleft()
        if node == goal:
            break
        for neighbor in sorted(edges.get(node, ())):
            if neighbor not in came_from:
                came_from[neighbor] = node
                frontier.append(neighbor)

    if goal not in came_from:
        return None

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def astar_path(edges, nodes, start, goal):
    """A* using straight-line (Euclidean) distance as the heuristic --
    admissible here because no real path between two points can be shorter
    than the straight line between them."""

    def heuristic(node_id):
        return math.dist(nodes[node_id], nodes[goal])

    def edge_cost(a, b):
        return math.dist(nodes[a], nodes[b])

    # Heap entries are (priority, node_id) tuples; ties compare node_id
    # strings lexicographically, which is enough to make the result
    # deterministic across runs without any extra bookkeeping.
    open_heap = [(heuristic(start), start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node == goal:
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in sorted(edges.get(node, ())):
            tentative = g_score[node] + edge_cost(node, neighbor)
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = node
                g_score[neighbor] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(neighbor), neighbor))

    return None


POLICIES = {"dfs": dfs_path, "bfs": bfs_path, "astar": astar_path}


def utility_astar_path(edges, nodes, obstacles, start, goal,
                         risk_weight=3.0, comfort_distance=40.0):
    """
    The utility-based agent's search: not "what's the shortest path?" but
    "what's the best path once we also weigh in how much we dislike
    hugging a wall?" Same A* machinery as astar_path -- the only thing
    that changes is what counts as "cost" for an edge:

        edge_cost = distance + risk_weight * max(0, comfort_distance - clearance)

    where `clearance` is how close that edge ever gets to an obstacle.
    An edge that stays farther than `comfort_distance` from every wall
    costs exactly its raw distance, same as plain astar_path; an edge that
    hugs a wall costs more, in proportion to how close and how heavily
    `risk_weight` penalizes that.

    Note on correctness: the heuristic below is still plain straight-line
    distance to the goal, which underestimates the TRUE cost here (since
    risk_weight * max(0, ...) is never negative) -- so it remains
    admissible and A* still finds the truly best (lowest-cost) path. It
    is just a less-informed heuristic than one that also accounted for
    risk would be, which only means more nodes get explored, not that the
    answer is wrong.
    """
    from grid_world import min_distance_to_obstacles

    edge_cost_cache = {}

    def edge_cost(a, b):
        key = frozenset((a, b))
        if key not in edge_cost_cache:
            dist = math.dist(nodes[a], nodes[b])
            clearance = min_distance_to_obstacles(nodes[a], nodes[b], obstacles)
            risk = max(0.0, comfort_distance - clearance) * risk_weight
            edge_cost_cache[key] = dist + risk
        return edge_cost_cache[key]

    def heuristic(node_id):
        return math.dist(nodes[node_id], nodes[goal])

    open_heap = [(heuristic(start), start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node == goal:
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in sorted(edges.get(node, ())):
            tentative = g_score[node] + edge_cost(node, neighbor)
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = node
                g_score[neighbor] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(neighbor), neighbor))

    return None


def path_length(nodes, path):
    """Total Euclidean length of a path, for comparing policies honestly --
    BFS's fewest-hops path is not always the physically shortest one."""
    if not path:
        return float("inf")
    return sum(math.dist(nodes[path[i]], nodes[path[i + 1]]) for i in range(len(path) - 1))


if __name__ == "__main__":
    from grid_world import build_grid_for_map, print_edge_facts

    import argparse
    parser = argparse.ArgumentParser(description="Compare DFS/BFS/A* on a static map's grid")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape", "default"])
    args = parser.parse_args()

    nodes, edges = build_grid_for_map(args.map)

    # Pick, as the start, the node that is deceptively close to the goal in
    # a straight line but has NO direct edge to it -- i.e. a real detour is
    # required. That's the genuinely interesting test case (a naive
    # straight-line/greedy approach would fail here); any node already
    # directly connected to "food" would make every policy look identical.
    blocked_candidates = [n for n in nodes if n != "food" and "food" not in edges[n]]
    if blocked_candidates:
        start = min(blocked_candidates, key=lambda n: math.dist(nodes[n], nodes["food"]))
    else:
        start = max((n for n in nodes if n != "food"),
                    key=lambda n: math.dist(nodes[n], nodes["food"]))
    goal = "food"

    print(f"Map: {args.map}  |  start={start} {nodes[start]}  goal={goal} {nodes[goal]}\n")

    for name, policy in POLICIES.items():
        path = policy(edges, nodes, start, goal)
        if path is None:
            print(f"{name.upper():6s}: no path found")
        else:
            length = path_length(nodes, path)
            print(f"{name.upper():6s}: {len(path)} nodes, length={length:7.1f}px  path={path}")
