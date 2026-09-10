# -*- coding: utf-8 -*-
"""
Coarse navigation grid ("the Prolog facts") used by goal-based search, CSP,
minimax, and the Q-learning agent later in the course.

This is the symbolic model of the world that the reflex-family agents
never had: a small graph of waypoints (nodes) connected by edges wherever
there is a clear line of sight between them, built once from a STATIC,
already-known obstacle layout. Deliberately coarse (roughly a dozen nodes)
so it can be drawn on a whiteboard and a search algorithm traced by hand.

Why the map matters
----------------------
ForagingEnv.reset() re-jitters the obstacle rectangles on every episode
for the "default" map (see foraging_env.py's reset()), but leaves
"u_shape" and "n_shape" untouched between episodes. That does NOT mean
this module only works on those two -- it means a grid is only ever valid
for the specific episode it was built from. build_grid_for_map() below
opens its own throwaway environment for quick one-off inspection (fine for
any map, including "default", as long as you only care about that one
snapshot). If you need a grid tied to a real, ongoing episode, build it
from that episode's own environment right after its own reset() instead --
see visualize_search.py for that pattern, which works on any map.
"""

import math

from edu_env import make_edu_env


# ---------------------------------------------------------------------
# Geometry helpers (segment/rectangle intersection, with a clearance
# margin so nodes and edges don't hug walls closer than the robot's own
# collision radius would allow in the real simulator).
# ---------------------------------------------------------------------

def _orientation(a, b, c):
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a, b, c):
    return (min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9 and
            min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9)


def _segments_intersect(p1, p2, p3, p4):
    o1, o2 = _orientation(p1, p2, p3), _orientation(p1, p2, p4)
    o3, o4 = _orientation(p3, p4, p1), _orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False


def _expand(rect, clearance):
    x, y, w, h = rect
    return (x - clearance, y - clearance, w + 2 * clearance, h + 2 * clearance)


def _point_in_rect(pt, rect):
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def _segment_intersects_rect(p1, p2, rect):
    if _point_in_rect(p1, rect) or _point_in_rect(p2, rect):
        return True
    x, y, w, h = rect
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return any(_segments_intersect(p1, p2, e0, e1) for e0, e1 in edges)


def _point_rect_distance(pt, rect):
    """Distance from a point to the nearest edge of a rectangle (0 if the
    point is inside it)."""
    x, y, w, h = rect
    dx = max(x - pt[0], 0, pt[0] - (x + w))
    dy = max(y - pt[1], 0, pt[1] - (y + h))
    return math.hypot(dx, dy)


def min_distance_to_obstacles(p1, p2, obstacles, samples=20):
    """
    Approximate how close the straight segment p1->p2 ever gets to any
    obstacle, by sampling points along it. Good enough for a coarse graph
    with short edges -- this is NOT meant to be exact computational
    geometry, just enough signal to tell "hugs a wall" from "passes well
    clear of everything" for a utility function. 0 means the segment
    touches or crosses an obstacle outright.
    """
    best = float("inf")
    for i in range(samples + 1):
        t = i / samples
        px = p1[0] + (p2[0] - p1[0]) * t
        py = p1[1] + (p2[1] - p1[1]) * t
        for rect in obstacles:
            d = _point_rect_distance((px, py), rect)
            if d < best:
                best = d
    return best


def has_line_of_sight(p1, p2, obstacles, clearance=20):
    """True if the straight segment p1->p2 avoids every obstacle by at
    least `clearance` pixels (roughly the robot's own collision margin)."""
    return not any(_segment_intersects_rect(p1, p2, _expand(rect, clearance))
                   for rect in obstacles)


def _point_is_clear(pt, obstacles, clearance):
    return not any(_point_in_rect(pt, _expand(rect, clearance)) for rect in obstacles)


# ---------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------

def _obstacle_corner_candidates(obstacles, clearance, world_width, world_height,
                                  boundary_margin=20):
    """
    Standard visibility-graph trick: a uniform grid can miss a narrow
    doorway that happens to fall between grid rows/columns (this is exactly
    what happened on the n_shape map -- see grid_world's module notes in
    README.md/TESTING.md for the story). Adding each obstacle's four
    corners, pushed outward by a bit more than the clearance margin, as
    extra candidate nodes fixes this robustly for any static map, without
    needing to hand-tune grid density per map.
    """
    push = clearance + 10
    candidates = []
    for (x, y, w, h) in obstacles:
        for cx, cy in [(x - push, y - push), (x + w + push, y - push),
                       (x + w + push, y + h + push), (x - push, y + h + push)]:
            if boundary_margin <= cx <= world_width - boundary_margin and \
               boundary_margin <= cy <= world_height - boundary_margin:
                candidates.append((cx, cy))
    return candidates


def _dedupe_candidates(points, min_separation=25):
    """Greedily drop candidates that land too close to one already kept,
    so obstacle corners don't clutter the graph with near-duplicate nodes."""
    kept = []
    for pt in points:
        if all(math.dist(pt, k) >= min_separation for k in kept):
            kept.append(pt)
    return kept


def build_grid(obstacles, world_width, world_height, food_pos=None,
                columns=3, rows=4, margin=90, clearance=20):
    """
    Lay out a `columns` x `rows` candidate grid over the world, add each
    obstacle's corners as extra candidates (so narrow doorways between grid
    rows/columns aren't missed), drop anything too close to an obstacle or
    already-kept candidate, then connect every remaining pair of nodes that
    has a clear line of sight.

    Returns
    -------
    nodes : dict[str, (float, float)]
        Node id -> pixel coordinates.
    edges : dict[str, set[str]]
        Node id -> set of neighboring node ids (undirected: if "n3" is in
        edges["n7"], then "n7" is in edges["n3"] too).
    """
    xs = [margin + i * (world_width - 2 * margin) / (columns - 1) for i in range(columns)]
    ys = [margin + j * (world_height - 2 * margin) / (rows - 1) for j in range(rows)]
    grid_candidates = [(x, y) for y in ys for x in xs]
    corner_candidates = _obstacle_corner_candidates(obstacles, clearance, world_width, world_height)

    all_candidates = _dedupe_candidates(grid_candidates + corner_candidates)

    nodes = {}
    for i, pt in enumerate(all_candidates):
        if _point_is_clear(pt, obstacles, clearance):
            nodes[f"n{i}"] = pt

    if food_pos is not None:
        nodes["food"] = (float(food_pos[0]), float(food_pos[1]))

    edges = {node_id: set() for node_id in nodes}
    items = list(nodes.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            id_a, pos_a = items[i]
            id_b, pos_b = items[j]
            if has_line_of_sight(pos_a, pos_b, obstacles, clearance):
                edges[id_a].add(id_b)
                edges[id_b].add(id_a)

    return nodes, edges


def nearest_visible_node(point, nodes, obstacles, clearance=20):
    """
    Connect an arbitrary continuous point (e.g. the robot's live position)
    to the closest grid node it has a clear line of sight to. Falls back to
    the plain closest node (ignoring obstacles) if none are visible, which
    should only happen if the point itself is in a spot the grid can't see
    at all -- worth investigating, not silently swallowing, if it happens.
    """
    visible = [(math.dist(point, pos), node_id) for node_id, pos in nodes.items()
               if has_line_of_sight(point, pos, obstacles, clearance)]
    if visible:
        return min(visible)[1]
    return min((math.dist(point, pos), node_id) for node_id, pos in nodes.items())[1]


def print_edge_facts(edges):
    """Print the graph as Prolog-style facts: edge(n3, n7)."""
    seen = set()
    for a, neighbors in sorted(edges.items()):
        for b in sorted(neighbors):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            print(f"edge({a}, {b}).")


# ---------------------------------------------------------------------
# Wiring this to the real simulator
# ---------------------------------------------------------------------

def extract_static_layout(map_name):
    """
    Build a throwaway environment, reset it once, and read off its
    obstacle rectangles and food position: a single snapshot, valid for
    exactly the episode it came from.

    For "u_shape" and "n_shape", obstacles never change between episodes
    (see foraging_env.py's reset()), so a snapshot happens to stay valid
    across many episodes too -- convenient, but not something to rely on
    for "default", whose obstacles ARE re-randomized on every reset(). If
    you need a grid for "default" (or for any map, really), the robust
    pattern is to build it from the SAME environment instance your agent
    is actually using, right after that instance's own reset() -- see
    visualize_search.py, which does exactly this and works on any map.
    This function stays around as a quick way to inspect a map's grid in
    isolation (as grid_world.py's own __main__ block does below).
    """
    env = make_edu_env(map_name=map_name, num_agents=1)
    env.reset()
    obstacles = list(env.obstacles)
    food_pos = tuple(env.food_pos)
    world_width, world_height = env.config.world_width, env.config.world_height
    env.close()
    return obstacles, food_pos, world_width, world_height


def build_grid_for_map(map_name, columns=3, rows=4, margin=90, clearance=20):
    """Convenience one-call version: static map name in, (nodes, edges) out."""
    obstacles, food_pos, width, height = extract_static_layout(map_name)
    return build_grid(obstacles, width, height, food_pos=food_pos,
                       columns=columns, rows=rows, margin=margin, clearance=clearance)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect the navigation grid for a static map")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape", "default"])
    args = parser.parse_args()

    nodes, edges = build_grid_for_map(args.map)

    print(f"Map: {args.map}")
    print(f"Nodes: {len(nodes)}  |  Edges: {sum(len(n) for n in edges.values()) // 2}\n")

    print("--- Prolog-style facts ---")
    print_edge_facts(edges)

    print("\n--- Node coordinates (pixels) ---")
    for node_id, pos in sorted(nodes.items()):
        print(f"  {node_id}: ({pos[0]:.0f}, {pos[1]:.0f})")
