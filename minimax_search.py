# -*- coding: utf-8 -*-
"""
Minimax (and alpha-beta pruning) for a predator-prey chase over the SAME
coarse navigation grid goal-based search already uses -- reusing the
infrastructure, not building a new one.

Why this is the one module that changes the observability assumption
------------------------------------------------------------------------
Every other module in this course keeps the robot's percept partial (8
proximity sensors, a food bearing, an odor). Classical minimax assumes
PERFECT information: both players know the exact game state, always. This
module deliberately breaks from partial observability -- both the
predator and the prey are assumed to know each other's exact position on
the grid at all times. That's not an oversight; it's what minimax, as
classically defined, requires. Worth saying explicitly in class: this is
a different assumption than every other module in this course, made on
purpose, not a relaxation snuck in quietly.

Game definition
------------------
State      : (predator_node, prey_node, turn), turn in {"predator","prey"}
Actions    : move to any node adjacent to the mover's current node (no
             "stay" option -- keeps the branching factor equal to node
             degree, and keeps the game from stalling out at a stand-off)
Terminal   : predator_node == prey_node (a capture)
Utility    : straight-line distance between predator and prey.
             0 at a capture (best possible for the predator, worst for
             the prey). The predator is the MIN player (wants distance
             small); the prey is the MAX player (wants distance large).
             This is a zero-sum game on a single shared quantity, which
             is exactly what makes it a natural fit for minimax as
             opposed to the multi-objective utility-based module.

Usage
-----
    python minimax_search.py --map u_shape --depth 4
"""

import argparse
import math

from grid_world import build_grid


def minimax_value(state, edges, nodes, depth, alpha=None, beta=None):
    """
    Plain minimax if alpha/beta are left as None; alpha-beta pruning if
    they're provided. Same function, same guaranteed answer either way --
    only the number of nodes visited differs. See minimax_decision() and
    alphabeta_decision() below for the two entry points, and TESTING.md
    for the actual measured difference in nodes explored.
    """
    predator_node, prey_node, turn = state
    if predator_node == prey_node:
        return 0.0
    if depth == 0:
        return math.dist(nodes[predator_node], nodes[prey_node])

    mover_node = predator_node if turn == "predator" else prey_node
    moves = sorted(edges[mover_node])

    if turn == "predator":  # MIN player: wants distance small
        best = float("inf")
        for nxt in moves:
            child = (nxt, prey_node, "prey")
            val = minimax_value(child, edges, nodes, depth - 1, alpha, beta)
            best = min(best, val)
            if beta is not None:
                if best <= (alpha if alpha is not None else float("-inf")):
                    break
                beta = min(beta, best)
        return best
    else:  # prey, MAX player: wants distance large
        best = float("-inf")
        for nxt in moves:
            child = (predator_node, nxt, "predator")
            val = minimax_value(child, edges, nodes, depth - 1, alpha, beta)
            best = max(best, val)
            if alpha is not None:
                if best >= (beta if beta is not None else float("inf")):
                    break
                alpha = max(alpha, best)
        return best


class _CountingDict(dict):
    """Tiny wrapper so we can count node visits without changing
    minimax_value's signature -- see nodes_explored in the two decision
    functions below."""
    counter = [0]

    def __getitem__(self, key):
        _CountingDict.counter[0] += 1
        return super().__getitem__(key)


def _decide(state, edges, nodes, depth, use_pruning):
    predator_node, prey_node, turn = state
    mover_node = predator_node if turn == "predator" else prey_node
    moves = sorted(edges[mover_node])

    counting_edges = _CountingDict(edges)
    _CountingDict.counter = [0]

    best_move, best_val = None, None
    alpha, beta = (float("-inf"), float("inf")) if use_pruning else (None, None)

    for nxt in moves:
        if turn == "predator":
            child = (nxt, prey_node, "prey")
        else:
            child = (predator_node, nxt, "predator")
        val = minimax_value(child, counting_edges, nodes, depth - 1, alpha, beta)

        if turn == "predator":
            if best_val is None or val < best_val:
                best_val, best_move = val, nxt
            if use_pruning:
                beta = min(beta, best_val)
        else:
            if best_val is None or val > best_val:
                best_val, best_move = val, nxt
            if use_pruning:
                alpha = max(alpha, best_val)

    return best_move, best_val, _CountingDict.counter[0]


def minimax_decision(state, edges, nodes, depth):
    """Returns (best_move, minimax_value, nodes_explored) for the player
    to move at `state`, using plain minimax (no pruning)."""
    return _decide(state, edges, nodes, depth, use_pruning=False)


def alphabeta_decision(state, edges, nodes, depth):
    """Same contract as minimax_decision, but with alpha-beta pruning.
    Guaranteed to return the exact same (best_move, minimax_value) as
    minimax_decision for the same inputs -- pruning only skips branches
    that provably can't change the answer, it never changes what the
    answer IS."""
    return _decide(state, edges, nodes, depth, use_pruning=True)


def simulate_chase(state, edges, nodes, depth, max_rounds=15):
    """
    Play the chase forward: both predator and prey use minimax (each
    looking `depth` plies ahead from their own turn) to pick every move,
    alternating, until a capture or `max_rounds` rounds pass. Returns the
    full trajectory as a list of (predator_node, prey_node) per ply,
    starting with the initial positions.

    Both sides being "equally smart" (same depth, same algorithm) is a
    deliberate choice -- it isolates the STRUCTURAL question (can the
    predator force a capture on this graph at all, given perfect play
    from both sides?) from any asymmetry in how clever either side is.
    """
    predator_node, prey_node, turn = state
    trajectory = [(predator_node, prey_node)]

    for _ in range(max_rounds):
        if predator_node == prey_node:
            break
        move, _, _ = minimax_decision((predator_node, prey_node, turn), edges, nodes, depth)
        if turn == "predator":
            predator_node = move
            turn = "prey"
        else:
            prey_node = move
            turn = "predator"
        trajectory.append((predator_node, prey_node))

    captured = predator_node == prey_node
    return trajectory, captured


# ---------------------------------------------------------------------
# True zero-sum variant: +1 / 0 / -1, with a REAL (not assumed) draw
# ---------------------------------------------------------------------
#
# Everything above this point scores a position by distance -- useful for
# the alpha-beta pruning demonstration (TESTING.md Section 12.1), but not
# how classical game-tree search is usually taught. This section plays it
# the textbook way instead:
#
#   +1  the predator captures the prey
#    0  a DRAW -- not "ran out of patience", but a state that repeats
#       along the same line of play. Since both players are deterministic
#       (always pick the same move given the same state), a repeated
#       state means the game is provably locked into that cycle forever.
#       That is exactly what a draw means in a game like chess, and it is
#       exactly what happened in the depth=4 evasion cycle found earlier.
#   -1  the search ran out of lookahead (depth reached 0) without finding
#       either a capture or a repeated state along this line. This is
#       NOT a proof the prey escapes -- it is an honest admission that
#       this depth wasn't enough to know. Searching deeper could reveal a
#       capture, a cycle, or just push the same uncertainty further out.
#
# `path_history` tracks the states visited along the CURRENT recursive
# line only (not across sibling branches, where revisiting a state is
# normal and not a cycle at all) -- that distinction is what makes the
# draw detection genuine rather than a guess.

# Everything above this point scores a position by distance -- useful for
# the alpha-beta pruning demonstration (TESTING.md Section 12.1), but not
# how classical game-tree search is usually taught. This section plays it
# the textbook way instead, with a genuine +1/0/-1 verdict at every leaf --
# no depth-limit placeholder standing in for "we don't know yet".
#
# SAFE_DISTANCE is the one new idea this needs: a threshold distance that
# defines "the prey got away". Every leaf of the search (whether reached
# by a capture, a REAL repeated state, or simply running out of lookahead)
# is scored by comparing the predator-prey distance AT THAT POINT against
# this threshold:
#
#   +1  the predator captures the prey (distance == 0) -- the prey lost.
#    0  the position is still in the "danger zone" (distance <=
#       SAFE_DISTANCE) but no capture has happened -- a draw. Nobody has
#       won yet, and if this exact state repeats (a real cycle), it never
#       will.
#   -1  the prey has reached SAFE_DISTANCE or farther -- the prey won.
#       This is now a real, evaluable fact about the position, not a
#       "ran out of patience" placeholder like the first version of this
#       module used.
#
# A cycle (a state repeating along the current line of play) and a plain
# depth cutoff are handled by the EXACT SAME rule: measure the distance
# right there and apply the threshold. This is actually more principled
# for the cycle case than it looks -- if the state provably repeats
# forever, the distance at that state repeats forever too, so whichever
# side of SAFE_DISTANCE it falls on is the position's true, permanent
# fate, not a guess.
#
# `path_history` tracks the states visited along the CURRENT recursive
# line only (not across sibling branches, where revisiting a state is
# normal and not a cycle at all) -- that distinction is what makes the
# draw detection genuine rather than a guess.

SAFE_DISTANCE = 250.0


def _leaf_value(predator_node, prey_node, nodes, safe_distance):
    d = math.dist(nodes[predator_node], nodes[prey_node])
    return -1.0 if d >= safe_distance else 0.0


def minimax_zerosum_value(state, edges, nodes, path_history, depth,
                            safe_distance=SAFE_DISTANCE, alpha=None, beta=None):
    predator_node, prey_node, turn = state

    if predator_node == prey_node:
        return 1.0
    if state in path_history:
        return _leaf_value(predator_node, prey_node, nodes, safe_distance)
    if depth == 0:
        return _leaf_value(predator_node, prey_node, nodes, safe_distance)

    new_history = path_history | {state}
    mover_node = predator_node if turn == "predator" else prey_node
    moves = sorted(edges[mover_node])

    if turn == "predator":  # MAX player: wants +1 (capture)
        best = float("-inf")
        for nxt in moves:
            child = (nxt, prey_node, "prey")
            val = minimax_zerosum_value(child, edges, nodes, new_history, depth - 1, safe_distance, alpha, beta)
            best = max(best, val)
            if alpha is not None:
                if best >= (beta if beta is not None else float("inf")):
                    break
                alpha = max(alpha, best)
        return best
    else:  # prey, MIN player: wants -1 (reach safety)
        best = float("inf")
        for nxt in moves:
            child = (predator_node, nxt, "predator")
            val = minimax_zerosum_value(child, edges, nodes, new_history, depth - 1, safe_distance, alpha, beta)
            best = min(best, val)
            if beta is not None:
                if best <= (alpha if alpha is not None else float("-inf")):
                    break
                beta = min(beta, best)
        return best


def _decide_zerosum(state, edges, nodes, depth, use_pruning, safe_distance=SAFE_DISTANCE):
    predator_node, prey_node, turn = state
    mover_node = predator_node if turn == "predator" else prey_node
    moves = sorted(edges[mover_node])

    best_move, best_val = None, None
    alpha, beta = (float("-inf"), float("inf")) if use_pruning else (None, None)
    history = frozenset({state})

    counting_edges = _CountingDict(edges)
    _CountingDict.counter = [0]

    for nxt in moves:
        child = (nxt, prey_node, "prey") if turn == "predator" else (predator_node, nxt, "predator")
        val = minimax_zerosum_value(child, counting_edges, nodes, history, depth - 1, safe_distance, alpha, beta)

        if turn == "predator":
            if best_val is None or val > best_val:
                best_val, best_move = val, nxt
            if use_pruning:
                alpha = max(alpha, best_val)
        else:
            if best_val is None or val < best_val:
                best_val, best_move = val, nxt
            if use_pruning:
                beta = min(beta, best_val)

    return best_move, best_val, _CountingDict.counter[0]


def zerosum_decision(state, edges, nodes, depth, safe_distance=SAFE_DISTANCE, use_pruning=True):
    """Returns (best_move, value) for the player to move at `state`, using
    the +1/0/-1 zero-sum evaluation. value is from the PREDATOR's point of
    view regardless of whose turn it is (predator wants it high, prey
    wants it low) -- same convention minimax_zerosum_value uses."""
    move, val, _ = _decide_zerosum(state, edges, nodes, depth, use_pruning, safe_distance)
    return move, val


def simulate_chase_zerosum(state, edges, nodes, depth, safe_distance=SAFE_DISTANCE,
                             max_rounds=15, verbose=False):
    """
    Like simulate_chase, but using the +1/0/-1 evaluation for move choice,
    AND checking the ACTUAL played trajectory (not just the search's
    internal lookahead) for a genuine repeated state -- if the real game
    revisits a state it has already been in, that's a real, provable draw
    happening live, not a hypothetical one buried inside the search.

    With verbose=True, prints each ply's decision: who moved, chosen move,
    value, and nodes explored WITH pruning vs WITHOUT -- concrete evidence
    of alpha-beta actually pruning THIS specific game, not just the
    abstract table from a single root position (Section 12.1).

    Returns (trajectory, outcome), outcome in {"capture", "draw", "prey_escaped"}.
    Since the state space is finite and both players are deterministic,
    a long enough max_rounds is GUARANTEED to end in one of these three --
    "prey_escaped" here means the ACTUAL distance is already >= safe_distance
    when the round budget runs out, which (thanks to the leaf rule above)
    is the same real verdict the search itself would have reached, not a
    fallback guess.
    """
    predator_node, prey_node, turn = state
    trajectory = [(predator_node, prey_node)]
    seen_states = {(predator_node, prey_node, turn)}

    if verbose:
        print(f"ply 0: predator={predator_node}  prey={prey_node}  (starting position)")

    outcome = None
    for ply in range(1, max_rounds + 1):
        if predator_node == prey_node:
            outcome = "capture"
            break

        cur_state = (predator_node, prey_node, turn)
        move, val, nodes_pruned = _decide_zerosum(cur_state, edges, nodes, depth, True, safe_distance)
        if verbose:
            _, _, nodes_plain = _decide_zerosum(cur_state, edges, nodes, depth, False, safe_distance)
            saved_pct = 100 * (1 - nodes_pruned / nodes_plain) if nodes_plain else 0
            mover = "predator" if turn == "predator" else "prey    "
            print(f"ply {ply}: {mover} at "
                  f"{predator_node if turn == 'predator' else prey_node} -> moves to {move}  "
                  f"(value={val:+.0f})  |  alpha-beta: {nodes_pruned} nodes, plain minimax: {nodes_plain} nodes "
                  f"({saved_pct:.0f}% saved)")

        if turn == "predator":
            predator_node = move
            turn = "prey"
        else:
            prey_node = move
            turn = "predator"

        new_state = (predator_node, prey_node, turn)
        trajectory.append((predator_node, prey_node))
        if predator_node == prey_node:
            outcome = "capture"
            if verbose:
                print(f"  -> CAPTURE at {predator_node}")
            break
        if new_state in seen_states:
            outcome = "draw"
            if verbose:
                print(f"  -> state (predator={predator_node}, prey={prey_node}, turn={turn}) "
                      f"already occurred earlier -> DRAW, this will repeat forever")
            break
        seen_states.add(new_state)

    if outcome is None:
        d = math.dist(nodes[predator_node], nodes[prey_node])
        outcome = "prey_escaped" if d >= safe_distance else "draw"
        if verbose:
            print(f"  -> round budget exhausted; distance={d:.0f} vs safe_distance={safe_distance:.0f} "
                  f"-> {outcome.upper()}")

    if verbose:
        predator_seq = " -> ".join(p for p, _ in trajectory)
        prey_seq = " -> ".join(q for _, q in trajectory)
        print(f"\nFull predator sequence: {predator_seq}")
        print(f"Full prey sequence:     {prey_seq}")

    return trajectory, outcome


if __name__ == "__main__":
    from edu_env import make_edu_env

    parser = argparse.ArgumentParser(description="Minimax vs alpha-beta on the predator-prey grid game")
    parser.add_argument("--map", default="u_shape", choices=["u_shape", "n_shape", "default", "csp_dense"])
    parser.add_argument("--depth", type=int, default=4, help="Plies to look ahead (predator+prey moves counted separately)")
    args = parser.parse_args()

    env = make_edu_env(map_name=args.map, num_agents=1)
    env.reset()
    nodes, edges = build_grid(env.obstacles, env.config.world_width, env.config.world_height,
                                food_pos=None)
    env.close()

    node_ids = sorted(nodes.keys())
    predator_start, prey_start = node_ids[0], node_ids[-1]
    state = (predator_start, prey_start, "predator")

    print(f"Map: {args.map}  |  {len(nodes)} nodes  |  depth={args.depth}")
    print(f"Predator starts at {predator_start} {nodes[predator_start]}")
    print(f"Prey starts at     {prey_start} {nodes[prey_start]}")
    print()

    move_mm, val_mm, nodes_mm = minimax_decision(state, edges, nodes, args.depth)
    move_ab, val_ab, nodes_ab = alphabeta_decision(state, edges, nodes, args.depth)

    print(f"Plain minimax  : predator should move to {move_mm} (value={val_mm:.1f}), {nodes_mm} nodes explored")
    print(f"Alpha-beta     : predator should move to {move_ab} (value={val_ab:.1f}), {nodes_ab} nodes explored")
    print(f"Same answer: {move_mm == move_ab and abs(val_mm - val_ab) < 1e-9}")
    print(f"Nodes saved by pruning: {nodes_mm - nodes_ab} ({100*(1 - nodes_ab/nodes_mm):.1f}%)")
