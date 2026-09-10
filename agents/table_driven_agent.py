# -*- coding: utf-8 -*-
"""
Table-driven agent -- discussion-only module.

A table-driven agent looks up its action in an explicit table indexed by
the entire percept (or percept sequence) seen so far. It needs no logic at
all: just a (potentially enormous) lookup table built in advance by the
designer.

This module does NOT implement a working table-driven controller for the
foraging task. Doing so would require materializing one table row per
possible percept -- and that is exactly the point we want students to see:
the approach is a valid theoretical baseline, but becomes impractical the
moment perception is even mildly rich.

Run this file directly to see how large that table would be under a few
discretization choices, and compare it to the much coarser grid (10-15
nodes) used later for goal-based search, CSP, and the Q-learning agent's
table. The contrast is the lesson: a Q-table is ALSO just a table -- the
difference is that nobody hand-fills it, and it is built over a deliberately
coarse, designer-chosen abstraction of the world instead of the raw percept.
"""


def table_size(num_sensors=8, bins_per_sensor=3, extra_discrete_features=2,
               bins_per_extra_feature=4):
    """
    Estimate how many rows a table-driven agent's lookup table would need
    for a single time step of the foraging task (no percept history, i.e.
    this already *undercounts* what a true table-driven agent would need,
    since it should index on the whole percept SEQUENCE, not just the
    current percept).

    Parameters
    ----------
    num_sensors : int
        Number of proximity sensors (8 in ForagingEnv).
    bins_per_sensor : int
        How many discrete buckets each continuous sensor reading is
        rounded into (e.g. 3 for "far / medium / close").
    extra_discrete_features : int
        Non-proximity percept features (food bearing, odor).
    bins_per_extra_feature : int
        Discretization granularity for those extra features.

    Returns
    -------
    int
        Number of distinct percepts, i.e. the number of rows the table
        needs (one action stored per row).
    """
    return (bins_per_sensor ** num_sensors) * \
           (bins_per_extra_feature ** extra_discrete_features)


if __name__ == "__main__":
    print("Table-driven agent: how big would the lookup table be?\n")

    scenarios = [
        ("very coarse (3 bins/sensor)", 3),
        ("coarse (5 bins/sensor)", 5),
        ("moderate (10 bins/sensor)", 10),
    ]

    for label, bins in scenarios:
        size = table_size(bins_per_sensor=bins)
        print(f"  {label:32s} -> {size:,} rows")

    print(
        "\nFor comparison, the navigation grid used later for goal-based "
        "search and Q-learning has 10-15 nodes total. The table-driven "
        "approach blows up long before we discretize anywhere near that "
        "coarsely, because it indexes on the FULL raw percept, not on a "
        "designer-chosen abstraction of the world (the grid)."
    )
