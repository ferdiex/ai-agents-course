# Commands per Activity (quick reference)

One block per activity in `exercises/AI_Agents_Exercises.docx`. Run
everything from inside `training_wheels/`. Details, expected output, and
troubleshooting for each command are in `TESTING.md`.

## Activity 1 — Table-Driven Agent

```bash
python3 agents/table_driven_agent.py
```

## Activity 2 — Simple Reflex vs. Model-Based Reflex

```bash
python3 demo_level0_reflex.py --agent simple --map u_shape --episodes 10
python3 demo_level0_reflex.py --agent model_based --map u_shape --episodes 10
```

## Activity 3 — CSP: Backtracking vs. Generate-and-Test

```bash
python3 csp_spawn.py --map csp_dense --seed 5
python3 csp_spawn.py --map csp_dense --seed 5 --dist_min 420 --dist_max 500 --max_attempts 5000
```

## Activity 4 — Minimax and Alpha-Beta Pruning

```bash
python3 minimax_search.py --map u_shape --depth 2
python3 minimax_search.py --map u_shape --depth 3
python3 minimax_search.py --map u_shape --depth 4
python3 minimax_search.py --map u_shape --depth 5
python3 minimax_search.py --map u_shape --depth 6

python3 visualize_minimax.py --map u_shape --depth 3
python3 visualize_minimax.py --map u_shape --depth 4 --rounds 20

python3 visualize_minimax.py --map u_shape --depth 3 --predator n0 --prey n11
python3 visualize_minimax.py --map u_shape --depth 3 --predator n0 --prey n11 --rounds 2 --safe_distance 300
```

## Activity 5 — Q-Learning

```bash
python3 qlearning.py --map u_shape --episodes 2000
python3 visualize_qlearning.py --map u_shape --episodes 2000
python3 visualize_qlearning.py --map u_shape --episodes 10
python3 demo_qlearning_live.py --map u_shape --episodes 2000
```

## Activity 6 — GA vs. Q-Learning

```bash
python3 qlearning.py --map u_shape --episodes 2000
python3 ga_learning.py --map u_shape --generations 100
python3 visualize_ga_population.py --map u_shape --generations 100
python3 demo_qlearning_live.py --map u_shape --episodes 2000
python3 demo_ga_live.py --map u_shape --generations 100
```

## Before any of the above — installation check

```bash
python3 verify_install.py
```
