"""
mcts.py -- STUB, stage B+.

Real MCTS (as distinct from the best-of-N in trace_gen.py) branches at the
CLAIM level, not the whole-trajectory level:
  - Node = state after verifying scaffold steps 1..k (bound variables e0..ek
    already computed and held as stateful memory).
  - At each node, branch into 2-3 candidate tool strategies for step k+1
    (e.g. check_spatial_relation vs vlm_probe for an ambiguous predicate).
  - Execute each branch; prune failures; expand survivors.
  - A full golden trajectory = a root-to-leaf path where every step
    resolved and the final answer matched ground truth.

This is worth building once stage A has produced a reviewed few-shot pool
(see prompts/fewshot_examples.py) -- it reuses verified partial state
across branches instead of regenerating whole trajectories from scratch,
much cheaper at scale, and gives genuinely different trajectories rather
than temperature-noise near-duplicates.

Not implemented yet -- do stage A first (trace_gen.py + filters + a
manually reviewed batch), then come back here.
"""

raise NotImplementedError(
    "mcts.py is a stage B+ placeholder. Use generation/trace_gen.py "
    "(best-of-N) for stage A."
)