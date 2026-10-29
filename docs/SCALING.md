# Scaling Notes - Self Improvement Loop

This project is deliberately small so the mechanism is exact and inspectable. This note
records how each part would scale and where the design would change.

## Cost of one iteration
- Let `P` be the number of training puzzles, `k` the samples per puzzle, `B` the step budget,
  and `A` the action-set size. One iteration samples `P * k` rollouts of up to `B` steps, each
  step scoring `A` actions, so generation is `O(P * k * B * A)`.
- Cloning is `O(E * A)` per step over `E` collected state-action pairs, repeated for the
  configured number of cloning steps.
- At the defaults (`P = 200`, `k = 12`, `B = 6`, `A = 8`) a full ten-iteration run finishes in
  seconds on one CPU core.

## What scales well
- **Generation is embarrassingly parallel** across puzzles and samples. The loop is written as
  a simple pass so it can be sharded across workers with no change to the algorithm.
- **The verifier is `O(B)`** and allocation-free, so filtering never dominates.

## What would change at larger scale
- **Replace the oracle.** The breadth-first oracle is `O(states)` and only exists because the
  value range is small. For a real task the rationalization hint would come from a stronger
  model or a search procedure, with no correctness guarantee, and the verifier would become the
  only trusted signal.
- **Replace the policy.** A log-linear policy caps the achievable accuracy. A neural policy
  would raise the ceiling but replace the explicit gradient with autograd; the loop around it is
  unchanged.
- **Stream, do not accumulate.** At large `P` the collected examples would be streamed to the
  cloning step rather than materialized in a list.

## Why the reference keeps the exact oracle
The exact oracle and log-linear policy make every number reproducible and every component
auditable. The admissibility feature it enables is what lifts greedy decoding; a search-based
decoder would be the next step to close the remaining gap to sampled pass@8.
