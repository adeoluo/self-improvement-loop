# Model / System Card - Self Improvement Loop

## Overview
- **What it is:** a from-scratch implementation of rejection-sampling self-training with a
  verifier (self-taught-reasoning style). A log-linear policy learns to solve a verifiable
  arithmetic reasoning task from its own verified rollouts, with no human labels.
- **Base model:** none. The policy is a single weight vector over hand-built features, trained
  by an explicit behavior-cloning gradient. No external model or dataset is used.
- **Version and date:** 0.1.0, 2026.

## Intended use
- **Primary use case:** a readable, reproducible demonstration of the self-improvement loop and
  the role each component plays (verifier filtering, rationalization, behavior cloning).
- **Primary users:** engineers and researchers studying verifier-based self-training and
  rejection-sampling fine-tuning.

## Out-of-scope / prohibited use
- Not a general reasoning system and not a solver for real-world planning or arithmetic.
- The absolute accuracy describes a toy task; do not cite it as a reasoning benchmark.
- The oracle hint relies on the task being small enough to solve exactly; the approach does not
  transfer unchanged to settings without an exact verifier or oracle.

## Performance
- Greedy pass@1 on held-out puzzles rises from 0.050 (untrained) to 0.733 (self-trained).
- Sampled pass@8 is 0.900; greedy solutions average 0.85 steps over the oracle optimum.
- Full method and ablations: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md).

## Limitations and failure modes
- Log-linear capacity caps greedy accuracy below sampled pass@8.
- Rationalization can bias the policy toward oracle-shortest paths that generalize slightly less
  well under greedy decoding.
- Some generated start/target pairs are unsolvable within the budget and are excluded from the
  splits; the policy is never asked to certify unsolvability.

## Ethical considerations
- No personal data, no network access, no external model. The task is synthetic arithmetic.
- The project illustrates a self-improvement mechanism; it has no autonomous capability beyond
  solving the bounded puzzle it is trained on.
