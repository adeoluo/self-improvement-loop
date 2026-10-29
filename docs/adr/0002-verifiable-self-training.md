# ADR-0002: A verifiable task for self-training, not a language model

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
The subject of this project is the self-improvement loop itself: generate, verify, filter,
and clone. A loop like this needs three things to be meaningful: a generator, a verifier that
is correct, and a way to measure generalization honestly. Using a neural language model as the
generator would make the verifier approximate, the training expensive, and the mechanism hard
to inspect, which would bury the very thing the project exists to demonstrate.

## Decision
Use a small, fully verifiable arithmetic reasoning task. From a start value, reach a target
within a step budget using a fixed action set. The verifier replays a trace and checks that it
stays in bounds, respects the budget, and lands exactly on the target. A breadth-first oracle
defines solvability and supplies rationalization hints.

Train and test splits are generated with seeded randomness, filtered to oracle-solvable
instances, and kept disjoint by construction, so held-out accuracy measures generalization
rather than memorization.

## Consequences
- The verifier and oracle are exact, so filtering and rationalization have no label noise.
- The whole mechanism fits in a few readable modules with no external dependencies.
- The absolute accuracy numbers describe a toy task; the claims are about the loop and its
  ablations, not about state-of-the-art reasoning. This is stated plainly in the README.
