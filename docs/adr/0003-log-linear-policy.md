# ADR-0003: A log-linear policy with an explicit behavior-cloning gradient

- **Status:** Accepted
- **Date:** 2026-01-01

## Context
The policy must have enough capacity to show real learning from verified traces, while keeping
the update rule inspectable. Two tensions pull against each other: a richer function
approximator raises the accuracy ceiling but hides the gradient behind autograd, and a trivial
policy keeps the gradient visible but cannot learn anything interesting.

## Decision
Use a log-linear policy: a single shared weight vector scores each action with a linear
function of hand-built features of the state and the action's effect, and a softmax over the
in-bounds actions forms the distribution. Training is behavior cloning on verified
state-action pairs, following the exact softmax gradient

```
grad log p(a | s) = phi(s, a) - sum_a' p(a' | s) phi(s, a')
```

written out directly, with no automatic differentiation. Zero-initialized weights give a
uniform policy, so any greedy success is attributable to learning.

The features combine progress toward the target, distance after the action, an overshoot
penalty, an admissibility flag (whether the target is still reachable additively within the
remaining budget), and per-action biases. The admissibility feature is the one that lets
greedy decoding avoid overshooting multiplications and lifts the greedy ceiling.

## Consequences
- The gradient is closed-form and matches the code line for line, which is the point.
- Capacity is limited: the greedy ceiling sits below sampled pass@8, a gap the README reports
  rather than hides.
- Features are engineered, not learned. A deep model would remove that hand-engineering at the
  cost of the explicit gradient this project is built to show.
