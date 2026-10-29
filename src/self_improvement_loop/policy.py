"""A log-linear policy over actions, trained from scratch by behavior cloning.

The policy scores each action with a linear function of hand-built features of the
current state and the action's effect, then forms a softmax distribution over the
actions that stay in bounds. Learning is behavior cloning: given state-action pairs
taken from verified traces, the policy follows the exact softmax gradient to raise
the log-probability of those actions. There is no autograd and no external library;
the gradient is written out directly.

With all weights at zero the policy is uniform over valid actions, so an untrained
policy solves few puzzles greedily. Self-training is what concentrates probability
onto the actions that make progress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from self_improvement_loop.environment import ACTIONS, Action, apply_action

State = tuple[int, int, int]  # (value, target, steps_remaining)


class RandomSource(Protocol):
    """Structural type for a random source exposing ``random() -> float``."""

    def random(self) -> float: ...


_SCALE = 100.0
_FAR = 20

# Names line up, in order, with the feature vector built by ``_features``.
FEATURE_NAMES: tuple[str, ...] = (
    "progress",
    "exact",
    "neg_distance_after",
    "overshoot",
    "moves_closer",
    "multiply_when_far",
    "reachable_additively",
    "bias",
    *(f"action::{a}" for a in ACTIONS),
)

_N_FEATURES = len(FEATURE_NAMES)
_ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


def _features(state: State, action: Action) -> list[float]:
    value, target, steps_remaining = state
    nxt = apply_action(value, action)
    dist_before = abs(target - value)
    dist_after = abs(target - nxt)
    crossed = (value - target) * (nxt - target) < 0
    phi = [0.0] * _N_FEATURES
    phi[0] = (dist_before - dist_after) / _SCALE
    phi[1] = 1.0 if nxt == target else 0.0
    phi[2] = -dist_after / _SCALE
    phi[3] = 1.0 if crossed and nxt != target else 0.0
    phi[4] = 1.0 if dist_after < dist_before else 0.0
    phi[5] = 1.0 if action in ("*2", "*3") and dist_before > _FAR else 0.0
    phi[6] = 1.0 if dist_after <= 3 * max(steps_remaining - 1, 0) else 0.0
    phi[7] = 1.0
    phi[8 + _ACTION_INDEX[action]] = 1.0
    return phi


@dataclass
class Policy:
    """A softmax policy with a single shared weight vector over features."""

    weights: list[float] = field(default_factory=lambda: [0.0] * _N_FEATURES)
    value_min: int = 0
    value_max: int = 200

    def valid_actions(self, value: int) -> list[Action]:
        """Return the actions whose result stays within representable bounds."""

        valid = [a for a in ACTIONS if self.value_min <= apply_action(value, a) <= self.value_max]
        return valid or list(ACTIONS)

    def _score(self, state: State, action: Action) -> float:
        return sum(w * f for w, f in zip(self.weights, _features(state, action), strict=True))

    def action_probs(self, state: State, *, temperature: float = 1.0) -> dict[Action, float]:
        """Return a softmax distribution over the in-bounds actions for ``state``."""

        value = state[0]
        actions = self.valid_actions(value)
        temp = max(temperature, 1e-6)
        scores = {a: self._score(state, a) / temp for a in actions}
        top = max(scores.values())
        weights = {a: math.exp(s - top) for a, s in scores.items()}
        total = sum(weights.values())
        return {a: w / total for a, w in weights.items()}

    def sample_action(self, state: State, rng: RandomSource, *, temperature: float = 1.0) -> Action:
        probs = self.action_probs(state, temperature=temperature)
        threshold = rng.random()
        cumulative = 0.0
        last = next(iter(probs))
        for action, prob in probs.items():
            cumulative += prob
            last = action
            if threshold <= cumulative:
                return action
        return last

    def greedy_action(self, state: State) -> Action:
        probs = self.action_probs(state, temperature=1.0)
        return max(probs, key=lambda a: probs[a])

    def clone_update(self, examples: list[tuple[State, Action]], *, learning_rate: float) -> float:
        """Take one behavior-cloning gradient step over ``examples``.

        Returns the mean negative log-likelihood before the update, a convenient
        training signal. The gradient of ``log p(a | s)`` is the feature vector of
        the taken action minus its expectation under the current policy.
        """

        if not examples:
            return 0.0
        gradient = [0.0] * _N_FEATURES
        total_nll = 0.0
        for state, action in examples:
            probs = self.action_probs(state, temperature=1.0)
            expected = [0.0] * _N_FEATURES
            for candidate, prob in probs.items():
                phi_c = _features(state, candidate)
                for i in range(_N_FEATURES):
                    expected[i] += prob * phi_c[i]
            phi_a = _features(state, action)
            for i in range(_N_FEATURES):
                gradient[i] += phi_a[i] - expected[i]
            total_nll -= math.log(max(probs.get(action, 1e-12), 1e-12))
        scale = learning_rate / len(examples)
        for i in range(_N_FEATURES):
            self.weights[i] += scale * gradient[i]
        return total_nll / len(examples)
