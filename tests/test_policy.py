"""Tests for the log-linear policy and behavior-cloning update."""

from __future__ import annotations

import random

from self_improvement_loop.environment import Puzzle
from self_improvement_loop.generation import rationalized_trace
from self_improvement_loop.policy import FEATURE_NAMES, Policy


def test_zero_initialized_policy_is_uniform() -> None:
    policy = Policy()
    probs = policy.action_probs((10, 20, 4))
    values = list(probs.values())
    assert all(abs(p - values[0]) < 1e-9 for p in values)
    assert abs(sum(values) - 1.0) < 1e-9


def test_valid_actions_mask_out_of_bounds() -> None:
    policy = Policy(value_min=0, value_max=200)
    # From 0, subtraction goes below zero and is excluded.
    actions = policy.valid_actions(0)
    assert "-1" not in actions
    assert "+1" in actions


def test_valid_actions_never_empty() -> None:
    policy = Policy(value_min=50, value_max=50)
    assert policy.valid_actions(50)  # falls back to the full action set


def test_action_probs_sum_to_one_after_training() -> None:
    policy = Policy()
    policy.weights = [0.5] * len(FEATURE_NAMES)
    probs = policy.action_probs((10, 40, 5), temperature=0.7)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_clone_update_raises_probability_of_cloned_action() -> None:
    puzzle = Puzzle(start=0, target=12, budget=6)
    trace = rationalized_trace(puzzle, Policy())
    assert trace is not None
    examples = trace.examples()
    policy = Policy()
    before = policy.action_probs(examples[0][0]).get(examples[0][1], 0.0)
    for _ in range(40):
        policy.clone_update(examples, learning_rate=0.3)
    after = policy.action_probs(examples[0][0]).get(examples[0][1], 0.0)
    assert after > before


def test_clone_update_on_empty_examples_is_noop() -> None:
    policy = Policy()
    assert policy.clone_update([], learning_rate=0.3) == 0.0
    assert policy.weights == [0.0] * len(FEATURE_NAMES)


def test_sample_action_is_reproducible_under_seed() -> None:
    policy = Policy()
    policy.weights = [0.3] * len(FEATURE_NAMES)
    state = (10, 50, 5)
    a = [policy.sample_action(state, random.Random(7)) for _ in range(3)]
    b = [policy.sample_action(state, random.Random(7)) for _ in range(3)]
    assert a == b


def test_greedy_action_matches_argmax_probability() -> None:
    policy = Policy()
    policy.weights = [0.4] * len(FEATURE_NAMES)
    state = (10, 50, 5)
    probs = policy.action_probs(state)
    assert policy.greedy_action(state) == max(probs, key=lambda k: probs[k])
