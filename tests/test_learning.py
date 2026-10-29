"""Tests for data splits, generation, the loop, and evaluation."""

from __future__ import annotations

import random

from self_improvement_loop.data import test_puzzles as held_out_puzzles
from self_improvement_loop.data import train_puzzles
from self_improvement_loop.environment import Puzzle, is_solvable, verify_trace
from self_improvement_loop.evaluation import evaluate_policy, pass_at_k
from self_improvement_loop.generation import greedy_trace, rationalized_trace, sample_trace
from self_improvement_loop.policy import Policy
from self_improvement_loop.self_improvement import SelfImprovementConfig, SelfTrainer


def test_splits_are_solvable_and_disjoint() -> None:
    train = train_puzzles(60, seed=0)
    test = held_out_puzzles(40, seed=0)
    assert all(is_solvable(p) for p in train)
    assert all(is_solvable(p) for p in test)
    train_keys = {(p.start, p.target) for p in train}
    test_keys = {(p.start, p.target) for p in test}
    assert train_keys.isdisjoint(test_keys)


def test_data_generation_is_deterministic() -> None:
    assert train_puzzles(30, seed=3) == train_puzzles(30, seed=3)


def test_sample_trace_reproducible_under_seed() -> None:
    puzzle = Puzzle(start=0, target=15, budget=6)
    policy = Policy()
    t1 = sample_trace(puzzle, policy, random.Random(1))
    t2 = sample_trace(puzzle, policy, random.Random(1))
    assert t1.actions == t2.actions


def test_rationalized_trace_solves_solvable_puzzle() -> None:
    puzzle = Puzzle(start=2, target=30, budget=6)
    trace = rationalized_trace(puzzle, Policy())
    assert trace is not None
    assert trace.solved
    assert verify_trace(puzzle, trace.actions)


def test_rationalized_trace_returns_none_when_unsolvable() -> None:
    assert rationalized_trace(Puzzle(start=0, target=199, budget=1), Policy()) is None


def test_self_training_improves_greedy_accuracy() -> None:
    train = train_puzzles(80, seed=0)
    test = held_out_puzzles(60, seed=0)
    config = SelfImprovementConfig(
        iterations=4, samples_per_puzzle=8, clone_steps=20, learning_rate=0.3, seed=0
    )
    result = SelfTrainer(config).train(train, test)
    baseline = result.history[0].test_accuracy
    final = result.history[-1].test_accuracy
    assert baseline < 0.2
    assert final > 0.5
    assert final > baseline


def test_self_training_is_deterministic() -> None:
    train = train_puzzles(40, seed=0)
    test = held_out_puzzles(30, seed=0)
    config = SelfImprovementConfig(iterations=3, samples_per_puzzle=6, clone_steps=15, seed=0)
    a = SelfTrainer(config).train(train, test)
    b = SelfTrainer(config).train(train, test)
    assert a.policy.weights == b.policy.weights


def test_unfiltered_cloning_underperforms_verified_filtering() -> None:
    train = train_puzzles(80, seed=0)
    test = held_out_puzzles(60, seed=0)
    shared = {
        "iterations": 5,
        "samples_per_puzzle": 8,
        "clone_steps": 20,
        "learning_rate": 0.3,
        "seed": 0,
    }
    filtered = SelfTrainer(SelfImprovementConfig(**shared)).train(train, test)
    unfiltered = SelfTrainer(
        SelfImprovementConfig(rationalize=False, filter_verified=False, **shared)
    ).train(train, test)
    assert filtered.history[-1].test_accuracy > unfiltered.history[-1].test_accuracy


def test_pass_at_k_is_monotonic_in_k() -> None:
    train = train_puzzles(60, seed=0)
    test = held_out_puzzles(40, seed=0)
    config = SelfImprovementConfig(iterations=3, samples_per_puzzle=8, clone_steps=15, seed=0)
    policy = SelfTrainer(config).train(train, test).policy
    p1 = pass_at_k(policy, test, k=1, seed=2)
    p8 = pass_at_k(policy, test, k=8, seed=2)
    assert 0.0 <= p1 <= p8 <= 1.0


def test_evaluate_reports_length_overhead() -> None:
    train = train_puzzles(60, seed=0)
    test = held_out_puzzles(40, seed=0)
    config = SelfImprovementConfig(iterations=3, samples_per_puzzle=8, clone_steps=15, seed=0)
    policy = SelfTrainer(config).train(train, test).policy
    report = evaluate_policy(policy, test)
    assert report.total == 40
    assert report.length_overhead >= 0.0
    assert "accuracy=" in report.format()


def test_greedy_trace_on_trivial_puzzle() -> None:
    trace = greedy_trace(Puzzle(start=5, target=5, budget=3), Policy())
    assert trace.solved
    assert trace.actions == []
