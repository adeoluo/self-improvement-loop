"""Evaluation: greedy accuracy, sampled pass@k, and solution quality.

Greedy accuracy is the primary metric. A puzzle counts as solved only when the
verifier accepts the greedy rollout, so the number cannot be inflated by lucky
sampling. Pass@k reports the probability that at least one of ``k`` sampled
rollouts solves a puzzle, which exposes how much headroom self-training still has.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from self_improvement_loop.environment import Puzzle, shortest_solution
from self_improvement_loop.generation import greedy_trace, sample_trace
from self_improvement_loop.policy import Policy


@dataclass(frozen=True)
class EvaluationReport:
    """Greedy evaluation of a policy over a puzzle set."""

    accuracy: float
    solved: int
    total: int
    mean_solution_length: float
    mean_optimal_length: float

    @property
    def length_overhead(self) -> float:
        """Average extra steps the greedy solutions take over the oracle optimum."""

        return self.mean_solution_length - self.mean_optimal_length

    def format(self) -> str:
        return (
            f"accuracy={self.accuracy:.4f} ({self.solved}/{self.total}), "
            f"mean_len={self.mean_solution_length:.2f}, "
            f"optimal_len={self.mean_optimal_length:.2f}"
        )


@dataclass(frozen=True)
class IterationReport:
    """One row of the self-improvement history."""

    iteration: int
    test_accuracy: float
    solved_by_sampling: int
    recovered_by_hint: int
    training_examples: int
    mean_nll: float


def evaluate_policy(policy: Policy, puzzles: list[Puzzle]) -> EvaluationReport:
    """Return greedy accuracy and solution-length statistics over ``puzzles``."""

    solved = 0
    solved_lengths: list[int] = []
    optimal_lengths: list[int] = []
    for puzzle in puzzles:
        trace = greedy_trace(puzzle, policy)
        if trace.solved:
            solved += 1
            solved_lengths.append(len(trace.actions))
            optimal = shortest_solution(puzzle)
            if optimal is not None:
                optimal_lengths.append(len(optimal))
    total = len(puzzles)
    return EvaluationReport(
        accuracy=solved / total if total else 0.0,
        solved=solved,
        total=total,
        mean_solution_length=_mean(solved_lengths),
        mean_optimal_length=_mean(optimal_lengths),
    )


def pass_at_k(
    policy: Policy, puzzles: list[Puzzle], *, k: int, seed: int = 0, temperature: float = 1.0
) -> float:
    """Return the fraction of puzzles solved by at least one of ``k`` sampled rollouts."""

    rng = random.Random(seed)
    solved = 0
    for puzzle in puzzles:
        if any(sample_trace(puzzle, policy, rng, temperature=temperature).solved for _ in range(k)):
            solved += 1
    return solved / len(puzzles) if puzzles else 0.0


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0
