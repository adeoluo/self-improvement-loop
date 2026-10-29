"""The self-improvement loop: generate, verify, filter, and clone.

Each iteration samples several rollouts per training puzzle, keeps only the traces
that the verifier accepts, optionally recovers unsolved puzzles with an oracle hint
(rationalization), and then takes behavior-cloning steps on the collected
state-action pairs. Measured on a held-out split, greedy accuracy rises across
iterations because probability mass concentrates on the actions that solve puzzles.

The loop deliberately trains only on verified traces. The verifier, not a human
label, is the supervision signal, which is the defining property of this family of
self-training methods.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from self_improvement_loop.environment import Action, Puzzle
from self_improvement_loop.evaluation import IterationReport, evaluate_policy
from self_improvement_loop.generation import rationalized_trace, sample_trace
from self_improvement_loop.policy import Policy, State


@dataclass(frozen=True)
class SelfImprovementConfig:
    """Hyperparameters for the self-improvement loop."""

    iterations: int = 10
    samples_per_puzzle: int = 12
    temperature: float = 1.0
    clone_steps: int = 30
    learning_rate: float = 0.3
    rationalize: bool = True
    filter_verified: bool = True
    seed: int = 0


@dataclass
class SelfImprovementResult:
    """The trained policy and per-iteration diagnostics."""

    policy: Policy
    history: list[IterationReport] = field(default_factory=list)


class SelfTrainer:
    """Runs rejection-sampling self-training over a set of puzzles."""

    def __init__(self, config: SelfImprovementConfig | None = None) -> None:
        self.config = config or SelfImprovementConfig()

    def _collect(
        self, puzzles: list[Puzzle], policy: Policy, rng: random.Random
    ) -> tuple[list[tuple[State, Action]], int, int]:
        examples: list[tuple[State, Action]] = []
        solved_by_sampling = 0
        recovered_by_hint = 0
        for puzzle in puzzles:
            best: list[tuple[State, Action]] | None = None
            for _ in range(self.config.samples_per_puzzle):
                trace = sample_trace(puzzle, policy, rng, temperature=self.config.temperature)
                if not self.config.filter_verified:
                    examples.extend(trace.examples())
                    continue
                if trace.solved and (best is None or len(trace.actions) < len(best)):
                    best = trace.examples()
            if not self.config.filter_verified:
                continue
            if best is not None:
                examples.extend(best)
                solved_by_sampling += 1
            elif self.config.rationalize:
                hinted = rationalized_trace(puzzle, policy)
                if hinted is not None and hinted.solved:
                    examples.extend(hinted.examples())
                    recovered_by_hint += 1
        return examples, solved_by_sampling, recovered_by_hint

    def train(self, train_set: list[Puzzle], test_set: list[Puzzle]) -> SelfImprovementResult:
        """Run the loop and return the trained policy with its iteration history."""

        rng = random.Random(self.config.seed)
        policy = Policy()
        history: list[IterationReport] = []

        baseline = evaluate_policy(policy, test_set)
        history.append(
            IterationReport(
                iteration=0,
                test_accuracy=baseline.accuracy,
                solved_by_sampling=0,
                recovered_by_hint=0,
                training_examples=0,
                mean_nll=0.0,
            )
        )

        for iteration in range(1, self.config.iterations + 1):
            examples, sampled, recovered = self._collect(train_set, policy, rng)
            last_nll = 0.0
            for _ in range(self.config.clone_steps):
                last_nll = policy.clone_update(examples, learning_rate=self.config.learning_rate)
            report = evaluate_policy(policy, test_set)
            history.append(
                IterationReport(
                    iteration=iteration,
                    test_accuracy=report.accuracy,
                    solved_by_sampling=sampled,
                    recovered_by_hint=recovered,
                    training_examples=len(examples),
                    mean_nll=last_nll,
                )
            )

        return SelfImprovementResult(policy=policy, history=history)
