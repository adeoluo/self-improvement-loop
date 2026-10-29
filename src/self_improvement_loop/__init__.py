"""Self Improvement Loop - recursive self-training with a verifier, from scratch."""

from __future__ import annotations

from self_improvement_loop.data import test_puzzles, train_puzzles
from self_improvement_loop.environment import (
    ACTIONS,
    Puzzle,
    apply_action,
    is_solvable,
    shortest_solution,
    verify_trace,
)
from self_improvement_loop.evaluation import (
    EvaluationReport,
    IterationReport,
    evaluate_policy,
    pass_at_k,
)
from self_improvement_loop.generation import (
    Trace,
    greedy_trace,
    rationalized_trace,
    sample_trace,
)
from self_improvement_loop.policy import Policy
from self_improvement_loop.self_improvement import (
    SelfImprovementConfig,
    SelfImprovementResult,
    SelfTrainer,
)

__version__ = "0.1.0"

__all__ = [
    "ACTIONS",
    "EvaluationReport",
    "IterationReport",
    "Policy",
    "Puzzle",
    "SelfImprovementConfig",
    "SelfImprovementResult",
    "SelfTrainer",
    "Trace",
    "__version__",
    "apply_action",
    "evaluate_policy",
    "greedy_trace",
    "is_solvable",
    "pass_at_k",
    "rationalized_trace",
    "sample_trace",
    "shortest_solution",
    "test_puzzles",
    "train_puzzles",
    "verify_trace",
]
