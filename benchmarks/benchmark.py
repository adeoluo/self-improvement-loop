"""A deterministic micro-benchmark for the self-improvement loop.

Runs the full loop on the bundled splits and reports wall-clock time, the greedy
accuracy before and after training, and sampled pass@8. Deterministic and offline;
run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from self_improvement_loop.data import test_puzzles, train_puzzles
from self_improvement_loop.evaluation import evaluate_policy, pass_at_k
from self_improvement_loop.self_improvement import SelfImprovementConfig, SelfTrainer


def main() -> None:
    train = train_puzzles(200, seed=0)
    test = test_puzzles(120, seed=0)

    start = time.perf_counter()
    result = SelfTrainer(SelfImprovementConfig()).train(train, test)
    elapsed = time.perf_counter() - start

    report = evaluate_policy(result.policy, test)
    p8 = pass_at_k(result.policy, test, k=8, seed=11)

    print(f"Train puzzles: {len(train)} | Test puzzles: {len(test)}")
    print(f"Full loop (10 iterations): {elapsed * 1000:.0f} ms")
    print(f"Greedy pass@1: {result.history[0].test_accuracy:.3f} -> {report.accuracy:.3f}")
    print(f"Sampled pass@8: {p8:.3f}")
    print(f"Length overhead vs oracle optimum: {report.length_overhead:.2f} steps")


if __name__ == "__main__":
    main()
