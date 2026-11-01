"""A large-workload scale benchmark for the self-improvement loop.

Runs verifier-filtered self-training over progressively larger pools of synthetic
puzzles and reports, per workload, the wall-clock time, throughput in training
puzzles per second, the peak traced memory, and the greedy pass@1 accuracy on a
fixed held-out split. It complements ``benchmark.py`` (a single fixed-size run) by
showing how the core training loop scales as the pool of training puzzles grows.

A reduced loop configuration keeps the total runtime bounded while preserving the
real operation: sampling rollouts, verifier filtering, rationalization, and
behavior-cloning updates. Timing and memory are captured in separate phases on
purpose: ``tracemalloc`` instruments every allocation and perturbs the wall-clock,
so the throughput is taken from an untraced run while the peak footprint comes from
a traced run of the same loop. Deterministic and offline;
run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from self_improvement_loop.data import generate_puzzles
from self_improvement_loop.environment import Puzzle
from self_improvement_loop.evaluation import evaluate_policy
from self_improvement_loop.self_improvement import SelfImprovementConfig, SelfTrainer

WORKLOADS: tuple[int, ...] = (80, 160, 320)
_ITERATIONS = 2
_SAMPLES_PER_PUZZLE = 4
_CLONE_STEPS = 12
_MEM_ITERATIONS = 2
_TEST = 120
_TEST_SEED = 777
_TRAIN_SEED = 0


@dataclass(frozen=True)
class ScalePoint:
    """One row of the scale table for a single workload size."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    accuracy: float


def _config(iterations: int) -> SelfImprovementConfig:
    return SelfImprovementConfig(
        iterations=iterations,
        samples_per_puzzle=_SAMPLES_PER_PUZZLE,
        clone_steps=_CLONE_STEPS,
        seed=0,
    )


def held_out_split(exclude_size: int = max(WORKLOADS)) -> list[Puzzle]:
    """Build a held-out test split disjoint from every training pool."""

    train = generate_puzzles(exclude_size, seed=_TRAIN_SEED)
    exclude = frozenset((puzzle.start, puzzle.target) for puzzle in train)
    return generate_puzzles(_TEST, seed=_TEST_SEED, exclude=exclude)


def measure(
    test: list[Puzzle],
    size: int,
    *,
    iterations: int = _ITERATIONS,
    mem_iterations: int = _MEM_ITERATIONS,
) -> ScalePoint:
    """Self-train on ``size`` puzzles and measure the cost and held-out accuracy."""

    train = generate_puzzles(size, seed=_TRAIN_SEED)

    tracemalloc.start()
    tracemalloc.reset_peak()
    SelfTrainer(_config(mem_iterations)).train(train, test)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    start = time.perf_counter()
    result = SelfTrainer(_config(iterations)).train(train, test)
    seconds = time.perf_counter() - start

    return ScalePoint(
        size=size,
        seconds=seconds,
        throughput=size / seconds if seconds > 0 else 0.0,
        peak_mib=peak / 1048576,
        accuracy=evaluate_policy(result.policy, test).accuracy,
    )


def main() -> None:
    print("Synthetic arithmetic puzzles; a pure-Python, from-scratch training loop.")
    print("Wall-clock is not competitive with compiled numerical libraries.")
    print(f"Self-training for {_ITERATIONS} iterations per workload; accuracy is greedy pass@1.\n")

    test = held_out_split()

    print(f"{'puzzles':>8}  {'seconds':>8}  {'puzzles/sec':>12}  {'peak MiB':>9}  {'accuracy':>8}")
    for size in WORKLOADS:
        point = measure(test, size)
        print(
            f"{point.size:>8}  {point.seconds:>8.3f}  {point.throughput:>12.1f}  "
            f"{point.peak_mib:>9.2f}  {point.accuracy:>8.3f}"
        )


if __name__ == "__main__":
    main()
