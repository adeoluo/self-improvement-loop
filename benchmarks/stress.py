"""A production-style stress benchmark for the solver service.

Trains once, then drives the /solve handler under sustained, concurrent load and
reports throughput with p50, p95, and p99 latency. Deterministic and offline;
run with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

from self_improvement_loop.data import test_puzzles
from self_improvement_loop.self_improvement import SelfImprovementConfig
from self_improvement_loop.service import SolverService, handle


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = SolverService.trained(config=SelfImprovementConfig(), train_count=200, test_count=120)
    rng = random.Random(0)
    puzzles = test_puzzles(120, seed=0)
    requests = [
        {"start": p.start, "target": p.target, "budget": p.budget}
        for p in (rng.choice(puzzles) for _ in range(5000))
    ]

    def one(body: dict[str, object]) -> float:
        start = time.perf_counter()
        handle("POST", "/solve", body, service)
        return (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        latencies = list(pool.map(one, requests))
    total_s = time.perf_counter() - start

    print(f"Requests: {len(requests)} | Workers: 8")
    print(f"Throughput: {len(requests) / total_s:.0f} req/sec")
    print(
        "Latency ms  "
        f"p50 {_percentile(latencies, 0.50):.3f}  "
        f"p95 {_percentile(latencies, 0.95):.3f}  "
        f"p99 {_percentile(latencies, 0.99):.3f}"
    )


if __name__ == "__main__":
    main()
