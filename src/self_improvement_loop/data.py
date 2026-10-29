"""Deterministic puzzle generation and fixed train and test splits.

Puzzles are sampled with a seeded generator and filtered to those the oracle can
solve within budget, so every instance has a known solution. The train and test
splits are disjoint by construction, which keeps evaluation honest: the policy is
measured on puzzles it never trained on.
"""

from __future__ import annotations

import random

from self_improvement_loop.environment import Puzzle, is_solvable

_START_RANGE = (0, 60)
_TARGET_RANGE = (0, 120)
_DEFAULT_BUDGET = 6


def generate_puzzles(
    count: int,
    *,
    seed: int,
    budget: int = _DEFAULT_BUDGET,
    exclude: frozenset[tuple[int, int]] | None = None,
) -> list[Puzzle]:
    """Return ``count`` distinct solvable puzzles drawn with a seeded generator."""

    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set(exclude or frozenset())
    puzzles: list[Puzzle] = []
    attempts = 0
    while len(puzzles) < count and attempts < count * 200:
        attempts += 1
        start = rng.randint(*_START_RANGE)
        target = rng.randint(*_TARGET_RANGE)
        key = (start, target)
        if key in seen or start == target:
            continue
        puzzle = Puzzle(start=start, target=target, budget=budget)
        if not is_solvable(puzzle):
            continue
        seen.add(key)
        puzzles.append(puzzle)
    if len(puzzles) < count:
        raise RuntimeError(f"could only generate {len(puzzles)} of {count} puzzles")
    return puzzles


def train_puzzles(
    count: int = 240, *, seed: int = 0, budget: int = _DEFAULT_BUDGET
) -> list[Puzzle]:
    """Return the training split."""

    return generate_puzzles(count, seed=seed, budget=budget)


def test_puzzles(count: int = 120, *, seed: int = 0, budget: int = _DEFAULT_BUDGET) -> list[Puzzle]:
    """Return the held-out test split, disjoint from :func:`train_puzzles`."""

    train = train_puzzles(seed=seed, budget=budget)
    exclude = frozenset((p.start, p.target) for p in train)
    return generate_puzzles(count, seed=seed + 1_000_003, budget=budget, exclude=exclude)
