"""The reasoning task, its verifier, and an exact oracle solver.

The task is a small, fully verifiable arithmetic reasoning puzzle. From a start
value, an agent applies a sequence of operations drawn from a fixed action set to
reach a target value within a step budget. Every step is deterministic and every
proposed solution can be checked exactly, which is what makes the task usable for
self-training: a trace either reaches the target or it does not.

The environment also exposes a breadth-first oracle that returns a shortest action
sequence to the target. The oracle is never used by the policy during ordinary
generation. It is used only to (a) construct solvable train and test splits and
(b) supply a corrective hint during rationalization, mirroring the hinted-rollout
step of self-taught reasoning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

Action = str

# The fixed action set. Each action maps a current integer value to the next one.
ACTIONS: tuple[Action, ...] = ("+1", "+2", "+3", "-1", "-2", "-3", "*2", "*3")

_VALUE_MIN = 0
_VALUE_MAX = 200


def apply_action(value: int, action: Action) -> int:
    """Return the value that results from applying ``action`` to ``value``."""

    if action == "+1":
        return value + 1
    if action == "+2":
        return value + 2
    if action == "+3":
        return value + 3
    if action == "-1":
        return value - 1
    if action == "-2":
        return value - 2
    if action == "-3":
        return value - 3
    if action == "*2":
        return value * 2
    if action == "*3":
        return value * 3
    raise ValueError(f"unknown action: {action!r}")


@dataclass(frozen=True)
class Puzzle:
    """A single task instance: reach ``target`` from ``start`` within ``budget`` steps."""

    start: int
    target: int
    budget: int

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")


def verify_trace(puzzle: Puzzle, actions: list[Action]) -> bool:
    """Return ``True`` if ``actions`` is a valid, in-budget solution to ``puzzle``.

    A trace is valid when it respects the step budget, keeps every intermediate
    value within representable bounds, uses only known actions, and lands exactly
    on the target. The empty trace solves a puzzle only when it already starts on
    the target.
    """

    if len(actions) > puzzle.budget:
        return False
    value = puzzle.start
    for action in actions:
        if action not in ACTIONS:
            return False
        value = apply_action(value, action)
        if value < _VALUE_MIN or value > _VALUE_MAX:
            return False
    return value == puzzle.target


def shortest_solution(puzzle: Puzzle) -> list[Action] | None:
    """Return a shortest in-bounds action sequence solving ``puzzle``, or ``None``.

    This is an exact breadth-first search over reachable values. It defines
    solvability and supplies rationalization hints; the learned policy never calls
    it during ordinary rollouts.
    """

    if puzzle.start == puzzle.target:
        return []
    start = puzzle.start
    if start < _VALUE_MIN or start > _VALUE_MAX:
        return None
    frontier: deque[int] = deque([start])
    came_from: dict[int, tuple[int, Action]] = {}
    depth = {start: 0}
    while frontier:
        value = frontier.popleft()
        if depth[value] >= puzzle.budget:
            continue
        for action in ACTIONS:
            nxt = apply_action(value, action)
            if nxt < _VALUE_MIN or nxt > _VALUE_MAX or nxt in depth:
                continue
            depth[nxt] = depth[value] + 1
            came_from[nxt] = (value, action)
            if nxt == puzzle.target:
                return _reconstruct(came_from, start, puzzle.target)
            frontier.append(nxt)
    return None


def _reconstruct(came_from: dict[int, tuple[int, Action]], start: int, target: int) -> list[Action]:
    path: list[Action] = []
    node = target
    while node != start:
        prev, action = came_from[node]
        path.append(action)
        node = prev
    path.reverse()
    return path


def is_solvable(puzzle: Puzzle) -> bool:
    """Return whether the puzzle admits an in-budget solution."""

    return shortest_solution(puzzle) is not None
