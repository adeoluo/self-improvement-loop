"""Tests for the environment: actions, verification, and the oracle solver."""

from __future__ import annotations

import pytest

from self_improvement_loop.environment import (
    ACTIONS,
    Puzzle,
    apply_action,
    is_solvable,
    shortest_solution,
    verify_trace,
)


def test_apply_action_covers_all_actions() -> None:
    assert apply_action(10, "+1") == 11
    assert apply_action(10, "+2") == 12
    assert apply_action(10, "+3") == 13
    assert apply_action(10, "-1") == 9
    assert apply_action(10, "-2") == 8
    assert apply_action(10, "-3") == 7
    assert apply_action(10, "*2") == 20
    assert apply_action(10, "*3") == 30


def test_apply_action_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        apply_action(1, "/2")


def test_puzzle_requires_positive_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        Puzzle(start=0, target=1, budget=0)


def test_verify_accepts_correct_trace() -> None:
    puzzle = Puzzle(start=0, target=7, budget=4)
    assert verify_trace(puzzle, ["+3", "*2", "+1"]) is True


def test_verify_rejects_wrong_final_value() -> None:
    puzzle = Puzzle(start=0, target=7, budget=4)
    assert verify_trace(puzzle, ["+3", "+3"]) is False


def test_verify_rejects_over_budget() -> None:
    puzzle = Puzzle(start=0, target=3, budget=1)
    assert verify_trace(puzzle, ["+1", "+1", "+1"]) is False


def test_verify_rejects_unknown_action() -> None:
    puzzle = Puzzle(start=0, target=1, budget=2)
    assert verify_trace(puzzle, ["??"]) is False


def test_verify_rejects_out_of_bounds_intermediate() -> None:
    puzzle = Puzzle(start=100, target=5, budget=3)
    # 100 * 3 = 300 exceeds the representable bound even if we could return later.
    assert verify_trace(puzzle, ["*3"]) is False


def test_empty_trace_solves_only_when_on_target() -> None:
    assert verify_trace(Puzzle(start=5, target=5, budget=1), []) is True
    assert verify_trace(Puzzle(start=5, target=6, budget=1), []) is False


def test_shortest_solution_is_valid_and_minimal() -> None:
    puzzle = Puzzle(start=0, target=12, budget=6)
    solution = shortest_solution(puzzle)
    assert solution is not None
    assert verify_trace(puzzle, solution)
    # No shorter solution exists for reaching 12 from 0 here.
    assert len(solution) <= 4


def test_shortest_solution_handles_already_solved() -> None:
    assert shortest_solution(Puzzle(start=9, target=9, budget=3)) == []


def test_unsolvable_within_budget_returns_none() -> None:
    puzzle = Puzzle(start=0, target=199, budget=1)
    assert shortest_solution(puzzle) is None
    assert is_solvable(puzzle) is False


def test_action_set_is_stable() -> None:
    assert ACTIONS == ("+1", "+2", "+3", "-1", "-2", "-3", "*2", "*3")
