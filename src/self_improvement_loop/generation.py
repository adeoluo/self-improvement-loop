"""Trace generation: sampled rollouts, greedy decoding, and rationalization.

A rollout walks the policy forward from the puzzle's start value, applying actions
until it lands on the target or exhausts the budget. Sampled rollouts explore;
greedy decoding is used for evaluation. Rationalization asks the oracle for a
shortest solution and rolls the policy forward while nudging it toward each oracle
action, which recovers training signal for puzzles the policy cannot yet solve on
its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from self_improvement_loop.environment import Action, Puzzle, apply_action, shortest_solution
from self_improvement_loop.policy import Policy, RandomSource, State


class _Chooser(Protocol):
    def __call__(self, state: State) -> Action: ...


@dataclass(frozen=True)
class Trace:
    """A rollout: the states visited, the actions taken, and whether it solved the puzzle."""

    puzzle: Puzzle
    states: list[State]
    actions: list[Action]
    solved: bool

    def examples(self) -> list[tuple[State, Action]]:
        """Return the state-action pairs for behavior cloning."""

        return list(zip(self.states, self.actions, strict=True))


def _rollout(puzzle: Puzzle, choose: _Chooser) -> Trace:
    value = puzzle.start
    states: list[State] = []
    actions: list[Action] = []
    for step in range(puzzle.budget):
        if value == puzzle.target:
            break
        state: State = (value, puzzle.target, puzzle.budget - step)
        action = choose(state)
        states.append(state)
        actions.append(action)
        value = apply_action(value, action)
    return Trace(puzzle=puzzle, states=states, actions=actions, solved=value == puzzle.target)


def sample_trace(
    puzzle: Puzzle, policy: Policy, rng: RandomSource, *, temperature: float = 1.0
) -> Trace:
    """Roll the policy forward, sampling each action at ``temperature``."""

    return _rollout(puzzle, lambda s: policy.sample_action(s, rng, temperature=temperature))


def greedy_trace(puzzle: Puzzle, policy: Policy) -> Trace:
    """Roll the policy forward, taking the most probable action at each step."""

    return _rollout(puzzle, policy.greedy_action)


def rationalized_trace(puzzle: Puzzle, policy: Policy) -> Trace | None:
    """Return a verified trace that follows the oracle's shortest solution.

    The oracle hint turns an unsolved puzzle into a usable training example. The
    returned trace records the policy's own states paired with the oracle actions,
    so cloning it teaches the policy to reproduce a correct path.
    """

    solution = shortest_solution(puzzle)
    if solution is None:
        return None
    value = puzzle.start
    states: list[State] = []
    actions: list[Action] = []
    for step, action in enumerate(solution):
        state: State = (value, puzzle.target, puzzle.budget - step)
        states.append(state)
        actions.append(action)
        value = apply_action(value, action)
    return Trace(puzzle=puzzle, states=states, actions=actions, solved=value == puzzle.target)
