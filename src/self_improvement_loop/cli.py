"""Command line interface for the self-improvement loop."""

from __future__ import annotations

import argparse

from self_improvement_loop.data import test_puzzles, train_puzzles
from self_improvement_loop.environment import Puzzle, verify_trace
from self_improvement_loop.evaluation import evaluate_policy, pass_at_k
from self_improvement_loop.generation import greedy_trace
from self_improvement_loop.self_improvement import SelfImprovementConfig, SelfTrainer


def _train(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    config = SelfImprovementConfig(
        iterations=args.iterations,
        samples_per_puzzle=args.samples,
        clone_steps=args.clone_steps,
        learning_rate=args.learning_rate,
        rationalize=not args.no_rationalize,
        seed=args.seed,
    )
    train_set = train_puzzles(args.train_count, seed=args.seed)
    test_set = test_puzzles(args.test_count, seed=args.seed)
    return SelfTrainer(config).train(train_set, test_set), test_set


def _cmd_train(args: argparse.Namespace) -> None:
    result, test_set = _train(args)
    print("Self-improvement history (test accuracy by iteration):")
    for row in result.history:
        print(
            f"  iter {row.iteration:>2}: acc={row.test_accuracy:.4f}  "
            f"sampled={row.solved_by_sampling:>3}  hint={row.recovered_by_hint:>3}  "
            f"examples={row.training_examples:>4}"
        )
    report = evaluate_policy(result.policy, test_set)
    print(f"Final greedy: {report.format()}")
    print(f"pass@8: {pass_at_k(result.policy, test_set, k=8, seed=args.seed + 1):.4f}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    result, test_set = _train(args)
    report = evaluate_policy(result.policy, test_set)
    print("Held-out greedy evaluation:")
    print(f"  {report.format()}")
    print(f"  length overhead vs oracle optimum: {report.length_overhead:.3f} steps")


def _cmd_solve(args: argparse.Namespace) -> None:
    result, _ = _train(args)
    puzzle = Puzzle(start=args.start, target=args.target, budget=args.budget)
    trace = greedy_trace(puzzle, result.policy)
    ok = verify_trace(puzzle, trace.actions)
    print(f"Puzzle: {args.start} -> {args.target} in <= {args.budget} steps")
    print(f"  actions: {' '.join(trace.actions) if trace.actions else '(none)'}")
    print(f"  solved:  {ok}")


def _cmd_serve(args: argparse.Namespace) -> None:
    from self_improvement_loop.service import serve

    serve(host=args.host, port=args.port)


def _add_training_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--clone-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--no-rationalize", action="store_true")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="self-improvement-loop", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run the loop and print the accuracy history")
    _add_training_flags(train)

    evaluate = sub.add_parser("evaluate", help="train, then report held-out greedy accuracy")
    _add_training_flags(evaluate)

    solve = sub.add_parser("solve", help="train, then solve a single puzzle greedily")
    _add_training_flags(solve)
    solve.add_argument("start", type=int)
    solve.add_argument("target", type=int)
    solve.add_argument("--budget", type=int, default=6)

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    handlers = {
        "train": _cmd_train,
        "evaluate": _cmd_evaluate,
        "solve": _cmd_solve,
        "serve": _cmd_serve,
    }
    args = parser.parse_args(argv)
    handlers[args.command](args)


if __name__ == "__main__":
    main()
