# AGENTS.md

Working notes for coding agents and contributors in **Self Improvement Loop**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Rejection-sampling self-training with a verifier, built from scratch: a policy that learns to solve reasoning puzzles from its own verified successes, with no human labels.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/self_improvement_loop/__init__.py`: Self Improvement Loop - recursive self-training with a verifier, from scratch.
- `src/self_improvement_loop/cli.py`: Command line interface for the self-improvement loop.
- `src/self_improvement_loop/data.py`: Deterministic puzzle generation and fixed train and test splits.
- `src/self_improvement_loop/environment.py`: The reasoning task, its verifier, and an exact oracle solver.
- `src/self_improvement_loop/evaluation.py`: Evaluation: greedy accuracy, sampled pass@k, and solution quality.
- `src/self_improvement_loop/generation.py`: Trace generation: sampled rollouts, greedy decoding, and rationalization.
- `src/self_improvement_loop/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/self_improvement_loop/policy.py`: A log-linear policy over actions, trained from scratch by behavior cloning.
- `src/self_improvement_loop/self_improvement.py`: The self-improvement loop: generate, verify, filter, and clone.
- `src/self_improvement_loop/service.py`: A standard-library HTTP service exposing the self-trained policy.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the HTTP service |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
