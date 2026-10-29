# Threat Model - Self Improvement Loop

## Assets
- The HTTP service's availability and the integrity of its responses.

## Trust boundaries
- The only external input is the JSON body of `POST /solve` and the request path. Everything
  else (puzzles, policy, oracle) is in-process and trusted.

## Entry points and mitigations

### `POST /solve`
- **Malformed JSON** is rejected with 400 before routing.
- **Type and range validation:** `start` and `target` must be integers in `0..200`; `budget`
  must be an integer in `1..12`. Booleans are rejected explicitly (a Python `bool` is an `int`
  subclass). Out-of-range or wrong-type input returns 400 with a message, never a stack trace.
- **Bounded work:** the budget cap bounds the number of decoding steps, so a single request
  cannot force unbounded computation.

### `GET` routes
- `/health`, `/ready`, `/evaluate`, `/history`, and `/metrics` take no user input and return
  fixed-shape payloads. `/metrics` emits Prometheus text and exposes only aggregate counters.

## Non-threats (by construction)
- **No secrets:** the service holds no credentials and makes no outbound calls.
- **No persistence:** state is the in-memory trained policy; there is no database or file write
  on the request path.
- **No code execution:** inputs are integers; there is no evaluation of user-supplied strings.

## Supply chain
- Zero runtime dependencies. Development tooling is pinned via `uv.lock`. CodeQL and dependency
  scanning run in CI.

## Residual risk
- A flood of requests could exhaust CPU; a production deployment would add a rate limiter and
  run behind a reverse proxy, as noted in [`OPERATIONS.md`](OPERATIONS.md).
