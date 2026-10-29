# Operations - Self Improvement Loop

## Running the service
```bash
make run                 # python -m self_improvement_loop serve
docker compose up --build
```
The service trains the policy once at startup, then serves requests. Training is a few seconds
on CPU; readiness is signaled by `GET /ready`.

## Health and readiness
- `GET /health` - liveness and the action set; always cheap.
- `GET /ready` - returns 200 once the policy is trained. Use it as the readiness probe so
  traffic is withheld until startup training completes.

## Observability
- `GET /metrics` exposes Prometheus text: per-route request counts, error counts, summed
  latency, and a latency histogram with buckets at 1, 5, 10, 50, 100, 500, and 1000 ms.
- Suggested alerts: error rate per route above a threshold, and p95 latency regressions on
  `/solve` and `/evaluate`.

## Capacity and limits
- Work per request is bounded by the step budget (`1..12`), so latency is predictable and
  largely constant. The service is stateless after startup and scales horizontally behind a
  load balancer.
- There is no database and no outbound call on the request path.

## Failure modes and responses
- **Startup training slow or failing:** readiness stays false; the orchestrator holds traffic.
- **Malformed or out-of-range input:** returns 400 with a message; not an availability concern.
- **CPU saturation under load:** add a rate limiter and more replicas; requests are independent.

## Rollout
- Deterministic builds (pinned `uv.lock`), CI gates on format, lint, type-check, test, and a
  package and Docker build. Roll forward by redeploying; there is no persistent state to migrate.
