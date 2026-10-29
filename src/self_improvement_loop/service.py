"""A dependency-free HTTP service exposing the self-trained policy.

At startup the service runs the self-improvement loop once over the bundled train
split and then answers solve and evaluation requests with the resulting policy. A
pure handler is separated from the socket binding so the routing logic is tested
without opening a port.

Endpoints
---------
``GET  /health``    liveness and action set.
``GET  /ready``     readiness once the policy is trained.
``GET  /metrics``   per-route request, error, and latency metrics (Prometheus text).
``POST /solve``     body ``{"start": int, "target": int, "budget": int?}``; returns a
                    greedy solution trace and whether the verifier accepts it.
``GET  /evaluate``  greedy accuracy and solution-length statistics on held-out data.
``GET  /history``   test accuracy per self-improvement iteration.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from self_improvement_loop.data import test_puzzles, train_puzzles
from self_improvement_loop.environment import ACTIONS, Puzzle, verify_trace
from self_improvement_loop.evaluation import IterationReport, evaluate_policy
from self_improvement_loop.generation import greedy_trace
from self_improvement_loop.observability import ServiceMetrics
from self_improvement_loop.policy import Policy
from self_improvement_loop.self_improvement import SelfImprovementConfig, SelfTrainer

JsonObject = dict[str, Any]
_MAX_BUDGET = 12


@dataclass(frozen=True)
class SolverService:
    """Holds the trained policy and answers solve and evaluation requests."""

    policy: Policy
    history: list[IterationReport]
    test_set: list[Puzzle]
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)

    @classmethod
    def trained(
        cls,
        *,
        config: SelfImprovementConfig | None = None,
        train_count: int = 200,
        test_count: int = 120,
    ) -> SolverService:
        cfg = config or SelfImprovementConfig()
        train_set = train_puzzles(train_count, seed=cfg.seed)
        test_set = test_puzzles(test_count, seed=cfg.seed)
        result = SelfTrainer(cfg).train(train_set, test_set)
        return cls(policy=result.policy, history=result.history, test_set=test_set)

    def solve(self, start: int, target: int, budget: int) -> JsonObject:
        puzzle = Puzzle(start=start, target=target, budget=budget)
        trace = greedy_trace(puzzle, self.policy)
        return {
            "start": start,
            "target": target,
            "budget": budget,
            "actions": trace.actions,
            "solved": verify_trace(puzzle, trace.actions),
            "steps": len(trace.actions),
        }

    def evaluate(self) -> JsonObject:
        report = evaluate_policy(self.policy, self.test_set)
        return {
            "accuracy": round(report.accuracy, 6),
            "solved": report.solved,
            "total": report.total,
            "mean_solution_length": round(report.mean_solution_length, 4),
            "mean_optimal_length": round(report.mean_optimal_length, 4),
            "length_overhead": round(report.length_overhead, 4),
        }

    def history_payload(self) -> JsonObject:
        return {
            "iterations": [
                {
                    "iteration": row.iteration,
                    "test_accuracy": round(row.test_accuracy, 6),
                    "solved_by_sampling": row.solved_by_sampling,
                    "recovered_by_hint": row.recovered_by_hint,
                    "training_examples": row.training_examples,
                }
                for row in self.history
            ]
        }


def _route(
    method: str, path: str, body: Mapping[str, Any], service: SolverService
) -> tuple[int, JsonObject]:
    if method == "GET" and path == "/health":
        return 200, {"status": "ok", "actions": list(ACTIONS)}
    if method == "GET" and path == "/evaluate":
        return 200, service.evaluate()
    if method == "GET" and path == "/history":
        return 200, service.history_payload()
    if method == "POST" and path == "/solve":
        parsed = _parse_solve(body)
        if isinstance(parsed, str):
            return 400, {"error": parsed}
        return 200, service.solve(*parsed)
    return 404, {"error": f"no route for {method} {path}"}


def _parse_solve(body: Mapping[str, Any]) -> tuple[int, int, int] | str:
    start = body.get("start")
    target = body.get("target")
    budget = body.get("budget", 6)
    if not isinstance(start, int) or isinstance(start, bool):
        return "'start' must be an integer"
    if not isinstance(target, int) or isinstance(target, bool):
        return "'target' must be an integer"
    if (
        not isinstance(budget, int)
        or isinstance(budget, bool)
        or budget < 1
        or budget > _MAX_BUDGET
    ):
        return f"'budget' must be an integer in 1..{_MAX_BUDGET}"
    if not (0 <= start <= 200) or not (0 <= target <= 200):
        return "'start' and 'target' must be in 0..200"
    return start, target, budget


def handle(
    method: str, path: str, body: Mapping[str, Any], service: SolverService
) -> tuple[int, JsonObject]:
    """Route a request, recording per-route metrics, and return status and payload."""

    if method == "GET" and path == "/ready":
        return 200, {"status": "ready", "actions": list(ACTIONS)}
    start = time.perf_counter()
    status, payload = _route(method, path, body, service)
    service.metrics.observe(
        f"{method} {path}", (time.perf_counter() - start) * 1000.0, error=status >= 400
    )
    return status, payload


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - exercised over sockets
    service: SolverService

    def _respond(self, status: int, payload: JsonObject) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = self.service.metrics.render_prometheus().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        status, payload = handle("GET", self.path, {}, self.service)
        self._respond(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._respond(400, {"error": "request body is not valid JSON"})
            return
        status, payload = handle("POST", self.path, body, self.service)
        self._respond(status, payload)

    def log_message(self, *_: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    """Start the HTTP service (blocking)."""

    handler = type("_BoundHandler", (_Handler,), {"service": SolverService.trained()})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"self-improvement-loop service listening on http://{host}:{port}")
    server.serve_forever()
