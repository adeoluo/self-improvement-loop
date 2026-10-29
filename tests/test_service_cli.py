"""Tests for the HTTP service routing and the CLI."""

from __future__ import annotations

from self_improvement_loop.cli import main
from self_improvement_loop.self_improvement import SelfImprovementConfig
from self_improvement_loop.service import SolverService, handle

_FAST = SelfImprovementConfig(iterations=2, samples_per_puzzle=6, clone_steps=10, seed=0)


def _service() -> SolverService:
    return SolverService.trained(config=_FAST, train_count=40, test_count=30)


def test_health_and_ready() -> None:
    service = _service()
    status, payload = handle("GET", "/health", {}, service)
    assert status == 200
    assert "+1" in payload["actions"]
    status, payload = handle("GET", "/ready", {}, service)
    assert status == 200
    assert payload["status"] == "ready"


def test_solve_route_returns_trace() -> None:
    service = _service()
    status, payload = handle("POST", "/solve", {"start": 0, "target": 12, "budget": 6}, service)
    assert status == 200
    assert payload["target"] == 12
    assert isinstance(payload["actions"], list)
    assert isinstance(payload["solved"], bool)


def test_solve_validates_input() -> None:
    service = _service()
    for body in (
        {"start": "x", "target": 5},
        {"start": 1, "target": "y"},
        {"start": 1, "target": 5, "budget": 0},
        {"start": 1, "target": 5, "budget": 99},
        {"start": -1, "target": 5},
        {"start": 1, "target": True},
    ):
        status, payload = handle("POST", "/solve", body, service)
        assert status == 400
        assert "error" in payload


def test_evaluate_and_history_routes() -> None:
    service = _service()
    status, payload = handle("GET", "/evaluate", {}, service)
    assert status == 200
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert payload["total"] == 30
    status, payload = handle("GET", "/history", {}, service)
    assert status == 200
    assert payload["iterations"][0]["iteration"] == 0


def test_unknown_route_is_404() -> None:
    status, payload = handle("GET", "/nope", {}, _service())
    assert status == 404
    assert "error" in payload


def test_metrics_recorded_after_handle() -> None:
    service = _service()
    handle("GET", "/evaluate", {}, service)
    rendered = service.metrics.render_prometheus()
    assert "service_requests_total" in rendered
    assert 'route="GET /evaluate"' in rendered


def test_cli_train_evaluate_and_solve(capsys) -> None:  # type: ignore[no-untyped-def]
    args = [
        "--iterations",
        "2",
        "--samples",
        "6",
        "--clone-steps",
        "10",
        "--train-count",
        "40",
        "--test-count",
        "30",
    ]
    main(["train", *args])
    main(["evaluate", *args])
    main(["solve", *args, "0", "12", "--budget", "6"])
    out = capsys.readouterr().out
    assert "Self-improvement history" in out
    assert "Final greedy" in out
    assert "solved:" in out
