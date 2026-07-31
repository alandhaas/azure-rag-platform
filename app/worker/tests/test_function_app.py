import json
import logging
from importlib import util
from pathlib import Path
from typing import cast

import azure.functions as func
import pytest
from rag_worker.function_app import app, live_health
from rag_worker.observability import request_id_context, resolve_request_id


def test_worker_function_app_imports() -> None:
    assert isinstance(app, func.FunctionApp)


def test_azure_functions_entrypoint_imports_src_package() -> None:
    entrypoint = Path(__file__).resolve().parents[1] / "function_app.py"
    spec = util.spec_from_file_location("worker_function_app_entrypoint", entrypoint)

    assert spec is not None
    assert spec.loader is not None

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert isinstance(module.app, func.FunctionApp)


def test_live_health_returns_ok() -> None:
    req = func.HttpRequest(
        method="GET",
        url="/api/health/live",
        headers={"x-request-id": "test-request-id"},
        body=b"",
    )

    response = live_health(req)

    assert response.status_code == 200
    assert json.loads(response.get_body()) == {"status": "ok"}
    assert response.headers["x-request-id"] == "test-request-id"
    assert request_id_context.get() is None


def test_live_health_generates_request_id() -> None:
    req = func.HttpRequest(
        method="GET",
        url="/api/health/live",
        body=b"",
    )

    response = live_health(req)

    assert response.status_code == 200
    assert response.headers["x-request-id"]


def test_resolve_request_id_uses_traceparent_trace_id() -> None:
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

    request_id = resolve_request_id(
        {
            "traceparent": f"00-{trace_id}-00f067aa0ba902b7-01",
        }
    )

    assert request_id == trace_id


def test_worker_logs_include_request_id(caplog: pytest.LogCaptureFixture) -> None:
    req = func.HttpRequest(
        method="GET",
        url="/api/health/live",
        headers={"x-request-id": "logged-request-id"},
        body=b"",
    )

    with caplog.at_level(logging.INFO, logger="rag_worker.functions"):
        live_health(req)

    request_ids = [
        cast(str | None, getattr(record, "request_id", None)) for record in caplog.records
    ]
    assert "logged-request-id" in request_ids
