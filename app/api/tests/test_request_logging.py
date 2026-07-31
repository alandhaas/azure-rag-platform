import logging
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient
from rag_api.main import create_app


def test_request_id_header_is_preserved() -> None:
    client = _test_client()

    response = cast(
        httpx.Response,
        client.get(
            "/health/live",
            headers={"x-request-id": "request-123"},
        ),
    )

    assert response.headers["x-request-id"] == "request-123"


def test_traceparent_trace_id_is_used_as_request_id() -> None:
    client = _test_client()
    trace_id = "0123456789abcdef0123456789abcdef"

    response = cast(
        httpx.Response,
        client.get(
            "/health/live",
            headers={"traceparent": f"00-{trace_id}-0123456789abcdef-01"},
        ),
    )

    assert response.headers["x-request-id"] == trace_id


def test_request_completion_is_logged_with_request_id(caplog: pytest.LogCaptureFixture) -> None:
    client = _test_client()

    with caplog.at_level(logging.INFO, logger="rag_api.requests"):
        client.get("/health/live", headers={"x-request-id": "request-123"})

    records = [
        record
        for record in caplog.records
        if record.name == "rag_api.requests" and "request_completed" in record.message
    ]
    assert records
    assert cast(str, records[0].__dict__["request_id"]) == "request-123"


def _test_client() -> Any:
    return TestClient(create_app())
