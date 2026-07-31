from typing import Any, cast

import httpx
from fastapi.testclient import TestClient
from rag_api.config import ApiSettings
from rag_api.main import create_app


def test_liveness_probe_returns_ok() -> None:
    client = _test_client()

    response = cast(httpx.Response, client.get("/health/live"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_readiness_probe_returns_ok() -> None:
    client = _test_client()

    response = cast(httpx.Response, client.get("/health/ready"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def _test_client() -> Any:
    return TestClient(create_app(ApiSettings()))
