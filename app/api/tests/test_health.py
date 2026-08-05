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


def test_cors_preflight_allows_configured_origin() -> None:
    client = TestClient(create_app(ApiSettings(CORS_ALLOWED_ORIGINS="https://app.example.com")))

    response = cast(
        httpx.Response,
        cast(Any, client).options(
            "/health/live",
            headers={
                "origin": "https://app.example.com",
                "access-control-request-method": "GET",
            },
        ),
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://app.example.com"
    assert response.headers["access-control-allow-credentials"] == "true"


def _test_client() -> Any:
    return TestClient(create_app(ApiSettings()))
