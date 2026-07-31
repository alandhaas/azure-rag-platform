"""Azure Functions entrypoint for the worker application."""

from __future__ import annotations

import json
import logging
import os
import time
from http import HTTPStatus

import azure.functions as func

from rag_worker.observability import (
    REQUEST_ID_HEADER,
    configure_logging,
    request_id_context,
    resolve_request_id,
)

configure_logging(os.getenv("LOG_LEVEL", "INFO"))

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
logger = logging.getLogger("rag_worker.functions")


@app.route(route="health/live", methods=["GET"])
def live_health(req: func.HttpRequest) -> func.HttpResponse:
    """Return a lightweight liveness response for local worker checks."""
    request_id = resolve_request_id(req.headers)
    token = request_id_context.set(request_id)
    started_at = time.perf_counter()

    try:
        response = func.HttpResponse(
            body=json.dumps({"status": "ok"}),
            status_code=HTTPStatus.OK,
            mimetype="application/json",
            headers={REQUEST_ID_HEADER: request_id},
        )
        logger.info(
            "function_completed name=live_health status_code=%s duration_ms=%.2f",
            response.status_code,
            _duration_ms(started_at),
            extra={"request_id": request_id},
        )
        return response
    except Exception:
        logger.exception(
            "function_failed name=live_health duration_ms=%.2f",
            _duration_ms(started_at),
            extra={"request_id": request_id},
        )
        raise
    finally:
        request_id_context.reset(token)


def _duration_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000
