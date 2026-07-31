"""Request logging and correlation middleware."""

import logging
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from rag_api.observability.context import request_id_context

RequestHandler = Callable[[Request], Awaitable[Response]]

REQUEST_ID_HEADER = "x-request-id"
TRACEPARENT_HEADER = "traceparent"

logger = logging.getLogger("rag_api.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Add request ids to logs and responses."""

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        request_id = _resolve_request_id(request)
        token = request_id_context.set(request_id)
        started_at = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = _duration_ms(started_at)
            logger.exception(
                "request_failed method=%s path=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        else:
            duration_ms = _duration_ms(started_at)
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "request_completed method=%s path=%s status_code=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
        finally:
            request_id_context.reset(token)


def _resolve_request_id(request: Request) -> str:
    request_id = request.headers.get(REQUEST_ID_HEADER)
    if request_id:
        return request_id

    trace_id = _trace_id_from_traceparent(request.headers.get(TRACEPARENT_HEADER))
    if trace_id:
        return trace_id

    return str(uuid4())


def _trace_id_from_traceparent(traceparent: str | None) -> str | None:
    if traceparent is None:
        return None

    parts = traceparent.split("-")
    if len(parts) < 4:
        return None

    trace_id = parts[1]
    if len(trace_id) != 32:
        return None

    return trace_id


def _duration_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000
