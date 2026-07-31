"""Correlation helpers shared by worker triggers."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

REQUEST_ID_HEADER = "x-request-id"
TRACEPARENT_HEADER = "traceparent"


def resolve_request_id(
    headers: Mapping[str, str] | None = None,
    fallback: str | None = None,
) -> str:
    """Resolve the request id used to correlate logs and Application Insights traces."""
    request_id = _header_value(headers, REQUEST_ID_HEADER)
    if request_id:
        return request_id

    trace_id = _trace_id_from_traceparent(_header_value(headers, TRACEPARENT_HEADER))
    if trace_id:
        return trace_id

    if fallback:
        return fallback

    return str(uuid4())


def _header_value(headers: Mapping[str, str] | None, name: str) -> str | None:
    if headers is None:
        return None

    for header_name, header_value in headers.items():
        if header_name.lower() == name:
            return header_value
    return None


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
