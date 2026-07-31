"""Logging configuration for the API application."""

import logging
from typing import Final

from rag_api.observability.context import request_id_context

LOG_FORMAT: Final = (
    "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
)


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get() or "-"
        return True


def configure_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    _ensure_request_id_filter(root_logger)

    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            _ensure_request_id_filter(handler)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _ensure_request_id_filter(handler)
    root_logger.addHandler(handler)


def _ensure_request_id_filter(logger_or_handler: logging.Logger | logging.Handler) -> None:
    has_filter = any(
        isinstance(existing_filter, RequestIdFilter)
        for existing_filter in logger_or_handler.filters
    )
    if not has_filter:
        logger_or_handler.addFilter(RequestIdFilter())
