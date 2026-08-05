"""Azure Functions entrypoint for the worker application."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from http import HTTPStatus
from typing import Protocol

import azure.functions as func

from rag_worker.commands import IngestionCommand, IngestionCommandError
from rag_worker.observability import (
    REQUEST_ID_HEADER,
    configure_logging,
    request_id_context,
    resolve_request_id,
)

configure_logging(os.getenv("LOG_LEVEL", "INFO"))

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
logger = logging.getLogger("rag_worker.functions")


class _EmbeddedDocumentResult(Protocol):
    @property
    def document_id(self) -> str: ...

    @property
    def chunks(self) -> Sequence[object]: ...


@app.queue_trigger(  # pyright: ignore[reportUnknownMemberType]
    arg_name="msg",
    queue_name="%INGESTION_QUEUE_NAME%",
    connection="AzureWebJobsStorage",
)
async def ingest_document(msg: func.QueueMessage) -> None:
    """Validate a queue message, then generate embeddings for document chunks."""
    started_at = time.perf_counter()

    try:
        command = IngestionCommand.from_json(msg.get_body())
    except IngestionCommandError:
        logger.exception(
            "function_failed name=ingest_document reason=invalid_command duration_ms=%.2f",
            _duration_ms(started_at),
        )
        raise

    request_id = resolve_request_id(fallback=command.correlation_id)
    token = request_id_context.set(request_id)

    try:
        logger.info(
            "ingestion_command_received document_id=%s blob_uri=%s",
            command.document_id,
            command.blob_uri,
            extra={"request_id": request_id},
        )
        embedded_document = await _run_document_indexing_pipeline(command)
        logger.info(
            "document_chunks_indexed document_id=%s chunk_count=%s",
            embedded_document.document_id,
            len(embedded_document.chunks),
            extra={"request_id": request_id},
        )
        logger.info(
            "function_completed name=ingest_document duration_ms=%.2f",
            _duration_ms(started_at),
            extra={"request_id": request_id},
        )
    except Exception:
        logger.exception(
            "function_failed name=ingest_document duration_ms=%.2f",
            _duration_ms(started_at),
            extra={"request_id": request_id},
        )
        raise
    finally:
        request_id_context.reset(token)


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


async def _run_document_indexing_pipeline(
    command: IngestionCommand,
) -> _EmbeddedDocumentResult:
    from rag_worker.dependencies import create_document_indexing_pipeline

    return await create_document_indexing_pipeline().process(command)
