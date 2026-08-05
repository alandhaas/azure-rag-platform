from __future__ import annotations

import json
import logging
from importlib import util
from pathlib import Path
from typing import cast

import azure.functions as func
import pytest
import rag_worker.function_app as function_app
from rag_worker.commands import IngestionCommand, IngestionCommandError
from rag_worker.function_app import app, ingest_document, live_health
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


async def test_ingest_document_validates_logs_and_generates_embeddings(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg = func.QueueMessage(
        id="queue-message-123",
        body=(
            b'{"document_id":"doc-123",'
            b'"blob_uri":"azurite://documents/doc-123.pdf",'
            b'"correlation_id":"queue-request-123"}'
        )
    )
    pipeline_calls: list[str] = []

    async def fake_pipeline(command: IngestionCommand) -> FakeEmbeddedDocument:
        pipeline_calls.append(command.document_id)
        return FakeEmbeddedDocument(document_id=command.document_id, chunks=(object(), object()))

    monkeypatch.setattr(function_app, "_run_document_indexing_pipeline", fake_pipeline)

    with caplog.at_level(logging.INFO, logger="rag_worker.functions"):
        await ingest_document(msg)

    request_ids = [
        cast(str | None, getattr(record, "request_id", None)) for record in caplog.records
    ]
    messages = [record.getMessage() for record in caplog.records]

    assert "queue-request-123" in request_ids
    assert any(
        "ingestion_command_received document_id=doc-123 "
        "blob_uri=azurite://documents/doc-123.pdf "
        "message_id=queue-message-123 dequeue_count=None" in message
        for message in messages
    )
    assert any(
        "document_chunks_indexed document_id=doc-123 chunk_count=2" in message
        for message in messages
    )
    assert pipeline_calls == ["doc-123"]
    assert request_id_context.get() is None


async def test_ingest_document_rejects_invalid_command() -> None:
    msg = func.QueueMessage(body=b"not-json")

    with pytest.raises(IngestionCommandError):
        await ingest_document(msg)

    assert request_id_context.get() is None


async def test_ingest_document_logs_invalid_command_with_retry_classification(
    caplog: pytest.LogCaptureFixture,
) -> None:
    msg = cast(
        func.QueueMessage,
        FakeQueueMessage(
            body=b"not-json",
            message_id="message-invalid",
            dequeue_count=5,
        ),
    )

    with caplog.at_level(logging.ERROR, logger="rag_worker.functions"):
        with pytest.raises(IngestionCommandError):
            await ingest_document(msg)

    messages = [record.getMessage() for record in caplog.records]
    request_ids = [
        cast(str | None, getattr(record, "request_id", None)) for record in caplog.records
    ]

    assert "message-invalid" in request_ids
    assert any(
        "failure_kind=permanent retryable=False reason=IngestionCommandError "
        "message_id=message-invalid dequeue_count=5 max_dequeue_count=5 "
        "will_poison_after_failure=True" in message
        for message in messages
    )
    assert request_id_context.get() is None


async def test_ingest_document_logs_transient_failure_before_retry(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg = cast(
        func.QueueMessage,
        FakeQueueMessage(
            body=(
                b'{"document_id":"doc-123",'
                b'"blob_uri":"azurite://documents/doc-123.txt",'
                b'"correlation_id":"retry-request-123"}'
            ),
            message_id="message-retry",
            dequeue_count=2,
        ),
    )

    class EmbeddingProviderError(Exception):
        pass

    async def failing_pipeline(command: IngestionCommand) -> FakeEmbeddedDocument:
        raise EmbeddingProviderError(command.document_id)

    monkeypatch.setattr(function_app, "_run_document_indexing_pipeline", failing_pipeline)

    with caplog.at_level(logging.ERROR, logger="rag_worker.functions"):
        with pytest.raises(EmbeddingProviderError):
            await ingest_document(msg)

    messages = [record.getMessage() for record in caplog.records]
    request_ids = [
        cast(str | None, getattr(record, "request_id", None)) for record in caplog.records
    ]

    assert "retry-request-123" in request_ids
    assert any(
        "failure_kind=transient retryable=True reason=EmbeddingProviderError "
        "message_id=message-retry dequeue_count=2 max_dequeue_count=5 "
        "will_poison_after_failure=False" in message
        for message in messages
    )
    assert request_id_context.get() is None


class FakeEmbeddedDocument:
    def __init__(self, *, document_id: str, chunks: tuple[object, ...]) -> None:
        self.document_id = document_id
        self.chunks = chunks


class FakeQueueMessage:
    def __init__(
        self,
        *,
        body: bytes,
        message_id: str,
        dequeue_count: int,
    ) -> None:
        self.id = message_id
        self.dequeue_count = dequeue_count
        self._body = body

    def get_body(self) -> bytes:
        return self._body
