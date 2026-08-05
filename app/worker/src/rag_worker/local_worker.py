"""Local queue worker for running ingestion without Azure Functions Core Tools."""

from __future__ import annotations

import asyncio
import base64
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any, NoReturn

from azure.core.exceptions import ResourceExistsError
from azure.storage.queue import QueueClient, TextBase64EncodePolicy

from rag_worker.commands import IngestionCommand, IngestionCommandError
from rag_worker.config import WorkerSettings, get_settings
from rag_worker.dependencies import create_document_indexing_pipeline
from rag_worker.observability import configure_logging, request_id_context, resolve_request_id
from rag_worker.retry import classify_ingestion_failure

logger = logging.getLogger("rag_worker.local")


@dataclass(frozen=True, slots=True)
class LocalQueueMessage:
    """Queue message data used for local ingestion logs and retry handling."""

    message: Any
    message_id: str | None
    dequeue_count: int | None
    content: str


class LocalQueueWorker:
    """Poll an Azure Storage Queue and run document ingestion locally."""

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings
        self._queue_client = QueueClient.from_connection_string(
            settings.storage_connection_string(),
            queue_name=settings.ingestion_queue_name,
        )
        self._poison_queue_client = QueueClient.from_connection_string(
            settings.storage_connection_string(),
            queue_name=f"{settings.ingestion_queue_name}-poison",
            message_encode_policy=TextBase64EncodePolicy(),
        )

    def run_forever(self, *, poll_interval_seconds: float = 2.0) -> NoReturn:
        logger.info(
            "local_worker_started queue_name=%s poll_interval_seconds=%.2f",
            self._settings.ingestion_queue_name,
            poll_interval_seconds,
        )
        _ensure_queue_exists(self._queue_client)
        _ensure_queue_exists(self._poison_queue_client)

        while True:
            processed = self.process_once()
            if not processed:
                time.sleep(poll_interval_seconds)

    def process_once(self) -> bool:
        messages = self._queue_client.receive_messages(
            messages_per_page=1,
            visibility_timeout=30,
        )
        for raw_message in messages:
            message = _to_local_queue_message(raw_message)
            self._process_message(message)
            return True
        return False

    def _process_message(self, message: LocalQueueMessage) -> None:
        started_at = time.perf_counter()

        try:
            command = IngestionCommand.from_json(message.content)
        except IngestionCommandError as exc:
            request_id = resolve_request_id(fallback=message.message_id)
            token = request_id_context.set(request_id)
            try:
                self._handle_failure(
                    message=message,
                    exc=exc,
                    started_at=started_at,
                    request_id=request_id,
                )
            finally:
                request_id_context.reset(token)
            return

        request_id = resolve_request_id(fallback=command.correlation_id)
        token = request_id_context.set(request_id)
        try:
            logger.info(
                "ingestion_command_received document_id=%s blob_uri=%s "
                "message_id=%s dequeue_count=%s",
                command.document_id,
                command.blob_uri,
                message.message_id,
                message.dequeue_count,
                extra={"request_id": request_id},
            )
            embedded_document = asyncio.run(
                create_document_indexing_pipeline(self._settings).process(command)
            )
            logger.info(
                "document_chunks_indexed document_id=%s chunk_count=%s",
                embedded_document.document_id,
                len(embedded_document.chunks),
                extra={"request_id": request_id},
            )
            self._queue_client.delete_message(message.message)
            logger.info(
                "local_worker_message_completed message_id=%s duration_ms=%.2f",
                message.message_id,
                _duration_ms(started_at),
                extra={"request_id": request_id},
            )
        except Exception as exc:
            self._handle_failure(
                message=message,
                exc=exc,
                started_at=started_at,
                request_id=request_id,
            )
        finally:
            request_id_context.reset(token)

    def _handle_failure(
        self,
        *,
        message: LocalQueueMessage,
        exc: BaseException,
        started_at: float,
        request_id: str,
    ) -> None:
        failure = classify_ingestion_failure(exc)
        will_poison = (not failure.retryable) or _will_poison_after_failure(
            message,
            retry_limit=self._settings.worker_retry_limit,
        )
        logger.exception(
            "local_worker_message_failed failure_kind=%s retryable=%s reason=%s "
            "message_id=%s dequeue_count=%s max_dequeue_count=%s "
            "will_poison_after_failure=%s duration_ms=%.2f",
            failure.kind,
            failure.retryable,
            failure.reason,
            message.message_id,
            message.dequeue_count,
            self._settings.worker_retry_limit,
            will_poison,
            _duration_ms(started_at),
            extra={"request_id": request_id},
        )
        if will_poison:
            self._poison_queue_client.send_message(message.content)
            self._queue_client.delete_message(message.message)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    _install_shutdown_logging()
    LocalQueueWorker(settings).run_forever()


def _ensure_queue_exists(queue_client: QueueClient) -> None:
    try:
        queue_client.create_queue()
    except ResourceExistsError:
        pass


def _to_local_queue_message(message: object) -> LocalQueueMessage:
    raw_content = getattr(message, "content", "")
    content = _decode_message_content(str(raw_content))
    return LocalQueueMessage(
        message=message,
        message_id=_non_empty_text(getattr(message, "id", None)),
        dequeue_count=_optional_int(getattr(message, "dequeue_count", None)),
        content=content,
    )


def _decode_message_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{"):
        return stripped
    try:
        return base64.b64decode(stripped).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return stripped


def _install_shutdown_logging() -> None:
    def handle_shutdown(signum: int, frame: object) -> NoReturn:
        logger.info("local_worker_stopped signal=%s", signum)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


def _non_empty_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _will_poison_after_failure(
    message: LocalQueueMessage,
    *,
    retry_limit: int,
) -> bool:
    if message.dequeue_count is None:
        return False
    return message.dequeue_count >= retry_limit


def _duration_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


if __name__ == "__main__":
    main()
