"""Azure Table Storage adapter for worker document status updates."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, Protocol, cast
from urllib.parse import urlparse

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from rag_core.ingestion import DocumentIngestionStatus, IngestionCommand, utc_now_iso

_PARTITION_KEY = "documents"
logger = logging.getLogger("rag_worker.status")


class _TableClient(Protocol):
    def create_table(self) -> object: ...

    def upsert_entity(self, entity: Mapping[str, object], **kwargs: object) -> object: ...

    def get_entity(self, *, partition_key: str, row_key: str) -> Mapping[str, object]: ...


class DocumentStatusStore:
    """Update document ingestion status rows from the worker."""

    def __init__(
        self,
        *,
        connection_string: str,
        table_name: str,
        queue_name: str,
        table_client: _TableClient | None = None,
    ) -> None:
        if not connection_string.strip():
            raise RuntimeError("Storage connection string is required.")
        if not table_name.strip():
            raise RuntimeError("Document status table name is required.")

        self._queue_name = queue_name
        self._table_client = table_client or cast(
            _TableClient,
            TableServiceClient.from_connection_string(
                connection_string,
            ).get_table_client(table_name=table_name),
        )

    def mark_processing(self, command: IngestionCommand) -> None:
        status = self._status_for_update(command).with_update(status="processing")
        self._upsert(status)

    def mark_indexed(self, command: IngestionCommand, *, chunk_count: int) -> None:
        status = self._status_for_update(command).with_update(
            status="indexed",
            chunk_count=chunk_count,
            error=None,
        )
        self._upsert(status)

    def mark_failed(self, command: IngestionCommand, *, error: str) -> None:
        status = self._status_for_update(command).with_update(
            status="failed",
            error=error,
        )
        self._upsert(status)

    def _status_for_update(self, command: IngestionCommand) -> _MutableStatus:
        self._ensure_table_exists()
        try:
            entity = self._table_client.get_entity(
                partition_key=_PARTITION_KEY,
                row_key=command.document_id,
            )
            return _entity_to_status(entity)
        except ResourceNotFoundError:
            timestamp = utc_now_iso()
            return _MutableStatus(
                document_id=command.document_id,
                status="queued",
                blob_uri=command.blob_uri,
                file_name=_file_name(command.blob_uri),
                content_type="application/pdf",
                queue_name=self._queue_name,
                correlation_id=command.correlation_id,
                created_at=timestamp,
                updated_at=timestamp,
            )

    def _upsert(self, status: _MutableStatus) -> None:
        self._table_client.upsert_entity(
            _status_to_entity(status.to_status()),
            mode=UpdateMode.REPLACE,
        )
        logger.info(
            "document_status_updated document_id=%s status=%s chunk_count=%s",
            status.document_id,
            status.status,
            status.chunk_count,
        )

    def _ensure_table_exists(self) -> None:
        try:
            self._table_client.create_table()
        except ResourceExistsError:
            pass


class _MutableStatus:
    def __init__(
        self,
        *,
        document_id: str,
        status: str,
        blob_uri: str,
        file_name: str,
        content_type: str,
        queue_name: str,
        correlation_id: str,
        created_at: str,
        updated_at: str,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        self.document_id = document_id
        self.status = status
        self.blob_uri = blob_uri
        self.file_name = file_name
        self.content_type = content_type
        self.queue_name = queue_name
        self.correlation_id = correlation_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.chunk_count = chunk_count
        self.error = error

    def with_update(
        self,
        *,
        status: str,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> _MutableStatus:
        return _MutableStatus(
            document_id=self.document_id,
            status=status,
            blob_uri=self.blob_uri,
            file_name=self.file_name,
            content_type=self.content_type,
            queue_name=self.queue_name,
            correlation_id=self.correlation_id,
            created_at=self.created_at,
            updated_at=utc_now_iso(),
            chunk_count=chunk_count,
            error=error,
        )

    def to_status(self) -> DocumentIngestionStatus:
        return DocumentIngestionStatus(
            document_id=self.document_id,
            status=cast(Any, self.status),
            blob_uri=self.blob_uri,
            file_name=self.file_name,
            content_type=self.content_type,
            queue_name=self.queue_name,
            correlation_id=self.correlation_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            chunk_count=self.chunk_count,
            error=self.error,
        )


def _status_to_entity(status: DocumentIngestionStatus) -> dict[str, object]:
    entity: dict[str, object] = {
        "PartitionKey": _PARTITION_KEY,
        "RowKey": status.document_id,
        "document_id": status.document_id,
        "status": status.status,
        "blob_uri": status.blob_uri,
        "file_name": status.file_name,
        "content_type": status.content_type,
        "queue_name": status.queue_name,
        "correlation_id": status.correlation_id,
        "created_at": status.created_at,
        "updated_at": status.updated_at,
        "chunk_count": status.chunk_count,
        "error": status.error,
    }
    return {key: value for key, value in entity.items() if value is not None}


def _entity_to_status(entity: Mapping[str, object]) -> _MutableStatus:
    return _MutableStatus(
        document_id=_text(entity, "document_id"),
        status=_text(entity, "status"),
        blob_uri=_text(entity, "blob_uri"),
        file_name=_text(entity, "file_name"),
        content_type=_text(entity, "content_type"),
        queue_name=_text(entity, "queue_name"),
        correlation_id=_text(entity, "correlation_id"),
        created_at=_text(entity, "created_at"),
        updated_at=_text(entity, "updated_at"),
        chunk_count=_optional_int(entity.get("chunk_count")),
        error=_optional_text(entity.get("error")),
    )


def _file_name(blob_uri: str) -> str:
    parsed = urlparse(blob_uri)
    name = PurePosixPath(parsed.path).name
    return name or "document.pdf"


def _text(entity: Mapping[str, object], key: str) -> str:
    value = entity.get(key)
    if isinstance(value, str):
        return value
    return ""


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
