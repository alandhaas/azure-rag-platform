"""Azure Table Storage adapter for document ingestion status."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from rag_core.ingestion import DocumentIngestionStatus, utc_now_iso

_PARTITION_KEY = "documents"


class DocumentStatusNotFoundError(LookupError):
    """Raised when a document status row does not exist."""


class _TableClient(Protocol):
    def create_table(self) -> object: ...

    def upsert_entity(self, entity: Mapping[str, object], **kwargs: object) -> object: ...

    def get_entity(self, *, partition_key: str, row_key: str) -> Mapping[str, object]: ...


class DocumentStatusRepository:
    """Read and write document ingestion status rows."""

    def __init__(
        self,
        *,
        connection_string: str,
        table_name: str,
        table_client: _TableClient | None = None,
    ) -> None:
        if not connection_string.strip():
            raise RuntimeError("Storage connection string is required.")
        if not table_name.strip():
            raise RuntimeError("Document status table name is required.")

        self._table_client = table_client or cast(
            _TableClient,
            TableServiceClient.from_connection_string(
                connection_string,
            ).get_table_client(table_name=table_name),
        )

    def create_queued(
        self,
        *,
        document_id: str,
        blob_uri: str,
        file_name: str,
        content_type: str,
        queue_name: str,
        correlation_id: str,
    ) -> DocumentIngestionStatus:
        timestamp = utc_now_iso()
        status = DocumentIngestionStatus(
            document_id=document_id,
            status="queued",
            blob_uri=blob_uri,
            file_name=file_name,
            content_type=content_type,
            queue_name=queue_name,
            correlation_id=correlation_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._ensure_table_exists()
        self._table_client.upsert_entity(
            _status_to_entity(status),
            mode=UpdateMode.REPLACE,
        )
        return status

    def get(self, document_id: str) -> DocumentIngestionStatus:
        try:
            entity = self._table_client.get_entity(
                partition_key=_PARTITION_KEY,
                row_key=document_id,
            )
        except ResourceNotFoundError as exc:
            raise DocumentStatusNotFoundError(document_id) from exc
        return _entity_to_status(entity)

    def _ensure_table_exists(self) -> None:
        try:
            self._table_client.create_table()
        except ResourceExistsError:
            pass


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


def _entity_to_status(entity: Mapping[str, object]) -> DocumentIngestionStatus:
    return DocumentIngestionStatus(
        document_id=_text(entity, "document_id"),
        status=cast(Any, _text(entity, "status")),
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
