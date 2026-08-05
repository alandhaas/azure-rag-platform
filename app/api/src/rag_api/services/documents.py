"""Document ingestion service for API uploads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePath
from typing import Any, Protocol, cast
from uuid import uuid4

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient, TextBase64EncodePolicy
from rag_core.ingestion import DocumentIngestionStatus, IngestionCommand

from rag_api.services.document_status import DocumentStatusRepository


class _BlobClient(Protocol):
    @property
    def url(self) -> str: ...

    def upload_blob(
        self,
        data: bytes,
        *,
        overwrite: bool,
        content_settings: ContentSettings,
        metadata: Mapping[str, str],
    ) -> object: ...


class _ContainerClient(Protocol):
    def list_blobs(self, *, include: list[str]) -> Iterable[object]: ...


class _BlobServiceClient(Protocol):
    def create_container(self, name: str) -> object: ...

    def get_blob_client(self, *, container: str, blob: str) -> _BlobClient: ...

    def get_container_client(self, container: str) -> _ContainerClient: ...


class _QueueClient(Protocol):
    def create_queue(self) -> object: ...

    def send_message(self, content: str) -> object: ...


class DocumentUploadValidationError(ValueError):
    """Raised when an uploaded document cannot be accepted for ingestion."""


@dataclass(frozen=True, slots=True)
class DocumentUploadResult:
    """Result of storing a document and enqueueing ingestion work."""

    document_id: str
    blob_uri: str
    queue_name: str
    correlation_id: str
    status: DocumentIngestionStatus


@dataclass(frozen=True, slots=True)
class StoredDocument:
    """Document file stored in the configured Blob container."""

    document_id: str | None
    blob_name: str
    blob_uri: str
    file_name: str
    content_type: str | None
    size_bytes: int
    created_at: str | None
    updated_at: str | None


class DocumentIngestionService:
    """Store uploaded PDFs and enqueue ingestion commands for the worker."""

    def __init__(
        self,
        *,
        connection_string: str,
        container_name: str,
        queue_name: str,
        status_repository: DocumentStatusRepository,
        blob_service_client: _BlobServiceClient | None = None,
        queue_client: _QueueClient | None = None,
    ) -> None:
        if not connection_string.strip():
            raise RuntimeError("Storage connection string is required.")
        if not container_name.strip():
            raise RuntimeError("Documents container name is required.")
        if not queue_name.strip():
            raise RuntimeError("Ingestion queue name is required.")

        self._container_name = container_name
        self._queue_name = queue_name
        self._status_repository = status_repository
        self._blob_service_client = blob_service_client or cast(
            _BlobServiceClient,
            BlobServiceClient.from_connection_string(connection_string),
        )
        self._queue_client = queue_client or cast(
            _QueueClient,
            QueueClient.from_connection_string(
                connection_string,
                queue_name=queue_name,
                message_encode_policy=TextBase64EncodePolicy(),
            ),
        )

    def upload_pdf(
        self,
        *,
        filename: str | None,
        content: bytes,
        content_type: str | None,
        correlation_id: str,
    ) -> DocumentUploadResult:
        if not correlation_id.strip():
            raise DocumentUploadValidationError("A correlation id is required.")
        if not content:
            raise DocumentUploadValidationError("Uploaded PDF cannot be empty.")
        if not _looks_like_pdf(filename=filename, content=content, content_type=content_type):
            raise DocumentUploadValidationError("Only PDF uploads are supported.")

        document_id = str(uuid4())
        blob_name = f"{document_id}/{_safe_pdf_filename(filename)}"
        blob_client = self._blob_service_client.get_blob_client(
            container=self._container_name,
            blob=blob_name,
        )

        self._ensure_storage_resources()
        blob_client.upload_blob(
            content,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/pdf"),
            metadata={
                "document_id": document_id,
                "correlation_id": correlation_id,
            },
        )

        command = IngestionCommand(
            document_id=document_id,
            blob_uri=blob_client.url,
            correlation_id=correlation_id,
        )
        status = self._status_repository.create_queued(
            document_id=document_id,
            blob_uri=blob_client.url,
            file_name=_safe_pdf_filename(filename),
            content_type="application/pdf",
            queue_name=self._queue_name,
            correlation_id=correlation_id,
        )
        self._queue_client.send_message(command.to_json())

        return DocumentUploadResult(
            document_id=document_id,
            blob_uri=blob_client.url,
            queue_name=self._queue_name,
            correlation_id=correlation_id,
            status=status,
        )

    def _ensure_storage_resources(self) -> None:
        try:
            self._blob_service_client.create_container(self._container_name)
        except ResourceExistsError:
            pass

        try:
            self._queue_client.create_queue()
        except ResourceExistsError:
            pass


class DocumentBlobListingService:
    """List document files stored in the configured Blob container."""

    def __init__(
        self,
        *,
        connection_string: str,
        container_name: str,
        blob_service_client: _BlobServiceClient | None = None,
    ) -> None:
        if not connection_string.strip():
            raise RuntimeError("Storage connection string is required.")
        if not container_name.strip():
            raise RuntimeError("Documents container name is required.")

        self._container_name = container_name
        self._blob_service_client = blob_service_client or cast(
            _BlobServiceClient,
            BlobServiceClient.from_connection_string(connection_string),
        )

    def list_documents(self) -> list[StoredDocument]:
        container_client = self._blob_service_client.get_container_client(self._container_name)

        try:
            blobs = list(container_client.list_blobs(include=["metadata"]))
        except ResourceNotFoundError:
            return []

        return [
            _stored_document_from_blob(
                blob=blob,
                blob_uri=self._blob_service_client.get_blob_client(
                    container=self._container_name,
                    blob=_blob_name(blob),
                ).url,
            )
            for blob in blobs
        ]


def _looks_like_pdf(
    *,
    filename: str | None,
    content: bytes,
    content_type: str | None,
) -> bool:
    has_pdf_name = filename is not None and PurePath(filename).suffix.lower() == ".pdf"
    has_pdf_type = content_type == "application/pdf"
    has_pdf_header = content.startswith(b"%PDF")
    return has_pdf_header and (has_pdf_name or has_pdf_type)


def _safe_pdf_filename(filename: str | None) -> str:
    if filename is None:
        return "document.pdf"

    name = PurePath(filename).name.strip()
    if not name:
        return "document.pdf"
    if PurePath(name).suffix.lower() != ".pdf":
        return f"{name}.pdf"
    return name


def _stored_document_from_blob(*, blob: object, blob_uri: str) -> StoredDocument:
    blob_name = _blob_name(blob)
    metadata = _blob_metadata(blob)
    content_settings = getattr(blob, "content_settings", None)
    content_type = getattr(content_settings, "content_type", None)
    size = getattr(blob, "size", 0)

    return StoredDocument(
        document_id=metadata.get("document_id") or _document_id_from_blob_name(blob_name),
        blob_name=blob_name,
        blob_uri=blob_uri,
        file_name=PurePath(blob_name).name,
        content_type=content_type if isinstance(content_type, str) else None,
        size_bytes=size if isinstance(size, int) else 0,
        created_at=_isoformat_or_none(getattr(blob, "creation_time", None)),
        updated_at=_isoformat_or_none(getattr(blob, "last_modified", None)),
    )


def _blob_name(blob: object) -> str:
    return str(cast(Any, blob).name)


def _blob_metadata(blob: object) -> Mapping[str, str]:
    metadata = getattr(blob, "metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    metadata_map = cast(Mapping[object, object], metadata)
    return {
        str(key): str(value)
        for key, value in metadata_map.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _document_id_from_blob_name(blob_name: str) -> str | None:
    first_segment = PurePath(blob_name).parts[0] if PurePath(blob_name).parts else ""
    if not first_segment.strip():
        return None
    return first_segment


def _isoformat_or_none(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None
