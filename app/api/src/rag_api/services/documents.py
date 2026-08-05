"""Document ingestion service for API uploads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePath
from typing import Protocol, cast
from uuid import uuid4

from azure.core.exceptions import ResourceExistsError
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


class _BlobServiceClient(Protocol):
    def create_container(self, name: str) -> object: ...

    def get_blob_client(self, *, container: str, blob: str) -> _BlobClient: ...


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
