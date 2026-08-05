from typing import Any

from fastapi.testclient import TestClient
from rag_api.config import ApiSettings
from rag_api.dependencies import (
    get_document_ingestion_service,
    get_document_status_repository,
)
from rag_api.main import create_app
from rag_api.services.document_status import DocumentStatusNotFoundError
from rag_api.services.documents import DocumentUploadResult, DocumentUploadValidationError
from rag_core.ingestion import DocumentIngestionStatus


class FakeDocumentIngestionService:
    def __init__(self) -> None:
        self.filename: str | None = None
        self.content: bytes | None = None
        self.content_type: str | None = None
        self.correlation_id: str | None = None

    def upload_pdf(
        self,
        *,
        filename: str | None,
        content: bytes,
        content_type: str | None,
        correlation_id: str,
    ) -> DocumentUploadResult:
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self.correlation_id = correlation_id
        return DocumentUploadResult(
            document_id="doc-123",
            blob_uri="http://127.0.0.1:10000/devstoreaccount1/documents/doc-123/test.pdf",
            queue_name="documents-to-ingest",
            correlation_id=correlation_id,
            status=_queued_status(),
        )


class RejectingDocumentIngestionService(FakeDocumentIngestionService):
    def upload_pdf(
        self,
        *,
        filename: str | None,
        content: bytes,
        content_type: str | None,
        correlation_id: str,
    ) -> DocumentUploadResult:
        raise DocumentUploadValidationError("Only PDF uploads are supported.")


def test_document_upload_endpoint_queues_pdf_with_request_id() -> None:
    service = FakeDocumentIngestionService()
    client = _test_client(service)

    response = client.post(
        "/documents",
        files={"file": ("test.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
        headers={"x-request-id": "request-123"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "document_id": "doc-123",
        "status": "queued",
        "blob_uri": "http://127.0.0.1:10000/devstoreaccount1/documents/doc-123/test.pdf",
        "queue_name": "documents-to-ingest",
        "request_id": "request-123",
    }
    assert service.filename == "test.pdf"
    assert service.content == b"%PDF-1.7\ncontent"
    assert service.content_type == "application/pdf"
    assert service.correlation_id == "request-123"
    assert response.headers["x-request-id"] == "request-123"


def test_document_status_endpoint_returns_current_status() -> None:
    client = _test_client(FakeDocumentIngestionService())

    response = client.get("/documents/doc-123")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-123",
        "status": "indexed",
        "blob_uri": "http://127.0.0.1:10000/devstoreaccount1/documents/doc-123/test.pdf",
        "file_name": "test.pdf",
        "content_type": "application/pdf",
        "queue_name": "documents-to-ingest",
        "request_id": "request-123",
        "created_at": "2026-08-05T20:00:00+00:00",
        "updated_at": "2026-08-05T20:01:00+00:00",
        "chunk_count": 3,
        "error": None,
    }


def test_document_status_endpoint_returns_not_found() -> None:
    client = _test_client(
        FakeDocumentIngestionService(),
        status_repository=MissingStatusRepository(),
    )

    response = client.get("/documents/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document status was not found."}


def test_document_upload_endpoint_returns_bad_request_for_invalid_upload() -> None:
    client = _test_client(RejectingDocumentIngestionService())

    response = client.post(
        "/documents",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF uploads are supported."}


def test_query_embedding_debug_endpoint_is_not_exposed() -> None:
    client = _test_client(FakeDocumentIngestionService())

    response = client.post("/queries/embedding", json={"text": "test"})

    assert response.status_code == 404


class FakeStatusRepository:
    def get(self, document_id: str) -> DocumentIngestionStatus:
        assert document_id == "doc-123"
        return _status()


class MissingStatusRepository:
    def get(self, document_id: str) -> DocumentIngestionStatus:
        raise DocumentStatusNotFoundError(document_id)


def _test_client(service: Any, *, status_repository: Any | None = None) -> Any:
    app = create_app(ApiSettings())
    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    app.dependency_overrides[get_document_status_repository] = (
        lambda: status_repository or FakeStatusRepository()
    )
    return TestClient(app)


def _status() -> DocumentIngestionStatus:
    return DocumentIngestionStatus(
        document_id="doc-123",
        status="indexed",
        blob_uri="http://127.0.0.1:10000/devstoreaccount1/documents/doc-123/test.pdf",
        file_name="test.pdf",
        content_type="application/pdf",
        queue_name="documents-to-ingest",
        correlation_id="request-123",
        created_at="2026-08-05T20:00:00+00:00",
        updated_at="2026-08-05T20:01:00+00:00",
        chunk_count=3,
    )


def _queued_status() -> DocumentIngestionStatus:
    return DocumentIngestionStatus(
        document_id="doc-123",
        status="queued",
        blob_uri="http://127.0.0.1:10000/devstoreaccount1/documents/doc-123/test.pdf",
        file_name="test.pdf",
        content_type="application/pdf",
        queue_name="documents-to-ingest",
        correlation_id="request-123",
        created_at="2026-08-05T20:00:00+00:00",
        updated_at="2026-08-05T20:00:00+00:00",
    )
