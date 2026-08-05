"""Document upload API routes."""

from __future__ import annotations

from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag_api.dependencies import (
    get_document_blob_listing_service,
    get_document_ingestion_service,
    get_document_status_repository,
)
from rag_api.observability.context import request_id_context
from rag_api.services.document_status import (
    DocumentStatusNotFoundError,
    DocumentStatusRepository,
)
from rag_api.services.documents import (
    DocumentBlobListingService,
    DocumentIngestionService,
    DocumentUploadValidationError,
)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str
    blob_uri: str
    queue_name: str
    request_id: str


class StoredDocumentResponse(BaseModel):
    document_id: str | None
    blob_name: str
    blob_uri: str
    file_name: str
    content_type: str | None
    size_bytes: int
    created_at: str | None
    updated_at: str | None


class DocumentListResponse(BaseModel):
    count: int
    request_id: str
    documents: list[StoredDocumentResponse]


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    blob_uri: str
    file_name: str
    content_type: str
    queue_name: str
    request_id: str
    created_at: str
    updated_at: str
    chunk_count: int | None = None
    error: str | None = None


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List stored documents",
    description="Lists all document files currently stored in the configured Blob container.",
)
async def list_documents(
    listing_service: Annotated[
        DocumentBlobListingService,
        Depends(get_document_blob_listing_service),
    ],
) -> DocumentListResponse:
    documents = listing_service.list_documents()
    return DocumentListResponse(
        count=len(documents),
        request_id=request_id_context.get() or "",
        documents=[
            StoredDocumentResponse(
                document_id=document.document_id,
                blob_name=document.blob_name,
                blob_uri=document.blob_uri,
                file_name=document.file_name,
                content_type=document.content_type,
                size_bytes=document.size_bytes,
                created_at=document.created_at,
                updated_at=document.updated_at,
            )
            for document in documents
        ],
    )


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=HTTPStatus.ACCEPTED,
    summary="Upload a PDF for ingestion",
    description="Stores a PDF and queues the worker to extract, chunk, embed, and index it.",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF document to index")],
    ingestion_service: Annotated[
        DocumentIngestionService,
        Depends(get_document_ingestion_service),
    ],
) -> DocumentUploadResponse:
    content = await file.read()
    request_id = request_id_context.get() or ""

    try:
        result = ingestion_service.upload_pdf(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
            correlation_id=request_id,
        )
    except DocumentUploadValidationError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return DocumentUploadResponse(
        document_id=result.document_id,
        status=result.status.status,
        blob_uri=result.blob_uri,
        queue_name=result.queue_name,
        request_id=result.correlation_id,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    summary="Get document ingestion status",
    description="Reads the current ingestion status for an uploaded document.",
)
async def get_document_status(
    document_id: str,
    status_repository: Annotated[
        DocumentStatusRepository,
        Depends(get_document_status_repository),
    ],
) -> DocumentStatusResponse:
    try:
        status = status_repository.get(document_id)
    except DocumentStatusNotFoundError as exc:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Document status was not found.",
        ) from exc

    return DocumentStatusResponse(
        document_id=status.document_id,
        status=status.status,
        blob_uri=status.blob_uri,
        file_name=status.file_name,
        content_type=status.content_type,
        queue_name=status.queue_name,
        request_id=status.correlation_id,
        created_at=status.created_at,
        updated_at=status.updated_at,
        chunk_count=status.chunk_count,
        error=status.error,
    )
