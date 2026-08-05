"""FastAPI dependency construction for shared RAG services."""

from typing import cast

from fastapi import Request
from rag_core.embeddings import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    OllamaEmbeddingProvider,
)
from rag_core.vectorstore import QdrantVectorStore, VectorStore

from rag_api.config import ApiSettings
from rag_api.services.document_status import DocumentStatusRepository
from rag_api.services.documents import DocumentIngestionService


def get_api_settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def create_embedding_provider(settings: ApiSettings) -> EmbeddingProvider:
    if settings.embedding_provider.lower() == "google":
        return GeminiEmbeddingProvider(settings.gemini_embedding_config())
    if settings.embedding_provider.lower() == "ollama":
        return OllamaEmbeddingProvider(settings.ollama_embedding_config())
    raise RuntimeError(f"Unsupported embedding provider: {settings.embedding_provider}")


def create_vector_store(settings: ApiSettings) -> VectorStore:
    return QdrantVectorStore(settings.qdrant_vector_store_config())


def create_document_ingestion_service(settings: ApiSettings) -> DocumentIngestionService:
    return DocumentIngestionService(
        connection_string=settings.storage_connection_string(),
        container_name=settings.documents_container_name,
        queue_name=settings.ingestion_queue_name,
        status_repository=create_document_status_repository(settings),
    )


def create_document_status_repository(settings: ApiSettings) -> DocumentStatusRepository:
    return DocumentStatusRepository(
        connection_string=settings.storage_connection_string(),
        table_name=settings.document_metadata_table_name,
    )


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return create_embedding_provider(get_api_settings(request))


def get_vector_store(request: Request) -> VectorStore:
    return create_vector_store(get_api_settings(request))


def get_document_ingestion_service(request: Request) -> DocumentIngestionService:
    return create_document_ingestion_service(get_api_settings(request))


def get_document_status_repository(request: Request) -> DocumentStatusRepository:
    return create_document_status_repository(get_api_settings(request))
