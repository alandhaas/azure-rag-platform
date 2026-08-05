from typing import Any

import pytest
from fastapi import Request
from rag_api.config import ApiSettings
from rag_api.dependencies import (
    create_document_ingestion_service,
    create_document_status_repository,
    create_embedding_provider,
    create_vector_store,
    get_api_settings,
)
from rag_api.main import create_app
from rag_api.services.document_status import DocumentStatusRepository
from rag_api.services.documents import DocumentIngestionService
from rag_core.embeddings import GeminiEmbeddingProvider, OllamaEmbeddingProvider
from rag_core.vectorstore import QdrantVectorStore


def test_get_api_settings_reads_settings_from_application_state() -> None:
    settings = _settings()
    app = create_app(settings)
    request = _request(app)

    assert get_api_settings(request) is settings


def test_create_embedding_provider_uses_gemini_settings_by_default() -> None:
    provider = create_embedding_provider(_settings())

    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.config.api_key == "test-key"
    assert provider.config.embedding_model == "gemini-embedding-001"
    assert provider.config.output_dimensionality == 768


def test_create_embedding_provider_can_use_ollama_settings() -> None:
    provider = create_embedding_provider(_settings(embedding_provider="ollama"))

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.config.base_url == "http://ollama.test"
    assert provider.config.embedding_model == "embedding-model"


def test_create_vector_store_uses_qdrant_settings() -> None:
    store = create_vector_store(_settings())

    assert isinstance(store, QdrantVectorStore)
    assert store.config.url == "http://qdrant.test:6333"
    assert store.config.collection_name == "documents"


def test_create_document_ingestion_service_uses_storage_settings() -> None:
    service = create_document_ingestion_service(_settings())

    assert isinstance(service, DocumentIngestionService)


def test_create_document_status_repository_uses_storage_settings() -> None:
    repository = create_document_status_repository(_settings())

    assert isinstance(repository, DocumentStatusRepository)


def test_create_embedding_provider_requires_selected_provider_settings() -> None:
    with pytest.raises(RuntimeError):
        create_embedding_provider(ApiSettings(EMBEDDING_PROVIDER="google", GEMINI_API_KEY=None))


def test_create_embedding_provider_rejects_unknown_provider() -> None:
    with pytest.raises(RuntimeError):
        create_embedding_provider(ApiSettings(EMBEDDING_PROVIDER="wat"))


def test_create_vector_store_requires_qdrant_settings() -> None:
    with pytest.raises(RuntimeError):
        create_vector_store(ApiSettings(QDRANT_URL=None, QDRANT_COLLECTION_NAME=None))


def test_create_document_ingestion_service_requires_storage_settings() -> None:
    with pytest.raises(RuntimeError):
        create_document_ingestion_service(ApiSettings(AzureWebJobsStorage=None))


def _settings(*, embedding_provider: str = "google") -> ApiSettings:
    return ApiSettings(
        EMBEDDING_PROVIDER=embedding_provider,
        GEMINI_API_KEY="test-key",
        GEMINI_EMBEDDING_MODEL="gemini-embedding-001",
        GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY=768,
        OLLAMA_BASE_URL="http://ollama.test",
        OLLAMA_EMBEDDING_MODEL="embedding-model",
        QDRANT_URL="http://qdrant.test:6333",
        QDRANT_COLLECTION_NAME="documents",
        AzureWebJobsStorage=(
            "DefaultEndpointsProtocol=http;"
            "AccountName=devstoreaccount1;"
            "AccountKey="
            "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
            "K1SZFPTOtr/KBHBeksoGMGw==;"
            "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
            "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
            "TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
        ),
        DOCUMENT_METADATA_TABLE_NAME="DocumentMetadata",
    )


def _request(app: Any) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "app": app,
        }
    )
