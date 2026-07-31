from typing import Any

import pytest
from fastapi import Request
from rag_api.config import ApiSettings
from rag_api.dependencies import (
    create_embedding_provider,
    create_vector_store,
    get_api_settings,
)
from rag_api.main import create_app
from rag_core.embeddings import OllamaEmbeddingProvider
from rag_core.vectorstore import QdrantVectorStore


def test_get_api_settings_reads_settings_from_application_state() -> None:
    settings = _settings()
    app = create_app(settings)
    request = _request(app)

    assert get_api_settings(request) is settings


def test_create_embedding_provider_uses_ollama_settings() -> None:
    provider = create_embedding_provider(_settings())

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.config.base_url == "http://ollama.test"
    assert provider.config.embedding_model == "embedding-model"


def test_create_vector_store_uses_qdrant_settings() -> None:
    store = create_vector_store(_settings())

    assert isinstance(store, QdrantVectorStore)
    assert store.config.url == "http://qdrant.test:6333"
    assert store.config.collection_name == "documents"


def test_create_embedding_provider_requires_ollama_settings() -> None:
    with pytest.raises(RuntimeError):
        create_embedding_provider(ApiSettings(OLLAMA_BASE_URL=None, OLLAMA_EMBEDDING_MODEL=None))


def test_create_vector_store_requires_qdrant_settings() -> None:
    with pytest.raises(RuntimeError):
        create_vector_store(ApiSettings(QDRANT_URL=None, QDRANT_COLLECTION_NAME=None))


def _settings() -> ApiSettings:
    return ApiSettings(
        OLLAMA_BASE_URL="http://ollama.test",
        OLLAMA_EMBEDDING_MODEL="embedding-model",
        QDRANT_URL="http://qdrant.test:6333",
        QDRANT_COLLECTION_NAME="documents",
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
