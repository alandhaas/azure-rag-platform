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


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return create_embedding_provider(get_api_settings(request))


def get_vector_store(request: Request) -> VectorStore:
    return create_vector_store(get_api_settings(request))
