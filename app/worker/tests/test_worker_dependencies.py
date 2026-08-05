from __future__ import annotations

import pytest
from rag_core.embeddings import GeminiEmbeddingProvider, OllamaEmbeddingProvider
from rag_worker.config import WorkerSettings
from rag_worker.dependencies import create_embedding_provider


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


def test_create_embedding_provider_rejects_unknown_provider() -> None:
    with pytest.raises(RuntimeError):
        create_embedding_provider(_settings(embedding_provider="wat"))


def _settings(*, embedding_provider: str = "google") -> WorkerSettings:
    return WorkerSettings(
        EMBEDDING_PROVIDER=embedding_provider,
        GEMINI_API_KEY="test-key",
        GEMINI_EMBEDDING_MODEL="gemini-embedding-001",
        GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY=768,
        OLLAMA_BASE_URL="http://ollama.test",
        OLLAMA_EMBEDDING_MODEL="embedding-model",
    )
