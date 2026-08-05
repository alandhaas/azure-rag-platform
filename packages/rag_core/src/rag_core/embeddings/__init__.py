"""Embedding domain models and provider contracts."""

from rag_core.embeddings.config import (
    DEFAULT_GEMINI_BASE_URL,
    EMBEDDING_PROVIDER_ENV,
    GEMINI_API_KEY_ENV,
    GEMINI_BASE_URL_ENV,
    GEMINI_EMBEDDING_MODEL_ENV,
    GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV,
    GOOGLE_API_KEY_ENV,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_EMBEDDING_MODEL_ENV,
    GeminiEmbeddingConfig,
    OllamaEmbeddingConfig,
)
from rag_core.embeddings.exceptions import (
    EmbeddingBatchSizeError,
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProviderError,
    EmbeddingValidationError,
    EmptyEmbeddingInputError,
)
from rag_core.embeddings.gemini import GeminiEmbeddingProvider
from rag_core.embeddings.models import Embedding
from rag_core.embeddings.ollama import OllamaEmbeddingProvider
from rag_core.embeddings.protocols import EmbeddingProvider
from rag_core.embeddings.validation import (
    batch_texts,
    validate_embedding_response,
    validate_texts,
)

__all__ = [
    "DEFAULT_GEMINI_BASE_URL",
    "EMBEDDING_PROVIDER_ENV",
    "Embedding",
    "EmbeddingBatchSizeError",
    "EmbeddingConfigurationError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingValidationError",
    "EmptyEmbeddingInputError",
    "GEMINI_API_KEY_ENV",
    "GEMINI_BASE_URL_ENV",
    "GEMINI_EMBEDDING_MODEL_ENV",
    "GEMINI_EMBEDDING_OUTPUT_DIMENSIONALITY_ENV",
    "GOOGLE_API_KEY_ENV",
    "GeminiEmbeddingConfig",
    "GeminiEmbeddingProvider",
    "OllamaEmbeddingConfig",
    "OllamaEmbeddingProvider",
    "OLLAMA_BASE_URL_ENV",
    "OLLAMA_EMBEDDING_MODEL_ENV",
    "batch_texts",
    "validate_embedding_response",
    "validate_texts",
]
