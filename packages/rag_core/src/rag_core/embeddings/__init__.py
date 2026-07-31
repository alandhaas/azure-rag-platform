"""Embedding domain models and provider contracts."""

from rag_core.embeddings.config import (
    OLLAMA_BASE_URL_ENV,
    OLLAMA_EMBEDDING_MODEL_ENV,
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
from rag_core.embeddings.models import Embedding
from rag_core.embeddings.ollama import OllamaEmbeddingProvider
from rag_core.embeddings.protocols import EmbeddingProvider
from rag_core.embeddings.validation import (
    batch_texts,
    validate_embedding_response,
    validate_texts,
)

__all__ = [
    "Embedding",
    "EmbeddingBatchSizeError",
    "EmbeddingConfigurationError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingValidationError",
    "EmptyEmbeddingInputError",
    "OllamaEmbeddingConfig",
    "OllamaEmbeddingProvider",
    "OLLAMA_BASE_URL_ENV",
    "OLLAMA_EMBEDDING_MODEL_ENV",
    "batch_texts",
    "validate_embedding_response",
    "validate_texts",
]
