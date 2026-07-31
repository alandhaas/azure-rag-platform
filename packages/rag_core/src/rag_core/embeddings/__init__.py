"""Embedding domain models and provider contracts."""

from rag_core.embeddings.exceptions import (
    EmbeddingBatchSizeError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingValidationError,
    EmptyEmbeddingInputError,
)
from rag_core.embeddings.models import Embedding
from rag_core.embeddings.protocols import EmbeddingProvider
from rag_core.embeddings.validation import (
    batch_texts,
    validate_embedding_response,
    validate_texts,
)

__all__ = [
    "Embedding",
    "EmbeddingBatchSizeError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingValidationError",
    "EmptyEmbeddingInputError",
    "batch_texts",
    "validate_embedding_response",
    "validate_texts",
]
