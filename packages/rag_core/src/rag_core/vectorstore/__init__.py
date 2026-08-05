"""Vector-store domain models and provider contracts."""

from rag_core.vectorstore.exceptions import (
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreProviderError,
    VectorStoreValidationError,
)
from rag_core.vectorstore.models import (
    VectorPoint,
    VectorSearchQuery,
    VectorSearchResult,
)
from rag_core.vectorstore.protocols import VectorStore
from rag_core.vectorstore.qdrant import (
    QDRANT_API_KEY_ENV,
    QDRANT_COLLECTION_NAME_ENV,
    QDRANT_URL_ENV,
    QDRANT_VECTOR_SIZE_ENV,
    QdrantVectorStore,
    QdrantVectorStoreConfig,
)

__all__ = [
    "QDRANT_API_KEY_ENV",
    "QDRANT_COLLECTION_NAME_ENV",
    "QDRANT_URL_ENV",
    "QDRANT_VECTOR_SIZE_ENV",
    "QdrantVectorStore",
    "QdrantVectorStoreConfig",
    "VectorPoint",
    "VectorSearchQuery",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreConfigurationError",
    "VectorStoreError",
    "VectorStoreProviderError",
    "VectorStoreValidationError",
]
