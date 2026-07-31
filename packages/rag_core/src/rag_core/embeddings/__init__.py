"""Embedding domain models and provider contracts."""

from rag_core.embeddings.models import Embedding
from rag_core.embeddings.protocols import EmbeddingProvider

__all__ = ["Embedding", "EmbeddingProvider"]
