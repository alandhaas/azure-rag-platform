"""Protocols implemented by embedding providers."""

from collections.abc import Sequence
from typing import Protocol

from rag_core.embeddings.models import Embedding


class EmbeddingProvider(Protocol):
    """Async batch embedding interface shared by API and worker code."""

    async def embed(self, texts: Sequence[str]) -> list[Embedding]: ...
