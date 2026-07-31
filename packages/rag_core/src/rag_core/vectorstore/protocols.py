"""Protocols implemented by vector-store adapters."""

from collections.abc import Sequence
from typing import Protocol

from rag_core.vectorstore.models import VectorPoint, VectorSearchQuery, VectorSearchResult


class VectorStore(Protocol):
    """Async vector-store operations shared by API and worker code."""

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def search(self, query: VectorSearchQuery) -> list[VectorSearchResult]: ...

    async def delete(self, point_ids: Sequence[str]) -> None: ...
