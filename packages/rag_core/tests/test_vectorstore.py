from collections.abc import Sequence

import pytest
from rag_core.embeddings import Embedding
from rag_core.vectorstore import (
    VectorPoint,
    VectorSearchQuery,
    VectorSearchResult,
    VectorStore,
    VectorStoreValidationError,
)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, VectorPoint] = {}

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        for point in points:
            self.points[point.id] = point

    async def search(self, query: VectorSearchQuery) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                point_id=point.id,
                score=1.0,
                payload=point.payload,
            )
            for point in list(self.points.values())[: query.top_k]
        ]

    async def delete(self, point_ids: Sequence[str]) -> None:
        for point_id in point_ids:
            self.points.pop(point_id, None)


def test_vector_point_requires_id() -> None:
    with pytest.raises(VectorStoreValidationError):
        VectorPoint(id=" ", embedding=_embedding())


def test_vector_search_query_requires_positive_top_k() -> None:
    with pytest.raises(VectorStoreValidationError):
        VectorSearchQuery(embedding=_embedding(), top_k=0)


def test_vector_search_result_requires_point_id() -> None:
    with pytest.raises(VectorStoreValidationError):
        VectorSearchResult(point_id="", score=0.5)


@pytest.mark.asyncio
async def test_vector_store_contract_supports_upsert_search_and_delete() -> None:
    store: VectorStore = InMemoryVectorStore()
    point = VectorPoint(
        id="chunk-1",
        embedding=_embedding(),
        payload={
            "document_id": "document-1",
            "chunk_index": 0,
            "content": "Example content",
        },
    )

    await store.upsert([point])
    results = await store.search(VectorSearchQuery(embedding=_embedding(), top_k=1))
    await store.delete([point.id])

    assert results == [
        VectorSearchResult(
            point_id="chunk-1",
            score=1.0,
            payload=point.payload,
        )
    ]
    assert store.points == {}


def _embedding() -> Embedding:
    return Embedding.from_iterable([0.1, 0.2, 0.3])
