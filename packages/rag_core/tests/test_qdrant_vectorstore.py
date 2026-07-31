from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from qdrant_client import models
from rag_core.embeddings import Embedding
from rag_core.vectorstore import (
    QDRANT_API_KEY_ENV,
    QDRANT_COLLECTION_NAME_ENV,
    QDRANT_URL_ENV,
    QdrantVectorStore,
    QdrantVectorStoreConfig,
    VectorPoint,
    VectorSearchQuery,
    VectorSearchResult,
    VectorStoreConfigurationError,
    VectorStoreProviderError,
    VectorStoreValidationError,
)


@dataclass(frozen=True, slots=True)
class FakeScoredPoint:
    id: str
    score: float
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class FakeQueryResponse:
    points: list[FakeScoredPoint]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.upserted_points: list[models.PointStruct] = []
        self.deleted_ids: list[str] = []
        self.search_filter: models.Filter | None = None
        self.closed = False

    async def upsert(
        self,
        *,
        collection_name: str,
        points: Sequence[models.PointStruct],
    ) -> object:
        self.collection_name = collection_name
        self.upserted_points = list(points)
        return None

    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        query_filter: models.Filter | None,
        limit: int,
        with_payload: bool,
    ) -> object:
        self.collection_name = collection_name
        self.search_vector = query
        self.search_filter = query_filter
        self.search_limit = limit
        self.with_payload = with_payload
        return FakeQueryResponse(
            points=[
                FakeScoredPoint(
                    id="chunk-1",
                    score=0.98,
                    payload={"document_id": "document-1"},
                )
            ]
        )

    async def delete(
        self,
        *,
        collection_name: str,
        points_selector: models.PointIdsList,
    ) -> object:
        self.collection_name = collection_name
        self.deleted_ids = [str(point_id) for point_id in points_selector.points]
        return None

    async def close(self) -> None:
        self.closed = True


def test_qdrant_config_reads_required_environment() -> None:
    env = _qdrant_env()

    config = QdrantVectorStoreConfig.from_env(env)

    assert config.url == env[QDRANT_URL_ENV]
    assert config.collection_name == env[QDRANT_COLLECTION_NAME_ENV]
    assert config.api_key == env[QDRANT_API_KEY_ENV]


@pytest.mark.parametrize(
    "missing_name",
    [QDRANT_URL_ENV, QDRANT_COLLECTION_NAME_ENV],
)
def test_qdrant_config_requires_environment(missing_name: str) -> None:
    env = _qdrant_env()
    env.pop(missing_name)

    with pytest.raises(VectorStoreConfigurationError):
        QdrantVectorStoreConfig.from_env(env)


@pytest.mark.asyncio
async def test_qdrant_vector_store_upserts_points() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(_config(), client=client)

    await store.upsert(
        [
            VectorPoint(
                id="chunk-1",
                embedding=_embedding(),
                payload={"document_id": "document-1"},
            )
        ]
    )

    assert client.collection_name == "collection-name"
    assert client.upserted_points[0].id == "chunk-1"
    assert client.upserted_points[0].vector == [0.1, 0.2, 0.3]
    assert client.upserted_points[0].payload == {"document_id": "document-1"}


@pytest.mark.asyncio
async def test_qdrant_vector_store_searches_with_payload_filter() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(_config(), client=client)

    results = await store.search(
        VectorSearchQuery(
            embedding=_embedding(),
            top_k=3,
            filter={"document_id": "document-1"},
        )
    )

    assert client.search_vector == [0.1, 0.2, 0.3]
    assert client.search_limit == 3
    assert client.with_payload is True
    assert client.search_filter is not None
    assert results == [
        VectorSearchResult(
            point_id="chunk-1",
            score=0.98,
            payload={"document_id": "document-1"},
        )
    ]


@pytest.mark.asyncio
async def test_qdrant_vector_store_deletes_points() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(_config(), client=client)

    await store.delete(["chunk-1", "chunk-2"])

    assert client.deleted_ids == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_qdrant_vector_store_closes_client() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(_config(), client=client)

    await store.close()

    assert client.closed is True


@pytest.mark.asyncio
async def test_qdrant_vector_store_rejects_unsupported_filter_value() -> None:
    store = QdrantVectorStore(_config(), client=FakeQdrantClient())

    with pytest.raises(VectorStoreValidationError):
        await store.search(VectorSearchQuery(embedding=_embedding(), top_k=1, filter={"tags": []}))


@pytest.mark.asyncio
async def test_qdrant_vector_store_rejects_invalid_query_response() -> None:
    class InvalidQdrantClient(FakeQdrantClient):
        async def query_points(
            self,
            *,
            collection_name: str,
            query: list[float],
            query_filter: models.Filter | None,
            limit: int,
            with_payload: bool,
        ) -> object:
            return object()

    store = QdrantVectorStore(_config(), client=InvalidQdrantClient())

    with pytest.raises(VectorStoreProviderError):
        await store.search(VectorSearchQuery(embedding=_embedding(), top_k=1))


def _qdrant_env() -> dict[str, str]:
    return {
        QDRANT_URL_ENV: "http://qdrant.test:6333",
        QDRANT_COLLECTION_NAME_ENV: "collection-name",
        QDRANT_API_KEY_ENV: "api-key",
    }


def _config() -> QdrantVectorStoreConfig:
    return QdrantVectorStoreConfig.from_env(_qdrant_env())


def _embedding() -> Embedding:
    return Embedding.from_iterable([0.1, 0.2, 0.3])
