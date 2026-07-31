from collections.abc import Sequence
from typing import Any, cast

import httpx
from fastapi.testclient import TestClient
from rag_api.config import ApiSettings
from rag_api.dependencies import get_embedding_provider, get_vector_store
from rag_api.main import create_app
from rag_core.embeddings import Embedding, EmbeddingProvider
from rag_core.vectorstore import VectorSearchQuery, VectorSearchResult, VectorStore


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.texts: Sequence[str] | None = None

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.texts = texts
        return [Embedding.from_iterable([0.1, 0.2, 0.3])]


class FakeVectorStore:
    def __init__(self) -> None:
        self.query: VectorSearchQuery | None = None

    async def upsert(self, points: Sequence[object]) -> None:
        raise NotImplementedError

    async def search(self, query: VectorSearchQuery) -> list[VectorSearchResult]:
        self.query = query
        return [
            VectorSearchResult(
                point_id="chunk-1",
                score=0.91,
                payload={
                    "document_id": "document-1",
                    "chunk_index": 3,
                    "content": "Relevant content",
                },
            )
        ]

    async def delete(self, point_ids: Sequence[str]) -> None:
        raise NotImplementedError


def test_query_retrieval_endpoint_searches_vector_store() -> None:
    embedder = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    client = _test_client(embedder, vector_store)

    response = cast(
        httpx.Response,
        client.post(
            "/queries/retrieval",
            json={
                "text": "What are the risks?",
                "top_k": 3,
                "filter": {"document_id": "document-1"},
            },
        ),
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "point_id": "chunk-1",
                "score": 0.91,
                "payload": {
                    "document_id": "document-1",
                    "chunk_index": 3,
                    "content": "Relevant content",
                },
            }
        ]
    }
    assert embedder.texts == ["What are the risks?"]
    assert vector_store.query == VectorSearchQuery(
        embedding=Embedding.from_iterable([0.1, 0.2, 0.3]),
        top_k=3,
        filter={"document_id": "document-1"},
    )
    assert response.headers["x-request-id"]


def test_query_retrieval_endpoint_validates_text() -> None:
    client = _test_client(FakeEmbeddingProvider(), FakeVectorStore())

    response = cast(httpx.Response, client.post("/queries/retrieval", json={"text": ""}))

    assert response.status_code == 422


def test_query_retrieval_endpoint_validates_top_k() -> None:
    client = _test_client(FakeEmbeddingProvider(), FakeVectorStore())

    response = cast(
        httpx.Response,
        client.post("/queries/retrieval", json={"text": "query", "top_k": 0}),
    )

    assert response.status_code == 422


def test_query_retrieval_endpoint_rejects_unsupported_filter_values() -> None:
    client = _test_client(FakeEmbeddingProvider(), FakeVectorStore())

    response = cast(
        httpx.Response,
        client.post("/queries/retrieval", json={"text": "query", "filter": {"tags": []}}),
    )

    assert response.status_code == 422


def _test_client(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> Any:
    app = create_app(ApiSettings())
    app.dependency_overrides[get_embedding_provider] = lambda: embedding_provider
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    return TestClient(app)
