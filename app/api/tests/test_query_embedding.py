from collections.abc import Sequence
from typing import Any, cast

import httpx
from fastapi.testclient import TestClient
from rag_api.config import ApiSettings
from rag_api.dependencies import get_embedding_provider
from rag_api.main import create_app
from rag_core.embeddings import Embedding, EmbeddingProvider


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.texts: Sequence[str] | None = None

    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        self.texts = texts
        return [Embedding.from_iterable([0.1, 0.2, 0.3])]


def test_query_embedding_endpoint_embeds_text() -> None:
    provider = FakeEmbeddingProvider()
    client = _test_client(provider)

    response = cast(
        httpx.Response,
        client.post("/queries/embedding", json={"text": "What is this document about?"}),
    )

    assert response.status_code == 200
    assert response.json() == {
        "embedding": [0.1, 0.2, 0.3],
        "dimension": 3,
    }
    assert provider.texts == ["What is this document about?"]
    assert response.headers["x-request-id"]


def test_query_embedding_endpoint_validates_text() -> None:
    client = _test_client(FakeEmbeddingProvider())

    response = cast(httpx.Response, client.post("/queries/embedding", json={"text": ""}))

    assert response.status_code == 422


def _test_client(provider: EmbeddingProvider) -> Any:
    app = create_app(ApiSettings())
    app.dependency_overrides[get_embedding_provider] = lambda: provider
    return TestClient(app)
