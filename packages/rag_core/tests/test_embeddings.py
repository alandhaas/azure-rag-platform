from collections.abc import Sequence

import pytest
from rag_core.embeddings import Embedding, EmbeddingProvider


class StaticEmbeddingProvider:
    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return [Embedding.from_iterable([float(len(text))]) for text in texts]


def test_embedding_from_iterable_stores_values_as_tuple() -> None:
    embedding = Embedding.from_iterable([0.1, 0.2, 0.3])

    assert embedding.values == (0.1, 0.2, 0.3)


@pytest.mark.asyncio
async def test_embedding_provider_contract_supports_async_batches() -> None:
    provider: EmbeddingProvider = StaticEmbeddingProvider()

    embeddings = await provider.embed(["one", "three"])

    assert [embedding.values for embedding in embeddings] == [(3.0,), (5.0,)]
