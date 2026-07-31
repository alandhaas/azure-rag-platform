from collections.abc import Sequence

import pytest
from rag_core.embeddings import (
    Embedding,
    EmbeddingBatchSizeError,
    EmbeddingDimensionError,
    EmbeddingProvider,
    EmbeddingValidationError,
    EmptyEmbeddingInputError,
    batch_texts,
    validate_embedding_response,
    validate_texts,
)


class StaticEmbeddingProvider:
    async def embed(self, texts: Sequence[str]) -> list[Embedding]:
        return [Embedding.from_iterable([float(len(text))]) for text in texts]


def test_embedding_from_iterable_stores_values_as_tuple() -> None:
    embedding = Embedding.from_iterable([0.1, 0.2, 0.3])

    assert embedding.values == (0.1, 0.2, 0.3)
    assert embedding.dimension == 3


def test_embedding_rejects_empty_vectors() -> None:
    with pytest.raises(EmbeddingDimensionError):
        Embedding.from_iterable([])


@pytest.mark.asyncio
async def test_embedding_provider_contract_supports_async_batches() -> None:
    provider: EmbeddingProvider = StaticEmbeddingProvider()

    embeddings = await provider.embed(["one", "three"])

    assert [embedding.values for embedding in embeddings] == [(3.0,), (5.0,)]


def test_validate_texts_returns_tuple_for_non_empty_texts() -> None:
    assert validate_texts(["hello", " world "]) == ("hello", " world ")


@pytest.mark.parametrize("texts", [[], [""], ["   "], ["valid", "\t"]])
def test_validate_texts_rejects_empty_inputs(texts: list[str]) -> None:
    with pytest.raises(EmptyEmbeddingInputError):
        validate_texts(texts)


def test_batch_texts_splits_validated_input() -> None:
    assert batch_texts(["one", "two", "three"], batch_size=2) == [
        ("one", "two"),
        ("three",),
    ]


def test_batch_texts_rejects_invalid_batch_size() -> None:
    with pytest.raises(EmbeddingBatchSizeError):
        batch_texts(["one"], batch_size=0)


def test_validate_embedding_response_returns_valid_embeddings() -> None:
    embeddings = [
        Embedding.from_iterable([0.1, 0.2]),
        Embedding.from_iterable([0.3, 0.4]),
    ]

    validated = validate_embedding_response(
        ["one", "two"],
        embeddings,
        expected_dimension=2,
    )

    assert validated == tuple(embeddings)


def test_validate_embedding_response_rejects_count_mismatch() -> None:
    with pytest.raises(EmbeddingValidationError):
        validate_embedding_response(
            ["one", "two"],
            [Embedding.from_iterable([0.1])],
        )


def test_validate_embedding_response_rejects_dimension_mismatch() -> None:
    with pytest.raises(EmbeddingDimensionError):
        validate_embedding_response(
            ["one", "two"],
            [
                Embedding.from_iterable([0.1, 0.2]),
                Embedding.from_iterable([0.3]),
            ],
        )


def test_validate_embedding_response_rejects_invalid_expected_dimension() -> None:
    with pytest.raises(EmbeddingDimensionError):
        validate_embedding_response(
            ["one"],
            [Embedding.from_iterable([0.1])],
            expected_dimension=0,
        )
