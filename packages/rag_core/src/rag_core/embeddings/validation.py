"""Validation helpers for embedding requests and responses."""

from collections.abc import Sequence

from rag_core.embeddings.exceptions import (
    EmbeddingBatchSizeError,
    EmbeddingDimensionError,
    EmbeddingValidationError,
    EmptyEmbeddingInputError,
)
from rag_core.embeddings.models import Embedding


def validate_texts(texts: Sequence[str]) -> tuple[str, ...]:
    """Return texts as an immutable batch after rejecting empty input."""

    batch = tuple(texts)
    if not batch:
        raise EmptyEmbeddingInputError("At least one text is required for embedding.")

    for index, text in enumerate(batch):
        if not text.strip():
            raise EmptyEmbeddingInputError(f"Text at index {index} is empty.")

    return batch


def batch_texts(texts: Sequence[str], batch_size: int) -> list[tuple[str, ...]]:
    """Split validated texts into fixed-size batches."""

    if batch_size < 1:
        raise EmbeddingBatchSizeError("Embedding batch size must be at least 1.")

    batch = validate_texts(texts)
    return [batch[index : index + batch_size] for index in range(0, len(batch), batch_size)]


def validate_embedding_response(
    texts: Sequence[str],
    embeddings: Sequence[Embedding],
    *,
    expected_dimension: int | None = None,
) -> tuple[Embedding, ...]:
    """Validate provider output count and dimensions against an embedding request."""

    batch = validate_texts(texts)
    response = tuple(embeddings)

    if len(response) != len(batch):
        raise EmbeddingValidationError(
            f"Expected {len(batch)} embeddings, received {len(response)}."
        )

    expected = _resolve_expected_dimension(response, expected_dimension)
    for index, embedding in enumerate(response):
        if embedding.dimension != expected:
            raise EmbeddingDimensionError(
                f"Embedding at index {index} has dimension {embedding.dimension}; "
                f"expected {expected}."
            )

    return response


def _resolve_expected_dimension(
    embeddings: Sequence[Embedding],
    expected_dimension: int | None,
) -> int:
    if expected_dimension is not None:
        if expected_dimension < 1:
            raise EmbeddingDimensionError("Expected embedding dimension must be at least 1.")
        return expected_dimension

    first_embedding = embeddings[0]
    return first_embedding.dimension
