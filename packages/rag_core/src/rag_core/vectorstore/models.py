"""Domain models for vector-store operations."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from rag_core.embeddings import Embedding
from rag_core.vectorstore.exceptions import VectorStoreValidationError

VectorPayload = Mapping[str, object]


def _empty_payload() -> VectorPayload:
    return {}


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """A vector and its payload ready to be stored."""

    id: str
    embedding: Embedding
    payload: VectorPayload = field(default_factory=_empty_payload)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise VectorStoreValidationError("Vector point id is required.")


@dataclass(frozen=True, slots=True)
class VectorSearchQuery:
    """A vector search request."""

    embedding: Embedding
    top_k: int
    filter: VectorPayload | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise VectorStoreValidationError("Vector search top_k must be at least 1.")


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """A scored vector search match."""

    point_id: str
    score: float
    payload: VectorPayload = field(default_factory=_empty_payload)

    def __post_init__(self) -> None:
        if not self.point_id.strip():
            raise VectorStoreValidationError("Vector search result point_id is required.")
