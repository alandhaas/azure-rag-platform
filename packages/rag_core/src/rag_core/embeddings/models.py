"""Domain models for embeddings."""

from collections.abc import Iterable
from dataclasses import dataclass

from rag_core.embeddings.exceptions import EmbeddingDimensionError


@dataclass(frozen=True, slots=True)
class Embedding:
    """Vector representation returned by an embedding provider."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise EmbeddingDimensionError("Embedding vectors must contain at least one value.")

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> Embedding:
        return cls(values=tuple(values))

    @property
    def dimension(self) -> int:
        return len(self.values)
